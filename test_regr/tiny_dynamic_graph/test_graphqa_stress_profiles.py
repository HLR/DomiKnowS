import json
from dataclasses import replace

import torch
import pytest

from domiknows.graph.executable import LogicDataset

from .to_run_dynamic_graphqa_global_constraints import (
    DEFAULT_CONFIG,
    ProcessResourceMonitor,
    StressConfig,
    build_stress_workload,
    dataset_fingerprint,
    evaluate_workload_predictions,
    global_loss_validation_gate,
    parse_args,
    summarize_item_records,
    train_stress_workload,
    workload_summary,
)


def _tiny_config():
    return replace(
        StressConfig.load(DEFAULT_CONFIG),
        feature_dim=4,
        hidden_dim=8,
        name_concepts=4,
        attribute_concepts=2,
        semantic_concepts=2,
        capability_concepts=2,
        kb_rules=12,
        examples=2,
        objects_per_example=4,
        active_distractors_per_example=2,
        epochs=1,
    )


def test_stress_workload_groups_all_scene_labels_into_one_optimizer_item():
    config = replace(_tiny_config(), examples=8)
    context = build_stress_workload(config, device="cpu")

    assert len(context.entries) == config.examples
    assert workload_summary(config)["optimizer_items_per_epoch"] == config.examples
    assert context.compiled_executable_formulas == 2
    assert context.reused_executable_rows == config.executable_rows - 2
    assert len(context.graph.executableLCs) == 2
    kb_facts = set(context.kb_facts)
    assert kb_facts
    assert all(
        (rule.predicate, rule.source_symbol, rule.target_symbol) in kb_facts
        for rule in context.rule_specs
    )
    assert {rule.projection for rule in context.rule_specs}.issubset({
        "name_to_semantic",
        "attribute_to_semantic",
        "semantic_to_capability",
        "name_to_capability",
    })
    assert all(example.anchor_fact in kb_facts for example in context.examples)
    assert all(
        example.anchor_fact == example.neighborhood_facts[0]
        for example in context.examples
    )
    assert all(
        set(example.neighborhood_facts).issubset(kb_facts)
        for example in context.examples
    )
    assert all(
        len(example.active_concepts)
        == 4 + config.active_distractors_per_example
        for example in context.examples
    )
    for example in context.examples:
        connected_symbols = set(example.anchor_fact[1:])
        for fact in example.neighborhood_facts[1:]:
            assert connected_symbols.intersection(fact[1:])
            connected_symbols.update(fact[1:])
    for _, item in context.entries:
        selected = item[LogicDataset.curr_lc_key]
        assert isinstance(selected, tuple)
        assert len(selected) == 2
        assert all(
            LogicDataset.KEYWORD_FMT.format(lc_name=name) in item for name in selected
        )
        bindings = item[LogicDataset.BINDINGS_KEY]
        assert sorted(len(bindings[name]) for name in selected) == [
            1,
            2 * config.objects_per_example,
        ]

    train_stress_workload(context, device="cpu")
    assert torch.isfinite(context.program.cmodel.last_executable_loss)
    assert torch.isfinite(context.program.cmodel.last_global_loss)
    assert context.graph._active_concepts is None


def test_stress_workload_reports_each_completed_epoch(capsys, tmp_path):
    config = replace(_tiny_config(), epochs=2)
    context = build_stress_workload(config, device="cpu")
    checkpoint_path = tmp_path / "stress-checkpoint.pt"

    train_stress_workload(context, device="cpu", output=checkpoint_path)

    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert [record["epoch"] for record in records] == [1, 2]
    assert all(record["event"] == "epoch_complete" for record in records)
    assert all(record["epochs"] == config.epochs for record in records)
    assert all(record["epoch_seconds"] > 0 for record in records)
    assert all(record["last_executable_loss"] is not None for record in records)
    assert all(record["last_global_loss"] is not None for record in records)
    assert context.epoch_timings == records
    assert [record["epoch"] for record in context.epoch_timings] == [1, 2]
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert list(checkpoint["epoch_timings"]) == records
    assert context.shared_model.forward_calls == config.examples * config.epochs
    assert len(context.item_records) == config.examples * config.epochs
    assert context.evaluation_records["item_summary"][
        "inactive_rule_evaluations"] == 0
    assert context.evaluation_records["gradient_diagnostic"][
        "shared_mlp_gradient_norm"] > 0
    assert context.evaluation_records["global_loss_validation_gate"]["passed"]


def test_global_loss_gate_accepts_satisfied_zeros_and_rejects_skipped_work():
    satisfied_zero = {
        "executable_loss": 1.0,
        "global_loss": 0.0,
        "active_rule_count": 3,
        "compiled_active_rule_count": 3,
    }
    diagnostic = {"global_loss": 2.0, "shared_mlp_gradient_norm": 1.0}

    summary = summarize_item_records([satisfied_zero])
    gate = global_loss_validation_gate(summary, diagnostic)
    assert gate["zero_global_loss_items"] == 1
    assert gate["zero_global_loss_with_no_active_rules"] == 0
    assert gate["zero_global_loss_with_rule_count_mismatch"] == 0
    assert gate["violated_probe_zero_loss"] is False
    assert gate["passed"] is True

    no_active = dict(satisfied_zero, active_rule_count=0,
                     compiled_active_rule_count=0)
    assert global_loss_validation_gate(
        summarize_item_records([no_active]), diagnostic)["passed"] is False

    mismatch = dict(satisfied_zero, compiled_active_rule_count=2)
    assert global_loss_validation_gate(
        summarize_item_records([mismatch]), diagnostic)["passed"] is False

    zero_probe = {"global_loss": 0.0, "shared_mlp_gradient_norm": 0.0}
    assert global_loss_validation_gate(summary, zero_probe)["passed"] is False


def test_synthetic_metrics_have_explicit_exact_set_and_rule_semantics():
    context = build_stress_workload(_tiny_config(), device="cpu")
    with torch.no_grad():
        for parameter in context.shared_model.parameters():
            parameter.zero_()

    metrics = evaluate_workload_predictions(context)

    assert metrics["concept_accuracy"] == 0.5
    assert metrics["miota_exact_set_accuracy"] == 0.0
    assert metrics["kb_rule_satisfaction"] == 1.0
    assert metrics["active_rules"]["min"] > 0


def test_dataset_fingerprint_ignores_global_weight_but_covers_features():
    config = _tiny_config()
    weighted = build_stress_workload(
        replace(config, global_weight=1.0), device="cpu")
    unweighted = build_stress_workload(
        replace(config, global_weight=0.0), device="cpu")

    assert dataset_fingerprint(weighted) == dataset_fingerprint(unweighted)
    original = dataset_fingerprint(unweighted)
    unweighted.examples[0].features[0, 0].add_(1.0)
    assert dataset_fingerprint(unweighted) != original


def test_resource_monitor_and_cli_measurement_overrides():
    with ProcessResourceMonitor("cpu") as monitor:
        _allocation = torch.ones(1024)
    resources = monitor.summary()
    assert resources["peak_cpu_rss_bytes"] > 0
    assert resources["peak_cuda_allocated_bytes"] == 0

    args = parse_args([
        "--global-weight", "0", "--results-json", "results.json",
        "--resume", "checkpoint.pt",
    ])
    assert args.global_weight == 0.0
    assert args.results_json == "results.json"
    assert args.resume == "checkpoint.pt"


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    two_epochs = replace(_tiny_config(), epochs=2)
    uninterrupted = build_stress_workload(two_epochs, device="cpu")
    train_stress_workload(uninterrupted, device="cpu")

    checkpoint_path = tmp_path / "resume.pt"
    first_epoch = build_stress_workload(
        replace(two_epochs, epochs=1), device="cpu")
    train_stress_workload(
        first_epoch, device="cpu", output=checkpoint_path)

    resumed = build_stress_workload(two_epochs, device="cpu")
    train_stress_workload(
        resumed,
        device="cpu",
        output=checkpoint_path,
        resume=checkpoint_path,
    )

    for name, expected in uninterrupted.shared_model.state_dict().items():
        assert torch.equal(resumed.shared_model.state_dict()[name], expected)
    assert [record["epoch"] for record in resumed.epoch_timings] == [1, 2]
    assert len(resumed.item_records) == two_epochs.examples * two_epochs.epochs
    assert resumed.shared_model.forward_calls == (
        two_epochs.examples * two_epochs.epochs)


def test_parameterized_templates_preserve_exact_compilation_training_loss():
    config = replace(_tiny_config(), examples=3)

    exact = build_stress_workload(config, device="cpu", parameterize_executable=False)
    train_stress_workload(exact, device="cpu")
    exact_executable = exact.program.cmodel.last_executable_loss.detach().clone()
    exact_global = exact.program.cmodel.last_global_loss.detach().clone()

    parameterized = build_stress_workload(
        config, device="cpu", parameterize_executable=True
    )
    train_stress_workload(parameterized, device="cpu")

    assert exact.compiled_executable_formulas > 2
    assert parameterized.compiled_executable_formulas == 2
    assert torch.allclose(
        parameterized.program.cmodel.last_executable_loss,
        exact_executable,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        parameterized.program.cmodel.last_global_loss,
        exact_global,
        atol=1e-6,
        rtol=1e-6,
    )


def test_amortized_profile_uses_compiled_per_grounding_dual_critic():
    context = build_stress_workload(
        _tiny_config(), device="cpu", program_profile="primal-dual-amortized"
    )

    assert context.program_profile == "primal-dual-amortized"
    assert context.program.cmodel.compile_lc is True
    assert context.program.cmodel.dual_granularity == "amortized"
    assert context.program.cmodel.dual_critic is not None
    assert set(context.program.cmodel.constr) == {
        rule.lcName for rule in context.rules.values()
    }

    before = (
        context.program.cmodel.dual_critic.constraint_embedding.weight.detach().clone()
    )
    train_stress_workload(context, device="cpu")
    after = context.program.cmodel.dual_critic.constraint_embedding.weight.detach()
    assert not torch.allclose(after, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_amortized_profile_aligns_compiled_features_and_duals_on_cuda():
    context = build_stress_workload(
        _tiny_config(), device="cuda", program_profile="primal-dual-amortized"
    )
    before = (
        context.program.cmodel.dual_critic.constraint_embedding.weight.detach().clone()
    )

    train_stress_workload(context, device="cuda")
    torch.cuda.synchronize()

    after = context.program.cmodel.dual_critic.constraint_embedding.weight.detach()
    assert after.device.type == "cuda"
    assert not torch.allclose(after, before)
