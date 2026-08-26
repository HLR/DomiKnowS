from dataclasses import replace

import torch
import pytest

from domiknows.graph.executable import LogicDataset

from .to_run_dynamic_graphqa_global_constraints import (
    DEFAULT_CONFIG,
    StressConfig,
    build_stress_workload,
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
