from types import SimpleNamespace

import pytest
import torch

from ..main import build_parser
from ..utils.stage2_training import resolve_parameter_policy, selection_rejections, validate_shared_optimizer
from ..program import JointReinforcementProgram
from ..models import FreeDecoding
from test_regr.VLABenchAgentInterface.simulation.program import (
    JointEpisode, PlannerReplayDecision, VLABenchHierarchicalReinforcementProgram,
)
from .test_joint_pipeline import joint_fixture as base_joint_fixture, make_planner, TinyController


@pytest.fixture
def joint_fixture():
    # Fresh graph ownership for each independent planner/optimizer experiment.
    return base_joint_fixture.__wrapped__()


def args(policy="auto", legacy=None):
    return SimpleNamespace(stage2_parameter_policy=policy, stage2_freeze_shared_backbone=legacy)


def test_new_runs_unfreeze_shared_adapters_and_resume_preserves_policy():
    parsed = build_parser().parse_args(["train-agent", "--two-stage"])
    assert resolve_parameter_policy(parsed) == "freeze_inactive"
    assert resolve_parameter_policy(parsed, {"stage": "stage1"}) == "freeze_inactive"
    assert resolve_parameter_policy(parsed, {"stage": "stage2", "metrics": {}}) == "freeze_shared"
    saved = {"stage": "stage2", "metrics": {"stage2_parameter_policy": "freeze_inactive"}}
    assert resolve_parameter_policy(parsed, saved) == "freeze_inactive"
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        resolve_parameter_policy(args("freeze_shared"), saved)
    assert resolve_parameter_policy(args(legacy=False)) == "freeze_inactive"
    assert resolve_parameter_policy(args(legacy=True)) == "freeze_shared"
    with pytest.raises(ValueError, match="conflicting"):
        resolve_parameter_policy(args("freeze_inactive", True))


def test_optimizer_must_include_trainable_shared_parameters(joint_fixture):
    _, runtime = joint_fixture
    planner = make_planner(runtime)
    optimizer = torch.optim.SGD(planner.label_heads.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="missing trainable"):
        validate_shared_optimizer(planner, optimizer, "freeze_inactive")
    validate_shared_optimizer(planner, optimizer, "freeze_shared")
    planner.model.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable"):
        validate_shared_optimizer(planner, optimizer, "freeze_inactive")


def test_training_rollouts_cannot_select_a_best_checkpoint_even_with_high_success():
    metrics = {"vlabench": {"success_rate": 1.0, "successful_task_count": 10,
                            "ik_truncation_rate": 0.0}, "retention_eligible": True}
    gate = dict(min_success=0.10, min_tasks=3, max_ik=0.25)
    assert selection_rejections(metrics, **gate) == ["fixed_seed_evaluation_missing"]
    metrics["stage2_evaluation"] = {"source": "training_rollouts"}
    assert selection_rejections(metrics, **gate) == ["fixed_seed_evaluation_missing"]
    metrics["stage2_evaluation"] = {"source": "fixed_seed_rollouts"}
    assert selection_rejections(metrics, **gate) == []
    metrics["vlabench"].update(success_rate=0.09, successful_task_count=2, ik_truncation_rate=0.3)
    metrics["retention_eligible"] = False
    assert selection_rejections(metrics, **gate) == ["vlabench_success_below_minimum",
        "successful_task_coverage_below_minimum", "ik_truncation_above_maximum", "baseline_or_retention_gate_failed"]


@pytest.mark.parametrize("field", ["success_rate", "ik_truncation_rate"])
def test_nonfinite_metrics_cannot_pass_selection(field):
    metrics = {"stage2_evaluation": {"source": "fixed_seed_rollouts"},
               "vlabench": {"success_rate": 1., "successful_task_count": 10, "ik_truncation_rate": 0.}}
    metrics["vlabench"][field] = float("nan")
    assert "nonfinite_rollout_metrics" in selection_rejections(metrics, min_success=.1, min_tasks=3, max_ik=.25)


@pytest.mark.parametrize("evaluate", [True, False])
def test_cli_stops_failed_preflight_and_never_selects_from_training_rollouts(
        joint_fixture, tmp_path, monkeypatch, evaluate):
    from .. import main
    from test_regr.VLABenchAgentInterface import dataset_protocol
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    parsed = build_parser().parse_args(["train-agent", "--two-stage", "--stage1-epochs", "0",
        "--stage2-epochs", "1", "--controller-warmup-steps", "0", "--stage1-eai-warmup-steps", "0", "--no-stage2-eval-planning",
        "--task", "select_toy", "--device", "cpu", "--output", str(tmp_path/"run"),
        "--stage2-eval-rollouts-per-task", "1" if evaluate else "0"])
    monkeypatch.setattr(dataset_protocol, "guard_training_dataset", lambda *a, **k: None)
    monkeypatch.setattr(main, "_prepare", lambda *a: (runtime, examples, examples,
        {"train": [], "validation": []}, {"train": [], "validation": []}))
    monkeypatch.setattr(main.JointQwenVLPlanner, "from_pretrained", lambda **k: planner)
    monkeypatch.setattr(main, "_controller", lambda *a: TinyController())
    monkeypatch.setattr(main, "_control_loaders", lambda *a: {"train": [], "validation": []})
    monkeypatch.setattr(main, "load_control_task_instructions", lambda *a: {})
    monkeypatch.setattr(main, "JointSolverPOIProgram", lambda *a, **k: SimpleNamespace(round_robin_cursor=0))
    monkeypatch.setattr(main, "_factory", lambda *a: None)
    monkeypatch.setattr(main, "_evaluate_eai", lambda *a, **k: {"goal_success": 1., "goal_recall": 1.})
    monkeypatch.setattr(main, "save_joint_checkpoint", lambda path, **k: str(path))
    records = []; monkeypatch.setattr(main, "_json", records.append)
    calls = []
    class Stage2:
        def __init__(self, *a, **k):
            assert k["parameter_policy"] == "freeze_inactive"
            self.round_robin_cursor = 0
        def evaluate_rollouts(self, *a, **k):
            return {"success_rate": 0., "positive_return_rate": 0., "ik_truncation_rate": 0.}
        def train_alternating_epoch(self, *a, **k):
            calls.append("train")
            return {"eai": {"success": 1., "reward": 1., "goal_recall": 1.},
                "vlabench": {"success_rate": 1., "successful_task_count": 10,
                             "ik_truncation_rate": 0., "return": 1., "steps": 1.}}
    monkeypatch.setattr(main, "JointReinforcementProgram", Stage2)
    main.command_train_agent(parsed)
    if evaluate:
        assert calls == []
        assert records[-1]["stage"] == "stage2-skipped"
        assert records[-1]["reason"] == "VLABench controller preflight gate"
    else:
        assert calls == ["train"]
        epoch = next(x for x in records if x["stage"] == "stage2")
        assert not epoch["best_candidate_eligible"]
        assert epoch["best_candidate_rejection_reasons"] == ["fixed_seed_evaluation_missing"]
        assert epoch["metrics"]["stage2_parameter_policy"] == "freeze_inactive"
        assert records[-1]["stage"] == "stage2-best-skipped"


@pytest.mark.parametrize("domain", ["eai", "vlabench"])
@pytest.mark.parametrize("policy", ["freeze_inactive", "freeze_shared"])
def test_pure_rl_replay_updates_shared_weights_only_when_policy_allows(
        joint_fixture, monkeypatch, domain, policy):
    examples, runtime = joint_fixture
    torch.manual_seed(42)
    planner = make_planner(runtime)
    planner.model.register_parameter("frozen_base", torch.nn.Parameter(torch.ones(3), requires_grad=False))
    program = JointReinforcementProgram(runtime, planner, TinyController(),
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.05),
        controller_optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01),
        env_factory=lambda **kwargs: None, parameter_policy=policy, dfa_enabled=False,
        eai_supervised_weight=0, supervised_weight=0, controller_bc_weight=0,
        eai_num_samples=2, ppo_epochs=1)
    shared_before = planner.model.embedding.weight.detach().clone()
    inactive = "vlabench" if domain == "eai" else "eai"
    inactive_before = planner.label_heads[inactive].weight.detach().clone()
    active_before = planner.label_heads[domain].weight.detach().clone()
    if domain == "eai":
        eos = runtime.eai_vocabulary.eos_label
        other = next(i for i in range(runtime.eai_vocabulary.label_count) if i != eos)
        trajectories = iter([[eos], [other, eos]])
        monkeypatch.setattr(program.eai_planner, "sample_labels_from_context",
                            lambda *a, **k: (next(trajectories), torch.zeros(())))
        from test_regr.EmbodiedAgentInterface import reward
        monkeypatch.setattr(reward, "evaluate_goal_satisfaction", lambda *a, **k: {"recall": 0., "is_success": False})
        rewards = iter([0., 1.])
        item = {**examples[0], "reward_function": lambda *a, **k: next(rewards)}
        metrics = program.train_eai_update(item)
    else:
        def update(parent, descriptors, **kwargs):
            vocab = runtime.vlabench_vocabulary
            labels = tuple(vocab.label_for_token(t) for t in
                           ["skill:pick", "arg:target_entity_name", "obj:0", "<eos>"])
            decision = PlannerReplayDecision({"instruction": "pick the object", "images": (),
                 "entity_table": ("object",)}, labels, FreeDecoding(vocab.label_count, vocab.eos_label), max_steps=10)
            loss = parent._update_planner([JointEpisode([decision], [], 1., False, True, 1, [1.])])
            return {"planner_loss": loss}
        monkeypatch.setattr(VLABenchHierarchicalReinforcementProgram, "train_joint_epoch", update)
        metrics = program.train_vlabench_update([{"task": "select_toy"}])
    evidence = metrics["planner_update"]
    assert evidence["finite"]
    assert torch.equal(inactive_before, planner.label_heads[inactive].weight)
    assert not torch.equal(active_before, planner.label_heads[domain].weight)
    assert torch.equal(planner.model.frozen_base, torch.ones(3))
    assert planner.model.frozen_base.requires_grad is False
    if policy == "freeze_inactive":
        assert not torch.equal(shared_before, planner.model.embedding.weight)
        assert evidence["shared_gradient_norm"] > 0
        assert evidence["sampled_shared_delta_l2"] > 0
        assert evidence["shared_optimizer_parameters"] == evidence["shared_trainable_parameters"] > 0
    else:
        assert torch.equal(shared_before, planner.model.embedding.weight)
        assert evidence["shared_trainable_parameters"] == 0
        assert evidence["sampled_shared_delta_l2"] == 0
    assert runtime.active_domain is None
