"""DFA-free RL must use the same full softmax for collection and replay."""

from types import SimpleNamespace

import pytest
import torch

from ..main import build_parser, _stage2_dfa_enabled
from ..models import FreeDecoding
from ..program import JointReinforcementProgram
from .test_stage1_dfa_loss import joint_fixture
from .test_joint_pipeline import make_planner, TinyController


def make_program(runtime, enabled=False, **kwargs):
    planner = make_planner(runtime)
    planner.free_decoding_min_steps = 7  # a prior evaluation override must not leak into RL
    controller = TinyController()
    program = JointReinforcementProgram(
        runtime, planner, controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.01),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **kwargs: None, dfa_enabled=enabled, **kwargs)
    return program, planner


def forbidden(*args, **kwargs):
    pytest.fail("DFA-free RL called a DFA binding/filter/acceptance path")


def test_stage2_cli_defaults_and_resume_mode_validation():
    parser = build_parser()
    assert parser.parse_args(["train-agent", "--two-stage"]).stage2_dfa == "on"
    assert parser.parse_args(["train-agent", "--two-stage"]).stage2_eai_anchor_weight == 0.5
    args = parser.parse_args(["train-agent", "--two-stage", "--stage2-dfa", "off"])
    assert not _stage2_dfa_enabled(args)
    assert not _stage2_dfa_enabled(args, {"stage": "stage1"})
    assert not _stage2_dfa_enabled(args, {"stage": "stage2", "metrics": {"stage2_dfa": "off"}})
    with pytest.raises(ValueError, match="DFA mode differs"):
        _stage2_dfa_enabled(args, {"stage": "stage2", "metrics": {}})


@pytest.mark.parametrize("domain", ["eai", "vlabench"])
def test_free_rollout_and_replay_match_full_softmax_without_binding(joint_fixture, monkeypatch, domain):
    from test_regr.EmbodiedAgentInterface.utils import constraint_compiler as eai_compiler
    from test_regr.VLABenchAgentInterface import generation_runtime as vla_runtime

    _, runtime = joint_fixture
    program, planner = make_program(runtime)
    assert planner.free_decoding and not program.planner_dfa_enabled
    assert planner.free_decoding_min_steps == 0
    monkeypatch.setattr(runtime, "dfa_for", forbidden)
    monkeypatch.setattr(eai_compiler, "bind_eai_policy", forbidden)
    monkeypatch.setattr(vla_runtime, "bind_vlabench_contextual_dfa", forbidden)
    vocabulary = planner.vocabulary(domain)
    context = {"instruction": "test", "images": (), "entity_table": ("button0", "bowl")}
    policy = program._policy_dfa(domain, context)
    assert isinstance(policy, FreeDecoding)
    assert len(policy.allowed_tokens(policy.start_state)) == vocabulary.label_count

    class RejectEverything:
        allowed_tokens = forbidden

    # Any passed DFA is ignored in free mode, including the facade's binding path.
    planner.eval()
    view = planner.for_domain(domain)
    labels, logprob = view.sample_labels(context, RejectEverything(), max_steps=3, deterministic=True)
    replayed = view.replay_labels_logprob(view.prepare_replay_context(context), labels,
                                         RejectEverything(), max_steps=3)
    torch.testing.assert_close(logprob, replayed)
    assert replayed.requires_grad
    # The fake backbone has the same hidden vector for every input position.
    hidden = planner.model.embedding(torch.tensor(1))
    expected = torch.log_softmax(planner.label_heads[domain](hidden), dim=-1)[labels].sum()
    torch.testing.assert_close(replayed, expected)
    from test_regr.VLABenchAgentInterface.simulation.program import PlannerReplayDecision
    evidence = PlannerReplayDecision(view.prepare_replay_context(context), tuple(labels), policy, max_steps=3)
    transported = evidence.logprob(view)
    torch.testing.assert_close(transported, logprob)
    (-transported).backward()
    assert torch.isfinite(planner.label_heads[domain].weight.grad).all()
    view.generate_plan(**context, dfa=RejectEverything(), max_steps=3)
    view.sample_with_logprob(**context, dfa=RejectEverything(), max_steps=3, return_labels=True)


def test_vlabench_free_mode_skips_pattern_productive_and_acceptance_filters(joint_fixture, monkeypatch):
    from test_regr.VLABenchAgentInterface.simulation import program as simulation

    _, runtime = joint_fixture
    program, _ = make_program(runtime)
    monkeypatch.setattr(runtime, "dfa_for", forbidden)
    monkeypatch.setattr(simulation, "_task_pattern_dfa", forbidden)
    monkeypatch.setattr(simulation, "ProductiveDFA", forbidden)
    monkeypatch.setattr(simulation, "dfa_accepts_plan", forbidden)
    policy = program._sampling_dfa_for_entities(("button0",), expected_pattern=("press",))
    assert isinstance(policy, FreeDecoding)
    monkeypatch.setattr(simulation, "validate_plan", lambda *args, **kw: SimpleNamespace(valid=True))
    monkeypatch.setattr(simulation, "materialize_plan", lambda *args, **kw: object())
    monkeypatch.setattr(simulation, "verify_plan_constraints", lambda *args: None)
    assert program._plan_rejection_reason([], ("button0",), dfa=policy) is None
    # Turning off DFA acceptance never disables schema checks.
    monkeypatch.setattr(simulation, "validate_plan", lambda *args, **kw: SimpleNamespace(valid=False, errors=("bad pointer",)))
    assert program._plan_rejection_reason([], ("button0",), dfa=policy) == "schema:bad pointer"


def test_default_mode_still_applies_dfa_filters(joint_fixture, monkeypatch):
    from test_regr.VLABenchAgentInterface.simulation import program as simulation

    _, runtime = joint_fixture
    program, planner = make_program(runtime, enabled=True)
    assert not planner.free_decoding
    calls = []
    monkeypatch.setattr(program, "_dfa_for_entities", lambda *args, **kw: "base")
    monkeypatch.setattr(simulation, "_task_pattern_dfa", lambda base, *args, **kw: calls.append(base) or "pattern")
    monkeypatch.setattr(simulation, "ProductiveDFA", lambda base: calls.append(base) or "productive")
    assert program._sampling_dfa_for_entities(("button0",), expected_pattern=("press",)) == "productive"
    assert calls == ["base", "pattern"]


def test_eai_rl_updates_with_free_sampling_and_free_replay(joint_fixture, monkeypatch):
    from test_regr.EmbodiedAgentInterface import reward

    examples, runtime = joint_fixture
    program, planner = make_program(runtime, eai_num_samples=2, eai_supervised_weight=0.0)
    monkeypatch.setattr(runtime, "dfa_for", forbidden)
    eos = int(runtime.eai_vocabulary.eos_label)
    action = next(i for i in range(runtime.eai_vocabulary.label_count) if i != eos)
    samples = iter(([action, eos], [eos]))

    def sample(context, policy, **kwargs):
        assert isinstance(policy, FreeDecoding)
        return list(next(samples)), torch.tensor(0.0)

    monkeypatch.setattr(program.eai_planner, "sample_labels_from_context", sample)
    monkeypatch.setattr(reward, "evaluate_goal_satisfaction", lambda *args, **kw: {"recall": 0.0, "is_success": False})
    item = dict(examples[0], reward_function=lambda labels, **kw: float(len(labels) > 1))
    before = planner.label_heads["eai"].weight.detach().clone()
    result = program.train_eai_update(item)
    assert result["samples"] == 2 and result["reward"] == 0.5
    assert not torch.equal(before, planner.label_heads["eai"].weight)


def test_cli_wires_free_stage2_and_records_mode_in_progress_and_epoch_checkpoints(joint_fixture, monkeypatch, tmp_path):
    from .. import main as cli
    from test_regr.VLABenchAgentInterface import dataset_protocol

    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    monkeypatch.setattr(dataset_protocol, "guard_training_dataset", lambda *args, **kw: None)
    monkeypatch.setattr(cli, "_prepare", lambda *args: (
        runtime, examples, examples, {"train": [object()], "validation": []}, {"train": [], "validation": []}))
    monkeypatch.setattr(cli.JointQwenVLPlanner, "from_pretrained", lambda **kw: planner)
    monkeypatch.setattr(cli, "_controller", lambda *args: controller)
    monkeypatch.setattr(cli, "_control_loaders", lambda *args: {"train": [None]})
    monkeypatch.setattr(cli, "load_control_task_instructions", lambda *args: {})
    monkeypatch.setattr(cli, "_factory", lambda *args: lambda **kw: None)

    class Stage2:
        round_robin_cursor = 0

        def __init__(self, *args, dfa_enabled, **kwargs):
            assert dfa_enabled is False
            assert kwargs["eai_supervised_weight"] == 0.0

        def train_alternating_epoch(self, *args, round_callback, **kwargs):
            round_callback(1, {"rounds": 1})
            return {"eai": {"success": 1.0, "reward": 1.0, "goal_recall": 1.0},
                    "vlabench": {"success_rate": 1.0, "successful_task_count": 3, "ik_truncation_rate": 0.0,
                                 "steps": 1.0, "return": 1.0}}

    monkeypatch.setattr(cli, "JointReinforcementProgram", Stage2)
    saved = {}

    def save(path, **kwargs):
        saved[path.name] = kwargs
        return path

    monkeypatch.setattr(cli, "save_joint_checkpoint", save)
    monkeypatch.setattr(cli, "load_joint_checkpoint", lambda path, **kw: saved[path.name])
    args = build_parser().parse_args([
        "train-agent", "--two-stage", "--stage2-dfa", "off", "--device", "cpu", "--output", str(tmp_path),
        "--stage1-epochs", "0", "--stage1-eai-warmup-steps", "0", "--controller-warmup-steps", "0",
        "--stage2-epochs", "1", "--stage2-eval-rollouts-per-task", "0",
        "--stage2-eai-anchor-weight", "0",
    ])
    cli.command_train_agent(args)
    assert {"joint_stage2_progress.pt", "joint_stage2_epoch_000.pt", "joint_stage2_best.pt"}.issubset(saved)
    assert all(payload["metrics"]["stage2_dfa"] == "off" for payload in saved.values())
