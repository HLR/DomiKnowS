import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from test_regr.EmbodiedAgentInterface.observed_reward import (
    prepare_observed_eai_goal,
    score_observed_eai_goal,
)
from test_regr.EmbodiedAgentInterface.world_graph import build_eai_world_graph

from .. import main as cli
from ..utils import constrained_stage2 as pilot
from .test_joint_pipeline import TinyController, make_planner
from .test_stage1_dfa_loss import joint_fixture


@pytest.fixture(scope="module")
def eai_world():
    return build_eai_world_graph()


def sample(goal, init="(closed fridge) (open cabinet)", domain="virtualhome"):
    return {
        "tl_goal": goal,
        "transition_model": f"""(define (problem observed) (:domain {domain})
            (:objects character - character apple_1 apple_2 fridge cabinet lamp_1 lamp_2 - object)
            (:init {init}) (:goal (inside apple_2 fridge)))""",
        "target_action_tokens": ["grab", "apple_2", "putback", "fridge"],
    }


def score(row, labels, world):
    goal = prepare_observed_eai_goal(row, world_bundle=world)
    return score_observed_eai_goal(labels, goal, world_bundle=world)


def test_reference_independent_observed_transition_and_wrong_instance(eai_world):
    row = sample("inside(apple.1, fridge)", init="(open fridge) (containers fridge)")
    valid = ["grab", "apple_1", "putback", "fridge", "<eos>"]
    wrong = ["grab", "apple_2", "putback", "fridge", "<eos>"]
    assert score(row, valid, eai_world)["is_success"]
    assert not score(row, wrong, eai_world)["is_success"]
    assert not score(row, ["grab", "apple_3", "putback", "fridge"], eai_world)["is_success"]
    assert not score(row, ["<eos>"], eai_world)["is_success"]
    edited = dict(row, target_action_tokens=["sleep"], target_action_labels=[0, 1])
    assert score(row, valid, eai_world) == score(edited, valid, eai_world)


def test_observed_initial_state_and_temporal_order(eai_world):
    assert score(sample("open(cabinet)"), ["<eos>"], eai_world)["is_success"]
    row = sample("open(fridge) then open(cabinet)", init="(closed fridge) (closed cabinet)")
    assert score(row, ["open", "fridge", "open", "cabinet"], eai_world)["is_success"]
    assert not score(row, ["open", "cabinet", "open", "fridge"], eai_world)["is_success"]
    row = sample("exists x. (grab(x)) then exists x. (type(x))")
    assert score(row, ["grab", "apple_1", "type", "cabinet"], eai_world)["is_success"]
    assert not score(row, ["type", "cabinet", "grab", "apple_1"], eai_world)["is_success"]


def test_virtualhome_class_only_observations_keep_action_ids_distinct(eai_world):
    row = sample("exists x. (type(x)) then ON(cabinet.17)")
    assert score(row, ["type", "fridge_3", "switchon", "cabinet_17"], eai_world)["is_success"]
    assert not score(row, ["type", "fridge_3", "switchon", "cabinet_27"], eai_world)["is_success"]


def test_behavior_typed_counting_uses_pddl_entities(eai_world):
    row = {
        "tl_goal": "forn 1. x0. (not book_n_02(x0) or inside(x0, carton.66))",
        "transition_model": """(define (problem observed) (:domain igibson)
            (:objects a_n_01_1 - agent book_n_02_1 book_n_02_2 - book_n_02
                carton_n_02_1 - carton_n_02)
            (:init (inside book_n_02_1 carton_n_02_1)) (:goal (open carton_n_02_1)))""",
    }
    assert score(row, [], eai_world)["is_success"]
    row["tl_goal"] = row["tl_goal"].replace("forn 1.", "forn 2.")
    assert not score(row, [], eai_world)["is_success"]


def test_ambiguous_behavior_identity_cannot_satisfy_negated_goal(eai_world):
    row = {
        "tl_goal": "not open(cabinet.100)",
        "transition_model": """(define (problem observed) (:domain igibson)
            (:objects cabinet_n_01_1 cabinet_n_01_2 - cabinet_n_01)
            (:init (open cabinet_n_01_1)))""",
    }
    outcome = score(row, [], eai_world)
    assert not outcome["is_success"] and outcome["unresolved"] == ("cabinet_100",)


def test_eai_dfa_accepts_empty_plan_from_observed_context(joint_fixture):
    _, runtime = joint_fixture
    goal = prepare_observed_eai_goal(sample("open(cabinet)"), world_bundle=runtime.world.eai)
    with runtime.domain_scope("eai"):
        dfa = runtime.dfa_for("eai", goal.policy_context())
        assert dfa.accepts([runtime.eai_vocabulary.eos_label])


def test_joint_cli_uses_only_the_constrained_pilot(joint_fixture, monkeypatch, tmp_path):
    from test_regr.VLABenchAgentInterface import dataset_protocol

    examples, runtime = joint_fixture
    planner, controller = make_planner(runtime), TinyController()
    monkeypatch.setattr(dataset_protocol, "guard_training_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_prepare", lambda *args: (
        runtime, examples, examples, {"train": [], "validation": []}, {"train": [], "validation": []}))
    monkeypatch.setattr(cli.JointQwenVLPlanner, "from_pretrained", lambda **kwargs: planner)
    monkeypatch.setattr(cli, "_controller", lambda *args: controller)
    monkeypatch.setattr(cli, "_control_loaders", lambda *args: {"train": [None]})
    monkeypatch.setattr(cli, "load_control_task_instructions", lambda *args: {})
    monkeypatch.setattr(cli, "_factory", lambda *args: lambda **kwargs: None)
    monkeypatch.setattr(cli, "load_joint_checkpoint", lambda *args, **kwargs: {
        "stage": "controller_warmup", "round_robin_cursor": 0, "metrics": {}})
    calls = []

    class Stage2:
        def __init__(self, *args, **kwargs):
            assert kwargs["dfa_enabled"] and kwargs["vlabench_grammar_only"]
            assert kwargs["parameter_policy"] == "freeze_shared"
            assert kwargs["planner_kl_weight"] == 0
            assert kwargs["execution_assistance"] == "off"

    monkeypatch.setattr(cli, "JointReinforcementProgram", Stage2)
    monkeypatch.setattr(pilot, "run_pilot", lambda stage2, **kwargs: calls.append(kwargs))
    warmup = tmp_path / "joint_controller_warmup.pt"
    args = cli.build_parser().parse_args([
        "train-agent", "--two-stage", "--resume", str(warmup), "--device", "cpu",
        "--output", str(tmp_path), "--stage2-planner-update", "constrained-es",
        "--stage2-training-seed", "43",
    ])
    cli.command_train_agent(args)
    assert calls[0]["mode"] == "constrained-es" and calls[0]["seed"] == 43
    assert calls[0]["initialization"] == warmup and calls[0]["resume"] is None
    assert len(calls[0]["evaluation_descriptors"]) == 10


def test_es_matches_z_scored_mirrored_update_and_isolates_active_head():
    active, untouched = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1)
    with torch.no_grad():
        active.weight.zero_()
    frozen = copy.deepcopy(untouched.state_dict())
    calls = []

    def rollout(case, seed, replay):
        calls.append((case, seed, active.weight.item()))
        return pilot.Rollout(active.weight.item() > 0)

    metrics = pilot.es_generation(active, [0, 1], rollout, 42, progress=lambda message: None)
    assert len(calls) == 16 and metrics["fitness_std"] == 0.5
    for first, second in zip(calls[::4], calls[2::4]):
        assert first[:2] == second[:2]
        assert first[2] == -second[2]
    directions = [abs(next(pilot._noise(tuple(active.parameters()), seed)).item())
                  for seed in metrics["noise_seeds"]]
    assert active.weight.item() == pytest.approx(pilot.STEP_SIZE * sum(directions) / 4)
    assert all(torch.equal(current, frozen[name]) for name, current in untouched.state_dict().items())


def test_es_restores_center_after_failure_and_tied_population():
    head = torch.nn.Linear(1, 2)
    before = copy.deepcopy(head.state_dict())

    def fail(*args):
        raise RuntimeError("rollout failed")

    with pytest.raises(RuntimeError, match="rollout failed"):
        pilot.es_generation(head, [0], fail, 42, progress=lambda message: None)
    tied = pilot.es_generation(head, [0], lambda *args: pilot.Rollout(False), 42, progress=lambda message: None)
    assert tied["tied"] and not tied["updated"]
    assert all(torch.equal(current, before[name]) for name, current in head.state_dict().items())


def test_reinforce_uses_binary_advantages_for_active_head():
    head = torch.nn.Linear(1, 1, bias=False)
    before = head.weight.item()
    outcomes = iter([True, False] * 4)

    def rollout(case, seed, replay):
        success = next(outcomes)
        return pilot.Rollout(success, (lambda: head.weight.sum() * (1 if success else -1),))

    metrics = pilot.reinforce_generation(head, [0], rollout, 42, progress=lambda message: None)
    assert metrics["updated"]
    assert head.weight.item() == pytest.approx(before + pilot.STEP_SIZE / 2)
    assert head.weight.grad is None


def test_reinforce_updates_when_candidate_fitness_ties_across_resets():
    head = torch.nn.Linear(1, 1, bias=False)
    before = head.weight.item()
    per_reset = [True] * 4 + [False] * 4
    draws = iter(per_reset + [not value for value in per_reset])

    def rollout(case, seed, replay):
        success = next(draws)
        return pilot.Rollout(success, (lambda: head.weight.sum() * (1 if success else -1),))

    metrics = pilot.reinforce_generation(head, [0, 1], rollout, 42, progress=lambda _: None)
    assert metrics["aggregate_fitness_tied"] and not metrics["tied"]
    assert metrics["mixed_resets"] == 2 and metrics["updated"]
    assert head.weight.item() == pytest.approx(before + pilot.STEP_SIZE / 2)


def test_pilot_resume_runs_only_remaining_generation_seeds(tmp_path, monkeypatch):
    planner = torch.nn.Module()
    planner.label_heads = torch.nn.ModuleDict({domain: torch.nn.Linear(1, 1) for domain in pilot.DOMAINS})
    initial_weights = copy.deepcopy(planner.state_dict())
    stage2 = SimpleNamespace(joint_planner=planner, controller=torch.nn.Linear(1, 1),
                             _execution_assistance_override=None,
                             joint_runtime=SimpleNamespace(world=SimpleNamespace(eai=None)))
    baseline = {"eai": {"success": 0.5, "goal_recall": 0.5, "successes": 44},
                "vlabench": {"success_rate": 0.1, "successes": 10, "successful_task_count": 3,
                             "positive_return_rate": 0.2, "ik_truncation_rate": 0.1},
                "stage2_evaluation": {"source": "fixed_seed_rollouts"}}
    monkeypatch.setattr(pilot, "_evaluation", lambda *args, **kwargs: copy.deepcopy(baseline))
    monkeypatch.setattr(pilot, "_eai_cases", lambda items, world: [
        (item, SimpleNamespace(domain="virtualhome", unresolved=())) for item in items])
    monkeypatch.setattr(pilot, "_training_cohorts", lambda *args: [
        {"eai": [], "vlabench": []} for _ in range(pilot.GENERATIONS)])
    visited = []
    monkeypatch.setattr(pilot, "_active_head", lambda stage2, domain: nullcontext(
        stage2.joint_planner.label_heads[domain]))

    def update(head, cohort, rollout, seed, **kwargs):
        visited.append(seed)
        with torch.no_grad():
            head.weight.add_(float(seed % 47))
        return {"tied": False, "updated": True}

    monkeypatch.setattr(pilot, "es_generation", update)
    initialization = tmp_path / "init.pt"
    initialization.write_bytes(b"warm checkpoint")
    options = dict(mode="constrained-es", seed=42, eai_train=[], eai_valid=[{}] * 88, descriptors=[],
                   evaluation_descriptors=[{}] * 10, output=tmp_path, settings={},
                   progress=lambda message: None)
    saved = {}

    def interrupt(path, state):
        if len(state["history"]) == 1:
            saved.update(copy.deepcopy(state))
            raise RuntimeError("interrupted after save")

    with pytest.raises(RuntimeError, match="interrupted"):
        pilot.run_pilot(stage2, **options, initialization=initialization, resume=None, save_checkpoint=interrupt)
    assert len(visited) == 1
    expected = saved["next_seed"]
    pilot.run_pilot(stage2, **options, initialization=None, resume=saved,
                    save_checkpoint=lambda path, state: torch.save(state, path))
    resumed = copy.deepcopy(planner.state_dict())
    assert visited[1] == expected
    planner.load_state_dict(initial_weights)
    pilot.run_pilot(stage2, **options, initialization=initialization, resume=None,
                    save_checkpoint=lambda path, state: torch.save(state, path))
    assert visited[:4] == visited[4:]
    for key, value in planner.state_dict().items():
        torch.testing.assert_close(value, resumed[key], atol=0, rtol=0)


def test_promotion_requires_paired_replication_and_source_gates():
    reports = []
    for seed in (42, 43):
        for mode in pilot.MODES[1:]:
            reports.append({"status": "complete", "protocol": "same", "initialization_sha256": "warm",
                            "mode": mode, "training_seed": seed, "cohort_id": str(seed),
                            "retention_rejections": [],
                            "final": {"eai": {"successes": 44},
                                      "vlabench": {"successes": 12 if mode == "constrained-es" else 10}}})
    assert pilot.promotion_candidate(reports)["training_seed"] == 42
    reports[2]["final"]["vlabench"]["successes"] = 10
    with pytest.raises(ValueError, match="replicate"):
        pilot.promotion_candidate(reports)
