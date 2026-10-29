from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from ..utils.evaluation_protocol import evaluation_protocol, training_data_protocol
from ..main import stage1_selection_key, build_parser
from ..eval_planning import build_eval_parser


def args(**kwargs):
    values = dict(seed=42, resume=None, holdout_tasks="select_book", split="heldout", input_probe="original")
    return SimpleNamespace(**(values | kwargs))


def test_requested_holdout_is_not_proof_that_a_legacy_model_did_not_train_on_it():
    report = evaluation_protocol(args(), {"metrics": {}})
    assert report["scope"] == "requested_task_holdout"
    assert not report["excluded_from_joint_training_verified"]


def test_verified_task_exclusion_requires_matching_training_and_resume_history():
    protocol = training_data_protocol(args())
    checkpoint = {"metrics": {"training_data_protocol": protocol}}
    assert evaluation_protocol(args(), checkpoint)["excluded_from_joint_training_verified"]
    assert training_data_protocol(args(resume="same.pt"), checkpoint)["holdout_history_verified"]
    assert not training_data_protocol(args(resume="legacy.pt"), {})["holdout_history_verified"]
    assert not training_data_protocol(args(resume_adapter="unknown-adapter"))["holdout_history_verified"]
    assert not training_data_protocol(args(resume="different.pt", holdout_tasks="select_toy"), checkpoint)["holdout_history_verified"]
    assert not evaluation_protocol(args(holdout_tasks="select_toy"), checkpoint)["excluded_from_joint_training_verified"]
    assert not evaluation_protocol(args(split="validation"), checkpoint)["excluded_from_joint_training_verified"]


def test_random_episode_validation_is_labeled_as_seen_task_evaluation():
    assert evaluation_protocol(args(split="validation", holdout_tasks=None), {})["scope"] == "episode_split_within_seen_tasks"


def test_checkpoint_selection_prefers_complete_plans_over_high_partial_pm():
    def metrics(strict, pm):
        return {"eai": {"goal_success": 0.9, "goal_recall": 0.9},
                "vlabench": {"strict_plan_match": strict, "exact_graph_match": pm, "valid": 1.0},
                "validation_loss": 0.1}
    assert stage1_selection_key(metrics(0.8, 0.8)) > stage1_selection_key(metrics(0.1, 1.0))


def test_both_joint_evaluation_entry_points_expose_input_audits():
    command = build_parser().parse_args(["evaluate-planning", "--checkpoint", "fixture.pt", "--audit-probes"])
    standalone = build_eval_parser().parse_args(["--checkpoint", "fixture.pt", "--audit-probes"])
    assert command.audit_probes and standalone.audit_probes


@pytest.mark.parametrize("example_count", [1, 2])
def test_paired_input_audit_uses_same_episodes_and_records_strict_drops(tmp_path, monkeypatch, example_count):
    from .. import eval_planning
    from test_regr.VLABenchAgentInterface.dataset import PlanningExample

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    selected = [PlanningExample(f"select_fruit/example{i}", "press", (
        {"name": "press", "params": {"target_entity_name": 0}},), (f"image{i}.png",), (), None,
        entities=("button0", "button1")) for i in range(example_count)]
    runtime = SimpleNamespace(domain_scope=lambda domain: nullcontext(),
                              eai_vocabulary=object(), vlabench_vocabulary=object())
    planner = SimpleNamespace(eval=lambda: None, for_domain=lambda domain: object())
    monkeypatch.setattr(eval_planning, "_prepare_eval", lambda *args: (runtime, {"validation": selected}))
    monkeypatch.setattr(eval_planning, "JointQwenVLPlanner", SimpleNamespace(from_pretrained=lambda **kw: planner))
    monkeypatch.setattr(eval_planning, "_controller", lambda *args: object())
    monkeypatch.setattr(eval_planning, "load_joint_checkpoint", lambda *args, **kw: {"stage": "stage1", "epoch": 0})
    monkeypatch.setattr(eval_planning, "_runtime_adapter", lambda *args: object())
    calls = []

    def evaluate(view, examples, adapter):
        assert [e.episode_id for e in examples] == [e.episode_id for e in selected]
        index = len(calls)
        if index == 1:
            assert all(not e.image_paths and not e.segmented_image_paths for e in examples)
        elif index == 2:
            assert all(a.image_paths != b.image_paths for a, b in zip(examples, selected))
        elif index == 3:
            for example in examples:
                assert example.entities == ("button1", "button0")
                assert example.operation_sequence[0]["params"]["target_entity_name"] == 1
        calls.append(examples)
        return {"strict_plan_match": (1.0, 0.5, 0.25, 0.75)[index], "graph_pm": 1.0,
                "planning_metrics": {"overall": {}, "dimensions": {}}}

    monkeypatch.setattr(eval_planning, "evaluate_planner", evaluate)
    arguments = build_eval_parser().parse_args(["--checkpoint", str(checkpoint), "--audit-probes", "--device", "cpu"])
    if example_count == 1:
        with pytest.raises(ValueError, match="at least two episodes"):
            eval_planning.run_planning_evaluation(arguments)
        return
    report = eval_planning.run_planning_evaluation(arguments)
    assert len(calls) == 4
    assert report["strict_plan_match"] == 1.0
    assert report["input_audit"]["no-images"]["strict_plan_match_drop"] == 0.5
    assert report["input_audit"]["shuffled-images"]["strict_plan_match_drop"] == 0.75
    assert report["input_audit"]["permuted-entities"]["strict_plan_match_drop"] == 0.25
    assert all(probe["changed_input_examples"] == 2 for probe in report["input_audit"].values())


def test_audit_probes_rejects_an_already_modified_reference():
    from ..eval_planning import run_planning_evaluation

    arguments = build_eval_parser().parse_args(["--checkpoint", "unused.pt", "--audit-probes", "--input-probe", "no-images"])
    with pytest.raises(ValueError, match="paired reference"):
        run_planning_evaluation(arguments)
