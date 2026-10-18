import json
from types import SimpleNamespace

import pytest
import torch

from test_regr.VLABenchAgentInterface.planning_metrics import (
    BENCHMARK_PATTERNS, DIMENSIONS, PlanningMetrics, planning_metadata,
    score_planning_metrics,
)
from test_regr.VLABenchAgentInterface.dataset import PlanningExample, load_planning_examples
from test_regr.VLABenchAgentInterface import training


def operation(name, entity=None, container=None):
    params = {}
    if entity is not None:
        params["target_entity_name"] = entity
    if container is not None:
        params["target_container_name"] = container
    return {"name": name, "params": params}


def subtask(entity="apple", container="bowl"):
    return [operation("pick", entity), operation("place", container=container)]


def test_perfect_plan_and_upstream_legacy_total():
    result = score_planning_metrics(subtask(), subtask())
    assert result == {"SR": 100, "PR": 100, "SPR": 100, "PM": 100, "score": 100, "legacy_score": 80}


def test_multiset_recall_extra_actions_and_role_qualified_arguments():
    reference = subtask() * 2
    short = score_planning_metrics(subtask(), reference)
    assert short == {"SR": 50, "PR": 50, "SPR": 50, "PM": 50, "score": 50, "legacy_score": 40}
    extra = score_planning_metrics(reference + subtask("banana"), reference)
    assert extra["score"] == 100  # recall does not penalize extra predicted occurrences
    swapped = [operation("pick", container="apple"), operation("place", entity="bowl")]
    result = score_planning_metrics(swapped, subtask())
    assert result["SR"] == 100
    assert result["PR"] == result["SPR"] == result["PM"] == 0


def test_skill_parameter_pairs_and_partial_dependency_credit():
    reference = subtask("apple") + subtask("banana")
    swapped = subtask("banana") + subtask("apple")
    sequential = score_planning_metrics(swapped, reference)
    assert sequential["SR"] == sequential["PR"] == sequential["SPR"] == 100
    assert sequential["PM"] == 0
    assert score_planning_metrics(swapped, reference, "Seq-independent")["PM"] == 100
    partial = score_planning_metrics(subtask("apple") + subtask("pear"), reference)
    assert partial["PM"] == 50  # node recall, not binary whole-plan equality
    misaligned = [operation("pick", "bowl"), operation("place", container="apple")]
    assert score_planning_metrics(misaligned, subtask())["SPR"] == 0


def test_official_pattern_order_is_independent_of_training_patterns():
    assert ("pick", "lift", "place") not in BENCHMARK_PATTERNS
    reference = [operation("pick", "apple"), operation("lift", "apple"), operation("place", container="bowl")]
    # Upstream recognizes pick-lift, then ignores the unmatched place.
    prediction = reference[:2]
    result = score_planning_metrics(prediction, reference)
    assert result["PM"] == 100
    assert result["SR"] == pytest.approx(200 / 3)
    pouring = [operation("pick", "apple"), operation("pour", container="bowl"), operation("place", container="plate")]
    assert score_planning_metrics(pouring[:2], pouring)["PM"] == 0


@pytest.mark.parametrize("prediction", [[], "not json", [None], [{"name": "pick", "params": []}], [operation("pick", ["apple"])]])
def test_malformed_predictions_remain_zero(prediction):
    assert all(value == 0 for value in score_planning_metrics(prediction, subtask()).values())


def test_readable_but_schema_invalid_predictions_are_not_reward_gated():
    prediction = [operation("pick"), operation("place", container="bowl")]
    result = score_planning_metrics(prediction, subtask())
    assert result["SR"] == 100
    assert result["PR"] == result["SPR"] == 50
    assert result["score"] == 65


@pytest.mark.parametrize("dependency", ["bad", {0: [1]}, {1: [3]}, {1: [2], 2: [1]}, {"1.5": [2]}, {1: "2"}])
def test_invalid_reference_dependencies_raise_even_for_empty_prediction(dependency):
    with pytest.raises(ValueError, match="dependency"):
        score_planning_metrics([], subtask() * 2, dependency)


def test_custom_dependencies_and_empty_reference_denominators():
    reference = subtask() + subtask("banana")
    assert score_planning_metrics(reference, reference, {"1": [2]})["PM"] == 100
    assert score_planning_metrics(subtask(), reference, {"1": [2]})["PM"] == 0
    assert all(value == 0 for value in score_planning_metrics(subtask(), []).values())
    assert score_planning_metrics([operation("press")], [operation("press")])["PR"] == 0
    with pytest.raises(ValueError, match="invalid reference"):
        score_planning_metrics([], "bad json")


def test_dimension_macro_is_not_pooled_and_unknown_tasks_are_excluded():
    tasks = ("select_fruit", "select_fruit_spatial", "select_fruit_common_sense",
             "select_fruit_semantic", "friction_qa", "cook_dishes")
    metrics = PlanningMetrics()
    perfect = score_planning_metrics(subtask(), subtask())
    zero = score_planning_metrics([], subtask())
    for task in tasks:
        metrics.update(perfect, task=task)
    metrics.update(zero, task=tasks[0])
    result = metrics.result()
    assert result["dimensions"]["M&T"]["examples"] == 2
    assert result["Macro"] == pytest.approx(550 / 6)
    assert result["overall"]["score"] == pytest.approx(600 / 7)
    assert result["legacy_Macro"] == pytest.approx(440 / 6)
    metrics.update(zero, task="unknown", dimension="Spatial")
    assert metrics.result()["Macro"] == result["Macro"]
    assert metrics.result()["unknown_task_count"] == 1
    assert metrics.result()["overall"]["score"] == 75


def test_missing_dimensions_and_empty_evaluation():
    metrics = PlanningMetrics()
    result = metrics.result()
    assert result["examples"] == 0
    assert result["Macro"] is result["legacy_Macro"] is None
    assert all(value is None for value in result["overall"].values())
    metrics.update(score_planning_metrics([], subtask()), task="select_fruit")
    assert metrics.result()["dimensions"]["M&T"]["score"] == 0
    assert metrics.result()["dimensions"]["Spatial"]["score"] is None
    assert metrics.result()["Macro"] is None
    assert tuple(metrics.result()["dimensions"]) == DIMENSIONS


@pytest.mark.parametrize("explicit, expected", [(None, "Seq-independent"), ("Sequential", "Sequential"), ({"1": []}, {"1": []})])
def test_dataset_metadata_and_dependency_precedence(tmp_path, explicit, expected):
    directory = tmp_path / "Complex" / "cook_dishes" / "example0"
    (directory / "output").mkdir(parents=True)
    (directory / "input").mkdir()
    payload = {"skill_sequence": subtask()}
    if explicit is not None:
        payload["dependency"] = explicit
    (directory / "output" / "operation_sequence.json").write_text(json.dumps(payload))
    example = load_planning_examples(tmp_path)[0]
    assert example.task == "cook_dishes"
    assert example.dimension == "Complex"
    assert example.dependency == expected
    assert example.as_reward_item()["task"] == "cook_dishes"
    assert planning_metadata("select_toy_spatial/example0") == ("select_toy_spatial", "M&T")
    assert planning_metadata("Spatial/unknown/example0") == ("unknown", None)


def test_config_dependency_and_payload_override(tmp_path):
    directory = tmp_path / "texas_holdem" / "example0"
    (directory / "output").mkdir(parents=True)
    (directory / "env_config").mkdir()
    config = directory / "env_config" / "episode.json"
    config.write_text(json.dumps({"dependency": "Sequential"}))
    output = directory / "output" / "operation_sequence.json"
    output.write_text(json.dumps({"skill_sequence": subtask()}))
    assert load_planning_examples(tmp_path)[0].dependency == "Sequential"
    output.write_text(json.dumps({"skill_sequence": subtask(), "dependency": "Seq-independent"}))
    assert load_planning_examples(tmp_path)[0].dependency == "Seq-independent"


class FakePlanner(torch.nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def generate_plan(self, **kwargs):
        assert not self.training
        return self.plan


@pytest.mark.parametrize("was_training", [True, False])
def test_evaluator_adds_metrics_and_restores_mode(monkeypatch, was_training):
    monkeypatch.setattr(training, "score_vlabench_plan", lambda *args, **kwargs: SimpleNamespace(
        total=0.65, valid=False, skill_match=1, entity_match=0.5,
        skill_with_entity_match=0.5, exact_graph_match=0,
    ))
    planner = FakePlanner(subtask())
    planner.train(was_training)
    runtime = SimpleNamespace(dfa=None, world_bundle=None, max_tokens=20)
    # Old constructors and episode metadata inference remain supported.
    example = PlanningExample("select_fruit/example0", "pick fruit", tuple(subtask()), (), (), None)
    result = training.evaluate_planner(planner, [example], runtime)
    assert result["reward"] == 0.65 and result["valid"] == 0
    assert result["planning_metrics"]["overall"]["score"] == 100
    assert result["planning_metrics"]["dimensions"]["M&T"]["examples"] == 1
    assert planner.training == was_training
    empty = training.evaluate_planner(planner, [], runtime)
    assert empty["examples"] == 0 and empty["planning_metrics"]["Macro"] is None


def test_evaluator_restores_mode_on_reference_error():
    planner = FakePlanner(subtask())
    runtime = SimpleNamespace(dfa=None, world_bundle=None, max_tokens=20)
    with pytest.raises(ValueError, match="invalid reference"):
        training.evaluate_planner(planner, [{"operation_sequence": "not json"}], runtime)
    assert planner.training


def test_evaluator_keeps_empty_prediction_in_denominator_with_real_reward():
    planner = FakePlanner(subtask())
    outputs = iter([subtask(), []])
    planner.generate_plan = lambda **kwargs: next(outputs)
    runtime = SimpleNamespace(dfa=None, world_bundle=None, max_tokens=20)
    example = PlanningExample(
        "select_fruit/example0", "pick fruit", tuple(subtask()), (), (), None,
        entities=("apple", "bowl"),
    )
    result = training.evaluate_planner(planner, [example, example], runtime)
    assert result["examples"] == 2
    assert result["reward"] == result["valid"] == 0.5
    assert result["planning_metrics"]["overall"]["score"] == 50
    assert result["planning_metrics"]["dimensions"]["M&T"]["examples"] == 2


@pytest.mark.parametrize("observation_mode", ["visual", "metadata-assisted"])
def test_evaluate_cli_all_and_default_split(monkeypatch, capsys, observation_mode):
    from test_regr.VLABenchAgentInterface import main
    parser = main.build_parser()
    args = parser.parse_args(["evaluate-planner", "--planning-dir", "unused", "--split", "all",
                              "--resume-adapter", "trained-adapter", "--observation-mode", observation_mode])
    assert parser.parse_args(["evaluate-planner", "--planning-dir", "unused"]).split == "test"
    examples = [object(), object()]
    vocabulary = SimpleNamespace(checksum="same", max_entities=64)
    runtime = SimpleNamespace(vocabulary=vocabulary)
    def load(*args, **kwargs):
        assert kwargs["observation_mode"] == observation_mode
        return examples
    monkeypatch.setattr(main, "load_planning_examples", load)
    monkeypatch.setattr(main, "build_vlabench_world_graph", lambda *args: None)
    monkeypatch.setattr(main.PlanVocabulary, "from_world", lambda *args: vocabulary)
    monkeypatch.setattr(main, "build_constraint_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(main, "_load_planner", lambda *args: None)
    seen = []
    def evaluate(_planner, selected, _runtime):
        seen.extend(selected)
        return {"examples": len(selected), "planning_metrics": {"Macro": None}}
    monkeypatch.setattr(main, "evaluate_planner", evaluate)
    main.command_evaluate_planner(args)
    assert seen == examples
    result = json.loads(capsys.readouterr().out)
    assert result["examples"] == 2 and result["observation_mode"] == observation_mode


@pytest.mark.parametrize("task, dimension", [
    ("select_billiards", "M&T"),
    ("select_billiards_spatial", "Spatial"),
    ("select_billiards_common_sense", "ComSense"),
    ("select_billiards_semantic", "Semantic"),
    ("thermal_expansion_qa", "PhysLaw"),
    ("reflection_qa", "PhysLaw"),
    ("speed_of_sound_qa", "PhysLaw"),
    ("magnetism_qa", "PhysLaw"),
])
def test_extended_task_metadata_and_dimension_aggregation(task, dimension):
    assert planning_metadata(f"{task}/example0") == (task, dimension)
    metrics = PlanningMetrics()
    scores = score_planning_metrics(subtask(), subtask())
    metrics.update(scores, task=task)
    result = metrics.result()
    assert result["unknown_task_count"] == 0
    assert result["dimensions"][dimension]["examples"] == 1
    assert result["dimensions"][dimension]["score"] == 100
    assert result["dimension_mapping_revision"] == "vlabench-planning-coverage-v2"


def test_full_450_episode_task_coverage_keeps_pooled_scores():
    # Historical test split: 373 previously classified plus 77 omitted episodes.
    old_tasks = {"select_fruit": 103, "select_fruit_spatial": 46,
                 "select_fruit_common_sense": 75, "select_fruit_semantic": 77,
                 "density_qa": 16, "cook_dishes": 56}
    omitted_tasks = {"select_billiards": 10, "select_billiards_spatial": 7,
                     "select_billiards_common_sense": 9, "select_billiards_semantic": 7,
                     "thermal_expansion_qa": 13, "reflection_qa": 14,
                     "speed_of_sound_qa": 6, "magnetism_qa": 11}
    metrics = PlanningMetrics()
    perfect = score_planning_metrics(subtask(), subtask())
    zero = score_planning_metrics([], subtask())
    for tasks, scores in ((old_tasks, perfect), (omitted_tasks, zero)):
        for task, count in tasks.items():
            for _ in range(count):
                metrics.update(scores, task=task)
    result = metrics.result()
    assert result["examples"] == 450
    assert result["unknown_task_count"] == 0
    assert {key: item["examples"] for key, item in result["dimensions"].items()} == {
        "M&T": 113, "Spatial": 53, "ComSense": 84, "Semantic": 84,
        "PhysLaw": 60, "Complex": 56,
    }
    assert sum(item["examples"] for item in result["dimensions"].values()) == 450
    assert result["overall"]["score"] == pytest.approx(37300 / 450)
    assert result["Macro"] == pytest.approx(sum(
        100 * original / total for original, total in
        ((103, 113), (46, 53), (75, 84), (77, 84), (16, 60), (56, 56))
    ) / 6)
