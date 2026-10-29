"""Complete-plan equality must reject the false positives accepted by partial PM."""

from types import SimpleNamespace

import pytest
import torch

from ..planning_metrics import strict_plan_match, score_planning_metrics
from ..dataset import PlanningExample
from ..training import evaluate_planner


def press(entity):
    return {"name": "press", "params": {"target_entity_name": entity}}


def test_extra_actions_get_full_benchmark_pm_but_fail_strict_equality():
    reference = [press(0)]
    prediction = reference + [press(1)]
    assert score_planning_metrics(prediction, reference)["PM"] == 100
    assert strict_plan_match(prediction, reference) == 0


@pytest.mark.parametrize("prediction", [[], [press(1)], [press(0), press(0)],
    [{"name": "press", "params": {"target_entity_name": 0, "extra": 1}}], "invalid json"])
def test_strict_equality_rejects_missing_wrong_duplicate_or_extra_parameters(prediction):
    assert strict_plan_match(prediction, [press(0)]) == 0


def test_strict_equality_grounds_names_and_preserves_order():
    assert strict_plan_match([press(0)], [press("button0")], ("button0", "button1")) == 1
    assert strict_plan_match([press(1), press(0)], [press(0), press(1)]) == 0
    with pytest.raises(ValueError, match="invalid reference"):
        strict_plan_match([], "bad reference")


def test_evaluator_reports_strict_and_pm_separately_with_dimension_counts():
    class Planner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.outputs = iter([[press(0)], [press(0), press(1)]])

        def generate_plan(self, **kwargs):
            return next(self.outputs)

    example = PlanningExample("press/example0", "press button0", (press("button0"),), (), (), None,
                              entities=("button0", "button1"), dimension="PhysLaw")
    result = evaluate_planner(Planner(), [example, example],
                             SimpleNamespace(dfa=None, world_bundle=None, max_tokens=20))
    assert result["strict_plan_match"] == 0.5
    assert result["graph_pm"] == result["exact_graph_match"] == 1.0
    assert result["strict_plan_match_by_dimension"]["PhysLaw"] == {"matches": 1, "examples": 2, "rate": 0.5}
    assert "partial" in result["metric_definitions"]["graph_pm"]
