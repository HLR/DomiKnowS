import json

import pytest
import torch

from test_regr.VLABenchAgentInterface.reward import (
    RolloutRewardAccumulator,
    make_vlabench_reward_function,
    score_vlabench_plan,
)


REFERENCE = [
    {"name": "pick", "params": {"target_entity_name": "apple"}},
    {"name": "place", "params": {"target_container_name": "bowl"}},
]
ENTITIES = ("apple", "bowl", "banana")


def test_perfect_plan_gets_one_not_upstream_point_eight():
    result = score_vlabench_plan(REFERENCE, REFERENCE, entity_table=ENTITIES)
    assert result.valid
    assert result.total == 1.0
    assert result.skill_match == result.entity_match == 1.0
    assert result.skill_with_entity_match == result.exact_graph_match == 1.0


def test_dense_components_preserve_partial_credit_but_hard_gate_invalid_schema():
    wrong_entity = [
        {"name": "pick", "params": {"target_entity_name": "banana"}},
        {"name": "place", "params": {"target_container_name": "bowl"}},
    ]
    result = score_vlabench_plan(wrong_entity, REFERENCE, entity_table=ENTITIES)
    assert result.valid
    assert result.skill_match == 1.0
    assert result.entity_match == 0.5
    assert result.skill_with_entity_match == 0.5
    assert result.exact_graph_match == 0.0
    assert result.total == pytest.approx(0.65)

    missing_parameter = [{"name": "pick", "params": {}}, {"name": "place", "params": {"target_container_name": "bowl"}}]
    invalid = score_vlabench_plan(missing_parameter, REFERENCE, entity_table=ENTITIES)
    assert not invalid.valid
    assert invalid.total == 0.0
    assert any("requires target_entity_name" in error for error in invalid.errors)


@pytest.mark.parametrize("prediction", ["not json", [], [{"name": "pick", "params": {"target_entity_name": "apple"}}]])
def test_malformed_empty_and_incomplete_patterns_are_zero(prediction):
    result = score_vlabench_plan(prediction, REFERENCE, entity_table=ENTITIES)
    assert not result.valid
    assert result.total == 0.0


def test_binary_mode_and_domiknows_reward_tensor_contract():
    item = {"task_id": "synthetic", "operation_sequence": REFERENCE, "entities": ENTITIES}
    closure = make_vlabench_reward_function(item, mode="binary")
    reward = closure(json.dumps(REFERENCE))
    assert reward.shape == torch.Size([1])
    assert reward.dtype == torch.float32
    assert reward.item() == 1.0
    assert closure.last_breakdown.valid
    assert make_vlabench_reward_function(item, mode="binary")("[]").item() == 0.0


def test_rollout_reward_formula_and_efficiency_gate():
    reward = RolloutRewardAccumulator(100)
    full = reward.finalize(True, progress=1, intention=1, steps=0)
    assert full.total == 1.0
    halfway = reward.finalize(True, progress=1, intention=1, steps=50)
    assert halfway.total == pytest.approx(0.975)
    failure = reward.finalize(False, progress=1, intention=1, steps=0)
    assert failure.efficiency == 0.0
    assert failure.total == pytest.approx(0.35)
    invalid = reward.finalize(True, progress=1, intention=1, steps=0, valid=False)
    assert invalid.total == 0.0


def test_rollout_accumulator_clamps_nonfinite_signals():
    reward = RolloutRewardAccumulator(10)
    reward.update(progress=float("nan"), intention=float("inf"), steps=20)
    result = reward.finalize(False)
    assert result.progress == result.intention == 0.0
    assert result.steps == 10
    assert 0.0 <= result.total <= 1.0
