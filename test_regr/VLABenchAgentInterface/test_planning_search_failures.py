"""Evaluation retains failures in the denominator and reports their cause."""
import pytest
import torch

from domiknows.generation import SemanticSearchLimitExceeded
from .graph import plan_to_tokens
from .training import build_constraint_runtime, evaluate_planner


@pytest.mark.parametrize("error_type,search_failures", [
    (SemanticSearchLimitExceeded, 1), (RuntimeError, 0),
])
def test_evaluation_reports_generation_failures_without_dropping_examples(error_type, search_failures):
    runtime = build_constraint_runtime(max_entities=1, max_operations=1,
                                       name_prefix="evaluation_search_failure")
    plan = [{"name": "press", "params": {"target_entity_name": 0}}]
    class Planner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def generate_plan(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise error_type("test search failure")
            return plan
    planner = Planner()
    examples = [{"episode_id": str(i), "entities": ("button0",),
                 "task": "select_painting", "operation_sequence": plan}
                for i in range(2)]
    result = evaluate_planner(planner, examples, runtime)
    assert result["examples"] == 2
    assert result["generation_failures"] == 1
    assert result["semantic_search_limit_failures"] == search_failures
    assert result["generation_errors"] == [{
        "example": 1, "episode_id": "0", "error_type": error_type.__name__,
        "message": "test search failure",
    }]
    assert result["skill_match"] == 0.5
    assert planner.training


def test_composite_plan_masks_with_long_horizon_retain_valid_completion_and_replay():
    runtime = build_constraint_runtime(max_entities=4, max_operations=4,
                                       name_prefix="composite_long_horizon")
    entities = ("apple", "bowl", "plate", "button0")
    policy = runtime.dfa_for("vlabench", {"entities": entities, "task": "cook_dishes"})
    plan = [{"name": name, "params": {key: target}} for name, key, target in (
        ("pick", "target_entity_name", 0), ("place", "target_container_name", 1),
        ("pick", "target_entity_name", 0), ("place", "target_container_name", 2),
    )]
    labels = [runtime.vocabulary.label_for_token(token)
              for token in plan_to_tokens(plan, entities, world=runtime.world_bundle)]
    masks = []
    for replay in (False, True):
        state = policy.start_state
        for index, label in enumerate(labels):
            mask = policy.allowed_tokens(state, remaining_steps=80-index)
            assert label in mask
            if replay:
                assert mask == masks[index]
            else:
                masks.append(mask)
            state = policy.step(state, label)
        assert policy.is_accepting(state)
