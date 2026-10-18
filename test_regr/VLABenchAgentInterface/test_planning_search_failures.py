"""Evaluation retains failures in the denominator and reports their cause."""
import pytest
import torch

from domiknows.generation import SemanticSearchLimitExceeded
from .graph import plan_to_tokens
from .training import build_constraint_runtime, evaluate_planner


@pytest.fixture(scope="module")
def wide_complex_runtime():
    return build_constraint_runtime(max_entities=64, max_operations=6,
                                    name_prefix="wide_complex_completion_bounds")


def complex_prefix(runtime, task):
    entities = tuple("bottle_" + str(i) for i in range(63)) + ("button0",)
    context = {"entities": entities, "task": task}
    policy = runtime.dfa_for("vlabench", context)
    plan = [{"name": name, "params": params} for name, params in (
        ("pick", {"target_entity_name": 0}), ("lift", {}), ("pull", {}),
        ("press", {"target_entity_name": 63}),
    )]
    labels = [runtime.vocabulary.label_for_token(token)
              for token in plan_to_tokens(plan, entities, world=runtime.world_bundle)]
    state = policy.start_state
    for label in labels[:-1]:
        state = policy.step(state, label)
        assert state is not None
    return policy, state, labels, context


@pytest.mark.parametrize("task", ["take_chemistry_experiment", "hammer_nail_and_hang_picture"])
def test_wide_complex_dead_ends_are_proved_without_enumerating_entity_combinations(wide_complex_runtime, task):
    policy, state, labels, context = complex_prefix(wide_complex_runtime, task)
    # Four operations have used neither place nor pour/push. No declared suffix
    # of at most two operations can supply both missing requirements.
    from .graph import _legal_skill_sequences
    schema = policy.adapter.schema
    complete, _ = _legal_skill_sequences(wide_complex_runtime.world_bundle, 6)
    prefix = ("pick", "lift", "pull", "press")
    assert not any(seq[:len(prefix)] == prefix and set(schema.required_skills) <= set(seq)
                   for seq in complete)
    bound = policy._reachability.completion_bound
    assert bound(state) is None
    assert policy.allowed_tokens(state, remaining_steps=80) == set()
    # Reproduce the previous 10,000-expansion failure at the SAME state/cap.
    policy._reachability.completion_bound = None
    with pytest.raises(SemanticSearchLimitExceeded):
        policy.allowed_tokens(state, remaining_steps=80)
    policy._reachability.completion_bound = bound
    assert policy.allowed_tokens(state, remaining_steps=80) == set()


def test_task_completion_bound_follows_lc_deactivation_addition_and_frozen_programs(wide_complex_runtime):
    from domiknows.graph.logicalConstrain import ifL, notL
    from .constraint_compiler import WorldConstraintDFA, compile_world_constraints
    world = wide_complex_runtime.world_bundle
    original, state, labels, context = complex_prefix(wide_complex_runtime, "take_chemistry_experiment")
    rule = next(rule for rule in world.graph.logicalConstrains.values() if rule.name == "composite_subgoal_completion")
    rule.active = False
    try:
        relaxed = WorldConstraintDFA(original.base_dfa, wide_complex_runtime.vocabulary, world,
                                     compile_world_constraints(world), context["entities"],
                                     requirements=original.adapter.requirements)
        assert relaxed.accepts(labels)
        assert relaxed._reachability.completion_bound(state) == 1
        assert wide_complex_runtime.vocabulary.eos_label in relaxed.allowed_tokens(state, 0)
        with world.graph:
            added = ifL(world.operation("op"), notL(world.premature_composite_eos("op")),
                        name="new_completion_rule_without_search_changes")
        restored = WorldConstraintDFA(original.base_dfa, wide_complex_runtime.vocabulary, world,
                                      compile_world_constraints(world), context["entities"],
                                      requirements=original.adapter.requirements)
        assert not restored.accepts(labels)
        assert restored._reachability.completion_bound(state) is None
        assert restored.allowed_tokens(state, 80) == set()
        # A cache for the same grammar/task must not leak across LC snapshots.
        assert relaxed.accepts(labels)
        assert not original.accepts(labels)
    finally:
        rule.active = True
        if "added" in locals():
            added.active = False


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


@pytest.mark.parametrize("task,middle", [
    ("take_chemistry_experiment", "pour"), ("hammer_nail_and_hang_picture", "push"),
])
def test_wide_complex_valid_completions_and_replay_remain_available(wide_complex_runtime, task, middle):
    runtime = wide_complex_runtime
    entities = tuple("bottle_" + str(i) for i in range(63)) + ("button0",)
    policy = runtime.dfa_for("vlabench", {"entities": entities, "task": task})
    plan = [{"name": "pick", "params": {"target_entity_name": 0}},
            {"name": middle, "params": {"target_container_name": 1} if middle == "pour" else {}},
            {"name": "place", "params": {"target_container_name": 1}}]
    labels = [runtime.vocabulary.label_for_token(token)
              for token in plan_to_tokens(plan, entities, world=runtime.world_bundle)]
    masks = []
    for replay in (False, True):
        state = policy.start_state
        for index, label in enumerate(labels):
            allowed = policy.allowed_tokens(state, 80-index)
            assert label in allowed
            if replay:
                assert allowed == masks[index]
            else:
                masks.append(allowed)
            state = policy.step(state, label)
        assert policy.is_accepting(state)
    # Structural bounds can be shared; scene-specific physical answers cannot.
    cache = policy.base_dfa.completion_bounds_cache
    before = len(cache)
    fixtures = tuple("table_" + str(i) for i in range(63)) + ("button0",)
    other = runtime.dfa_for("vlabench", {"entities": fixtures, "task": task})
    assert len(cache) == before
    assert other.adapter is not policy.adapter
    assert not other.accepts(labels)
    assert policy.accepts(labels)
