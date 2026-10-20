"""Independent semantic examples for completed plans and decoding prefixes."""

import pytest
from ..semantics import PlanSemantics, PlanSemanticState
from ..world_graph import SKILL_ARGUMENTS, build_vlabench_world_graph, materialize_plan, verify_plan_constraints


@pytest.mark.parametrize("skill,params", [
    ("place", {"target_container_name": 2}), ("pour", {"target_container_name": 2}),
    ("pull", {}), ("open_door", {"target_entity_name": 3}),
])
def test_pick_history_survives_intervening_actions_without_mutating_prior_state(skill, params):
    semantics = PlanSemantics(("apple", "pear", "bowl", "door"), SKILL_ARGUMENTS)
    initial = semantics.initial_state()
    picked = semantics.advance(initial, "pick", {"target_entity_name": 0})
    later = semantics.advance(picked, skill, params)
    assert not initial.visited_picks
    assert picked.visited_picks == later.visited_picks == frozenset({"apple"})
    assert semantics.evaluate_operation(later, "pick", {"target_entity_name": "apple"}).repeated_pick_entity
    assert not semantics.evaluate_operation(later, "pick", {"target_entity_name": 1}).repeated_pick_entity


def test_operation_schema_semantics_distinguish_missing_invalid_and_forbidden_arguments():
    semantics = PlanSemantics(("apple", "floor"), SKILL_ARGUMENTS)
    state = PlanSemanticState()
    assert semantics.evaluate_operation(state, "pick", {"target_entity_name": 0}).valid_operation
    missing = semantics.evaluate_operation(state, "pick", {})
    assert not missing.has_target_entity and not missing.valid_operation
    assert not semantics.evaluate_operation(state, "pick", {"target_entity_name": True}).valid_pointer
    assert not semantics.evaluate_operation(state, "pick", {"target_entity_name": 0, "target_container_name": 1}).valid_operation
    assert semantics.evaluate_operation(state, "pick", {"target_entity_name": 1}).targets_static_fixture
    assert not semantics.evaluate_operation(state, "unknown", {}).valid_skill_type


def test_prefix_evaluation_resolves_mixed_pointers_without_mutating_state():
    semantics = PlanSemantics({2: "apple", 7: "bowl"}, SKILL_ARGUMENTS)
    initial = PlanSemanticState()
    held = semantics.advance(initial, "pick", {"target_entity_name": 2})
    assert initial.held_entity is None
    assert held.held_entity == "apple"
    assert semantics.evaluate_operation(held, "place", {"target_container_name": "apple"}).is_self_placement
    assert not semantics.evaluate_operation(held, "place", {"target_container_name": "7"}).is_self_placement
    assert not semantics.evaluate_operation(held, "place", {"target_container_name": 0}).valid_pointer
    assert not PlanSemantics((), SKILL_ARGUMENTS).evaluate_operation(initial, "pick", {"target_entity_name": 0}).valid_pointer


def test_opening_one_container_does_not_open_another():
    semantics = PlanSemantics(("cooler_a", "cooler_b", "cooler_a_door"), SKILL_ARGUMENTS)
    state = semantics.advance(PlanSemanticState(), "open_door", {"target_entity_name": 2})
    assert not semantics.evaluate_operation(state, "place", {"target_container_name": 0}).unopened_container_placement
    assert semantics.evaluate_operation(state, "place", {"target_container_name": 1}).unopened_container_placement


def test_pouring_retains_vessel_and_inserting_releases_held_object():
    semantics = PlanSemantics(("jug", "bowl"), SKILL_ARGUMENTS)
    state = semantics.advance(PlanSemanticState(), "pick", {"target_entity_name": 0})
    poured = semantics.advance(state, "pour", {"target_container_name": 1})
    assert poured.held_entity == "jug"
    assert semantics.evaluate_operation(poured, "place", {"target_container_name": 0}).is_self_placement
    inserted = semantics.advance(state, "insert", {"target_container_name": 1})
    assert inserted.held_entity is None


def test_completion_is_evaluated_at_eos_not_during_a_valid_prefix():
    semantics = PlanSemantics(("apple", "bowl"), SKILL_ARGUMENTS)
    state = semantics.advance(PlanSemanticState(), "pick", {"target_entity_name": 0})
    assert semantics.premature_composite_eos(state, min_operations=2, required_skills=("pick", "place"))
    assert not semantics.evaluate_operation(state, "place", {"target_container_name": 1}).is_self_placement
    state = semantics.advance(state, "place", {"target_container_name": 1})
    assert not semantics.premature_composite_eos(state, min_operations=2, required_skills=("pick", "place"))


def test_materialization_grounds_cross_operation_predicates():
    world = build_vlabench_world_graph("test_shared_prefix_grounding")
    def op(skill, **params):
        return {"name": skill, "params": params}
    entities = ("apple", "pear", "bowl", "plate")
    self_plan = [op("pick", target_entity_name=0), op("place", target_container_name="apple")]
    result = verify_plan_constraints(materialize_plan(self_plan, entities, world), world)
    assert result.results["anti_self_placement"]["satisfied"] == 50.0
    assert result.score < 1.0
    changed_destination = [op("pick", target_entity_name=0), op("place", target_container_name=2),
                           op("pick", target_entity_name=1), op("place", target_container_name=3)]
    result = verify_plan_constraints(materialize_plan(changed_destination, entities, world, task_context={"task_target_container": "bowl"}), world)
    assert result.results["container_destination_consistency"]["satisfied"] < 100.0
    same_destination = changed_destination[:-1] + [op("place", target_container_name="bowl")]
    result = verify_plan_constraints(materialize_plan(same_destination, entities, world), world)
    assert result.score == 1.0


def test_materialization_preserves_container_identity_and_repeated_pick_aliases():
    world = build_vlabench_world_graph("test_shared_identity_grounding")
    plan = [{"name": "pick", "params": {"target_entity_name": 0}},
            {"name": "pick", "params": {"target_entity_name": "apple"}}]
    result = verify_plan_constraints(materialize_plan(plan, ("apple",), world), world)
    assert result.results["no_redundant_repeated_picks"]["satisfied"] == 0.0
    plan = [{"name": "pick", "params": {"target_entity_name": 0}},
            {"name": "open_door", "params": {"target_entity_name": 0}},
            {"name": "pick", "params": {"target_entity_name": 2}},
            {"name": "place", "params": {"target_container_name": 1}}]
    result = verify_plan_constraints(materialize_plan(plan, ("cooler_a", "cooler_b", "apple"), world), world)
    assert result.results["open_before_enclosed_placement"]["satisfied"] < 100.0
