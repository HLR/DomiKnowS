"""Request facts affect both compiled decoding and completed-plan verification."""
from itertools import product
import pytest
from .training import build_constraint_runtime
from .graph import plan_to_tokens
from .world_graph import materialize_plan, verify_plan_constraints, validate_plan

ENTITIES = ("apple", "cup", "bowl", "cooler")

def operation(skill, target=None):
    return {"name": skill, "params": {} if target is None else {
        "target_container_name" if skill in ("place", "insert", "pour") else "target_entity_name": target}}

def labels(runtime, plan):
    return tuple(runtime.vocabulary.label_for_token(x) for x in plan_to_tokens(plan, ENTITIES, world=runtime.world_bundle))

def verified(runtime, policy, plan):
    root = materialize_plan(plan, ENTITIES, runtime.world_bundle, policy=policy)
    result = verify_plan_constraints(root, runtime.world_bundle)
    return result is None or result.score == 1.0

@pytest.fixture(scope="module")
def runtime():
    return build_constraint_runtime(max_entities=4, max_operations=4, name_prefix="grounded_vla")

@pytest.mark.parametrize("facts,plan", [
    ({"destinations": {"apple": "bowl"}}, [operation("pick", 0), operation("place", 3)]),
    ({"initial_state": {"held": {"right": "cup"}}, "capacities": {"right": 1}}, [operation("pick", 0), operation("place", 2)]),
    ({"entity_properties": {"apple": {"graspable": False}}}, [operation("pick", 0), operation("place", 2)]),
    ({"entity_properties": {"cup": {"pourable": False}}}, [operation("pick", 1), operation("pour", 2)]),
    ({"entity_properties": {"cooler": {"openable": False}}}, [operation("pick", 0), operation("open_door", 3)]),
    ({"compatibility": [{"subject": "apple", "destination": "bowl", "allowed": False}]}, [operation("pick", 0), operation("place", 2)]),
    ({"milestones": [{"name": "opened", "action": "open_door"}, {"name": "placed", "action": "place", "requires": ["opened"]}]}, [operation("pick", 0), operation("place", 2)]),
    ({"goals": {"positive": [["at", "apple", "bowl"]]}}, [operation("pick", 0), operation("place", 3)]),
    ({"protected_entities": ["apple"]}, [operation("pick", 0), operation("place", 2)]),
])
def test_grounded_counterexamples(runtime, facts, plan):
    policy = runtime.dfa_for("vlabench", {"entities": ENTITIES, "planning_facts": facts})
    assert not policy.accepts(labels(runtime, plan))
    assert not verified(runtime, policy, plan)


def test_object_specific_destinations_and_unknown_report(runtime):
    facts = {"destinations": {"apple": "bowl", "cup": "cooler"}, "initial_state": {"positive": [["open", "cooler"]]}}
    plan = [operation("pick", 0), operation("place", 2), operation("pick", 1), operation("place", 3)]
    policy = runtime.dfa_for("vlabench", {"entities": ENTITIES, "planning_facts": facts})
    assert policy.accepts(labels(runtime, plan))
    assert verified(runtime, policy, plan)
    root = materialize_plan(plan, ENTITIES, runtime.world_bundle, policy=policy)
    assert not root.attributes["grounding_report"]["fully_grounded"]


def test_continuation_and_already_satisfied_noop(runtime):
    facts = {"initial_state": {"held": {"right": "apple"}}, "goals": {"positive": [["at", "apple", "bowl"]]}}
    context = {"entities": ENTITIES, "planning_facts": facts, "task_skill_sequence": ["place"], "task_target_entity": "apple"}
    policy = runtime.dfa_for("vlabench", context)
    plan = [operation("place", 2)]
    assert policy.accepts(labels(runtime, plan))
    assert validate_plan(plan, entity_table=ENTITIES, policy=policy).valid
    assert verified(runtime, policy, plan)
    facts = {"initial_state": {"positive": [["at", "apple", "bowl"]]}, "goals": {"positive": [["at", "apple", "bowl"]]}}
    policy = runtime.dfa_for("vlabench", {"entities": ENTITIES, "planning_facts": facts})
    assert policy.accepts(labels(runtime, []))
    assert validate_plan([], entity_table=ENTITIES, policy=policy).valid
    assert verified(runtime, policy, [])


def test_bounded_completed_verification_and_token_masks(runtime):
    from test_regr.generation.constraint_equivalence import finite_language_dfa, assert_bounded_equivalence
    from domiknows.generation.dfa.predicate_monitor import PredicateMonitorDFA
    facts = {"destinations": {"apple": "bowl", "cup": "cooler"}, "entity_properties": {"apple": {"graspable": True}, "cup": {"graspable": False}}}
    policy = runtime.dfa_for("vlabench", {"entities": ENTITIES, "planning_facts": facts})
    oracle = {}
    for subject, destination in product(range(2), (2, 3)):
        plan = [operation("pick", subject), operation("place", destination)]
        expected = subject == 0 and destination == 2  # Independent task oracle.
        encoded = labels(runtime, plan)
        oracle[encoded] = expected
        assert policy.accepts(encoded) == expected
        assert verified(runtime, policy, plan) == expected
        if expected:
            state = policy.start_state
            for label in encoded:
                assert label in policy.allowed_tokens(state, remaining_steps=len(encoded))
                state = policy.step(state, label)
    grammar = finite_language_dfa(oracle)
    restricted = PredicateMonitorDFA(grammar, policy.program, policy.adapter)
    audit = assert_bounded_equivalence(grammar, restricted, oracle.__getitem__, max_steps=7,
                                      terminal_symbols={runtime.vocabulary.eos_label})
    assert audit.candidates == 4 and audit.accepted == 1 and audit.checked_masks > 10


@pytest.mark.parametrize("family,facts,plan", [
    ("destination", {"destinations": {"apple": "bowl"}}, [operation("pick", 0), operation("place", 3)]),
    ("holding", {"effects": {"pick": {"release_hands": ["right"]}}}, [operation("pick", 0), operation("lift")]),
    ("capacity", {"initial_state": {"held": {"right": "cup"}}, "capacities": {"right": 1}}, [operation("pick", 0), operation("lift")]),
    ("affordance", {"entity_properties": {"apple": {"graspable": False}}}, [operation("pick", 0), operation("lift")]),
    ("compatibility", {"compatibility": [{"subject": "apple", "destination": "bowl", "allowed": False}]}, [operation("pick", 0), operation("place", 2)]),
    ("milestone_order", {"milestones": [{"name": "first", "action": "open_door"}, {"name": "last", "action": "place", "requires": ["first"]}]}, [operation("pick", 0), operation("place", 2)]),
    ("goal_completion", {"goals": {"positive": [["at", "apple", "bowl"]]}}, [operation("pick", 0), operation("place", 3)]),
    ("preservation", {"protected_entities": ["apple"]}, [operation("pick", 0), operation("lift")]),
])
def test_each_lc_changes_enforcement_without_decoder_edits(family, facts, plan):
    from .world_graph import build_vlabench_world_graph
    from domiknows.graph.logicalConstrain import andL, ifL
    world = build_vlabench_world_graph("editable_grounded_"+family)
    for rule in world.graph.logicalConstrains.values():
        if getattr(rule, "headLC", True):
            rule.active = rule.name == "grounded__"+family
    runtime = build_constraint_runtime(world, max_entities=4, max_operations=2, name_prefix="edit_"+family)
    context = {"entities": ENTITIES, "planning_facts": facts}
    original = runtime.dfa_for("vlabench", context)
    encoded = labels(runtime, plan)
    assert not original.accepts(encoded)
    assert not verified(runtime, original, plan)
    rule = next(rule for rule in world.graph.logicalConstrains.values() if rule.name == "grounded__"+family)
    rule.active = False
    runtime = build_constraint_runtime(world, max_entities=4, max_operations=2, name_prefix="relaxed_"+family)
    relaxed = runtime.dfa_for("vlabench", context)
    assert relaxed.accepts(encoded)
    assert verified(runtime, relaxed, plan)
    parent = world.plan if family == "goal_completion" else world.operation
    with world.graph:
        ifL(andL(parent("x"), world.grounded_constraints[family+"__applicable"]("x")),
            world.grounded_constraints[family+"__satisfied"]("x"), name="replacement_"+family)
    runtime = build_constraint_runtime(world, max_entities=4, max_operations=2, name_prefix="restored_"+family)
    restored = runtime.dfa_for("vlabench", context)
    assert not restored.accepts(encoded)
    assert not verified(runtime, restored, plan)
    assert not original.accepts(encoded)
