"""Semantic checks that declarations, rather than rule names, control EAI."""

from itertools import product
from types import SimpleNamespace
import pytest
import torch
from domiknows.graph.logicalConstrain import ifL, notL
from ..world_graph import build_eai_world_graph
from ..utils.world_runtime import (
    evaluate_default_world_constraints,
    materialize_world_trajectory,
    verify_world_constraints,
    _precondition_status,
)
from ..utils.goal_constraints import (
    build_goal_operator,
    compile_goal_operator,
    evaluate_goal_operator,
)
from ..utils.state_effects import build_state_effect_graph, compile_state_effects
from ..utils.effect_runtime import apply_event_effects
from ..utils.task_knowledge import build_task_relevance_graph, task_semantic_action_permissions
from ..reward import _ActionEvent, _Atom, _Then, _EvalContext, _first_satisfaction
from .test_grounded_constraints import policy_for, encode, verify


@pytest.mark.parametrize("kind", ("and", "or", "forall", "exists", "forn"))
def test_goal_boolean_and_count_constraints_match_independent_truth_tables(kind):
    for arity in range(5):
        for values in product((False, True), repeat=arity):
            for count in range(arity + 2):
                expected = {
                    "and": all(values),
                    "forall": all(values),
                    "or": any(values),
                    "exists": any(values),
                    "forn": sum(values) == count,
                }[kind]
                assert evaluate_goal_operator(kind, values, count) is expected
    assert evaluate_goal_operator("not", (False,))
    assert not evaluate_goal_operator("not", (True,))


def test_temporal_witness_formula_and_strict_order():
    for holds, before_start in product((False, True), repeat=2):
        assert evaluate_goal_operator("temporal_witness", (holds, before_start)) == (
            holds and not before_start
        )
    goal = _Then((_Atom("open", ("cup",)), _Atom("closed", ("cup",))))
    context = _EvalContext(("cup",), {"cup": {"cup"}})
    assert _first_satisfaction(goal, [{("open", "cup"), ("closed", "cup")}], [], context) is None
    assert _first_satisfaction(goal, [{("closed", "cup")}, {("open", "cup")}], [], context) is None
    assert _first_satisfaction(goal, [{("open", "cup")}, {("closed", "cup")}], [], context) == 1


def test_goal_and_task_formulas_can_be_changed():
    bundle = build_goal_operator("and", 2)
    rule = next(lc for lc in bundle.graph.logicalConstrains.values() if lc.headLC)
    # Negate the actual conjunction without changing any operator dispatch code.
    with bundle.graph:
        edited = notL(rule.e[2])
    edited.headLC = False
    rule.e[2] = edited
    rule.name = "changed_formula_with_arbitrary_name"
    assert not evaluate_goal_operator("and", (True, True), program=compile_goal_operator(bundle))
    assert evaluate_goal_operator("and", (True, False), program=compile_goal_operator(bundle))
    assert task_semantic_action_permissions({"natural_language_description": "wash the cup"}) == (
        "clean",
    )
    assert (
        task_semantic_action_permissions(
            {"target_action_tokens": ["clean", "cup"], "tl_goal": "READ(book)"}
        )
        == ()
    )
    graph, cues, permitted = build_task_relevance_graph()
    from domiknows.generation.dfa.predicate_monitor import (
        PredicateBinding,
        compile_predicate_constraints,
    )

    rule = next(lc for lc in graph.logicalConstrains.values() if lc.headLC)
    rule.active = False
    program = compile_predicate_constraints(
        graph,
        [PredicateBinding(c, cue, "task") for cue, c in cues.items()]
        + [PredicateBinding(permitted, "permitted", "task")],
    )
    assert task_semantic_action_permissions({"tl_goal": "READ(book)"}, program=program) == (
        "clean",
    )


def test_container_access_truth_table_and_argument_membership():
    for opened, closed, not_open, not_closed in product((False, True), repeat=4):
        facts = {
            (name, "bowl")
            for name, value in zip(
                ("open", "closed", "not_open", "not_closed"), (opened, closed, not_open, not_closed)
            )
            if value
        }
        applicable, satisfied = _precondition_status(
            "destination_open_if_known", "right_place_inside", ("bowl",), facts
        )
        assert applicable == bool(facts)
        assert satisfied == (not facts or ((opened or not_closed) and not closed and not not_open))
    for args in ((), ("apple_1",), ("apple_1", "bowl_2"), ("apple_1", "unknown_3")):
        applicable, satisfied = _precondition_status(
            "argument_available_in_task", "pour", args, set(), ("apple", "bowl")
        )
        assert applicable
        assert satisfied == (
            bool(args) and all(arg.rsplit("_", 1)[0] in {"apple", "bowl"} for arg in args)
        )


def test_editing_precondition_changes_decoding_fast_and_graph_verification():
    world = build_eai_world_graph("editable_eai_precondition", include_default_constraints=True)
    rule = next(
        lc
        for lc in world.default_constraints
        if lc.name == "action_precondition__right_place_inside__source_holding"
    )
    for lc in world.default_constraints:
        lc.active = lc is rule
    facts = {"initial_state": {"held": {"right": "apple"}, "positive": [["open", "bowl"]]}}
    events = [_ActionEvent("right_place_inside", ("bowl",))]
    original, vocab = policy_for(world, facts)
    tokens = encode(vocab, ("right_place_inside", "bowl"))
    assert original.accepts(tokens) and verify(world, facts, events)
    with world.graph:
        edited = ifL(rule.e[0], notL(world.facts["selected_hand_occupied"]("operation")))
    edited.headLC = False
    rule.e = edited.e
    rule.name = "arbitrary_name_does_not_select_a_policy"
    changed, _ = policy_for(world, facts)
    assert not changed.accepts(tokens)
    assert not verify(world, facts, events)
    assert (
        evaluate_default_world_constraints(
            [set(), set()], events, world, planning_facts=facts, entity_universe=("apple", "bowl")
        ).score
        == 0.0
    )
    assert original.accepts(tokens)  # Compiled snapshots remain immutable.
    key = next(key for key, lc in world.graph.logicalConstrains.items() if lc is rule)
    del world.graph.logicalConstrains[key]
    relaxed, _ = policy_for(world, facts)
    assert relaxed.accepts(tokens) and verify(world, facts, events)


def test_verifier_reads_current_primitive_nodes_instead_of_aggregate_flags():
    world = build_eai_world_graph("eai_observation_nodes", include_default_constraints=True)
    for lc in world.default_constraints:
        lc.active = lc.name == "grounded__affordance"
    facts = {"entity_properties": {"cooler": {"openable": False}}}
    prepared = SimpleNamespace(
        task_id="primitive",
        entity_universe=("cooler",),
        tracked_binary_pairs=(),
        planning_facts=facts,
    )
    root = materialize_world_trajectory(
        prepared, [set(), set()], [_ActionEvent("open", ("cooler",))], world
    )
    pending, nodes = [root], []
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.getChildDataNodes() or ())
    action = next(node for node in nodes if node.ontologyNode is world.action)
    action.attributes[f"<{world.grounded_constraints['affordance__satisfied'].name}>"] = (
        torch.tensor([-30.0, 30.0])
    )
    assert verify_world_constraints(root, world).score == 0.0
    key = next(
        key
        for key, lc in world.graph.logicalConstrains.items()
        if lc.name == "grounded__affordance"
    )
    # Independently execute the original DomiKnowS LC verifier on these nodes.
    assert root.verifyResultsLC(key="/local/argmax")[key]["satisfied"] == 0.0
    requirement = next(
        node for node in nodes if node.ontologyNode is world.grounded_constraints.requirement
    )
    requirement.attributes[f"<{world.grounded_constraints.holds.name}>"] = torch.tensor(
        [-30.0, 30.0]
    )
    assert verify_world_constraints(root, world).score == 1.0
    assert root.verifyResultsLC(key="/local/argmax")[key]["satisfied"] == 100.0


def test_effect_rules_establish_remove_and_preserve_correlated_facts():
    source = {("closed", "bowl"), ("not_open", "bowl"), ("closed", "cooler")}
    held = {"left": None, "right": None}
    result = apply_event_effects(_ActionEvent("open", ("bowl",)), source, held)
    assert result == {("open", "bowl"), ("not_closed", "bowl"), ("closed", "cooler")}
    source = {
        ("holds_lh", "character", "apple"),
        ("holds_rh", "character", "cup"),
        ("inside", "apple", "cooler"),
    }
    result = apply_event_effects(
        _ActionEvent("left_place_inside", ("bowl",)), source, {"left": "apple", "right": "cup"}
    )
    assert result == {("inside", "apple", "bowl"), ("holds_rh", "character", "cup")}


def test_removing_effect_formula_changes_state_without_action_name_dispatch():
    bundle = build_state_effect_graph()
    source = {("closed", "bowl")}
    event = _ActionEvent("open", ("bowl",))
    original = compile_state_effects(bundle)
    rule = next(
        lc for lc in bundle.graph.logicalConstrains.values() if lc.name == "opening_effects"
    )
    rule.name = "name_is_only_a_label"
    assert ("open", "bowl") in apply_event_effects(
        event, source, {"left": None, "right": None}, program=compile_state_effects(bundle)
    )
    rule.active = False
    assert (
        apply_event_effects(
            event, source, {"left": None, "right": None}, program=compile_state_effects(bundle)
        )
        == source
    )
    assert ("open", "bowl") in apply_event_effects(
        event, source, {"left": None, "right": None}, program=original
    )


def test_documented_declaration_counts_match_the_configured_graphs():
    from collections import Counter
    from html import unescape
    from pathlib import Path
    from ..graph import create_generation_graph

    world = build_eai_world_graph("documented_eai", include_default_constraints=True)
    assert len(world.default_constraints) == 62
    assert Counter(spec.kind for spec in world.default_preconditions) == {
        "argument_available_in_task": 26,
        "placement_source_ready": 17,
        "release_source_ready": 5,
        "pour_source_ready": 1,
        "destination_open_if_known": 5,
    }
    graph, _ = create_generation_graph(
        vocab=("<eos>", "open", "close", "clean", "sleep", "cup", "door"),
        object_tokens=("cup", "door"),
        action_sequence_tokens=("open", "close", "clean", "sleep"),
        action_object_constraint_tokens={"open": ("door",), "close": ("door",), "clean": ("cup",)},
    )
    rules = [lc for lc in graph.logicalConstrains.values() if lc.headLC]
    assert len(rules) == 6
    assert Counter(type(lc).__name__ for lc in rules) == {"ifL": 4, "atMostAL": 1, "andL": 1}
    effects = build_state_effect_graph()
    effects = [lc for lc in effects.graph.logicalConstrains.values() if lc.headLC]
    assert len(effects) == 26
    document = unescape(
        (Path(__file__).parents[1] / "docs/EAILogicalConstraints.md").read_text(encoding="utf-8")
    )
    assert all(f'name="{lc.name}"' in document for lc in effects)
