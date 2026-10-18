"""Semantic compiler regressions: formulas, scene binding and token decisions."""

from itertools import product

import pytest

from domiknows.graph.logicalConstrain import andL, atMostAL, existsL, ifL, notL, orL, sumL
from .constraint_compiler import WorldConstraintCompilationError, WorldConstraintDFA
from .graph import PlanVocabulary, bind_vlabench_contextual_dfa, compile_planner_dfa, create_planner_generation_graph, plan_to_tokens
from .world_graph import build_vlabench_world_graph, materialize_plan, verify_plan_constraints


def runtime(*, builders=(), defaults=True, entities=("apple", "bowl"), task=None, operations=4, planning_facts=None):
    world = build_vlabench_world_graph("test_compiled_world", constraint_builders=builders,
                                      include_default_constraints=defaults)
    vocab = PlanVocabulary.from_world(world, max_entities=4)
    graph, generation = create_planner_generation_graph(world, vocab, max_operations=operations)
    base = compile_planner_dfa(graph, generation, world, vocab, max_operations=operations)
    context = {"entities": entities, "task": task, "planning_facts": planning_facts}
    return world, vocab, bind_vlabench_contextual_dfa(base, vocab, context)


def labels(vocab, *tokens):
    return [vocab.label_for_token(token) for token in tokens]


def advance(dfa, vocab, *tokens):
    state = dfa.start_state
    for label in labels(vocab, *tokens):
        state = dfa.step(state, label)
        assert state is not None
    return state


def test_actual_formula_controls_decoding_and_names_are_only_diagnostics():
    def forbid_self(w):
        ifL(w.operation("x"), notL(w.is_self_placement("x")), name="arbitrary_rule_name")
    _, vocab, restricted = runtime(builders=(forbid_self,), defaults=False)
    _, _, unrestricted = runtime(defaults=False)
    sequence = labels(vocab, "skill:pick", "arg:target_entity_name", "obj:0",
                      "skill:place", "arg:target_container_name", "obj:0", "<eos>")
    assert not restricted.accepts(sequence)
    assert unrestricted.accepts(sequence)
    assert [rule.name for rule in restricted.rules] == ["arbitrary_rule_name"]


def test_nested_boolean_formula_and_new_supported_constraint_change_language():
    def choose(w):
        ifL(w.operation("x"), orL(w.skills["press"]("x"),
                                 andL(w.skills["pick"]("x"), notL(w.targets_static_fixture("x")))),
            name="boolean_policy")
    _, vocab, dfa = runtime(builders=(choose,), defaults=False, entities=("floor", "apple"))
    state = advance(dfa, vocab, "skill:pick", "arg:target_entity_name")
    assert vocab.label_for_token("obj:0") not in dfa.allowed_tokens(state)
    assert vocab.label_for_token("obj:1") in dfa.allowed_tokens(state)
    assert dfa.accepts(labels(vocab, "skill:press", "arg:target_entity_name", "obj:1", "<eos>"))


@pytest.mark.parametrize("builder, diagnostic", [
    (lambda w: sumL(w.operation("x"), name="value_rule"), "value_rule"),
    (lambda w: atMostAL(w.operation("x"), limit=1, name="accumulated_rule"), "accumulated_rule"),
])
def test_unsupported_world_formula_fails_compilation(builder, diagnostic):
    with pytest.raises(WorldConstraintCompilationError, match=diagnostic):
        runtime(builders=(builder,))


def test_disabled_unsupported_rule_is_not_compiled():
    def inactive(w):
        existsL(w.operation("x"), name="inactive_quantifier", active=False)
    _, _, dfa = runtime(builders=(inactive,))
    assert "inactive_quantifier" not in {rule.name for rule in dfa.rules}


def test_pointer_static_and_self_constraints_remove_object_tokens():
    _, vocab, dfa = runtime(entities=("apple", "floor", "bowl"))
    assert isinstance(dfa, WorldConstraintDFA)
    state = advance(dfa, vocab, "skill:pick", "arg:target_entity_name")
    allowed = dfa.allowed_tokens(state)
    assert vocab.label_for_token("obj:1") not in allowed
    assert vocab.label_for_token("obj:3") not in allowed
    state = advance(dfa, vocab, "skill:pick", "arg:target_entity_name", "obj:0", "skill:place", "arg:target_container_name")
    assert dfa.allowed_tokens(state) == {vocab.label_for_token("obj:1"), vocab.label_for_token("obj:2")}


def test_no_valid_placement_returns_empty_set_without_relaxing_constraints():
    _, vocab, dfa = runtime(entities=("apple", "cooler"))
    state = advance(dfa, vocab, "skill:pick", "arg:target_entity_name", "obj:0", "skill:place", "arg:target_container_name")
    assert dfa.allowed_tokens(state) == set()


def test_completion_rule_controls_eos_even_at_operation_limit():
    _, vocab, dfa = runtime(task="cook_dishes", operations=2)
    state = advance(dfa, vocab, "skill:pick", "arg:target_entity_name", "obj:0",
                    "skill:place", "arg:target_container_name", "obj:1")
    assert vocab.eos_label not in dfa.allowed_tokens(state)
    assert dfa.step(state, vocab.eos_label) is None
    assert not dfa.is_accepting(state)
    _, _, no_completion = runtime(defaults=False, task="cook_dishes", operations=2)
    assert no_completion.accepts(labels(vocab, "skill:pick", "arg:target_entity_name", "obj:0",
                                       "skill:place", "arg:target_container_name", "obj:1", "<eos>"))


def test_transition_monitor_preserves_destination_across_intervening_pick():
    _, vocab, dfa = runtime(entities=("apple", "pear", "bowl", "plate"), planning_facts={"destinations": {"apple": "bowl", "pear": "bowl"}})
    state = advance(dfa, vocab, "skill:pick", "arg:target_entity_name", "obj:0", "skill:place",
                    "arg:target_container_name", "obj:2", "skill:pick", "arg:target_entity_name",
                    "obj:1", "skill:place", "arg:target_container_name")
    assert dfa.allowed_tokens(state) == {vocab.label_for_token("obj:2")}


def test_bounded_plans_agree_with_independent_world_verification():
    world, vocab, dfa = runtime(entities=("apple", "cooler"), operations=2)
    for pattern in world.subtask_patterns:
        if len(pattern) > 2:
            continue
        for pointers in product(range(2), repeat=sum(len(world.skill_arguments[skill]) for skill in pattern)):
            arguments = iter(pointers)
            plan = [{"name": skill, "params": {key: next(arguments) for key in world.skill_arguments[skill]}}
                    for skill in pattern]
            encoded = [vocab.label_for_token(token) for token in plan_to_tokens(plan, ("apple", "cooler"), world=world)]
            verified = verify_plan_constraints(materialize_plan(plan, ("apple", "cooler"), world), world)
            assert dfa.accepts(encoded) == (verified.score == 1.0), plan
