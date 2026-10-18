"""Audit world logic against real materialization and independently named cases."""
import pytest

from domiknows.graph.logicalConstrain import andL, ifL, notL, orL
from test_regr.generation.constraint_equivalence import assert_bounded_equivalence
from .constraint_compiler import compile_world_constraints
from .graph import (
    PlanVocabulary, bind_vlabench_contextual_dfa, compile_planner_dfa,
    create_planner_generation_graph, plan_to_tokens, tokens_to_plan,
)
from .world_graph import build_vlabench_world_graph, materialize_plan, verify_plan_constraints


def compile_policy(world, entities, *, operations=2, task=None):
    vocab = PlanVocabulary.from_world(world, max_entities=max(3, len(entities)))
    graph, generation = create_planner_generation_graph(world, vocab, max_operations=operations)
    grammar = compile_planner_dfa(graph, generation, world, vocab, max_operations=operations)
    policy = bind_vlabench_contextual_dfa(grammar, vocab, {"entities": entities, "task": task})
    return vocab, grammar, policy


def encode(plan, vocab, world, entities):
    return tuple(vocab.label_for_token(token) for token in plan_to_tokens(plan, entities, world=world))


def verified(plan, entities, world, *, task=None):
    result = verify_plan_constraints(materialize_plan(plan, entities, world, task_name=task), world)
    return result is None or result.score == 1.0


@pytest.mark.parametrize("entities,task", [
    (("apple", "bowl"), None),
    (("apple", "cooler"), None),
    (("apple", "floor"), None),
    (("apple", "bowl"), "cool_drink"),
    ({0: "apple", 2: "bowl"}, None),
    (("apple", "bowl"), "cook_dishes"),
])
def test_all_small_scene_paths_and_bounded_masks_match_completed_verifier(entities, task):
    world = build_vlabench_world_graph("exhaustive_world_equivalence")
    vocab, grammar, monitor = compile_policy(world, entities, task=task)
    def oracle(sequence):
        plan = tokens_to_plan([vocab.token_for_label(label) for label in sequence], world=world)
        return verified(plan, entities, world, task=task)
    audit = assert_bounded_equivalence(
        grammar, monitor, oracle, max_steps=7, terminal_symbols={vocab.eos_label},
    )
    assert audit.candidates > 40 and audit.checked_prefixes > 10
    assert audit.rejected > 0
    assert (audit.accepted == 0) == (task == "cook_dishes")


def op(skill, pointer=None):
    if skill in {"pick", "press", "open_door"}:
        params = {"target_entity_name": pointer}
    elif skill in {"place", "pour", "insert"}:
        params = {"target_container_name": pointer}
    else:
        params = {}
    return {"name": skill, "params": params}


@pytest.mark.parametrize("entities,plan,expected", [
    (("apple", "bowl"), [op("pick", 0), op("place", 1)], True),
    (("apple", "bowl"), [op("pick", "apple"), op("place", "bowl")], True),
    (("apple", "bowl"), [op("pick", 0), op("place", "apple")], False),
    (("apple", "bowl"), [op("pick", 0), op("place", 0)], False),
    (("apple", "bowl"), [op("pick", 2), op("place", 1)], False),
    (("apple", "floor"), [op("pick", 1), op("lift")], False),
    (("apple", "cooler"), [op("pick", 0), op("place", 1)], False),
    (("apple", "cooler"), [op("pick", 0), op("open_door", 1), op("pick", 0), op("place", 1)], False),
    (("apple", "cooler_a", "cooler_b"), [op("pick", 0), op("open_door", 1), op("pick", 0), op("place", 2)], False),
    (("apple", "bowl", "plate"), [op("pick", 0), op("place", 1), op("pick", 0), op("place", 2)], False),
    (("apple", "bowl", "plate"), [op("pick", 0), op("place", 1), op("pick", 0), op("place", 1)], False),
    (("apple", "bowl", "pear", "plate"), [op("pick", 0), op("place", 1), op("pick", 2), op("place", 3)], True),
    (("apple", "bowl"), [op("pick", 0), op("pour", 1), op("place", 0)], False),
])
def test_independent_semantic_counterexamples(entities, plan, expected):
    world = build_vlabench_world_graph("independent_semantic_examples")
    vocab, grammar, monitor = compile_policy(world, entities, operations=len(plan))
    sequence = encode(plan, vocab, world, entities)
    assert grammar.accepts(sequence), "counterexample must exercise world logic, not syntax"
    assert verified(plan, entities, world) == expected
    assert monitor.accepts(sequence) == expected


def test_deactivating_and_adding_rules_changes_both_verified_and_decoded_languages():
    entities = ("apple", "bowl")
    world = build_vlabench_world_graph("mutable_constraint_equivalence")
    vocab, _, original = compile_policy(world, entities)
    self_placement = [op("pick", 0), op("place", 0)]
    sequence = encode(self_placement, vocab, world, entities)
    assert not original.accepts(sequence) and not verified(self_placement, entities, world)
    rule = next(rule for rule in world.graph.logicalConstrains.values() if rule.name == "anti_self_placement")
    rule.active = False
    _, _, relaxed = compile_policy(world, entities)
    assert relaxed.accepts(sequence)
    assert verified(self_placement, entities, world)
    assert "anti_self_placement" in compile_world_constraints(world).coverage.inactive
    # Existing compiled policies retain their original logical snapshot.
    assert not original.accepts(sequence)
    with world.graph:
        ifL(world.operation("x"), notL(world.is_self_placement("x")), name="new_rule_without_decoder_changes")
    _, _, restored = compile_policy(world, entities)
    assert not restored.accepts(sequence)
    assert not verified(self_placement, entities, world)
    assert "new_rule_without_decoder_changes" in compile_world_constraints(world).coverage.compiled



def test_no_active_world_constraints_is_consistent_with_verification():
    world = build_vlabench_world_graph("all_inactive_equivalence")
    for constraint in world.graph.logicalConstrains.values():
        constraint.active = False
    assert not world.has_constraints
    entities = ("apple", "cooler")
    vocab, _, monitor = compile_policy(world, entities)
    plan = [op("pick", 0), op("place", 0)]
    assert monitor.accepts(encode(plan, vocab, world, entities))
    assert verify_plan_constraints(materialize_plan(plan, entities, world), world) is None



def test_new_boolean_rule_matches_completed_verifier_for_all_small_paths():
    def boolean_policy(w):
        ifL(w.operation("x"), orL(w.skills["press"]("x"),
                                  andL(w.skills["pick"]("x"), notL(w.targets_static_fixture("x")))),
            name="new_boolean_rule")
    world = build_vlabench_world_graph("boolean_world_equivalence", constraint_builders=(boolean_policy,))
    entities = ("floor", "button0")
    vocab, grammar, monitor = compile_policy(world, entities)
    def oracle(sequence):
        plan = tokens_to_plan([vocab.token_for_label(label) for label in sequence], world=world)
        return verified(plan, entities, world)
    audit = assert_bounded_equivalence(
        grammar, monitor, oracle, max_steps=7, terminal_symbols={vocab.eos_label},
    )
    assert audit.accepted > 0 and audit.rejected > audit.accepted
    assert audit.checked_masks > audit.checked_prefixes
