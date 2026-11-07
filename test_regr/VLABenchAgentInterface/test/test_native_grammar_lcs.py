"""The native JSON decoder runs on the compiled ``graph.py`` grammar and accepts the hand-written language exactly."""
import json

import pytest

from ..generation_runtime import compile_plan_language, profile_vocabulary
from ..graph import create_plan_grammar_graph
from ..native_planning import NativePlanDFA, native_schema
from .legacy_native_grammar import LegacyNativePlanDFA

EOS = 1000
BYTES = {b: bytes([b]) for b in range(32, 127)} | {9: b"\t", 10: b"\n", 13: b"\r"}
# JSON string escapes are lexical (p spells p); they multiply states without touching the grammar.
LEXICAL = tuple(b for b in BYTES if b != ord("\\"))
CONFIGS = [
    ("benchmark", 3, 3, None),
    ("benchmark", 2, 4, None),
    ("benchmark", 0, 2, None),
    ("world", 2, 3, None),
    ("legacy", 2, 3, None),
    ("benchmark", 2, 3, (("pick", "place"), ("press",))),
]


def pair(profile, entities, operations, patterns):
    kwargs = dict(entity_count=entities, max_operations=operations, grammar_profile=profile, patterns=patterns)
    return NativePlanDFA(BYTES, EOS, **kwargs), LegacyNativePlanDFA(BYTES, EOS, **kwargs)


def lexical(state):
    """The lexer's fields; the grammar field differs in kind (compiled DFA state vs. prefix tuples)."""
    return None if state is None else (state[0],) + state[2:]


def labels(plan):
    return list(json.dumps(plan, separators=(",", ":")).encode()) + [EOS]


@pytest.mark.parametrize("profile,entities,operations,patterns", CONFIGS)
def test_the_inventory_is_the_hand_written_one(profile, entities, operations, patterns):
    new, old = pair(profile, entities, operations, patterns)
    assert new.actions == old.actions
    assert new.skill_arguments == dict(old.skill_arguments)
    assert new.operations == old.operations and new.ends == old.ends and new.end_action == old.end_action
    assert new.possible == old.possible and new.node_actions == old.node_actions


@pytest.mark.parametrize("profile,entities,operations,patterns", CONFIGS)
def test_every_reachable_byte_transition_matches(profile, entities, operations, patterns):
    new, old = pair(profile, entities, operations, patterns)
    start = (new.start_state, old.start_state)
    assert lexical(start[0]) == lexical(start[1])
    seen, frontier, accepting = {start}, [start], 0
    while frontier:
        state, legacy = frontier.pop()
        for byte in LEXICAL:
            following, expected = new.byte_step(state, byte), old.byte_step(legacy, byte)
            assert lexical(following) == lexical(expected), (state, legacy, chr(byte))
            if following is not None and (following, expected) not in seen:
                seen.add((following, expected))
                frontier.append((following, expected))
        accepting += state[2] == "closed"
        assert len(seen) < 200_000
    assert (accepting > 0) == (entities > 0) and (len(seen) > 10 or entities == 0)


@pytest.mark.parametrize("profile", ["benchmark", "world", "legacy"])
def test_full_size_pattern_admission_matches(profile):
    operations = 8 if profile == "legacy" else 16
    new, old = pair(profile, 64, operations, None)
    assert new.actions == old.actions
    start = (new.start_state[1], old.start_state[1], 0)
    seen, frontier = {start}, [start]
    while frontier:
        grammar, prefixes, count = frontier.pop()
        assert new.plan_closes(grammar) == (() in prefixes and count > 0)
        # At the budget the old DFA refused the operation bytes; the compiled budget refuses every skill.
        expected = old.allowed_skills(prefixes, count) if count < operations else frozenset()
        assert new.allowed_skills(grammar) == expected, (prefixes, count)
        for skill in expected:
            following = (new.next_semantics(grammar, skill), old.next_semantics(prefixes, skill), count + 1)
            if following not in seen:
                seen.add(following)
                frontier.append(following)
    assert len(seen) > operations


def test_benchmark_plans_are_accepted_and_broken_ones_refused():
    new, old = pair("benchmark", 4, 4, None)
    good = [{"name": "pick", "params": {"target_entity_name": 1}}, {"name": "place", "params": {"target_container_name": 2}}]
    cases = [good, good[:1], [], [{"name": "pick", "params": {"target_entity_name": 4}}, good[1]],
             [{"name": "pick", "params": {}}, good[1]], good + good + good]
    for plan in cases:
        assert new.accepts(labels(plan)) == old.accepts(labels(plan))
    escaped = list(rb'[{"name":"pick","params":{"target_entity_name":1}},') + labels(good[1:])[1:]
    assert new.accepts(escaped) == old.accepts(escaped) == True
    assert new.accepts(labels(good)) and not new.accepts(labels(good[:1])) and not new.accepts(labels([]))


def benchmark_grammar(entities=4, operations=4):
    arguments, patterns = native_schema("benchmark")
    vocabulary = profile_vocabulary(arguments, entities)
    return vocabulary, *create_plan_grammar_graph(vocabulary, patterns, max_operations=operations,
                                                  graph_name="test_native_benchmark_grammar")


def test_the_native_decoder_runs_on_the_graph_py_declaration():
    dfa = NativePlanDFA(BYTES, EOS, entity_count=4, max_operations=4)
    bundle = dfa.language.bundle
    assert [rule.name for rule in bundle.planner_grammar_rules] == [
        "plan_progress_positions", "plan_operation_arguments", "plan_pattern_steps", "plan_only_completed_subtasks_end"]
    assert bundle.planner_grammar is bundle.planner_grammar_rules[0]
    assert all(rule in dfa.language.graph.logicalConstrains.values() for rule in bundle.planner_grammar_rules)
    assert dfa.language.vocabulary.skills == tuple(native_schema("benchmark")[0])


def test_deactivating_the_declared_grammar_changes_the_language():
    lone_pick = [{"name": "pick", "params": {"target_entity_name": 1}}]
    vocabulary, graph, bundle = benchmark_grammar()
    dfa = lambda: NativePlanDFA(BYTES, EOS, entity_count=4, max_operations=4,
                                language=compile_plan_language(graph, bundle, vocabulary))
    assert not dfa().accepts(labels(lone_pick))
    bundle.planner_grammar.active = False
    assert dfa().accepts(labels(lone_pick))


def test_the_plan_frame_rules_are_named_and_editable():
    vocabulary, graph, bundle = benchmark_grammar(operations=2)
    assert [rule.name for rule in bundle.plan_frame_rules] == [
        "plan_requires_eos", "plan_eos_closure", "plan_token_budget", "plan_operation_budget"]
    assert all(rule in graph.logicalConstrains.values() for rule in bundle.plan_frame_rules)
    step = {"name": "pick", "params": {"target_entity_name": 1}}, {"name": "place", "params": {"target_container_name": 2}}
    two_subtasks = [*step, *step]
    dfa = lambda: NativePlanDFA(BYTES, EOS, entity_count=4, max_operations=4,
                                language=compile_plan_language(graph, bundle, vocabulary))
    assert not dfa().accepts(labels(two_subtasks))  # four operations over a budget of two
    for rule in bundle.plan_frame_rules[2:]:
        rule.active = False
    assert dfa().accepts(labels(two_subtasks))

