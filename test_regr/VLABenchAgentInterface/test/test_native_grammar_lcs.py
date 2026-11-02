"""The native plan grammar is compiled from logical constraints and accepts the hand-written language exactly."""
import json

import pytest

from ..native_planning import NativePlanDFA, native_schema
from ..official.grammar_graph import build_grammar_graph
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


@pytest.mark.parametrize("profile,entities,operations,patterns", CONFIGS)
def test_the_compiled_inventory_is_the_hand_written_one(profile, entities, operations, patterns):
    new, old = pair(profile, entities, operations, patterns)
    assert new.actions == old.actions
    assert new.skill_arguments == dict(old.skill_arguments)
    assert new.operations == old.operations and new.ends == old.ends and new.end_action == old.end_action
    assert new.possible == old.possible and new.node_actions == old.node_actions


@pytest.mark.parametrize("profile,entities,operations,patterns", CONFIGS)
def test_every_reachable_byte_transition_matches(profile, entities, operations, patterns):
    new, old = pair(profile, entities, operations, patterns)
    assert new.start_state == old.start_state
    seen, frontier, accepting = {new.start_state}, [new.start_state], 0
    while frontier:
        state = frontier.pop()
        for byte in LEXICAL:
            following = new.byte_step(state, byte)
            assert following == old.byte_step(state, byte), (state, chr(byte))
            if following is not None and following not in seen:
                seen.add(following)
                frontier.append(following)
        accepting += state[2] == "closed"
        assert len(seen) < 200_000
    assert (accepting > 0) == (entities > 0) and (len(seen) > 10 or entities == 0)


@pytest.mark.parametrize("profile", ["benchmark", "world", "legacy"])
def test_full_size_pattern_admission_matches(profile):
    operations = 8 if profile == "legacy" else 16
    new, old = pair(profile, 64, operations, None)
    assert new.actions == old.actions
    index, _, _ = new.semantic_automaton()
    for prefixes in index:
        for count in range(operations):
            assert new.allowed_skills(prefixes, count) == old.allowed_skills(prefixes, count)
        # At the budget the old DFA refused the operation bytes; the budget formula refuses every skill.
        assert not new.allowed_skills(prefixes, operations)
        for count in range(operations + 1):
            assert new.plan_closes(prefixes, count) == (() in prefixes and count > 0)


def test_benchmark_plans_are_accepted_and_broken_ones_refused():
    new, old = pair("benchmark", 4, 4, None)
    def labels(plan):
        return list(json.dumps(plan, separators=(",", ":")).encode()) + [EOS]
    good = [{"name": "pick", "params": {"target_entity_name": 1}}, {"name": "place", "params": {"target_container_name": 2}}]
    cases = [good, good[:1], [], [{"name": "pick", "params": {"target_entity_name": 4}}, good[1]],
             [{"name": "pick", "params": {}}, good[1]], good + good + good]
    for plan in cases:
        assert new.accepts(labels(plan)) == old.accepts(labels(plan))
    escaped = list(rb'[{"name":"\u0070ick","params":{"target_entity_name":1}},') + labels(good[1:])[1:]
    assert new.accepts(escaped) == old.accepts(escaped) == True
    assert new.accepts(labels(good)) and not new.accepts(labels(good[:1])) and not new.accepts(labels([]))


def grammar(**changes):
    arguments, patterns = native_schema("benchmark")
    return build_grammar_graph(arguments, patterns)


def test_deactivating_a_formula_changes_the_language():
    def labels(plan):
        return list(json.dumps(plan, separators=(",", ":")).encode()) + [EOS]
    lone_pick = [{"name": "pick", "params": {"target_entity_name": 1}}]
    far_id = [{"name": "press", "params": {"target_entity_name": 7}}]
    graph = grammar()
    rules = {rule.name: rule for rule in graph.graph.logicalConstrains.values()}
    dfa = lambda: NativePlanDFA(BYTES, EOS, entity_count=4, max_operations=4, grammar_graph=graph)
    assert not dfa().accepts(labels(lone_pick)) and not dfa().accepts(labels(far_id))
    rules["plan_completes_subtask_pattern"].active = False
    assert dfa().accepts(labels(lone_pick))
    rules["target_within_id_bound"].active = False
    assert dfa().accepts(labels(far_id))
    assert len(dfa().actions) > len(NativePlanDFA(BYTES, EOS, entity_count=4, max_operations=4).actions)


def test_the_grammar_graph_declares_every_rule():
    names = [rule.name for rule in grammar().compiled().rules]
    assert names == ["operation_exactly_one_skill",
                     *(f"skill_arguments__{s}" for s in native_schema("benchmark")[0]),
                     "target_within_id_bound", "operation_within_budget", "operation_continues_subtask_pattern",
                     "plan_has_an_operation", "plan_completes_subtask_pattern"]
