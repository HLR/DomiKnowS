import json
from itertools import product

import pytest

from ..native_planning import NativePlanDFA
from ..official.causal_dfa import CausalPlanDFA
from ..official.causal_graph import build_causal_graph

EOS = 200


def character_dfa(cls=CausalPlanDFA, entity_count=12, **kwargs):
    tokens = {byte: bytes([byte]) for byte in [9, 10, 13, *range(32, 127)]}
    tokens[130] = b'}},{"name":"place","params":{"target_container_name":'
    return cls(tokens, EOS, entity_count=entity_count, **kwargs)


def op(name, entity=None, container=None):
    params = {}
    if entity is not None:
        params["target_entity_name"] = entity
    if container is not None:
        params["target_container_name"] = container
    return {"name": name, "params": params}


def labels(plan):
    return list(json.dumps(plan, separators=(",", ":")).encode()) + [EOS]


def accepted_by_both(plan):
    return character_dfa(NativePlanDFA).accepts(labels(plan)), character_dfa().accepts(labels(plan))


@pytest.mark.parametrize("plan", [
    [op("pick", 2), op("place", container=2)],                                    # self-placement
    [op("pick", 2), op("pour", container=3), op("place", container=3)],           # repeated destination
    [op("pick", 2), op("place", container=3), op("pick", 3), op("place", container=4)],  # destination picked
    [op("pick", 2), op("push", container=2), op("pull")],
])
def test_causal_rules_reject_what_the_grammar_accepts(plan):
    assert accepted_by_both(plan) == (True, False)


@pytest.mark.parametrize("plan", [
    [op("pick", 2), op("place", container=3)],
    [op("pick", 3), op("pour", container=1), op("place", container=0),
     op("pick", 5), op("pour", container=1), op("place", container=0),
     op("pick", 5), op("pour", container=1), op("place", container=0)],  # take_chemistry_experiment shape
    [op("pick", 2), op("place", container=3), op("pick", 4), op("place", container=3)],
    [op("press", 3)],
    [op("pick", 1), op("place", container=10)],
])
def test_reference_shapes_stay_accepted(plan):
    assert accepted_by_both(plan) == (True, True)


def test_pretty_printed_plans_never_reach_a_dead_end():
    dfa = character_dfa(entity_count=30)  # "2" may still become 20..29
    state = dfa.start_state
    for label in b'[{"name":"pick","params":{"target_entity_name":2}},{"name":"place","params":{"target_container_name":2':
        state = dfa.step(state, label)
    assert state is not None
    assert dfa.step(state, ord("\n")) is None and dfa.step(state, ord("}")) is None
    assert dfa.step(state, ord("1")) is not None
    state = dfa.start_state
    pretty = json.dumps([op("pick", 2), op("place", container=3)], indent=4).encode()
    for label in pretty:
        state = dfa.step(state, label)
        assert state is not None and dfa.allowed_tokens(state)


def test_forbidden_ids_are_pruned_before_the_operation_closes():
    dfa = character_dfa()
    state = dfa.start_state
    for label in labels([op("pick", 2)])[:-2]:  # stop before the closing "]" of the list
        state = dfa.step(state, label)
    state = dfa.step(state, ord(","))
    for label in b'{"name":"place","params":{"target_container_name":':
        state = dfa.step(state, label)
    allowed = dfa.allowed_tokens(state)
    assert ord("2") not in allowed and ord("3") in allowed
    # Multi-byte tokens are checked the same way.
    start = dfa.start_state
    for label in b'[{"name":"pick","params":{"target_entity_name":2':
        start = dfa.step(start, label)
    assert dfa.step(start, 130) is not None
    assert dfa.step(dfa.step(start, 130), ord("2")) is None


def reference_admits(skill, pointer, causal):
    """Independent pre-refactor semantics, without graph helpers or compilation."""
    subject, destinations, former = causal
    if skill == "pick":
        return pointer not in former
    if skill in {"place", "insert", "pour", "push"}:
        return pointer != subject and pointer not in destinations
    return True


def test_all_bounded_observations_match_independent_causal_rules():
    dfa = character_dfa(entity_count=2)
    histories = ((), (0,), (1,), (0, 1))
    former_sets = tuple(map(frozenset, histories))
    assert len(dfa.causal_program.rules) == 3
    for subject, destinations, former in product((None, 0, 1), histories, former_sets):
        causal = subject, destinations, former
        for action, (skill, pointer) in enumerate(dfa.actions):
            assert dfa.admits(action, causal) == reference_admits(skill, pointer, causal)


@pytest.mark.parametrize("rule_name,plan", [
    ("anti_self_placement", [op("pick", 0), op("place", container=0)]),
    ("distinct_subtask_destinations", [op("pick", 0), op("pour", container=1),
                                       op("place", container=1)]),
    ("destinations_stay_put", [op("pick", 0), op("place", container=1),
                               op("pick", 1), op("place", container=0)]),
])
@pytest.mark.parametrize("change", ["remove", "deactivate"])
def test_declaration_changes_control_actual_token_acceptance(rule_name, plan, change):
    declarations = build_causal_graph()
    rejecting = character_dfa(causal_graph=declarations)
    for key, rule in list(declarations.graph.logicalConstrains.items()):
        if rule.name == rule_name:
            if change == "remove":
                del declarations.graph.logicalConstrains[key]
            else:
                rule.active = False
    relaxed = character_dfa(causal_graph=declarations)
    assert not rejecting.accepts(labels(plan))  # existing compiled snapshots stay unchanged
    assert relaxed.accepts(labels(plan))
    assert len(relaxed.causal_program.rules) == 2


def test_formula_body_and_not_rule_name_determines_admission():
    from domiknows.graph.logicalConstrain import ifL

    declarations = build_causal_graph()
    for key, rule in list(declarations.graph.logicalConstrains.items()):
        if rule.name == "anti_self_placement":
            del declarations.graph.logicalConstrains[key]
        else:
            rule.name = "label_" + rule.name
    # Keep the former name, but reverse its meaning for place operations.
    with declarations.graph:
        ifL(declarations.skills["place"]("operation"),
            declarations.observations["target_is_subject"]("operation"),
            name="anti_self_placement")
    dfa = character_dfa(causal_graph=declarations)
    assert dfa.accepts(labels([op("pick", 0), op("place", container=0)]))
    assert not dfa.accepts(labels([op("pick", 0), op("place", container=1)]))


def test_bounded_plans_and_every_prefix_keep_original_pruning():
    class ReferenceCausalDFA(CausalPlanDFA):
        def admits(self, action, causal):
            return reference_admits(*self.actions[action], causal)

    current = character_dfa(entity_count=2)
    reference = character_dfa(ReferenceCausalDFA, entity_count=2)
    subtasks = [
        [op("pick", subject), op("place", container=destination)]
        for subject, destination in product(range(2), repeat=2)
    ] + [[op("press", 0)], [op("pick", 0), op("pour", container=1), op("place", container=1)]]
    for count in range(3):
        for parts in product(subtasks, repeat=count):
            plan = [operation for part in parts for operation in part]
            assert current.accepts(labels(plan)) == reference.accepts(labels(plan))
            left, right = current.start_state, reference.start_state
            for label in labels(plan):
                assert current.allowed_tokens(left) == reference.allowed_tokens(right)
                left, right = current.step(left, label), reference.step(right, label)
                assert (left is None) == (right is None)
                if left is None:
                    break


def test_press_resets_local_history_but_preserves_former_destinations():
    dfa = character_dfa(entity_count=3)
    causal = 0, (1,), frozenset({1})
    press = dfa.action_index[("press", 2)]
    following = dfa.advance(press, causal)
    assert following == (None, (), frozenset({1}))
    assert dfa.admits(dfa.action_index[("place", 1)], following)
    assert not dfa.admits(dfa.action_index[("pick", 1)], following)


def test_removing_all_causal_formulas_recovers_native_grammar():
    declarations = build_causal_graph()
    declarations.graph.logicalConstrains.clear()
    relaxed = character_dfa(entity_count=2, causal_graph=declarations)
    native = character_dfa(NativePlanDFA, entity_count=2)
    assert not relaxed.causal_program.rules
    plans = [
        [], [op("pick", 0)], [op("press", 1)],
        [op("pick", 0), op("place", container=0)],
        [op("pick", 0), op("pour", container=1), op("place", container=1)],
        [op("pick", 0), op("place", container=1), op("pick", 1), op("place", container=0)],
    ]
    for plan in plans:
        assert relaxed.accepts(labels(plan)) == native.accepts(labels(plan))
