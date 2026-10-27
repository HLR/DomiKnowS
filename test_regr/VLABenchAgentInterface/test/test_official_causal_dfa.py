import json

import pytest

from ..native_planning import NativePlanDFA
from ..official.causal_dfa import CausalPlanDFA

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
