"""Hand-aware semantic compilation from observations, independently of labels."""
from itertools import product
from types import SimpleNamespace

import pytest
from domiknows.generation.dfa.core import DFA
from domiknows.generation.dfa.vocabulary import TokenVocabulary
from test_regr.EmbodiedAgentInterface.utils.constraint_compiler import CompiledEAIPlannerDFA, EAIExecutor, bind_eai_policy, compile_world_constraints
from test_regr.EmbodiedAgentInterface.reward import _ActionEvent
from test_regr.EmbodiedAgentInterface.world_graph import build_eai_world_graph, materialize_world_trajectory, verify_world_constraints


ENTITIES = ("apple", "cup", "bowl", "cooler", "character")
TOKENS = ("left_grasp", "right_grasp", "left_release", "right_release", "left_place_inside", "right_place_inside", "open", "close", "pour", *ENTITIES, "<eos>")


@pytest.fixture(scope="module")
def world():
    return build_eai_world_graph("grounded_eai", include_default_constraints=True)


def policy_for(world, facts, **extra):
    vocab = TokenVocabulary(TOKENS, eos_token="<eos>")
    eos = vocab.label_for_token("<eos>")
    grammar = DFA(frozenset((0, 1)), frozenset(range(len(TOKENS))),
                  {(0, i): 1 if i == eos else 0 for i in range(len(TOKENS))}, 0, frozenset((1,)))
    template = CompiledEAIPlannerDFA.from_dfa(grammar, compile_world_constraints(world), world_constraint_bundle=world, generation_graph=world.graph)
    context = {"entities": ENTITIES, "planning_facts": facts, **extra}
    return bind_eai_policy(template, vocab, context), vocab


def encode(vocab, tokens):
    return [vocab.label_for_token(t) for t in (*tokens, "<eos>")]


def verify(world, facts, events):
    prepared = SimpleNamespace(entity_universe=ENTITIES, tracked_binary_pairs=frozenset(), task_id="grounded", planning_facts=facts)
    executor = EAIExecutor(world, {"planning_facts": facts}, ENTITIES)
    # Materialization receives no reference-derived state at all.
    root = materialize_world_trajectory(prepared, [set() for _ in range(len(events)+1)], events, world)
    evaluation = verify_world_constraints(root, world)
    return evaluation is None or evaluation.score == 1.0


CASES = [
    ("destination", {"initial_state": {"held": {"left": "apple"}}, "destinations": {"apple": "bowl"}}, ("left_place_inside", "cooler")),
    ("holding", {"initial_state": {"held": {"left": "cup"}}}, ("left_release", "apple")),
    ("capacity", {"initial_state": {"held": {"left": "cup"}}, "capacities": {"left": 1}}, ("left_grasp", "apple")),
    ("affordance", {"entity_properties": {"cooler": {"openable": False}}}, ("open", "cooler")),
    ("compatibility", {"initial_state": {"held": {"left": "apple"}}, "compatibility": [{"subject": "apple", "destination": "bowl", "allowed": False}]}, ("left_place_inside", "bowl")),
    ("milestone_order", {"milestones": [{"name": "first", "action": "close"}, {"name": "second", "action": "open", "requires": ["first"]}]}, ("open", "cooler")),
    ("goal_completion", {"initial_state": {"negative": [["open", "cooler"]]}, "goals": {"positive": [["open", "cooler"]]}}, ("left_grasp", "apple")),
    ("preservation", {"initial_state": {"positive": [["open", "cooler"]]}, "preserve": {"positive": [["open", "cooler"]]}}, ("close", "cooler")),
]


@pytest.mark.parametrize("family,facts,tokens", CASES)
def test_each_family_is_compiled_and_can_be_deactivated(family, facts, tokens):
    world = build_eai_world_graph("eai_edit_"+family, include_default_constraints=True)
    # Isolate this LC, so overlapping legacy rules cannot hide its contribution.
    for rule in world.graph.logicalConstrains.values():
        if getattr(rule, "headLC", False):
            rule.active = rule.name == "grounded__"+family
    policy, vocab = policy_for(world, facts)
    encoded = encode(vocab, tokens)
    assert not policy.accepts(encoded)
    events = [_ActionEvent(tokens[0], tokens[1:])]
    assert not verify(world, facts, events)
    rule = next(rule for rule in world.graph.logicalConstrains.values() if rule.name == "grounded__"+family)
    rule.active = False
    relaxed, _ = policy_for(world, facts)
    assert relaxed.accepts(encoded)
    assert verify(world, facts, events)
    assert not policy.accepts(encoded)  # Previous compilation remains immutable.


@pytest.mark.parametrize("comparison", [{"criterion": "max"}, {"criterion": "rank", "rank": 1}])
def test_observed_numeric_target_selection_is_shared_without_changing_hand_semantics(world, comparison):
    facts = {"selections": [{"action": "left_grasp", **comparison,
                              "candidates": ["apple", "cup"],
                              "scores": {"apple": 1, "cup": 2}}]}
    policy, vocab = policy_for(world, facts)
    for entity, expected in (("apple", False), ("cup", True)):
        tokens = ("left_grasp", entity)
        assert policy.accepts(encode(vocab, tokens)) == expected
        assert verify(world, facts, [_ActionEvent("left_grasp", (entity,))]) == expected
    # A requirement for the left hand's action does not silently constrain right_grasp.
    assert policy.accepts(encode(vocab, ("right_grasp", "apple")))


def test_hand_specific_continuation_and_idempotence(world):
    facts = {"initial_state": {"held": {"left": "apple", "right": "cup"}, "positive": [["open", "bowl"]]}, "destinations": {"apple": "bowl", "cup": "cooler"}}
    policy, vocab = policy_for(world, facts)
    assert policy.accepts(encode(vocab, ("left_place_inside", "bowl")))
    assert not policy.accepts(encode(vocab, ("right_place_inside", "bowl")))
    executor = EAIExecutor(world, {"planning_facts": facts}, ENTITIES)
    state, _, _ = executor.advance(executor.planning.initial, "left_place_inside", ("bowl",))
    assert dict(state.holdings) == {"left": None, "right": "cup"}
    assert ("holds_rh", "character", "cup") in state.facts.positive
    state, checks, _ = executor.advance(state, "left_place_inside", ("bowl",))
    assert next(c for c in checks if c.family == "holding").satisfied is True


def test_negative_goals_and_noop_completion(world):
    unknown, vocab = policy_for(world, {"goals": {"positive": [["open", "cooler"]]}})
    assert not unknown.accepts(encode(vocab, ()))
    facts = {"initial_state": {"positive": [["open", "cooler"]]}, "goals": {"negative": [["open", "cooler"]]}}
    policy, vocab = policy_for(world, facts)
    assert not policy.accepts(encode(vocab, ()))
    assert policy.accepts(encode(vocab, ("close", "cooler")))
    facts["initial_state"] = {"negative": [["open", "cooler"]]}
    policy, vocab = policy_for(world, facts)
    assert policy.accepts(encode(vocab, ()))


def test_observation_of_a_protected_entity_does_not_modify_it(world):
    executor = EAIExecutor(world, {"planning_facts": {"protected_entities": ["cooler"]}}, ENTITIES)
    _, checks, _ = executor.advance(executor.planning.initial, "lookat", ("cooler",))
    assert next(c for c in checks if c.family == "preservation").satisfied is True


def test_observed_grasp_replaces_hand_facts_and_release_alias_empties_hand(world):
    executor = EAIExecutor(world, {"planning_facts": {"initial_state": {"held": {"right": "cup"}}}}, ENTITIES)
    state, _, _ = executor.advance(executor.planning.initial, "right_grasp", ("apple",))
    assert ("holds_rh", "character", "cup") not in state.facts.positive
    assert ("holds_rh", "character", "apple") in state.facts.positive
    state, _, _ = executor.advance(state, "right_realease", ("apple",))
    assert dict(state.holdings)["right"] is None
    assert ("holds_rh", "character", "apple") not in state.facts.positive
    executor = EAIExecutor(world, {"planning_facts": {
        "initial_state": {"held": {"right": "apple"}},
        "effects": {"open": {"release_hands": ["right"]}},
    }}, ENTITIES)
    state, _, _ = executor.advance(executor.planning.initial, "open", ("cooler",))
    assert dict(state.holdings)["right"] is None
    assert state.facts.truth(("holds_rh", "character", "apple")) is False


def test_reference_labels_never_initialize_state_and_requests_are_fresh(world):
    facts = {"initial_state": {"held": {"left": None, "right": "cup"}}}
    policy, vocab = policy_for(world, facts, output=["left_grasp", "apple"], tl_goal="inside(apple,bowl)", initial_state=[("holds_lh", "character", "apple")])
    assert dict(policy.adapter.start_state.grounded.holdings)["left"] is None
    assert not policy.accepts(encode(vocab, ("left_place_inside", "bowl")))
    fresh, _ = policy_for(world, {"initial_state": {"held": {"left": "apple"}}})
    assert fresh.accepts(encode(vocab, ("left_place_inside", "bowl")))
    assert policy.adapter.executor.audit([])[2]["checks"]
    types = ["apple"]
    bound, vocab = policy_for(world, {}, generation_entity_types=types)
    types.append("cup")
    assert bound.adapter.executor.context["generation_entity_types"] == ("apple",)
    assert not bound.accepts(encode(vocab, ("open", "cup")))


def test_bounded_hands_match_independent_oracle(world):
    from test_regr.generation.constraint_equivalence import finite_language_dfa, assert_bounded_equivalence
    from domiknows.generation.dfa.predicate_monitor import PredicateMonitorDFA
    facts = {"initial_state": {"held": {"left": "apple", "right": "cup"}}, "destinations": {"apple": "bowl", "cup": "cooler"}}
    policy, vocab = policy_for(world, facts)
    oracle = {}
    for side, destination in product(("left", "right"), ("bowl", "cooler")):
        tokens = (side+"_place_inside", destination)
        expected = (side, destination) in (("left", "bowl"), ("right", "cooler"))
        encoded = encode(vocab, tokens)
        oracle[tuple(encoded)] = expected
        assert policy.accepts(encoded) == expected
        assert verify(world, facts, [_ActionEvent(tokens[0], tokens[1:])]) == expected
        if expected:
            state = policy.start_state
            for label in encoded:
                assert label in policy.allowed_tokens(state, remaining_steps=len(encoded))
                state = policy.step(state, label)
    grammar = finite_language_dfa(oracle)
    restricted = PredicateMonitorDFA(grammar, policy.program, policy.adapter)
    audit = assert_bounded_equivalence(grammar, restricted, oracle.__getitem__, max_steps=3,
                                      terminal_symbols={vocab.eos_label})
    assert audit.candidates == 4 and audit.accepted == 2 and audit.checked_masks > 3
