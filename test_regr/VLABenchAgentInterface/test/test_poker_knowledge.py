"""Independent category fixtures and frozen upstream differential cases."""
import json
from pathlib import Path

import pytest

from domiknows.generation.grounded_facts import FactState, GroundedState
from ..knowledge.poker_lcs import build_poker_knowledge, parse_poker_selection, PokerMonitor, UPSTREAM_REVISION


@pytest.fixture(scope="module")
def knowledge():
    return build_poker_knowledge()


def selection(cards, **kwargs):
    return parse_poker_selection({"cards": [{"entity": f"card{i}", "rank": r, "suit": s, "face_observed": True}
                                             for i, (r, s) in enumerate(cards)], "destination": "mat", **kwargs}, str)


def state(n, selected):
    return GroundedState(facts=FactState(frozenset(("at", f"card{i}", "mat") for i in selected),
                                        frozenset(("at", f"card{i}", "mat") for i in range(n) if i not in selected)))


CASES = [
    (10, [(10,"hearts"),(11,"hearts"),(12,"hearts"),(13,"hearts"),(14,"hearts")], [0,1,2,3,4]),
    (9, [(5,"clubs"),(6,"clubs"),(7,"clubs"),(8,"clubs"),(9,"clubs")], [0,1,2,3,4]),
    (8, [(8,"hearts"),(8,"clubs"),(8,"diamonds"),(8,"spades"),(14,"hearts")], [0,1,2,3]),
    (7, [(3,"hearts"),(3,"clubs"),(3,"diamonds"),(12,"spades"),(12,"hearts")], [0,1,2,3,4]),
    (6, [(2,"spades"),(4,"spades"),(7,"spades"),(9,"spades"),(13,"spades")], [0,1,2,3,4]),
    (5, [(14,"clubs"),(2,"hearts"),(3,"clubs"),(4,"diamonds"),(5,"spades")], [0,1,2,3,4]),
    (4, [(6,"hearts"),(6,"clubs"),(6,"diamonds"),(9,"spades"),(13,"hearts")], [0,1,2]),
    (3, [(2,"hearts"),(2,"clubs"),(8,"diamonds"),(8,"spades"),(13,"hearts")], [0,1,2,3]),
    (2, [(5,"hearts"),(5,"clubs"),(8,"diamonds"),(10,"spades"),(13,"hearts")], [0,1]),
    (1, [(2,"hearts"),(5,"clubs"),(8,"diamonds"),(10,"spades"),(13,"hearts")], [4]),
]


@pytest.mark.parametrize("rank,cards,required", CASES)
def test_all_categories_and_variable_sized_groups(knowledge, rank, cards, required):
    monitor = PokerMonitor(knowledge, selection(cards))
    valid = monitor.completion(state(len(cards), required))[0]
    assert valid.satisfied is True
    assert f"poker__category_{rank}__required_cards" in {r.name for r in valid.requirements}
    assert monitor.completion(state(len(cards), required[1:]))[0].satisfied is False
    # Upstream requires containment of its returned group, not exclusion of extras.
    assert monitor.completion(state(len(cards), range(len(cards))))[0].satisfied is True


def test_ties_preserve_observed_order_instead_of_kickers(knowledge):
    cards = [(2,"hearts"),(2,"clubs"),(2,"diamonds"),(14,"spades"),(14,"clubs"),(14,"hearts"),(7,"spades")]
    monitor = PokerMonitor(knowledge, selection(cards))
    # Earlier twos-full-of-aces wins over later aces-full-of-twos.
    assert monitor.completion(state(7, [0,1,2,3,4]))[0].satisfied is True
    assert monitor.completion(state(7, [0,1,3,4,5]))[0].satisfied is False


def test_upstream_ace_low_flush_quirk_is_preserved(knowledge):
    cards = [(14,"clubs"),(2,"clubs"),(3,"clubs"),(4,"clubs"),(5,"clubs")]
    check = PokerMonitor(knowledge, selection(cards)).completion(state(5, range(5)))[0]
    assert "poker__category_10__required_cards" in {r.name for r in check.requirements}


def test_unknown_faces_order_or_destination_do_not_reveal_hidden_cards(knowledge):
    cards = [{"entity": "ace_of_hearts_face_down", "rank": 14, "suit": "hearts", "face_observed": False}]
    hidden = parse_poker_selection({"cards": cards, "destination": "mat"}, str)
    assert hidden.cards[0][1:] == (None, None)
    for block in (hidden, selection(CASES[0][1], order_known=False), parse_poker_selection(None, str)):
        assert PokerMonitor(knowledge, block).completion(state(5, range(5)))[0].satisfied is None


def test_editing_the_category_formula_changes_enforcement():
    knowledge = build_poker_knowledge()
    monitor = PokerMonitor(knowledge, selection(CASES[8][1]))
    bad = state(5, [])
    assert monitor.completion(bad)[0].satisfied is False
    knowledge.enforce_constraints["poker__category_2__required_cards"].active = False
    assert PokerMonitor(knowledge, monitor.selection).completion(bad)[0].satisfied is True


def test_pinned_upstream_differential_cases(knowledge):
    fixture = json.loads(Path(__file__).with_name("poker_upstream_cases.json").read_text())
    assert fixture["revision"] == UPSTREAM_REVISION
    assert len(fixture["source_sha256"]) == 64
    for case in fixture["cases"]:
        monitor = PokerMonitor(knowledge, selection(case["cards"]))
        required = case["required_indices"]
        check = monitor.completion(state(len(case["cards"]), required))[0]
        assert check.satisfied is True, case
        assert f"poker__category_{case['rank']}__required_cards" in {r.name for r in check.requirements}
        assert monitor.completion(state(len(case["cards"]), required[1:]))[0].satisfied is False, case


def test_rejects_hidden_winning_targets_and_duplicate_faces():
    with pytest.raises(ValueError):
        parse_poker_selection({"winning_cards": ["ace"]}, str)
    with pytest.raises(ValueError):
        selection([(2,"hearts"),(2,"hearts")])


def test_poker_completion_verification_decoding_and_reachability_agree():
    from ..training import build_constraint_runtime
    from ..plan_codec import plan_to_tokens
    from ..world_graph import materialize_plan, verify_plan_constraints
    from ..reachability import ReachablePlanDFA
    runtime = build_constraint_runtime(max_entities=6, max_operations=4, name_prefix="poker_integration")
    entities = tuple(f"card{i}" for i in range(5)) + ("mat",)
    cards = CASES[8][1]
    context = {"task":"texas_holdem", "entities":entities, "planning_facts":{
        "initial_state":{"held":{"right":None}, "negative":[["at", e, "mat"] for e in entities[:5]]},
        "capacities":{"right":1},
        "entity_properties":{e:{"graspable":e in entities[:2], "openable":False, "closeable":False, "pourable":False} for e in entities},
        "destinations":dict.fromkeys(entities[:5], "mat"),
        "compatibility":[{"subject":e,"destination":"mat","allowed":True} for e in entities[:5]],
        "poker_selection":{"cards":[{"entity":i,"rank":r,"suit":s,"face_observed":True} for i,(r,s) in enumerate(cards)],
                           "destination":"mat"}}}
    policy = runtime.dfa_for("vlabench", context)
    plan = [{"name":s,"params":{role:p}} for subject in (0,1)
            for s,role,p in (("pick","target_entity_name",subject),("place","target_container_name",5))]
    encode = lambda p: [runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(p, world=runtime.world_bundle)]
    assert policy.accepts(encode(plan))
    assert not policy.accepts(encode(plan[:2]))
    for value, expected in ((plan, True), (plan[:2], False)):
        root = materialize_plan(value, entities, runtime.world_bundle, policy=policy)
        assert (verify_plan_constraints(root, runtime.world_bundle).score == 1) is expected
    query = policy.action_reachability()
    assert query.can_reach(remaining_actions=4) is True
    assert query.can_reach(remaining_actions=3) is False
    tokens = {i:bytes([i]) for i in range(32,127)}
    native = ReachablePlanDFA(tokens, 200, policy=policy, entity_count=6,
                              grammar_profile="world", max_operations=4, causal=False)
    state_ = native.start_state
    sequence = list(json.dumps(plan).encode()) + [200]
    for token in sequence:
        assert token in native.allowed_tokens(state_)
        state_ = native.step(state_, token)
    assert native.is_accepting(state_)
