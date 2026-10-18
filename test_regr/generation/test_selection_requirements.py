"""Numeric target eligibility uses supplied observations, not option positions."""
import pytest
from itertools import product

from domiknows.generation.grounded import GroundedState, Operation, PlanningFacts, report


def facts(**requirement):
    return PlanningFacts.parse({"selections": [{"action": "choose", "candidates": ["a", "b", "c"],
                                                **requirement}]}, entities=("a", "b", "c"))


def eligible(planning, entity):
    _, checks = planning.evaluate(GroundedState(), Operation("choose", subject=entity))
    return next(c for c in checks if c.family == "affordance"), checks


@pytest.mark.parametrize("criterion,threshold,expected", [
    ("min", None, {"b"}), ("max", None, {"a", "c"}),
    ("below", 2, {"b"}), ("above", 1, {"a", "c"}),
])
def test_numeric_selection_and_ties_have_independent_expected_answers(criterion, threshold, expected):
    planning = facts(criterion=criterion, threshold=threshold, scores={"a": 3, "b": 1, "c": 3})
    for entity in ("a", "b", "c"):
        check, _ = eligible(planning, entity)
        assert check.applicable
        assert check.satisfied == (entity in expected)


def test_missing_observations_do_not_conceal_a_known_loser_or_prove_a_winner():
    planning = facts(criterion="max", scores={"a": 3, "b": 1})
    loser, checks = eligible(planning, "b")
    assert loser.satisfied is False
    assert report(checks)["violations"] >= 1
    candidate, checks = eligible(planning, "a")
    assert candidate.satisfied is None
    assert not report(checks)["fully_grounded"]
    assert any(c.family == "selection" and c.missing for c in checks)


def test_destination_selection_and_action_applicability_are_generic():
    planning = facts(argument="destination", criterion="min", scores={"a": 3, "b": 1, "c": 2})
    _, checks = planning.evaluate(GroundedState(), Operation("choose", destination="b"))
    assert next(c for c in checks if c.family == "affordance").satisfied is True
    _, checks = planning.evaluate(GroundedState(), Operation("other", destination="a"))
    assert not next(c for c in checks if c.family == "affordance").applicable


@pytest.mark.parametrize("change", [
    {"criterion": "guess"}, {"argument": "hidden_answer"}, {"candidates": []},
    {"candidates": ["a", "a"]}, {"scores": {"a": float("nan")}},
    {"scores": {"a": float("inf")}}, {"scores": {"a": True}},
    {"scores": {"a": 10**1000}}, {"scores": {"unknown": 1}},
    {"candidates": ["a"], "scores": {"b": 1}},
])
def test_malformed_or_unresolved_numeric_requirements_fail_closed(change):
    with pytest.raises(ValueError):
        facts(**{"criterion": "max", **change})


@pytest.mark.parametrize("values", list(product((1, 2, 3), repeat=3)))
@pytest.mark.parametrize("rank", (0, 1, 2))
def test_rank_matches_independent_order_statistic_including_ties(values, rank):
    scores = dict(zip(("a", "b", "c"), values))
    planning = facts(criterion="rank", rank=rank, scores=scores)
    expected = sorted(values)[rank]
    for entity, value in scores.items():
        check, _ = eligible(planning, entity)
        assert check.satisfied == (value == expected)


def test_rank_partial_observations_enforce_known_violations():
    planning = facts(criterion="rank", rank=0, scores={"a": 1, "b": 3})
    loser, _ = eligible(planning, "b")
    possible, checks = eligible(planning, "a")
    assert loser.satisfied is False
    assert possible.satisfied is None
    assert not report(checks)["fully_grounded"]
    highest = facts(criterion="rank", rank=2, scores={"a": 1, "b": 3})
    assert eligible(highest, "a")[0].satisfied is False


@pytest.mark.parametrize("changes", [
    {"rank": None}, {"rank": True}, {"rank": -1}, {"rank": 3}, {"rank": 1.0},
    {"criterion": "min", "rank": 1},
])
def test_invalid_rank_requirements_are_rejected(changes):
    with pytest.raises(ValueError):
        facts(**{"criterion": "rank", **changes})
