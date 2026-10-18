"""Independent symbolic counterexamples for the shared planning state."""
import pytest

from domiknows.generation.grounded import FactState, PlanningFacts, Operation, observations, report


ENTITIES = ("apple", "cup", "bowl", "cooler")


def check(raw, operation, **effects):
    facts = PlanningFacts.parse(raw, entities=ENTITIES)
    state, checks = facts.evaluate(facts.initial, operation, **effects)
    return facts, state, {c.family: c for c in checks}


@pytest.mark.parametrize("raw,operation,family", [
    ({"destinations": {"apple": "bowl"}}, Operation("place", "apple", "cooler"), "destination"),
    ({"initial_state": {"held": {"right": "cup"}}}, Operation("lift", "apple", requires_held=True), "holding"),
    ({"initial_state": {"held": {"right": "cup"}}, "capacities": {"right": 1}}, Operation("pick", "apple", grasp=True), "capacity"),
    ({"capacities": {"right": 0}}, Operation("pick", "apple", grasp=True), "capacity"),
    ({"entity_properties": {"apple": {"graspable": False}}}, Operation("pick", "apple", affordances=(("apple", "graspable"),)), "affordance"),
    ({"compatibility": [{"subject": "apple", "destination": "bowl", "allowed": False}]}, Operation("place", "apple", "bowl"), "compatibility"),
    ({"milestones": [{"name": "first", "action": "open"}, {"name": "last", "action": "place", "requires": ["first"]}]}, Operation("place", "apple", "bowl"), "milestone_order"),
    ({"protected_entities": ["apple"]}, Operation("pick", "apple", writes=("apple",)), "preservation"),
])
def test_known_violations_survive_missing_other_grounding(raw, operation, family):
    _, _, checks = check(raw, operation)
    assert checks[family].satisfied is False
    assert observations(checks.values())[family + "__applicable"]
    assert report(checks.values())["violations"] >= 1


def test_goals_and_invariants_use_post_effect_state():
    facts, state, checks = check({
        "initial_state": {"positive": [["open", "cooler"]]},
        "goals": {"positive": [["at", "apple", "bowl"]]},
        "preserve": {"positive": [["open", "cooler"]]},
    }, Operation("place", "apple", "bowl"), positive=(("at", "apple", "bowl"),), negative=(("open", "cooler"),))
    assert facts.completion(state).satisfied is True
    assert checks["preservation"].satisfied is False


def test_unknown_is_not_known_empty_or_known_false():
    _, _, checks = check({}, Operation("lift", "apple", requires_held=True))
    assert checks["holding"].satisfied is None
    assert not observations(checks.values())["holding__applicable"]
    assert not report(checks.values())["fully_grounded"]
    _, _, checks = check({"initial_state": {"held": {"right": None}}}, Operation("lift", "apple", requires_held=True))
    assert checks["holding"].satisfied is False


def test_milestones_require_explicit_predecessor_progress():
    raw = {"initial_state": {"held": {"right": "apple"}}, "milestones": [
        {"name": "opened", "action": "open", "subject": "cooler"},
        {"name": "placed", "action": "place", "requires": ["opened"]},
    ]}
    facts = PlanningFacts.parse(raw, entities=ENTITIES)
    assert facts.completion(facts.initial).satisfied is False
    state, _ = facts.evaluate(facts.initial, Operation("open", "cooler"))
    state, checks = facts.evaluate(state, Operation("place", "apple", "bowl"))
    assert next(c for c in checks if c.family == "milestone_order").satisfied is True
    assert facts.completion(state).satisfied is True


def test_functional_effect_refutes_wrong_goal_destination():
    facts, state, _ = check({"goals": {"positive": [["at", "apple", "bowl"]]}},
        Operation("place", "apple", "cooler"), positive=(("at", "apple", "cooler"),), clear_relations=(("at", "apple"),))
    assert facts.completion(state).satisfied is False


def test_explicit_effect_refines_default_and_state_is_immutable():
    raw = {"initial_state": {"held": {"right": "apple"}}, "effects": {
        "open": {"negative": [["open", "$subject"]], "release_hands": ["right"]}}}
    facts, state, _ = check(raw, Operation("open", "cooler"), positive=(("open", "cooler"),))
    assert state.facts.truth(("open", "cooler")) is False
    assert dict(state.holdings)["right"] is None
    assert dict(facts.initial.holdings)["right"] == "apple"
    assert hash(state)


def test_declared_effect_cannot_change_another_protected_entity():
    _, _, checks = check({"protected_entities": ["cooler"],
        "effects": {"pick": {"positive": [["open", "cooler"]]}}}, Operation("pick", "apple"))
    assert checks["preservation"].satisfied is False


@pytest.mark.parametrize("raw", [
    {"initial_state": {"positive": [["open", "cooler"]], "negative": [["open", "cooler"]]}},
    {"destinations": {"missing": "bowl"}},
    {"effects": {"open": {"positive": [["open", "$missing"]]}}},
    {"milestones": [{"name": "a", "action": "open", "requires": ["a"]}]},
])
def test_invalid_explicit_grounding_is_rejected(raw):
    with pytest.raises(ValueError):
        PlanningFacts.parse(raw, entities=ENTITIES)
