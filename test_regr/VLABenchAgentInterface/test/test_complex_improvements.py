from itertools import product
import pytest
import torch

from domiknows.generation.dfa.pattern_language import BoundedPatternLanguage
from domiknows.generation.decision_trace import decision_record
from domiknows.generation.grounded import PlanningFacts, Operation
from ..complex_grounding import complex_planning_facts
from ..graph import plan_to_tokens, tokens_to_plan
from ..planning_augmentation import permute_entities
from ..training import build_constraint_runtime
from ..world_graph import validate_plan, materialize_plan, verify_plan_constraints


def op(name, destination=None, subject=None):
    return {"name": name, "params": {**({"target_container_name": destination} if destination is not None else {}), **({"target_entity_name": subject} if subject is not None else {})}}


def test_compact_language_matches_independent_enumeration():
    patterns = (("a",), ("a", "b"), ("b", "a"))
    language = BoundedPatternLanguage(patterns, 6)
    sentences = set()
    def extend(prefix):
        for p in patterns:
            if len(prefix + p) <= 6 and prefix + p not in sentences:
                sentences.add(prefix + p)
                extend(prefix + p)
    extend(())
    for n in range(7):
        for word in product("ab", repeat=n):
            assert (word in language) == (word in sentences)


def test_empty_and_epsilon_declared_languages(runtime):
    from ..graph import _domain_dfa
    for sentences, accepts_empty in ((frozenset(), False), (frozenset({()}), True)):
        dfa = _domain_dfa(runtime.world_bundle, runtime.vocabulary, 12, sentences)
        state = dfa.step(dfa.start_state, runtime.vocabulary.eos_label)
        assert (state is not None and dfa.is_accepting(state)) == accepts_empty


@pytest.fixture(scope="module")
def runtime():
    return build_constraint_runtime(max_entities=6, max_operations=12, name_prefix="complex_improvements")


def agrees(runtime, entities, facts, plan, task=None):
    policy = runtime.dfa_for("vlabench", {"entities": entities, "planning_facts": facts, "task": task})
    labels = [runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(plan, entities, world=runtime.world_bundle)]
    decoded = policy.accepts(labels)
    result = verify_plan_constraints(materialize_plan(plan, entities, runtime.world_bundle, policy=policy), runtime.world_bundle)
    assert decoded == (result is None or result.score == 1)
    return decoded


def test_push_optional_role_roundtrip_and_independent_validation(runtime):
    for plan in ([op("pick", subject=0), op("push")], [op("pick", subject=0), op("push", destination=1)]):
        tokens = plan_to_tokens(plan, ("book", "shelf"), world=runtime.world_bundle)
        assert tokens_to_plan(tokens) == plan
        assert validate_plan(plan, entity_table=("book", "shelf")).valid
        assert agrees(runtime, ("book", "shelf"), None, plan)
    assert not validate_plan([op("pick", subject=0), op("push", destination=99)], entity_table=("book", "shelf")).valid
    assert not agrees(runtime, ("book", "shelf"), None, [op("pick", subject=0), op("push", destination=0)])


def test_nine_operation_book_plan_and_wrong_order(runtime):
    entities = ("century17_first_book", "century19_second_book", "century21_third_book", "shelf")
    facts = complex_planning_facts("book_rearrange", "earliest to latest publication", entities)
    plan = [action for i in range(3) for action in (op("pick", subject=i), op("push", destination=3), op("pull"))]
    assert agrees(runtime, entities, facts, plan, "book_rearrange")
    wrong = plan[3:6] + plan[:3] + plan[6:]
    assert not agrees(runtime, entities, facts, wrong, "book_rearrange")
    assert not agrees(runtime, entities, facts, plan[:-3], "book_rearrange")


def test_hammer_tool_milestones_and_destination(runtime):
    entities = ("hammer", "nail", "counter", "the night_painting")
    facts = complex_planning_facts("hammer_nail_and_hang_picture", "Hang the The Night on the wall", entities)
    plan = [op("pick", subject=0), op("push", destination=1), op("place", destination=2), op("pick", subject=3), op("push", destination=1)]
    assert agrees(runtime, entities, facts, plan, "hammer_nail_and_hang_picture")
    assert not agrees(runtime, entities, facts, [op("pick", subject=3), op("push", destination=1)], "hammer_nail_and_hang_picture")


def test_repeated_action_milestones_advance_from_source_state_only():
    facts = PlanningFacts.parse({"milestones": [{"name": "first", "action": "pour"}, {"name": "second", "action": "pour", "requires": ["first"]}]})
    state, checks = facts.evaluate(facts.initial, Operation("pour"))
    assert state.milestones == {"first"}
    assert next(c for c in checks if c.family == "milestone_order").satisfied
    assert facts.completion(state).satisfied is False
    state, _ = facts.evaluate(state, Operation("pour"))
    assert state.milestones == {"first", "second"}


def test_recipe_goals_and_unavailable_inputs():
    facts = complex_planning_facts("cook_dishes", "Prepare tomato_and_egg_stir_fry", ("plate", "tomato", "egg", "fish"))
    # The goals are rules in knowledge/complex_lcs.py; the facts carry the bound recipe and plate.
    from ..complex_grounding import bound_subjects
    assert facts["complex_request"]["recipes"] == ["tomato_and_egg_stir_fry"]
    assert facts["complex_request"]["roles"]["plate"] == ["plate"]
    assert bound_subjects("cook_dishes", facts["complex_request"]) == ["tomato", "egg"]
    assert facts["required_skill_sequence"] == ["pick", "place"] * 2
    assert complex_planning_facts("cook_dishes", "unknown dish", ("plate",))["complex_grounding_report"]["status"] == "unavailable"
    assert complex_planning_facts("texas_holdem", "win", ("card",))["complex_grounding_report"]["status"] == "unavailable"


@pytest.mark.parametrize("instruction", [
    "Mix Na2CO3 solution and HCl solution to produce NaCl and CO2.",
    "Mix Na2CO3 and HCl solutions to produce NaCl and CO2.",
    "Measure solutions of Na2CO3 and HCl, resulting in NaCl and CO2.",
    "Neutralize a solution using Na2CO3 and HCl, resulting in NaCl.",
])
def test_chemical_products_are_not_reactant_goals(instruction):
    facts = complex_planning_facts("take_chemistry_experiment", instruction, ("Na2CO3", "HCl", "NaCl", "flask", "table"))
    subjects = set(facts["complex_request"]["roles"]["reagent"])
    assert subjects == {"Na2CO3", "HCl"}


def test_action_specific_destinations_and_duplicate_rejection():
    facts = PlanningFacts.parse({"action_destinations": [
        {"action": "pour", "subject": "cup", "destination": "flask"},
        {"action": "place", "subject": "cup", "destination": "table"}]}, entities=("cup", "flask", "table"))
    assert facts.destination_for("pour", "cup") == "flask"
    assert facts.destination_for("place", "cup") == "table"
    with pytest.raises(ValueError, match="duplicate"):
        PlanningFacts.parse({"action_destinations": [
            {"action": "pour", "subject": "cup", "destination": "flask"},
            {"action": "pour", "subject": "cup", "destination": "table"}]}, entities=("cup", "flask", "table"))


def test_pointer_permutation_preserves_entity_meaning():
    ctx, plan = permute_entities({"entity_table": ("button0", "button1", "button2")}, [op("press", subject=1)], [2, 0, 1])
    assert ctx["entity_table"][plan[0]["params"]["target_entity_name"]] == "button1"


def test_complex_grounding_ignores_references_and_hidden_conditions(tmp_path):
    import json
    from ..dataset import load_planning_examples
    episode = tmp_path / "Complex/book_rearrange/example0"
    for name in ("input", "output", "env_config"):
        (episode / name).mkdir(parents=True)
    (episode / "input/instruction.txt").write_text("Arrange the books from earliest to latest publication")
    config = {"task": {"components": [{"name": name} for name in ("century17_first_book", "century21_last_book", "shelf")],
                       "conditions": {"order": {"entities": ["hidden_wrong_answer"]}}}}
    (episode / "env_config/scene.json").write_text(json.dumps(config))
    output = episode / "output/operation_sequence.json"
    output.write_text(json.dumps({"skill_sequence": [op("pick", subject=0)]}))
    first = load_planning_examples(tmp_path)[0]
    output.write_text(json.dumps({"skill_sequence": [op("pick", subject=1)]}))
    config["task"]["conditions"] = {"answer": {"target_entity": "other_hidden_answer"}}
    (episode / "env_config/scene.json").write_text(json.dumps(config))
    second = load_planning_examples(tmp_path)[0]
    assert first.entities == second.entities == ("century17_first_book", "century21_last_book", "shelf")
    assert first.planning_facts == second.planning_facts
    assert first.operation_sequence != second.operation_sequence


def test_decision_observation_is_read_only(runtime):
    logits = torch.arange(runtime.vocabulary.label_count, dtype=torch.float, requires_grad=True)
    record = decision_record(logits, {2, 3}, 3, runtime.vocabulary)
    assert record["chosen"] == runtime.vocabulary.token_for_label(3)
    assert sum(record["allowed_probabilities"].values()) == pytest.approx(1)
    assert logits.requires_grad
