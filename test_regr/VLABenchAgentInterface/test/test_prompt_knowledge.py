"""--prompt-constraints metadata: every task-level LC in English, and the facts its rules read for the episode."""
import json
from fnmatch import fnmatchcase
from types import SimpleNamespace

import pytest

from .. import main, native_planning, prompt_protocol
from ..dataset import PlanningExample
from ..knowledge.prompt_knowledge import (
    GENERIC_RULES, SCENE_RULES, TASK_RULES, UNWORDED, episode_facts, rule_patterns, unworded, worded_by,
)
from ..models import planner_prompt
from ..official.prompt_constraints import (
    CAUSAL_RULES, GRAMMAR_RULES, OFFICIAL_RULES, prompt_constraints_text, prompt_rules,
)
from ..world_graph import build_vlabench_world_graph

RELAYED = ("physics_knowledge", "complex_knowledge", "poker_knowledge", "articulation_knowledge")


@pytest.fixture(scope="module")
def world():
    return build_vlabench_world_graph(graph_name="test_prompt_knowledge_world", knowledge_level="task",
                                      task_identity="oracle")


@pytest.fixture(autouse=True)
def prompt_off():
    yield
    prompt_protocol.apply_prompt_knowledge_argument(SimpleNamespace(prompt_constraints=None))


def declared_names(world):
    """Named LCs of the metadata level's groups, and every family the relayed knowledge declares."""
    names = {lc.name for group in ("causal", "generic", "scene", "task") for lc in world.knowledge_groups[group]}
    for attribute in RELAYED:
        names |= set(getattr(world, attribute).get().families)
    return names


def test_every_metadata_lc_is_worded_or_left_out_with_a_reason(world):
    rules = prompt_rules("metadata")
    missing = sorted(name for name in declared_names(world) if not worded_by(name, rules) and not unworded(name))
    assert not missing
    left_out = sorted(name for name in declared_names(world) if unworded(name))
    assert left_out == ["observation_supported_references"]


def test_every_rule_pattern_words_some_declared_lc(world):
    names = declared_names(world)
    for rule in GENERIC_RULES + SCENE_RULES + TASK_RULES:
        for pattern in rule.lcs:
            assert any(fnmatchcase(name, pattern) for name in names), pattern


def test_metadata_is_the_grammar_causal_and_world_rules_without_the_official_ones():
    assert prompt_rules("metadata") == GRAMMAR_RULES + CAUSAL_RULES + GENERIC_RULES + SCENE_RULES + TASK_RULES
    assert not set(OFFICIAL_RULES) & set(prompt_rules("metadata"))
    text = prompt_constraints_text("metadata", grammar_profile="world", max_operations=12)
    numbers = [line.split(".")[0] for line in text.splitlines() if line[:1].isdigit()]
    assert numbers == [str(number) for number in range(1, len(prompt_rules("metadata")) + 1)]
    assert "labeled image" not in text and "Task rules:" in text and "Scene rules:" in text


def test_the_world_schema_words_optional_arguments():
    schema = ({"pick": ("target_entity_name",), "push": ()}, (("pick", "push"),), {"push": ("target_container_name",)})
    text = prompt_constraints_text("grammar", max_operations=4, schema=schema)
    assert "push takes optionally target_container_name" in text and '["pick", "push"]' in text


QA_TABLE = ("table", "button0", "stone_box", "button1", "wood_box", "button2", "metal_ball", "lab_1")


def qa_context(phrases=("medium",)):
    return {"task": "density_qa", "instruction": "Choose the object with the medium density.", "entity_table": QA_TABLE,
            "planning_facts": {"physics_selection": {
                "action": "press", "argument": "subject", "candidates": ["button0", "button1", "button2"],
                "phrases": list(phrases), "property": "density_qa:qualitative_model_v1",
                "attributes": {"button0": {"material": "stone"}, "button1": {"material": "wood"},
                               "button2": {"material": "metal"}}}}}


def test_qa_facts_state_the_options_the_order_and_the_applied_criterion(world):
    facts = episode_facts(qa_context(), world)
    assert "Task: density_qa." in facts and "Skill sequence of the task: press." in facts
    assert "Answer buttons: 1 (button0), 3 (button1), 5 (button2)." in facts
    assert "The object of button 3 (button1): material wood." in facts
    assert ("Declared order of density by material, from lowest to highest: "
            "wood < rubber < water < glass < stone < metal.") in facts
    assert any("middle of the options" in line for line in facts)
    assert "Static fixtures: 0 (table), 7 (lab_1)." in facts
    assert not any(line.startswith("Buttons: ") for line in facts)  # listed once, as answer buttons


def test_the_first_phrase_the_physics_lcs_read_decides_the_stated_criterion(world):
    context = {**qa_context(), "task": "magnetism_qa"}
    context["planning_facts"]["physics_selection"]["phrases"] = ["non_magnetic", "magnetic"]
    stated = [line for line in episode_facts(context, world) if line.startswith("The instruction says")]
    assert stated == ['The instruction says "non-magnetic" or "non magnetic", which asks for the option the lowest '
                      'in the declared order.']


def test_task_identity_selects_the_stated_sequence_and_the_facts_hold_no_reference(world):
    context = {"task": "select_fruit", "instruction": "Put the apple in the plate", "entity_table": ("table", "apple", "plate")}
    assert "Skill sequence of the task: pick, place." in episode_facts(context, world)
    assert "Skill sequence of the task: pick, lift." in episode_facts({**context, "task": "select_chemistry_tube"}, world)
    reference = [{"name": "pick", "params": {"target_entity_name": 1}}]
    with_reference = {**context, "operation_sequence": reference, "goal": "apple on plate"}
    assert episode_facts(with_reference, world) == episode_facts(context, world)


def test_complex_facts_state_roles_recipes_and_years(world):
    table = ("counter", "plate_seen", "tomato", "egg", "steak", "kitchen_2")
    context = {"task": "cook_dishes", "instruction": "Cook tomato_and_egg_stir_fry", "entity_table": table,
               "planning_facts": {"complex_request": {
                   "task": "cook_dishes", "entities": list(table), "recipes": ["tomato_and_egg_stir_fry"],
                   "roles": {"plate": ["plate_seen"], "counter": ["counter"]}, "years": {}, "books_keyed": False,
                   "asks_reverse": False, "formulae": []}}}
    facts = episode_facts(context, world)
    assert "Recipe named by the instruction: tomato_and_egg_stir_fry, whose ingredients are tomato, egg." in facts
    assert "Scene roles: counter: 0 (counter); plate: 1 (plate_seen)." in facts


def test_graph_planner_prompt_carries_rules_and_facts_only_when_on():
    vocabulary = SimpleNamespace(skills=("pick", "place"))
    context = {"task": "select_fruit", "instruction": "Put the apple in the plate", "entity_table": ("table", "apple", "plate")}
    plain = planner_prompt(context["instruction"], context["entity_table"], vocabulary)
    assert prompt_protocol.prompt_knowledge_text(context) is None and prompt_protocol.prompt_knowledge_record() is None
    args = SimpleNamespace(prompt_constraints="metadata", max_operations=9)
    assert prompt_protocol.apply_prompt_knowledge_argument(args) == "metadata"
    assert (args.knowledge, args.observation_mode, args.no_dfa) == ("metadata", "metadata-assisted", True)
    assert (args.stage1_dfa_loss, args.stage2_dfa) == ("off", "off")
    knowledge = prompt_protocol.prompt_knowledge_text(context)
    assert "Use at most 9 skill calls in total." in knowledge and "- Skill sequence of the task: pick, place." in knowledge
    prompt = planner_prompt(context["instruction"], context["entity_table"], vocabulary, knowledge=knowledge)
    assert prompt.replace(f"{knowledge}\n", "") == plain
    assert prompt.index("### Facts about this episode") < prompt.index("Objects:")
    record = prompt_protocol.prompt_knowledge_record()
    assert record["level"] == "metadata" and record["max_operations"] == 9 and len(record["rules_sha256"]) == 64


@pytest.mark.parametrize("flags", [
    {"no_dfa_official": True}, {"knowledge": "official"}, {"task_identity": "predicted"}, {"ce_dfa_only": True},
    {"constraint_profile": "research_v1"}, {"research_perception": "image-vlm"},
])
def test_prompt_constraints_refuse_what_would_enforce_or_perceive(flags):
    with pytest.raises(ValueError, match="omit"):
        prompt_protocol.apply_prompt_knowledge_argument(SimpleNamespace(prompt_constraints="metadata", **flags))


def test_a_checkpoint_keeps_its_prompt_rules():
    prompt_protocol.apply_prompt_knowledge_argument(SimpleNamespace(prompt_constraints="metadata", max_operations=12))
    record = prompt_protocol.prompt_knowledge_record()
    prompt_protocol.validate_prompt_knowledge({"prompt_knowledge": record}, "evaluation")
    with pytest.raises(ValueError, match="same --prompt-constraints"):
        prompt_protocol.validate_prompt_knowledge({}, "evaluation")
    prompt_protocol.apply_prompt_knowledge_argument(SimpleNamespace(prompt_constraints=None))
    prompt_protocol.validate_prompt_knowledge({}, "evaluation")
    with pytest.raises(ValueError, match="same --prompt-constraints"):
        prompt_protocol.validate_prompt_knowledge({"prompt_knowledge": record}, "resume")


def parse(*flags):
    return main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "unused", *flags])


def test_the_official_protocol_adds_each_episode_s_facts_after_the_rules(tmp_path, monkeypatch, capsys):
    from .test_prompt_constraints import PLAN, official_episode
    from ..official import official_protocol as op
    data, out = tmp_path / "data", tmp_path / "out"
    official_episode(data)
    example = PlanningExample("select_fruit/example0", "Put the apple in the bowl", tuple(PLAN), (), (), None,
                              entities=("apple", "bowl"), task="select_fruit")
    loaded = {}
    monkeypatch.setattr(main, "load_planning_examples", lambda *a, **kw: loaded.update(kw) or [example])
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", lambda *a, **kw: object())
    seen = []
    reply = "```json\n" + json.dumps(PLAN) + "\n```"
    monkeypatch.setattr(op, "make_vlm_generator", lambda *a, **kw: lambda items: seen.append(items) or reply)
    main.command_evaluate_planner(main.build_parser().parse_args([
        "evaluate-planner", "--planning-dir", str(data), "--protocol", "official", "--prompt-constraints", "metadata",
        "--official-output-dir", str(out), "--no-load-in-4bit"]))
    capsys.readouterr()
    rules = prompt_constraints_text("metadata", grammar_profile="benchmark", max_operations=op.DFA_MAX_OPERATIONS)
    text = seen[0][0][1]
    assert loaded["observation_mode"] == "metadata-assisted"
    assert text.startswith(f"{op.load_official_prompt()}\n\n{rules}\n\n### Facts about this episode:")
    assert "- Task: select_fruit." in text and "- Skill sequence of the task: pick, place." in text
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["episode_facts"] is True and run["prompt_constraints"]["level"] == "metadata"
    row = json.loads((out / "outputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["episode_facts"].startswith("### Facts about this episode:")


def test_frozen_vlm_states_each_episode_s_facts(monkeypatch, capsys):
    from .test_prompt_constraints import Native, PLAN
    from .. import dataset
    args = parse("--prompt-constraints", "metadata", "--split", "all", "--max-operations", "9", "--no-load-in-4bit")
    example = PlanningExample("select_fruit/example0", "Put apple in bowl", tuple(PLAN), (), (), None,
                              entities=("apple", "bowl"), task="select_fruit")
    seen = {}
    monkeypatch.setattr(dataset, "load_scene_planning_examples",
                        lambda *a, **kw: seen.update(kw) or [example])
    planner = Native([json.dumps(PLAN)])
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", lambda *a, **kw: planner)
    main.command_evaluate_planner(args)
    result = json.loads(capsys.readouterr().out)
    assert seen["observation_mode"] == "metadata-assisted"
    stated = planner.calls[0]["prompt_constraints"]
    assert stated.startswith(prompt_constraints_text("metadata", grammar_profile="world", max_operations=9))
    assert "- Task: select_fruit." in stated and "- Skill sequence of the task: pick, place." in stated
    assert result["dfa_constrained"] is False and result["knowledge"] is None
    assert result["prompt_constraints"]["level"] == "metadata"
