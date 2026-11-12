"""--prompt-constraints metadata: every LC of --knowledge metadata in English, plus the episode's facts, with no DFA.

Mirrors ``test_prompt_constraints.py`` (the official level) for ``official/prompt_metadata.py``.
"""
import json
from fnmatch import fnmatchcase
from types import SimpleNamespace

import pytest

from .. import dataset, main, native_planning, prompt_protocol
from ..dataset import PlanningExample
from ..models import planner_prompt
from ..official import official_protocol as op
from ..official.prompt_constraints import (
    CAUSAL_RULES, EPISODE_LEVELS, GRAMMAR_RULES, OFFICIAL_RULES, prompt_constraints_record, prompt_constraints_text,
    prompt_rules,
)
from ..official.prompt_metadata import (
    GENERIC_RULES, METADATA_SECTIONS, SCENE_RULES, TASK_RULES, UNWORDED, episode_facts, episode_facts_text,
    episode_prompt_text, unworded, worded_by,
)
from ..world_graph import build_vlabench_world_graph
from .test_prompt_constraints import PLAN, Native, official_episode

RELAYED = ("physics_knowledge", "complex_knowledge", "poker_knowledge", "articulation_knowledge")
METADATA_RULES = GENERIC_RULES + SCENE_RULES + TASK_RULES
FRUIT = {"task": "select_fruit", "instruction": "Put the apple in the plate", "entity_table": ("table", "apple", "plate")}


@pytest.fixture(scope="module")
def world():
    return build_vlabench_world_graph(graph_name="test_prompt_metadata_world", knowledge_level="task",
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


def fruit_example(**fields):
    return PlanningExample("select_fruit/example0", "Put the apple in the bowl", tuple(PLAN), (), (), None,
                           entities=("apple", "bowl"), task="select_fruit", **fields)


# ---------------------------------------------------------------------------------------------------------------
# The rules.

def test_every_rule_matches_a_declared_logical_constraint(world):
    rules = prompt_rules("metadata")
    names = declared_names(world)
    missing = sorted(name for name in names if not worded_by(name, rules) and not unworded(name))
    assert not missing
    assert sorted(name for name in names if unworded(name)) == ["observation_supported_references"]
    assert all(rule.lcs for rule in METADATA_RULES)
    for rule in METADATA_RULES:
        for pattern in rule.lcs:
            assert any(fnmatchcase(name, pattern) for name in names), pattern
    assert set(UNWORDED) == {"observation_supported_references", "official__*"}


def test_rules_name_no_episode_object_id_or_answer():
    # The rules are the same for every episode: ids and answers reach the prompt only through the facts.
    for rule in METADATA_RULES:
        assert not any(character.isdigit() for character in rule.text), rule.text
    text = prompt_constraints_text("metadata", grammar_profile="benchmark", max_operations=16)
    assert "Facts about this episode" not in text and "the facts below" in text


def test_metadata_is_the_grammar_causal_and_world_rules_without_the_official_ones():
    assert "metadata" in EPISODE_LEVELS and not EPISODE_LEVELS & {"grammar", "causal", "official"}
    assert prompt_rules("metadata") == GRAMMAR_RULES + CAUSAL_RULES + METADATA_RULES
    assert not set(OFFICIAL_RULES) & set(prompt_rules("metadata"))
    assert [title for title, _ in METADATA_SECTIONS.values()] == ["General rules", "Scene rules", "Task rules"]


def test_text_is_worded_from_the_language_of_the_path():
    benchmark = prompt_constraints_text("metadata", grammar_profile="benchmark", max_operations=7)
    world = prompt_constraints_text("metadata", grammar_profile="world", max_operations=7)
    assert "at most 7 skill calls" in benchmark and "at most 7 skill calls" in world
    assert "open_laptop" not in benchmark and "open_laptop takes target_entity_name" in world
    assert all(mark not in benchmark for mark in ("{", "}", "ifL", "DFA", "LC"))
    assert [line.split(".")[0] for line in benchmark.splitlines() if line[:1].isdigit()] == [
        str(number) for number in range(1, len(prompt_rules("metadata")) + 1)]
    with pytest.raises(ValueError, match="max-operations"):
        prompt_constraints_text("metadata", grammar_profile="benchmark", max_operations=0)


def test_the_world_schema_words_optional_arguments():
    schema = ({"pick": ("target_entity_name",), "push": ()}, (("pick", "push"),), {"push": ("target_container_name",)})
    text = prompt_constraints_text("metadata", max_operations=4, schema=schema)
    assert "push takes optionally target_container_name" in text and '["pick", "push"]' in text


def test_the_sections_follow_the_grammar_and_causal_rules_and_leave_out_the_image_rules():
    text = prompt_constraints_text("metadata", grammar_profile="benchmark", max_operations=12)
    titles = [line for line in text.splitlines() if line.endswith(":") and not line.startswith("#")]
    assert titles == ["Plan structure:", "Causal rules:", "General rules:", "Scene rules:", "Task rules:"]
    assert "labeled image" not in text
    assert "General rules" not in prompt_constraints_text("official", grammar_profile="benchmark", max_operations=12)


# ---------------------------------------------------------------------------------------------------------------
# The episode's facts.

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
    assert "Skill sequence of the task: pick, place." in episode_facts(FRUIT, world)
    assert "Skill sequence of the task: pick, lift." in episode_facts({**FRUIT, "task": "select_chemistry_tube"}, world)
    reference = [{"name": "pick", "params": {"target_entity_name": 1}}]
    with_reference = {**FRUIT, "operation_sequence": reference, "goal": "apple on plate"}
    assert episode_facts(with_reference, world) == episode_facts(FRUIT, world)


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


def test_facts_text_is_a_bulleted_section_after_the_rules(world):
    text = episode_facts_text(FRUIT, world)
    assert text.startswith("### Facts about this episode:\n\n- Task: select_fruit.")
    assert all(line.startswith("- ") for line in text.splitlines()[2:])
    rules = prompt_constraints_text("metadata", max_operations=9)
    stated = episode_prompt_text(rules, fruit_example())
    assert stated.startswith(f"{rules}\n\n### Facts about this episode:\n\n")


# ---------------------------------------------------------------------------------------------------------------
# The flag.

def parse(*flags):
    return main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "unused", *flags])


def test_flag_selects_metadata_only_by_name():
    assert parse().prompt_constraints is None
    assert parse("--prompt-constraints").prompt_constraints == "official"
    assert parse("--prompt-constraints", "metadata").prompt_constraints == "metadata"


@pytest.mark.parametrize("flags,named", [
    (["--generation-mode", "graph"], "--generation-mode graph"),
    (["--knowledge", "metadata"], "--knowledge"),
    (["--constraints", "causal"], "--constraints"),
    (["--task-identity", "predicted"], "--task-identity"),
    (["--press-question"], "--press-question"),
    (["--pattern-first"], "--pattern-first"),
    (["--hmm", "best.pt"], "--hmm"),
    (["--perception-cache", "cache"], "--perception-cache"),
])
@pytest.mark.parametrize("protocol", ["internal", "official"])
def test_cli_refuses_flags_that_enforce_or_observe(flags, named, protocol):
    args = parse("--prompt-constraints", "metadata", "--protocol", protocol, *flags)
    with pytest.raises(ValueError, match=f"omit .*{named}"):
        main.command_evaluate_planner(args)


@pytest.mark.parametrize("flags", [
    {"no_dfa_official": True}, {"knowledge": "official"}, {"task_identity": "predicted"}, {"ce_dfa_only": True},
    {"constraint_profile": "research_v1"}, {"research_perception": "image-vlm"},
])
def test_training_refuses_what_would_enforce_or_perceive(flags):
    with pytest.raises(ValueError, match="omit"):
        prompt_protocol.apply_prompt_knowledge_argument(SimpleNamespace(prompt_constraints="metadata", **flags))


# ---------------------------------------------------------------------------------------------------------------
# The paths.

def test_graph_planner_prompt_carries_rules_and_facts_only_when_on():
    vocabulary = SimpleNamespace(skills=("pick", "place"))
    plain = planner_prompt(FRUIT["instruction"], FRUIT["entity_table"], vocabulary)
    assert prompt_protocol.prompt_knowledge_text(FRUIT) is None and prompt_protocol.prompt_knowledge_record() is None
    args = SimpleNamespace(prompt_constraints="metadata", max_operations=9)
    assert prompt_protocol.apply_prompt_knowledge_argument(args) == "metadata"
    assert (args.knowledge, args.observation_mode, args.no_dfa) == ("metadata", "metadata-assisted", True)
    assert (args.stage1_dfa_loss, args.stage2_dfa) == ("off", "off")
    knowledge = prompt_protocol.prompt_knowledge_text(FRUIT)
    assert "Use at most 9 skill calls in total." in knowledge and "- Skill sequence of the task: pick, place." in knowledge
    prompt = planner_prompt(FRUIT["instruction"], FRUIT["entity_table"], vocabulary, knowledge=knowledge)
    assert prompt.replace(f"{knowledge}\n", "") == plain
    assert prompt.index("### Facts about this episode") < prompt.index("Objects:")
    record = prompt_protocol.prompt_knowledge_record()
    assert record["level"] == "metadata" and record["max_operations"] == 9 and len(record["rules_sha256"]) == 64


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


def test_internal_cli_states_each_episode_s_facts_on_the_frozen_vlm(monkeypatch, capsys):
    args = parse("--prompt-constraints", "metadata", "--split", "all", "--max-operations", "9", "--no-load-in-4bit")
    def forbidden(*args, **kwargs):
        pytest.fail("prompt constraints entered the graph generation path")
    for name in ("build_constraint_runtime", "_load_planner"):
        monkeypatch.setattr(main, name, forbidden)
    seen = {}
    monkeypatch.setattr(dataset, "load_scene_planning_examples", lambda *a, **kw: seen.update(kw) or [fruit_example()])
    planner = Native([json.dumps(PLAN)])
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", lambda *a, **kw: planner)
    main.command_evaluate_planner(args)
    result = json.loads(capsys.readouterr().out)
    rules = prompt_constraints_text("metadata", grammar_profile="world", max_operations=9)
    assert seen["observation_mode"] == "metadata-assisted"
    stated = planner.calls[0]["prompt_constraints"]
    assert stated == episode_prompt_text(rules, fruit_example())
    assert "- Task: select_fruit." in stated and "- Skill sequence of the task: pick, place." in stated
    assert result["generation_mode"] == "frozen-vlm" and result["dfa_constrained"] is False
    assert result["knowledge"] is None and result["task_identity"] is None
    assert result["prompt_constraints"] == {**prompt_constraints_record("metadata", rules),
                                            "episode_facts": "per episode, after the rules"}


def test_internal_cli_refuses_another_observation_mode():
    args = parse("--prompt-constraints", "metadata", "--observation-mode", "metadata-assisted")
    with pytest.raises(ValueError, match="selects the metadata observations"):
        main.command_evaluate_planner(args)


def test_official_cli_adds_each_episode_s_facts_after_the_rules_without_dfa(tmp_path, monkeypatch, capsys):
    data, out = tmp_path / "data", tmp_path / "out"
    official_episode(data)
    loaded = {}
    monkeypatch.setattr(main, "load_planning_examples", lambda *a, **kw: loaded.update(kw) or [fruit_example()])
    monkeypatch.setattr(main, "official_perception_options", lambda *_: pytest.fail("prompt constraints use no perception"))
    monkeypatch.setattr(op, "make_grammar_dfa_generator", lambda *a, **kw: pytest.fail("prompt constraints use no DFA"))
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", lambda *a, **kw: object())
    seen = []
    reply = "```json\n" + json.dumps(PLAN) + "\n```"
    monkeypatch.setattr(op, "make_vlm_generator", lambda *a, **kw: lambda items: seen.append(items) or reply)
    main.command_evaluate_planner(main.build_parser().parse_args([
        "evaluate-planner", "--planning-dir", str(data), "--protocol", "official", "--prompt-constraints", "metadata",
        "--official-output-dir", str(out), "--no-load-in-4bit"]))
    output = capsys.readouterr().out
    summary = json.loads(output[output.index("{"):])
    rules = prompt_constraints_text("metadata", grammar_profile="benchmark", max_operations=op.DFA_MAX_OPERATIONS)
    assert loaded["observation_mode"] == "metadata-assisted"
    assert seen[0][0] == ["text", f"{op.load_official_prompt()}\n\n{episode_prompt_text(rules, fruit_example())}"]
    assert summary["standard"] is False and summary["dfa_constrained"] is False
    assert summary["decoding"] == "greedy+prompt-constraints-metadata"
    assert summary["prompt_sha256"] == op.PROMPT_SHA256
    assert summary["prompt_constraints"] == prompt_constraints_record("metadata", rules)
    assert summary["examples"] == 1 and summary["format_errors"] == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["episode_facts"] is True and run["prompt_constraints"]["level"] == "metadata"
    row = json.loads((out / "outputs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["episode_facts"] == episode_facts_text_for_example()


def episode_facts_text_for_example():
    from ..official.prompt_metadata import episode_facts_for
    return episode_facts_for(fruit_example())


def test_official_cli_refuses_an_episode_without_scene_configuration(tmp_path, monkeypatch):
    data = tmp_path / "data"
    official_episode(data)
    monkeypatch.setattr(main, "load_planning_examples", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="needs the scene configuration"):
        main.command_evaluate_planner(main.build_parser().parse_args([
            "evaluate-planner", "--planning-dir", str(data), "--protocol", "official", "--prompt-constraints",
            "metadata", "--official-output-dir", str(tmp_path / "out"), "--no-load-in-4bit"]))


def test_a_resumed_run_refuses_to_add_or_drop_the_facts(tmp_path):
    root = tmp_path / "data"
    official_episode(root)
    examples = op.load_official_examples(root)
    reply = "```json\n" + json.dumps(PLAN) + "\n```"
    record = prompt_constraints_record("metadata", "RULES")
    common = dict(output_dir=tmp_path / "out", model_name="fake", prompt_constraints=record)
    op.run_official_protocol(examples, lambda items: reply, **common, episode_facts=lambda example: "FACTS")
    with pytest.raises(ValueError, match="different configuration"):
        op.run_official_protocol(examples, lambda items: reply, **common)
