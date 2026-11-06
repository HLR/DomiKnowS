"""--no-dfa-official on the frozen VLM: no scene names reach the prompt, and scoring keeps the real inventory."""
import json

import pytest
import torch

from .. import dataset, main, native_planning, prompt_protocol
from ..dataset import PlanningExample
from ..knowledge import default_level, set_default_level
from ..official.prompt_constraints import prompt_constraints_record, prompt_constraints_text
from ..research import config
from ..research.config import apply_research_arguments

PLAN = [{"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}}]
ENTITIES = ("stone_cylinder", "answer_button_wood")


@pytest.fixture(autouse=True)
def restore_protocol():
    variables = [prompt_protocol._ANONYMOUS_OBJECTS, config._DEFAULT, config._PERCEPTION_OPTIONS,
                 config._PERCEPTION_POLICY, config._COMPILED_LOSS, config._LOSS_MODE, config._TRAIN_OPTIONS]
    tokens = [(variable, variable.set(variable.get())) for variable in variables]
    level = default_level()
    yield
    for variable, token in reversed(tokens):
        variable.reset(token)
    set_default_level(level)


def parse(*flags):
    args = main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "unused", *flags])
    apply_research_arguments(args)
    return args


def test_frozen_prompt_keeps_names_without_the_flag():
    parse("--generation-mode", "frozen-vlm")
    prompt = native_planning.native_prompt("Do it", ENTITIES)
    assert all(name in prompt for name in ENTITIES)


def test_frozen_prompt_removes_names_with_the_flag():
    parse("--no-dfa-official")
    prompt = native_planning.native_prompt("Do it", ENTITIES, constraints="RULES")
    assert all(name not in prompt for name in ENTITIES)
    assert '"0": "numbered object 0"' in prompt and '"1": "numbered object 1"' in prompt
    assert "RULES" in prompt and "Instruction: Do it" in prompt


class Native:
    def __init__(self, outputs):
        self.model = torch.nn.Linear(1, 1)
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, item, **kwargs):
        self.calls.append(kwargs)
        return next(self.outputs), []


@pytest.mark.parametrize("flags,level", [([], None), (["--prompt-constraints"], "official")])
def test_cli_implies_frozen_vlm_and_scores_with_the_real_inventory(monkeypatch, capsys, flags, level):
    args = parse("--no-dfa-official", "--split", "all", "--no-load-in-4bit", *flags)
    assert args.knowledge == "grammar"  # set by the flag; the frozen path has no DFA to select
    def forbidden(*args, **kwargs):
        pytest.fail("the frozen baseline entered the graph generation path")
    for name in ("load_planning_examples", "build_constraint_runtime", "build_vlabench_world_graph", "_load_planner"):
        monkeypatch.setattr(main, name, forbidden)
    example = PlanningExample("select_fruit/example0", "Put apple in bowl", tuple(PLAN), (), (), None, entities=ENTITIES)
    monkeypatch.setattr(dataset, "load_scene_planning_examples", lambda *a, **kw: [example])
    planner = Native([json.dumps(PLAN)])
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", lambda *a, **kw: planner)
    main.command_evaluate_planner(args)
    result = json.loads(capsys.readouterr().out)
    assert result["generation_mode"] == "frozen-vlm" and result["dfa_constrained"] is False
    assert result["no_dfa_official"] is True and result["planner_input_protocol"] == "anonymous_objects"
    assert result["knowledge"] is None and result["task_identity"] is None
    assert result["planning_metrics"]["overall"]["score"] == 100
    if level:
        text = prompt_constraints_text(level, grammar_profile="world", max_operations=12)
        assert result["prompt_constraints"] == prompt_constraints_record(level, text)
        assert planner.calls[0]["prompt_constraints"] == text
    else:
        assert "prompt_constraints" not in result and "prompt_constraints" not in planner.calls[0]


@pytest.mark.parametrize("flags,message", [
    (["--protocol", "official"], "already shows no object names"),
    (["--generation-mode", "graph"], "uses no DFA"),
])
def test_cli_refuses_a_dfa_with_the_flag(flags, message):
    args = parse("--no-dfa-official", *flags)
    with pytest.raises(ValueError, match=message):
        main.command_evaluate_planner(args)


def test_prompt_constraints_still_refuse_an_explicit_knowledge_level_without_the_flag():
    args = main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "unused", "--prompt-constraints",
                                           "--knowledge", "grammar"])
    with pytest.raises(ValueError, match="omit .*--knowledge"):
        main.command_evaluate_planner(args)
