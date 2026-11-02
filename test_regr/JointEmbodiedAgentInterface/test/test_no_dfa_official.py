from types import SimpleNamespace

import pytest

from test_regr.VLABenchAgentInterface.models import planner_prompt
from test_regr.VLABenchAgentInterface.prompt_protocol import (
    planner_input_protocol, validate_training_input_protocol,
)
from test_regr.VLABenchAgentInterface.research.config import (
    apply_research_arguments, default_perception_options,
)
from test_regr.VLABenchAgentInterface.knowledge import default_level, set_default_level
from ..eval_planning import build_eval_parser
from ..main import build_parser, _dfa_loss_config, _stage2_dfa_enabled


@pytest.fixture(autouse=True)
def restore_protocol():
    # ContextVar settings must not escape into subsequent training/evaluation tests.
    from test_regr.VLABenchAgentInterface import prompt_protocol
    from test_regr.VLABenchAgentInterface.research import config
    variables = [prompt_protocol._ANONYMOUS_OBJECTS, config._DEFAULT, config._PERCEPTION_OPTIONS,
                 config._COMPILED_LOSS, config._LOSS_MODE, config._TRAIN_OPTIONS]
    tokens = [(v, v.set(v.get())) for v in variables]
    level = default_level()
    yield
    for var, token in reversed(tokens):
        var.reset(token)
    set_default_level(level)


@pytest.mark.parametrize("argv", [
    ["train-agent", "--ce-only", "--no-dfa-official"],
    ["train-agent", "--two-stage", "-no-dfa-official"],
    ["evaluate-planning", "--checkpoint", "unused.pt", "--no-dfa-official"],
])
def test_joint_commands_disable_masks_and_loss_without_perception(argv):
    args = build_parser().parse_args(argv)
    apply_research_arguments(args)
    assert args.no_dfa and args.knowledge == "grammar"
    assert _dfa_loss_config(args) is None
    assert not _stage2_dfa_enabled(args)
    assert default_perception_options() is None
    assert planner_input_protocol() == "anonymous_objects"


@pytest.mark.parametrize("flag", ["--no-dfa-official", "-no-dfa-official"])
def test_eval_prompt_removes_every_name_and_button_hint_but_preserves_pointers(flag):
    args = build_eval_parser().parse_args(["--checkpoint", "unused.pt", flag])
    apply_research_arguments(args)
    entities = ["red_box", "answer_button_stone", "answer_button_wood"]
    before = entities[:]
    reference = [{"name": "press", "params": {"target_entity_name": 2}}]
    prompt = planner_prompt("Select the denser material", entities, SimpleNamespace(skills=("pick", "press")))
    assert all(name not in prompt for name in entities)
    assert "answer buttons" not in prompt and "<button_id>" not in prompt
    assert all(f"{i}: numbered object {i}" in prompt for i in range(3))
    assert "Instruction: Select the denser material" in prompt
    assert entities == before and entities[reference[0]["params"]["target_entity_name"]] == "answer_button_wood"


def test_existing_no_dfa_still_retains_named_object_prompt():
    args = build_eval_parser().parse_args(["--checkpoint", "unused.pt", "--no-dfa"])
    apply_research_arguments(args)
    prompt = planner_prompt("Select an object", ["box", "answer_button"], SimpleNamespace(skills=("press",)))
    assert "0: box" in prompt and "1: answer_button" in prompt
    assert "answer buttons" in prompt
    assert planner_input_protocol() == "named_objects"


@pytest.mark.parametrize("extra", [
    ["--knowledge", "official"], ["--constraint-profile", "research_v1"],
    ["--research-perception", "image-vlm"], ["--observation-mode", "metadata-assisted"],
])
def test_rejects_conflicting_observation_and_constraint_modes(extra):
    args = build_eval_parser().parse_args(["--checkpoint", "unused.pt", "--no-dfa-official", *extra])
    with pytest.raises(ValueError, match="--no-dfa-official"):
        apply_research_arguments(args)


@pytest.mark.parametrize("mode", ["--ce-dfa-only", "--ce-dfa-mass-only"])
def test_rejects_masked_training(mode):
    args = build_parser().parse_args(["train-agent", mode, "--no-dfa-official"])
    with pytest.raises(ValueError, match="DFA-masked CE"):
        apply_research_arguments(args)


def test_resume_cannot_silently_change_prompt_protocol():
    args = build_eval_parser().parse_args(["--checkpoint", "unused.pt", "--no-dfa-official"])
    apply_research_arguments(args)
    with pytest.raises(ValueError, match="input protocol differs"):
        validate_training_input_protocol({})  # historical named-object checkpoint
    validate_training_input_protocol({"planner_input_protocol": "anonymous_objects"})
    apply_research_arguments(SimpleNamespace())
    with pytest.raises(ValueError, match="input protocol differs"):
        validate_training_input_protocol({"planner_input_protocol": "anonymous_objects"})
