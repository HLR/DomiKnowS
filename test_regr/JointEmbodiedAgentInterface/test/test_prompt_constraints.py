"""--prompt-constraints metadata on the Joint planner: rules and facts in the prompt, nothing enforcing them."""
from types import SimpleNamespace

import pytest

from test_regr.VLABenchAgentInterface import prompt_protocol
from test_regr.VLABenchAgentInterface.knowledge import default_level, set_default_level
from ..eval_planning import build_eval_parser
from ..main import _dfa_loss_config, _stage2_dfa_enabled, build_parser
from ..models import JointQwenVLPlanner


@pytest.fixture(autouse=True)
def restore():
    level = default_level()
    yield
    prompt_protocol.apply_prompt_knowledge_argument(SimpleNamespace(prompt_constraints=None))
    set_default_level(level)


@pytest.mark.parametrize("parse", [
    lambda: build_parser().parse_args(["train-agent", "--two-stage", "--prompt-constraints", "metadata"]),
    lambda: build_parser().parse_args(["train-agent", "--ce-only", "--prompt-constraints", "metadata"]),
    lambda: build_parser().parse_args(["evaluate-planning", "--checkpoint", "unused.pt", "--prompt-constraints",
                                       "metadata"]),
    lambda: build_eval_parser().parse_args(["--checkpoint", "unused.pt", "--prompt-constraints", "metadata"]),
])
def test_commands_tell_the_rules_and_disable_every_dfa_use(parse):
    args = parse()
    assert prompt_protocol.apply_prompt_knowledge_argument(args) == "metadata"
    assert args.no_dfa and args.knowledge == "metadata" and args.observation_mode == "metadata-assisted"
    assert _dfa_loss_config(args) is None
    assert not _stage2_dfa_enabled(args)
    assert prompt_protocol.planner_input_protocol() == "named_objects"


def test_the_option_is_off_by_default():
    args = build_parser().parse_args(["train-agent", "--two-stage"])
    assert args.prompt_constraints is None and prompt_protocol.apply_prompt_knowledge_argument(args) is None


def test_joint_prompt_states_the_episode_facts_only_with_the_option():
    stub = SimpleNamespace(vocabulary=lambda domain: SimpleNamespace(skills=("pick", "place")))
    context = {"instruction": "Put the apple in the plate", "entity_table": ("table", "apple", "plate"),
               "task": "select_fruit"}
    plain = JointQwenVLPlanner._prompt_base(stub, "vlabench", context)
    assert "select_fruit" not in plain
    prompt_protocol.apply_prompt_knowledge_argument(SimpleNamespace(prompt_constraints="metadata", max_operations=12))
    told = JointQwenVLPlanner._prompt_base(stub, "vlabench", context)
    assert "- Task: select_fruit." in told and "- Skill sequence of the task: pick, place." in told
    assert "Never pick a static fixture" in told and "- Static fixtures: 0 (table)." in told
    assert told.replace(prompt_protocol.prompt_knowledge_text(context) + "\n", "") == plain
    # Only the observed fields reach the prompt: a reference plan changes nothing.
    poisoned = {**context, "operation_sequence": [{"SECRET_GOLD": 1}], "target_plan_labels": [999]}
    assert JointQwenVLPlanner._prompt_base(stub, "vlabench", poisoned) == told
