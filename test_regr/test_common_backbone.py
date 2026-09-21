"""Regression tests for the common embodied planner backbone selection."""

from types import SimpleNamespace

from test_regr.common_backbone import (
    COMMON_VLM_MODEL_ID,
    is_vision_language_config,
    language_hidden_size,
)
from test_regr.JointEmbodiedAgentInterface.models import DEFAULT_MODEL_ID
from test_regr.VLABenchAgentInterface.models import PLANNER_MODEL_ID


def test_embodied_planners_share_qwen3_vl_default():
    assert COMMON_VLM_MODEL_ID == "Qwen/Qwen3-VL-8B-Instruct"
    assert PLANNER_MODEL_ID == DEFAULT_MODEL_ID == COMMON_VLM_MODEL_ID


def test_vision_language_detection_preserves_text_only_models():
    assert is_vision_language_config(SimpleNamespace(model_type="qwen3_vl"))
    assert is_vision_language_config(SimpleNamespace(model_type="qwen2_5_vl"))
    assert not is_vision_language_config(SimpleNamespace(model_type="qwen3"))
    assert not is_vision_language_config(None)


def test_qwen3_vl_uses_nested_language_hidden_size():
    config = SimpleNamespace(
        model_type="qwen3_vl",
        text_config=SimpleNamespace(hidden_size=4096),
        vision_config=SimpleNamespace(hidden_size=1152),
    )
    assert language_hidden_size(config) == 4096
    assert language_hidden_size(SimpleNamespace(hidden_size=1024)) == 1024
