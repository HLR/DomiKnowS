"""Shared planner-backbone defaults for the three embodied-agent workflows."""

COMMON_VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def is_vision_language_config(config) -> bool:
    """Whether a Transformers config needs the image-text model loader."""
    return getattr(config, "model_type", None) in {"qwen3_vl", "qwen2_5_vl", "qwen2_vl"}


def language_hidden_size(config) -> int | None:
    """Resolve the language width from text-only or multimodal configs."""
    for candidate in (config, getattr(config, "text_config", None), getattr(config, "language_config", None)):
        size = getattr(candidate, "hidden_size", None) or getattr(candidate, "d_model", None)
        if size is not None:
            return int(size)
    return None
