from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, List

import torch


_YES_VALUES = {"1", "true", "t", "yes", "y", "positive", "pos"}
_NO_VALUES = {"0", "false", "f", "no", "n", "negative", "neg"}


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]+", "", text)
    return text


def _flatten_generator_output(generator_output: Any) -> List[Any]:
    if isinstance(generator_output, dict):
        for key in ("generated_text", "text", "output", "answer", "prediction", "predictions"):
            if key in generator_output:
                return _flatten_generator_output(generator_output[key])
        return [generator_output]
    if isinstance(generator_output, (list, tuple)):
        flattened: List[Any] = []
        for item in generator_output:
            flattened.extend(_flatten_generator_output(item))
        return flattened
    if isinstance(generator_output, torch.Tensor):
        if generator_output.ndim == 0:
            return [generator_output.item()]
        if generator_output.ndim == 1:
            return generator_output.tolist()
        return generator_output.detach().cpu().tolist()
    return [generator_output]


def _coerce_logic_labels(logic_label: Any, batch_size: int) -> torch.Tensor:
    if isinstance(logic_label, torch.Tensor):
        label_tensor = logic_label.detach().clone()
        if label_tensor.ndim == 0:
            label_tensor = label_tensor.reshape(1)
        label_tensor = label_tensor.to(dtype=torch.float32).flatten()
        if label_tensor.numel() == 1 and batch_size > 1:
            label_tensor = label_tensor.repeat(batch_size)
        return label_tensor

    if isinstance(logic_label, (list, tuple)):
        values = [_coerce_logic_labels(item, 1).item() for item in logic_label]
        label_tensor = torch.tensor(values, dtype=torch.float32)
        if label_tensor.numel() == 1 and batch_size > 1:
            label_tensor = label_tensor.repeat(batch_size)
        return label_tensor

    normalized = _normalize_text(logic_label)
    if normalized in _YES_VALUES:
        return torch.ones(batch_size, dtype=torch.float32)
    if normalized in _NO_VALUES:
        return torch.zeros(batch_size, dtype=torch.float32)

    try:
        numeric = float(logic_label)
    except (TypeError, ValueError):
        numeric = 0.0
    label_tensor = torch.full((batch_size,), numeric, dtype=torch.float32)
    return label_tensor


def reward_from_generator(generator_output: Any, logic_label: Any) -> torch.Tensor:
    """Return a binary reward tensor for generator output against logic labels.

    The reward is 1 when the generator output matches the expected label implied
    by ``logic_label`` and 0 otherwise. The function accepts raw generator output
    in the common Hugging Face / custom wrapper shapes:

    - plain string
    - list/tuple of strings
    - dict with a generated-text field
    - torch tensor
    """

    outputs = _flatten_generator_output(generator_output)
    if not outputs:
        return torch.zeros(1, dtype=torch.float32)

    labels = _coerce_logic_labels(logic_label, len(outputs))
    rewards = torch.zeros(len(outputs), dtype=torch.float32)

    for idx, output in enumerate(outputs):
        expected = int(labels[idx].item() >= 0.5)
        normalized_output = _normalize_text(output)
        if expected == 1:
            rewards[idx] = 1.0 if normalized_output in _YES_VALUES else 0.0
        else:
            rewards[idx] = 1.0 if normalized_output in _NO_VALUES else 0.0

    return rewards


@dataclass(frozen=True)
class RewardProgramConfig:
    """Lightweight placeholder for the generic reward example."""

    program: str = "TODO"
    reward_name: str = "reward_from_generator"



def make_reward_function(logic_str: str, logic_label: Any):
    """Build a Python reward function bound to a sample's logical target.

    This keeps the same simple yes/no reward behavior while attaching the
    sample's logic string and label for downstream inspection.
    """

    def _reward(generator_output: Any) -> torch.Tensor:
        return reward_from_generator(generator_output, logic_label)

    _reward.logic_str = logic_str
    _reward.logic_label = logic_label
    return _reward
