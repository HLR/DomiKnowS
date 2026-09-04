"""Reusable reward helpers for :mod:`domiknows.reinforcement`.

The reinforcement program intentionally accepts plain Python reward callables.
This module keeps common boilerplate out of examples: generated-output
flattening, text/label coercion, reward tensor normalization, and small generic
binary/count reward factories.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable

import torch


_YES_VALUES = {"1", "true", "t", "yes", "y", "positive", "pos", "one"}
_NO_VALUES = {"0", "false", "f", "no", "n", "negative", "neg", "zero"}
_OUTPUT_KEYS = ("generated_text", "text", "output", "answer", "prediction", "predictions")
_CONTEXT_KEYS = ("data_item", "datanode", "samples", "targets")


def normalize_text(value: Any, keep_numeric_symbols: bool = True) -> str:
    """Normalize generated text and labels for simple reward matching."""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if keep_numeric_symbols:
        text = re.sub(r"[^\w\s\.\-\+]+", "", text)
    else:
        text = re.sub(r"[^\w\s]+", "", text)
    return text


def flatten_generator_output(generator_output: Any) -> list[Any]:
    """Flatten common generator output shapes into a list of scalar-ish values."""
    if isinstance(generator_output, dict):
        for key in _OUTPUT_KEYS:
            if key in generator_output:
                return flatten_generator_output(generator_output[key])
        return [generator_output]
    if isinstance(generator_output, (list, tuple)):
        flattened: list[Any] = []
        for item in generator_output:
            flattened.extend(flatten_generator_output(item))
        return flattened
    if torch.is_tensor(generator_output):
        if generator_output.numel() == 0:
            return []
        if generator_output.ndim == 0:
            return [generator_output.detach().cpu().item()]
        return generator_output.detach().cpu().reshape(-1).tolist()
    return [generator_output]


def coerce_label_tensor(logic_label: Any, batch_size: int) -> torch.Tensor:
    """Coerce yes/no/numeric labels to a float tensor of length ``batch_size``."""
    if torch.is_tensor(logic_label):
        label_tensor = logic_label.detach().clone().to(dtype=torch.float32).flatten()
        if label_tensor.numel() == 1 and batch_size > 1:
            label_tensor = label_tensor.repeat(batch_size)
        return label_tensor

    if isinstance(logic_label, (list, tuple)):
        values = [coerce_label_tensor(item, 1).item() for item in logic_label]
        label_tensor = torch.tensor(values, dtype=torch.float32)
        if label_tensor.numel() == 1 and batch_size > 1:
            label_tensor = label_tensor.repeat(batch_size)
        return label_tensor

    normalized = normalize_text(logic_label)
    if normalized in _YES_VALUES:
        return torch.ones(batch_size, dtype=torch.float32)
    if normalized in _NO_VALUES:
        return torch.zeros(batch_size, dtype=torch.float32)

    try:
        numeric = float(logic_label)
    except (TypeError, ValueError):
        numeric = 0.0
    return torch.full((batch_size,), numeric, dtype=torch.float32)


def binary_label(value: Any, default: int = 0) -> int:
    """Return ``1`` for yes-like values, ``0`` for no-like values, else default."""
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return binary_label(value.detach().cpu().reshape(-1)[0].item(), default=default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric == 0.0:
            return 0
        if numeric == 1.0:
            return 1
        return default

    normalized = normalize_text(value)
    if normalized in _YES_VALUES:
        return 1
    if normalized in _NO_VALUES:
        return 0

    try:
        numeric = float(normalized)
    except ValueError:
        return default
    if numeric == 0.0:
        return 0
    if numeric == 1.0:
        return 1
    return default


def binary_label_name(
    value: Any,
    true_label: str = "yes",
    false_label: str = "no",
    default: str | None = None,
) -> str:
    """Map yes/no-like values to caller-chosen canonical label names."""
    label = binary_label(value, default=-1)
    if label == 1:
        return true_label
    if label == 0:
        return false_label
    if default is not None:
        return default
    return false_label


def as_reward_tensor(value: Any, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalize reward outputs to a tensor for reduction or loss code."""
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype) if device is not None else value.to(dtype=dtype)
    if isinstance(value, (list, tuple)):
        if not value:
            return torch.zeros(1, device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)
    return torch.tensor([float(value)], device=device, dtype=dtype)


def _callable_context_keys(fn: Callable[..., Any]) -> tuple[bool, set[str]]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False, set()

    accepted: set[str] = set()
    accepts_kwargs = False
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
        elif name in _CONTEXT_KEYS and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(name)
    return accepts_kwargs, accepted


def call_reward_function(
    reward_fn: Callable[..., Any],
    generator_output: Any,
    *,
    data_item: Any = None,
    datanode: Any = None,
    samples: Any = None,
    targets: Any = None,
) -> Any:
    """Call old-style or context-aware reward functions.

    A one-argument reward keeps receiving only ``generator_output``. If the
    function explicitly accepts reinforcement context keywords, those values are
    passed by name.
    """
    context = {
        "data_item": data_item,
        "datanode": datanode,
        "samples": samples,
        "targets": targets,
    }
    accepts_kwargs, accepted = _callable_context_keys(reward_fn)
    # Preserve old one-argument reward functions, but pass context to rewards
    # that explicitly ask for it via named keywords or **kwargs.
    if accepts_kwargs:
        return reward_fn(generator_output, **context)
    if accepted:
        return reward_fn(generator_output, **{key: context[key] for key in accepted})
    return reward_fn(generator_output)


def make_reward_function(fn: Callable[..., Any], **metadata):
    """Wrap ``fn`` as a reward callable and attach metadata attributes."""
    # Metadata is intentionally stored as attributes so examples can inspect the
    # active reward without changing ReinforcementProgram's callable interface.
    def _reward(generator_output: Any, **context) -> Any:
        return call_reward_function(fn, generator_output, **context)

    for key, value in metadata.items():
        setattr(_reward, key, value)
    return _reward


def binary_match_reward(generator_output: Any, logic_label: Any) -> torch.Tensor:
    """Reward yes/no generated outputs against yes/no labels."""
    outputs = flatten_generator_output(generator_output)
    if not outputs:
        return torch.zeros(1, dtype=torch.float32)

    labels = coerce_label_tensor(logic_label, len(outputs))
    rewards = torch.zeros(len(outputs), dtype=torch.float32)
    for idx, output in enumerate(outputs):
        expected = int(labels[idx].item() >= 0.5)
        predicted = binary_label(output, default=-1)
        rewards[idx] = 1.0 if predicted == expected else 0.0
    return rewards


def _numeric_value(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return _numeric_value(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(normalize_text(value))
    except ValueError:
        return None


def _matches_expected(output: Any, expected_value: Any) -> bool:
    expected_binary = binary_label(expected_value, default=-1)
    if expected_binary in (0, 1):
        return binary_label(output, default=-1) == expected_binary

    expected_number = _numeric_value(expected_value)
    output_number = _numeric_value(output)
    if expected_number is not None and output_number is not None:
        return output_number == expected_number

    return normalize_text(output) == normalize_text(expected_value)


def count_reward(
    generator_output: Any,
    expected_value: Any,
    expected_count: int,
    mode: str = "exact",
) -> torch.Tensor:
    """Reward generated outputs by exact/threshold count of ``expected_value``."""
    outputs = flatten_generator_output(generator_output)
    count = sum(1 for output in outputs if _matches_expected(output, expected_value))

    normalized_mode = normalize_text(mode)
    if normalized_mode in {"atleast", "atleastl", "at_least"}:
        passed = count >= expected_count
    elif normalized_mode in {"atmost", "atmostl", "at_most"}:
        passed = count <= expected_count
    elif normalized_mode in {"exact", "exactl", "equals"}:
        passed = count == expected_count
    else:
        raise ValueError(
            f"Unsupported reward mode {mode!r}; expected exact, at_least, or at_most."
        )

    return torch.tensor([1.0 if passed else 0.0], dtype=torch.float32)


def make_binary_reward_function(logic_str: str, logic_label: Any):
    """Build a binary reward closure bound to a logical label."""
    def _reward(generator_output: Any) -> torch.Tensor:
        return binary_match_reward(generator_output, logic_label)

    _reward.logic_str = logic_str
    _reward.logic_label = logic_label
    return _reward


def _logic_field(logic_label: Any, key: str, default: Any) -> Any:
    if isinstance(logic_label, dict):
        return logic_label.get(key, default)
    return getattr(logic_label, key, default)


def make_count_reward_function(logic_str: str, logic_label: Any):
    """Build a count reward closure from dict/object metadata."""
    expected_value = _logic_field(logic_label, "expected_value", "zero")
    expected_count = int(_logic_field(logic_label, "expected_count", 0))
    mode = _logic_field(logic_label, "mode", "exact")

    def _reward(generator_output: Any) -> torch.Tensor:
        return count_reward(generator_output, expected_value, expected_count, mode=mode)

    _reward.logic_str = logic_str
    _reward.logic_label = logic_label
    _reward.expected_value = expected_value
    _reward.expected_count = expected_count
    _reward.mode = mode
    return _reward
