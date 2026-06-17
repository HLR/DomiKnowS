from __future__ import annotations

import re
from typing import Any, List, Optional

import torch


_YES_VALUES = {"1", "true", "t", "yes", "y", "positive", "pos"}
_NO_VALUES = {"0", "false", "f", "no", "n", "negative", "neg"}
_COUNT_EXPR = re.compile(
    r"(?P<lhs>count\s*\([^)]+\)|suml\s*\([^)]+\)|\d+(?:\.\d+)?)\s*"
    r"(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<rhs>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-\+]+", "", text)
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
        return generator_output.detach().cpu().reshape(-1).tolist()
    return [generator_output]


def _coerce_logic_label_tensor(logic_label: Any, batch_size: int) -> torch.Tensor:
    if isinstance(logic_label, torch.Tensor):
        label_tensor = logic_label.detach().clone().to(dtype=torch.float32).flatten()
        if label_tensor.numel() == 1 and batch_size > 1:
            label_tensor = label_tensor.repeat(batch_size)
        return label_tensor

    if isinstance(logic_label, (list, tuple)):
        values = [_coerce_logic_label_tensor(item, 1).item() for item in logic_label]
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
    return torch.full((batch_size,), numeric, dtype=torch.float32)


def _extract_number(text: Any) -> Optional[float]:
    normalized = _normalize_text(text)
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _count_expression_truth(logic_str: str, generator_output: Any) -> Optional[bool]:
    match = _COUNT_EXPR.search(logic_str or "")
    if not match:
        return None

    lhs = match.group("lhs").strip().lower()
    op = match.group("op")
    rhs = float(match.group("rhs"))
    predicted = _extract_number(generator_output)

    if predicted is None and lhs.startswith(("count", "suml")):
        return None
    if predicted is None:
        try:
            predicted = float(lhs)
        except ValueError:
            return None

    if op == ">":
        return predicted > rhs
    if op == ">=":
        return predicted >= rhs
    if op == "<":
        return predicted < rhs
    if op == "<=":
        return predicted <= rhs
    if op == "==":
        return predicted == rhs
    if op == "!=":
        return predicted != rhs
    return None


def reward_from_generator(generator_output: Any, logic_str: str, logic_label: Any) -> torch.Tensor:
    outputs = _flatten_generator_output(generator_output)
    if not outputs:
        return torch.zeros(1, dtype=torch.float32)

    labels = _coerce_logic_label_tensor(logic_label, len(outputs))
    rewards = torch.zeros(len(outputs), dtype=torch.float32)

    for idx, output in enumerate(outputs):
        expected_value = labels[idx].item()
        normalized_output = _normalize_text(output)
        count_truth = _count_expression_truth(logic_str, output)

        if count_truth is not None:
            rewards[idx] = 1.0 if count_truth == bool(round(expected_value)) else 0.0
            continue

        if expected_value in (0.0, 1.0):
            if int(expected_value) == 1:
                rewards[idx] = 1.0 if normalized_output in _YES_VALUES else 0.0
            else:
                rewards[idx] = 1.0 if normalized_output in _NO_VALUES else 0.0
            continue

        predicted_number = _extract_number(output)
        if predicted_number is not None:
            rewards[idx] = 1.0 if predicted_number == expected_value else 0.0
        else:
            rewards[idx] = 1.0 if normalized_output == _normalize_text(expected_value) else 0.0

    return rewards
