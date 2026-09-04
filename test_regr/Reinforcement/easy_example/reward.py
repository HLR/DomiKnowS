from __future__ import annotations

import re
from typing import Any, Optional

import torch

from domiknows.reinforcement.rewards import (
    binary_match_reward,
    coerce_label_tensor,
    flatten_generator_output,
    normalize_text,
)


_COUNT_EXPR = re.compile(
    r"(?P<lhs>count\s*\([^)]+\)|suml\s*\([^)]+\)|\d+(?:\.\d+)?)\s*"
    r"(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<rhs>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_number(text: Any) -> Optional[float]:
    # Generated answers can be text such as "count is 2"; use the first number
    # when this reward is evaluating numeric count expressions.
    match = re.search(r"-?\d+(?:\.\d+)?", normalize_text(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _count_expression_truth(logic_str: str, generator_output: Any) -> Optional[bool]:
    # This parser is intentionally example-local: it recognizes the simple
    # count/sum logical strings used by the easy reinforcement regression test.
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
    # Normalize the generated answer shape first so the same reward works for
    # scalars, tensors, lists, tuples, and dict outputs from decoders.
    outputs = flatten_generator_output(generator_output)
    if not outputs:
        return torch.zeros(1, dtype=torch.float32)

    labels = coerce_label_tensor(logic_label, len(outputs))
    rewards = torch.zeros(len(outputs), dtype=torch.float32)

    for idx, output in enumerate(outputs):
        expected_value = labels[idx].item()
        count_truth = _count_expression_truth(logic_str, output)

        # Prefer the count-expression interpretation when the logic string has
        # one; otherwise fall back to binary or exact numeric/text matching.
        if count_truth is not None:
            rewards[idx] = 1.0 if count_truth == bool(round(expected_value)) else 0.0
            continue

        if expected_value in (0.0, 1.0):
            rewards[idx] = binary_match_reward([output], expected_value).mean()
            continue

        predicted_number = _extract_number(output)
        if predicted_number is not None:
            rewards[idx] = 1.0 if predicted_number == expected_value else 0.0
        else:
            rewards[idx] = 1.0 if normalize_text(output) == normalize_text(expected_value) else 0.0

    return rewards
