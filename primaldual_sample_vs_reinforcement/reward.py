from __future__ import annotations

from typing import Any

import torch

from domiknows.reinforcement.rewards import count_reward, make_count_reward_function


def reward_from_count(
    generator_output: Any,
    expected_value: Any,
    expected_count: int,
    mode: str = "exact",
) -> torch.Tensor:
    """Score generated labels by whether they meet a count target."""
    # Keep the example-facing function name stable while delegating the generic
    # exact/at_least/at_most count scoring to the shared reinforcement helper.
    return count_reward(generator_output, expected_value, expected_count, mode=mode)
