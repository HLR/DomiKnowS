from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from domiknows.reinforcement.rewards import (
    binary_match_reward,
    make_binary_reward_function,
)


def reward_from_generator(generator_output: Any, logic_label: Any) -> torch.Tensor:
    """Return a binary reward tensor for generated yes/no answers."""
    return binary_match_reward(generator_output, logic_label)


@dataclass(frozen=True)
class RewardProgramConfig:
    """Lightweight placeholder for the generic reward example."""

    program: str = "TODO"
    reward_name: str = "reward_from_generator"


def make_reward_function(logic_str: str, logic_label: Any):
    """Build a Python reward function bound to a sample's logical target."""
    return make_binary_reward_function(logic_str, logic_label)
