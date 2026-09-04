"""Reward-driven (reinforcement-learning) training for DomiKnowS.

Exposes :class:`ReinforcementProgram`, which optimizes a graph against a reward
function by sampling discrete decodings from the model's predicted
distributions.
"""

from .reinforcement_program import ReinforcementProgram, ReinforcementModel
from .sampling import (
    sample_assignments,
    decoding_logprob,
    importance_weighted_loss,
    reinforce_loss,
)
from .constraint_reward import constraint_satisfaction_reward
from .visualization import ReinforcementVisualizer
from .rewards import (
    as_reward_tensor,
    binary_label,
    binary_label_name,
    binary_match_reward,
    call_reward_function,
    coerce_label_tensor,
    count_reward,
    flatten_generator_output,
    make_binary_reward_function,
    make_count_reward_function,
    make_reward_function,
    normalize_text,
)

__all__ = [
    "ReinforcementProgram",
    "ReinforcementModel",
    "sample_assignments",
    "decoding_logprob",
    "importance_weighted_loss",
    "reinforce_loss",
    "constraint_satisfaction_reward",
    "ReinforcementVisualizer",
    "as_reward_tensor",
    "binary_label",
    "binary_label_name",
    "binary_match_reward",
    "call_reward_function",
    "coerce_label_tensor",
    "count_reward",
    "flatten_generator_output",
    "make_binary_reward_function",
    "make_count_reward_function",
    "make_reward_function",
    "normalize_text",
]
