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

__all__ = [
    "ReinforcementProgram",
    "ReinforcementModel",
    "sample_assignments",
    "decoding_logprob",
    "importance_weighted_loss",
    "reinforce_loss",
    "constraint_satisfaction_reward",
    "ReinforcementVisualizer",
]
