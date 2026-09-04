from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn


def create_dataset(N: int, M: int) -> List[Dict[str, Any]]:
    return [{
        'a': [0],
        'b': [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
        'label': [1] * M,
    }]


class TestTrainLearner(nn.Module):
    """Tiny per-element classifier: maps each ``b`` feature vector to 2 logits."""

    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 2),
        )

    def forward(self, relation_tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        return self.layers(x)


def return_contain(b: torch.Tensor, _: Any) -> torch.Tensor:
    """Connect every ``b`` instance to the single ``a`` container."""
    return torch.ones(len(b)).unsqueeze(-1)
