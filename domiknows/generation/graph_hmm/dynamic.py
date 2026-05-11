"""Dynamic relational constraint helpers for DomiKnowS-aware HMMs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class DynamicConstraintContext:
    """Context passed to dynamic HMM transition hooks."""

    step: int
    prefix: tuple[Any, ...]
    belief: torch.Tensor | None = None
    sequence: tuple[Any, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorizedStateSpace:
    """Flat HMM state ids backed by named relational factors."""

    factor_names: tuple[str, ...]
    states: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_factors(cls, factors: Mapping[str, Sequence[Any]]) -> "FactorizedStateSpace":
        if not factors:
            raise ValueError("factors must not be empty")
        names = tuple(factors.keys())
        values = [tuple(factors[name]) for name in names]
        if any(not value for value in values):
            raise ValueError("factor value lists must not be empty")
        return cls(names, tuple(product(*values)))

    def __len__(self) -> int:
        return len(self.states)

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self.format_state(index) for index in range(len(self)))

    def format_state(self, state_id: int) -> str:
        values = self.state_tuple(state_id)
        return "|".join(f"{name}={value}" for name, value in zip(self.factor_names, values))

    def state_tuple(self, state_id: int) -> tuple[Any, ...]:
        try:
            return self.states[state_id]
        except IndexError as exc:
            raise ValueError(f"state_id {state_id} is out of range") from exc

    def state_dict(self, state_id: int) -> dict[str, Any]:
        return dict(zip(self.factor_names, self.state_tuple(state_id)))

    def state_id(self, **factors: Any) -> int:
        values = tuple(factors[name] for name in self.factor_names)
        try:
            return self.states.index(values)
        except ValueError as exc:
            raise ValueError(f"unknown factorized state {factors!r}") from exc

    def state_mask(
        self,
        predicate: Callable[[Mapping[str, Any]], bool],
        *,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        values = [1.0 if predicate(self.state_dict(index)) else 0.0 for index in range(len(self))]
        return torch.tensor(values, dtype=dtype, device=device)

    def transition_mask(
        self,
        predicate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
        *,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        mask = torch.zeros((len(self), len(self)), dtype=dtype, device=device)
        for src in range(len(self)):
            src_state = self.state_dict(src)
            for dst in range(len(self)):
                if predicate(src_state, self.state_dict(dst)):
                    mask[src, dst] = 1.0
        return mask


def transition_energy_matrix(
    values,
    *,
    shape: tuple[int, int] | None = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Return a finite non-negative transition energy matrix."""

    energy = torch.as_tensor(values, dtype=dtype, device=device)
    if shape is not None and tuple(energy.shape) != tuple(shape):
        raise ValueError(f"transition energy must have shape {shape}, got {tuple(energy.shape)}")
    if energy.ndim != 2:
        raise ValueError("transition energy must be a rank-2 matrix")
    if not torch.isfinite(energy).all():
        raise ValueError("transition energy must contain only finite values")
    if (energy < 0).any():
        raise ValueError("transition energy must be non-negative")
    return energy


def apply_transition_energy(transition_probs, energy, weight: float = 1.0) -> torch.Tensor:
    """Softly penalize transitions by multiplying by ``exp(-weight * energy)``."""

    if weight < 0:
        raise ValueError("weight must be non-negative")
    transition = torch.as_tensor(transition_probs)
    penalty = transition_energy_matrix(
        energy,
        shape=tuple(transition.shape),
        dtype=transition.dtype,
        device=transition.device,
    )
    return transition * torch.exp(-weight * penalty)
