"""Dynamic constraint helpers for graph-aware sequence models.

This module provides lightweight building blocks used by constrained HMM and
spectral components:
- a runtime context object passed to dynamic constraint hooks,
- a factorized state-space utility that maps relational factors to flat ids,
- helpers for validating and applying soft transition-energy penalties.

The goal is to keep dynamic-constraint logic explicit and reusable while the
core model code stays focused on inference and learning.
"""

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
    # Free-form metadata (for example state names, symbols, or external tags).
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FiniteStateDynamicConstraint:
    """Finite-state hard abstraction for DFA export of dynamic constraints.

    Arbitrary ``dynamic_transition`` callbacks may depend on unbounded prefix
    history, beliefs, or external state, so they cannot be exported exactly as a
    finite DFA.  This wrapper lets callers provide the finite control state
    needed for exact product construction when their dynamic hard constraint is
    regular.
    """

    start_state: Any
    transition_mask: Callable[[Any, frozenset[int], Mapping[str, Any]], Any]
    advance: Callable[[Any, Any, frozenset[int], Mapping[str, Any]], Any]
    is_accepting: Callable[[Any], bool] | None = None
    name: str | None = None

    def accepts(self, state: Any) -> bool:
        """Return whether a terminal dynamic state is accepting."""

        if self.is_accepting is None:
            return True
        return bool(self.is_accepting(state))


@dataclass(frozen=True)
class FactorizedStateSpace:
    """Flat HMM state ids backed by named relational factors."""

    factor_names: tuple[str, ...]
    states: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_factors(cls, factors: Mapping[str, Sequence[Any]]) -> "FactorizedStateSpace":
        """Create all possible states from the Cartesian product of factors."""
        if not factors:
            raise ValueError("factors must not be empty")
        names = tuple(factors.keys())
        values = [tuple(factors[name]) for name in names]
        if any(not value for value in values):
            raise ValueError("factor value lists must not be empty")
        # Each concrete state is one tuple in the product space.
        return cls(names, tuple(product(*values)))

    def __len__(self) -> int:
        """Return total number of flat states in the factorized space."""
        return len(self.states)

    @property
    def state_names(self) -> tuple[str, ...]:
        """Human-readable names for all flat state ids."""
        return tuple(self.format_state(index) for index in range(len(self)))

    def format_state(self, state_id: int) -> str:
        """Format one state id as ``factor=value|...``."""
        values = self.state_tuple(state_id)
        return "|".join(f"{name}={value}" for name, value in zip(self.factor_names, values))

    def state_tuple(self, state_id: int) -> tuple[Any, ...]:
        """Map a flat state id to its factor-value tuple."""
        try:
            return self.states[state_id]
        except IndexError as exc:
            raise ValueError(f"state_id {state_id} is out of range") from exc

    def state_dict(self, state_id: int) -> dict[str, Any]:
        """Map a flat state id to a ``{factor: value}`` dictionary."""
        return dict(zip(self.factor_names, self.state_tuple(state_id)))

    def state_id(self, **factors: Any) -> int:
        """Return the flat id for a provided factor assignment."""
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
        """Build a 1D mask over states from a predicate on state dictionaries."""
        values = [1.0 if predicate(self.state_dict(index)) else 0.0 for index in range(len(self))]
        return torch.tensor(values, dtype=dtype, device=device)

    def transition_mask(
        self,
        predicate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
        *,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        """Build a pairwise transition mask from source/destination predicates."""
        mask = torch.zeros((len(self), len(self)), dtype=dtype, device=device)
        for src in range(len(self)):
            src_state = self.state_dict(src)
            for dst in range(len(self)):
                # Evaluate legality in relational space, store in flat matrix.
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
    # Higher energy values shrink probabilities more strongly.
    return transition * torch.exp(-weight * penalty)
