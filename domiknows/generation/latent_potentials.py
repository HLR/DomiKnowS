"""Latent transition potentials for HMM and WFA dynamics.

These helpers reweight automata transition dynamics without changing the
symbolic DFA enforcement path.  HMM potentials are compatibility factors over
hidden-state transitions and are renormalized row-wise.  WFA potentials preserve
signed transition semantics and are not normalized.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


PotentialValues = torch.Tensor | Sequence[Sequence[float]] | Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LatentTransitionPotential:
    """Transition compatibility factors for latent automata dynamics.

    Args:
        values: A tensor-like object or callable. Tensor values are broadcast
            against transition tensors. Callable values receive the transition
            tensor being reweighted and must return a broadcast-compatible
            tensor.
        log_space: When ``True``, *values* are interpreted as log-factors.
            HMM factors are exponentiated before row normalization; WFA
            ``mode="add"`` treats log-space values as additive scores.
        name: Optional display/debug name.
    """

    values: PotentialValues
    log_space: bool = False
    name: str | None = None

    def tensor_for(self, reference: torch.Tensor) -> torch.Tensor:
        if callable(self.values):
            raw = self.values(reference)
        else:
            raw = self.values
        tensor = torch.as_tensor(raw, dtype=reference.dtype, device=reference.device)
        if not torch.isfinite(tensor).all():
            raise ValueError("transition potential values must be finite")
        return tensor


def apply_hmm_transition_potential(
    transition_probs: torch.Tensor,
    potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None,
    *,
    eps: float | None = None,
) -> torch.Tensor:
    """Apply a non-negative HMM transition potential and row-normalize."""

    if potential is None:
        return transition_probs
    if transition_probs.dim() != 2 or transition_probs.shape[0] != transition_probs.shape[1]:
        raise ValueError("HMM transition_probs must have shape [states, states]")
    wrapped = _coerce_potential(potential)
    factors = wrapped.tensor_for(transition_probs)
    if wrapped.log_space:
        factors = torch.exp(factors)
    if torch.any(factors < 0):
        raise ValueError("HMM transition potential factors must be non-negative")
    try:
        weighted = transition_probs * factors
    except RuntimeError as exc:
        raise ValueError("HMM transition potential must broadcast to [states, states]") from exc
    if weighted.shape != transition_probs.shape:
        raise ValueError("HMM transition potential must broadcast to [states, states]")
    if eps is None:
        eps = torch.finfo(transition_probs.dtype).eps
    row_sums = weighted.sum(dim=-1, keepdim=True)
    if torch.any(row_sums <= eps):
        raise ValueError("HMM transition potential produced an all-zero transition row")
    return weighted / row_sums


def apply_wfa_transition_potential(
    transitions: torch.Tensor,
    potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None,
    *,
    mode: str = "multiply",
) -> torch.Tensor:
    """Apply a WFA transition potential while preserving signed scores."""

    if potential is None:
        return transitions
    if transitions.dim() != 3 or transitions.shape[1] != transitions.shape[2]:
        raise ValueError("WFA transitions must have shape [labels, states, states]")
    mode = str(mode).lower()
    if mode not in {"multiply", "add"}:
        raise ValueError("mode must be 'multiply' or 'add'")
    wrapped = _coerce_potential(potential)
    factors = wrapped.tensor_for(transitions)
    if mode == "multiply" and wrapped.log_space:
        factors = torch.exp(factors)
    try:
        result = transitions * factors if mode == "multiply" else transitions + factors
    except RuntimeError as exc:
        raise ValueError("WFA transition potential must broadcast to [labels, states, states]") from exc
    if result.shape != transitions.shape:
        raise ValueError("WFA transition potential must broadcast to [labels, states, states]")
    return result


def forbid_hmm_transition(from_state: int, to_state: int, state_count: int, strength: float = 0.0) -> LatentTransitionPotential:
    """Return an HMM potential that sets one transition factor to *strength*."""

    values = _ones_state_matrix(state_count)
    values[int(from_state), int(to_state)] = float(strength)
    return LatentTransitionPotential(values, name=f"forbid_{from_state}_to_{to_state}")


def penalize_hmm_transition(from_state: int, to_state: int, state_count: int, penalty: float = 0.1) -> LatentTransitionPotential:
    """Return an HMM potential that softly downweights one transition."""

    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    return forbid_hmm_transition(from_state, to_state, state_count, strength=penalty)


def transition_potential_matrix(values: Sequence[Sequence[float]], *, log_space: bool = False, name: str | None = None) -> LatentTransitionPotential:
    """Create a named transition potential from a matrix-like object."""

    return LatentTransitionPotential(values, log_space=log_space, name=name)


def combine_transition_potentials(
    potentials: Sequence[LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]]],
    *,
    name: str | None = None,
) -> LatentTransitionPotential | None:
    """Compose transition potentials by multiplying their compatibility factors.

    Log-space inputs are exponentiated before composition so the returned
    potential is a normal non-negative factor.  ``None`` is returned for an
    empty sequence, which is convenient for optional model kwargs.
    """

    wrapped = [_coerce_potential(potential) for potential in potentials if potential is not None]
    if not wrapped:
        return None

    def _combined(reference: torch.Tensor) -> torch.Tensor:
        result = torch.ones_like(reference)
        for potential in wrapped:
            factors = potential.tensor_for(reference)
            if potential.log_space:
                factors = torch.exp(factors)
            result = result * factors
        return result

    display = name or "combined_" + "_".join(p.name or "potential" for p in wrapped)
    return LatentTransitionPotential(_combined, log_space=False, name=display)


def _coerce_potential(
    potential: LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]],
) -> LatentTransitionPotential:
    if isinstance(potential, LatentTransitionPotential):
        return potential
    return LatentTransitionPotential(potential)


def _ones_state_matrix(state_count: int) -> torch.Tensor:
    state_count = int(state_count)
    if state_count < 1:
        raise ValueError("state_count must be at least 1")
    return torch.ones((state_count, state_count), dtype=torch.float32)
