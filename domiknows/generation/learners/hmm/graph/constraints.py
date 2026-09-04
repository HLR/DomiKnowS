"""Constraint and projection utilities for graph-constrained HMM models.

This module centralizes small, reusable helpers that validate masks,
combine static constraint sources, and project probability distributions
or matrices onto legal support.

It is intentionally framework-light so higher-level HMM and spectral
components can rely on consistent numeric behavior and clear constraint
reporting semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


@dataclass
class ConstraintApplicationReport:
    """Small report describing what graph/constraint pieces were compiled."""

    applied: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def add_applied(self, message: str) -> None:
        """Record one successfully compiled constraint fragment."""
        self.applied.append(message)

    def add_unsupported(self, message: str) -> None:
        """Record one graph/constraint fragment that could not be compiled."""
        self.unsupported.append(message)

    def extend(self, other: "ConstraintApplicationReport") -> None:
        """Merge another report into this instance."""
        self.applied.extend(other.applied)
        self.unsupported.extend(other.unsupported)


@dataclass(frozen=True)
class TransitionMaskSpec:
    """Explicit non-negative mask over hidden-state transitions."""

    mask: Any
    name: str | None = None


@dataclass(frozen=True)
class EmissionMaskSpec:
    """Explicit non-negative mask over hidden-state emissions."""

    mask: Any
    name: str | None = None


@dataclass(frozen=True)
class AllowedTransitionsSpec:
    """Allowed transition pairs; all other transitions are masked out."""

    transitions: Sequence[tuple[Any, Any]]
    name: str | None = None


@dataclass(frozen=True)
class ForbiddenTransitionsSpec:
    """Forbidden transition pairs; every other transition is left unchanged."""

    transitions: Sequence[tuple[Any, Any]]
    name: str | None = None


@dataclass(frozen=True)
class AllowedEmissionsSpec:
    """Allowed ``(state, symbol)`` emission pairs; all others are masked out."""

    emissions: Sequence[tuple[Any, Any]]
    name: str | None = None


@dataclass(frozen=True)
class ForbiddenEmissionsSpec:
    """Forbidden ``(state, symbol)`` emission pairs."""

    emissions: Sequence[tuple[Any, Any]]
    name: str | None = None


@dataclass(frozen=True)
class StatePredicateTransitionSpec:
    """Transition mask built from factorized-state predicates.

    The predicate receives ``(source_state_dict, destination_state_dict)`` and
    should return ``True`` for allowed transitions.
    """

    predicate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    name: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ConstraintDFAExportSpec:
    """Marker for regular constraints exported separately as observable DFAs."""

    dfa: Any = None
    name: str | None = None
    description: str | None = None


def validate_mask(
    mask,
    shape: tuple[int, ...],
    *,
    name: str = "mask",
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return *mask* as a non-negative tensor with the expected shape."""

    tensor = torch.as_tensor(mask, dtype=dtype, device=device)
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if (tensor < 0).any():
        raise ValueError(f"{name} must be non-negative")
    return tensor


def _fallback_distribution(mask: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Build a deterministic recovery distribution for degenerate mask rows."""
    allowed = mask > 0
    if allowed.any():
        # Keep support strictly on allowed entries and add a tiny floor.
        return allowed.to(dtype=mask.dtype) + smoothing * allowed.to(dtype=mask.dtype)
    # Preserve strict semantics: a fully forbidden row stays fully forbidden.
    return torch.zeros_like(mask)


def project_distribution(probs, mask, smoothing: float = 1e-6) -> torch.Tensor:
    """Project a distribution through a non-negative mask and normalize.

    Forbidden entries remain zero whenever the mask has at least one allowed
    entry. If a mask row is entirely zero, the returned row remains all zeros.
    """

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    probs_t = torch.as_tensor(probs)
    mask_t = torch.as_tensor(mask, dtype=probs_t.dtype, device=probs_t.device)
    if probs_t.shape != mask_t.shape:
        raise ValueError(f"probs and mask must have the same shape, got {probs_t.shape} and {mask_t.shape}")
    if not torch.isfinite(probs_t).all():
        raise ValueError("probs must contain only finite values")
    if (probs_t < 0).any():
        raise ValueError("probs must be non-negative")
    if (mask_t < 0).any():
        raise ValueError("mask must be non-negative")

    masked = probs_t * mask_t
    total = masked.sum()
    if total <= 0:
        # Degenerate case: no allowed mass survives masking.
        masked = _fallback_distribution(mask_t, smoothing)
        total = masked.sum()
    if total <= 0:
        # All-zero masks remain all-zero to preserve strict forbidden support.
        return masked
    # Clamp the denominator for numerical stability in low-precision regimes.
    return masked / total.clamp_min(torch.finfo(masked.dtype).tiny)


def project_matrix_rows(matrix, mask, smoothing: float = 1e-6) -> torch.Tensor:
    """Apply :func:`project_distribution` independently to each matrix row."""

    matrix_t = torch.as_tensor(matrix)
    mask_t = torch.as_tensor(mask, dtype=matrix_t.dtype, device=matrix_t.device)
    if matrix_t.ndim != 2 or mask_t.ndim != 2:
        raise ValueError("matrix and mask must be rank-2 tensors")
    if matrix_t.shape != mask_t.shape:
        raise ValueError(f"matrix and mask must have the same shape, got {matrix_t.shape} and {mask_t.shape}")
    # Row-wise projection keeps transition/emission semantics intact.
    rows = [project_distribution(row, row_mask, smoothing=smoothing) for row, row_mask in zip(matrix_t, mask_t)]
    return torch.stack(rows, dim=0)


def project_matrix(matrix, mask, smoothing: float = 1e-6) -> torch.Tensor:
    """Alias for row-wise projection of a transition or emission matrix."""

    return project_matrix_rows(matrix, mask, smoothing=smoothing)


def normalize_matrix_rows(matrix) -> torch.Tensor:
    """Normalize positive rows and leave all-zero rows unchanged."""

    matrix_t = torch.as_tensor(matrix)
    if matrix_t.ndim != 2:
        raise ValueError("matrix must be a rank-2 tensor")
    if not torch.isfinite(matrix_t).all():
        raise ValueError("matrix must contain only finite values")
    if (matrix_t < 0).any():
        raise ValueError("matrix must be non-negative")
    totals = matrix_t.sum(dim=1, keepdim=True)
    # Preserve all-zero rows exactly (instead of forcing a fallback), which is
    # important when a caller intentionally represents blocked transitions.
    return torch.where(totals > 0, matrix_t / totals.clamp_min(torch.finfo(matrix_t.dtype).tiny), matrix_t)


def combine_masks(
    masks: Iterable[torch.Tensor | None],
    shape: tuple[int, ...],
    *,
    name: str = "mask",
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Multiply several masks, treating ``None`` as an all-ones mask."""

    combined = torch.ones(shape, dtype=dtype, device=device)
    for index, mask in enumerate(masks):
        if mask is None:
            continue
        # Validation enforces shared shape, finite values, and non-negativity.
        combined = combined * validate_mask(mask, shape, name=f"{name}[{index}]", device=device, dtype=dtype)
    return combined
