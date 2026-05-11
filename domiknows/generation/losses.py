"""Unified loss helpers for DomiKnowS generation experiments."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from .latent_constraints import LatentLossItem


@dataclass(frozen=True)
class GenerationLossWeights:
    """Weights for the standard generation training-loss components."""

    supervised: float = 1.0
    pmd: float = 1.0
    latent: float = 0.0
    allowed_mass: float = 0.0
    automata: float = 0.0

    def __post_init__(self):
        for name in ("supervised", "pmd", "latent", "allowed_mass", "automata"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} weight must be non-negative")


@dataclass(frozen=True)
class GenerationLossBreakdown:
    """Total training loss plus named component losses."""

    total: torch.Tensor
    supervised: torch.Tensor
    pmd: torch.Tensor
    latent: torch.Tensor
    allowed_mass: torch.Tensor
    automata: torch.Tensor
    latent_items: tuple[LatentLossItem, ...] = ()

    def as_float_dict(self) -> dict[str, float]:
        return {
            "model_loss": _as_float(self.supervised),
            "constraint_loss": _as_float(self.pmd),
            "latent_loss": _as_float(self.latent),
            "allowed_mass_loss": _as_float(self.allowed_mass),
            "automata_aux_loss": _as_float(self.automata),
            "total_loss": _as_float(self.total),
        }


def compute_generation_training_loss(
    *,
    supervised_loss=None,
    model_loss=None,
    pmd_loss=None,
    constraint_loss=None,
    latent_loss=None,
    allowed_mass_loss_value=None,
    automata_aux_loss=None,
    weights: GenerationLossWeights | None = None,
    latent_items: Sequence[LatentLossItem] = (),
) -> GenerationLossBreakdown:
    """Combine common DomiKnowS generation loss components.

    The function is intentionally tensor-light and task-agnostic: callers may
    compute ``program.cmodel(...)``, latent losses, and automata auxiliary
    losses however their program is structured, then combine them here.
    """
    weights = weights or GenerationLossWeights()
    supervised = _coerce_loss(supervised_loss if supervised_loss is not None else model_loss)
    pmd = _coerce_loss(pmd_loss if pmd_loss is not None else constraint_loss, reference=supervised)
    latent = _coerce_loss(latent_loss, reference=supervised)
    allowed = _coerce_loss(allowed_mass_loss_value, reference=supervised)
    automata = _coerce_loss(automata_aux_loss, reference=supervised)
    total = (
        weights.supervised * supervised
        + weights.pmd * pmd
        + weights.latent * latent
        + weights.allowed_mass * allowed
        + weights.automata * automata
    )
    return GenerationLossBreakdown(
        total=total,
        supervised=supervised,
        pmd=pmd,
        latent=latent,
        allowed_mass=allowed,
        automata=automata,
        latent_items=tuple(latent_items),
    )


def token_probs_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Convert model log-probabilities to probabilities for latent losses."""
    if not isinstance(log_probs, torch.Tensor):
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
    return torch.exp(log_probs)


def _coerce_loss(value, *, reference: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if reference is not None:
        return reference.new_tensor(float(value or 0.0))
    return torch.tensor(float(value or 0.0), dtype=torch.float32)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value or 0.0)
