"""Loss helpers for automata-backed learner heads."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ...dfa import DFA
from ..hmm.head import HMMGenerationHead
from ..hmm.prompt_conditioned_head import PromptConditionedHMMGenerationHead
from ..wfa.prompt_conditioned_head import PromptConditionedSpectralWFAGenerationHead
from ..wfa.head import SpectralWFAGenerationHead
from .utils import TransitionPotentialInput, _empty_or_prompt, _target_labels

__all__ = ["allowed_mass_loss", "hmm_sequence_nll", "wfa_sequence_energy_loss"]

def hmm_sequence_nll(
    head: HMMGenerationHead | PromptConditionedHMMGenerationHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    transition_potential: TransitionPotentialInput = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log-likelihood of a target label sequence under an HMM head."""
    if not isinstance(head, (HMMGenerationHead, PromptConditionedHMMGenerationHead)):
        raise TypeError("hmm_sequence_nll expects an HMM generation head")
    device = head.transition_logits.device
    labels = _target_labels(target_labels, head.pad_size, device=device)
    prompt = _empty_or_prompt(instruction_tokens, device)
    log_probs = head(None, prompt, labels, transition_potential=transition_potential)
    return F.nll_loss(log_probs, labels, reduction=reduction)


def wfa_sequence_energy_loss(
    head: SpectralWFAGenerationHead | PromptConditionedSpectralWFAGenerationHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    transition_potential: TransitionPotentialInput = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Energy-style supervised loss for a WFA head.

    Signed WFA next-symbol scores are interpreted as logits and optimized with
    cross-entropy against the target compact labels.
    """
    if not isinstance(head, (SpectralWFAGenerationHead, PromptConditionedSpectralWFAGenerationHead)):
        raise TypeError("wfa_sequence_energy_loss expects a spectral WFA generation head")
    device = head.transitions.device if hasattr(head, "transitions") else head.initial.device
    labels = _target_labels(target_labels, head.pad_size, device=device)
    prompt = _empty_or_prompt(instruction_tokens, device)
    log_probs = head(None, prompt, labels, transition_potential=transition_potential)
    return F.nll_loss(log_probs, labels, reduction=reduction)


def allowed_mass_loss(
    probs: torch.Tensor,
    dfa: DFA,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Softly encourage probability mass on labels allowed by a DFA.

    The DFA state is advanced along the greedy label path for each sequence.
    The loss remains differentiable with respect to the probability mass placed
    on labels that are valid at those visited states.  This is an auxiliary
    training signal; hard correctness still comes from DFA decoding.
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if probs.dim() == 2:
        batched = probs.unsqueeze(0)
        squeeze = True
    elif probs.dim() == 3:
        batched = probs
        squeeze = False
    else:
        raise ValueError("probs must have shape [seq_len, labels] or [batch, seq_len, labels]")

    losses = []
    eps = torch.finfo(batched.dtype).eps
    label_count = batched.shape[-1]
    for batch_idx in range(batched.shape[0]):
        state = dfa.start_state
        step_losses = []
        for step_idx in range(batched.shape[1]):
            allowed = sorted(
                int(label)
                for label in dfa.allowed_tokens(state, remaining_steps=batched.shape[1] - step_idx)
                if 0 <= int(label) < label_count
            )
            if allowed:
                allowed_index = torch.tensor(allowed, dtype=torch.long, device=batched.device)
                mass = batched[batch_idx, step_idx].index_select(0, allowed_index).sum()
                step_losses.append(-torch.log(mass.clamp_min(eps)))
            label = int(torch.argmax(batched[batch_idx, step_idx]).item())
            next_state = dfa.step(state, label)
            if next_state is not None:
                state = next_state
        if step_losses:
            losses.append(torch.stack(step_losses).mean())
        else:
            losses.append(batched.new_zeros(()))

    result = torch.stack(losses)
    if squeeze and reduction == "none":
        return result[0]
    if reduction == "none":
        return result
    if reduction == "sum":
        return result.sum()
    return result.mean()
