"""Loss helpers for automata-backed learner heads."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...dfa import DFA
from .utils import TransitionPotentialInput, _empty_or_prompt, _target_labels

if TYPE_CHECKING:
    from ..hmm.discrete.discreteHMMLearner import HMMGenerationHead
    from ..wfa.prompt_conditioned_head import PromptConditionedSpectralWFAGenerationHead
    from ..wfa.head import SpectralWFAGenerationHead

__all__ = ["allowed_label_sets", "allowed_log_mass", "allowed_mass_loss", "hmm_sequence_nll", "masked_cross_entropy",
           "wfa_sequence_energy_loss"]

def hmm_sequence_nll(
    head: HMMGenerationHead,
    target_labels: torch.Tensor | Sequence[int],
    *,
    instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    transition_potential: TransitionPotentialInput = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log-likelihood of a target label sequence under an HMM head."""
    from ..hmm.discrete.discreteHMMLearner import HMMGenerationHead

    if not isinstance(head, HMMGenerationHead):
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
    from ..wfa.head import SpectralWFAGenerationHead
    from ..wfa.prompt_conditioned_head import PromptConditionedSpectralWFAGenerationHead

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
    path: torch.Tensor | Sequence[int] | None = None,
    score_steps: Sequence[int] | None = None,
    lookahead: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Softly encourage probability mass on labels allowed by a DFA.

    The DFA state is advanced along the greedy label path for each sequence, or along ``path`` (a label sequence
    of shape ``[seq_len]`` or ``[batch, seq_len]``, e.g. the teacher-forced target) when one is given. With
    ``score_steps`` only those step indices are scored (the state still advances at every step), which bounds the
    cost when computing the allowed set of a step is expensive. With ``lookahead=False`` the allowed set is the labels
    valid at the step (``allowed_tokens`` without a remaining-step budget), skipping the search for a completion.
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

    score_set = None if score_steps is None else {int(i) for i in score_steps}
    losses = []
    eps = torch.finfo(batched.dtype).eps
    label_count = batched.shape[-1]
    if path is not None:
        path = torch.as_tensor(path, dtype=torch.long)
        path = path.unsqueeze(0) if path.dim() == 1 else path
        if tuple(path.shape) != tuple(batched.shape[:2]):
            raise ValueError("path must match the [batch, seq_len] shape of probs")
    for batch_idx in range(batched.shape[0]):
        state = dfa.start_state
        step_losses = []
        for step_idx in range(batched.shape[1]):
            scored = score_steps is None or step_idx in score_set
            allowed = [] if not scored else sorted(
                int(label)
                for label in (dfa.allowed_tokens(state, remaining_steps=batched.shape[1] - step_idx - 1)
                              if lookahead else dfa.allowed_tokens(state))
                if 0 <= int(label) < label_count
            )
            if allowed:
                allowed_index = torch.tensor(allowed, dtype=torch.long, device=batched.device)
                mass = batched[batch_idx, step_idx].index_select(0, allowed_index).sum()
                step_losses.append(-torch.log(mass.clamp_min(eps)))
            label = int(path[batch_idx, step_idx]) if path is not None else int(torch.argmax(batched[batch_idx, step_idx]).item())
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


def allowed_label_sets(
    dfa: DFA,
    path: torch.Tensor | Sequence[int],
    label_count: int,
    *,
    score_steps: Sequence[int] | None = None,
    lookahead: bool = True,
    max_steps: int | None = None,
) -> dict[int, list[int]]:
    """Labels the DFA allows at each step along ``path`` (step -> sorted labels; steps with none are omitted).

    The state advances along ``path`` at every step; ``score_steps`` restricts which steps compute a set and
    ``lookahead=False`` takes the labels valid at the step without searching for a completion.
    With lookahead, ``max_steps`` is the decoder's total generation budget, not the gold path length.
    Omitting it preserves the legacy path-length budget for existing callers.
    """
    labels = [int(label) for label in (path.tolist() if torch.is_tensor(path) else path)]
    budget = len(labels) if max_steps is None else int(max_steps)
    if budget < len(labels):
        raise ValueError("max_steps cannot be shorter than the teacher-forced path")
    wanted = None if score_steps is None else {int(step) for step in score_steps}
    state = dfa.start_state
    sets: dict[int, list[int]] = {}
    for step, label in enumerate(labels):
        if wanted is None or step in wanted:
            valid = (dfa.allowed_tokens(state, remaining_steps=budget - step - 1) if lookahead
                     else dfa.allowed_tokens(state))
            allowed = sorted(int(item) for item in valid if 0 <= int(item) < label_count)
            if allowed:
                sets[step] = allowed
        following = dfa.step(state, label)
        if following is not None:
            state = following
    return sets


def allowed_log_mass(log_probs: torch.Tensor, sets: dict[int, Sequence[int]]) -> torch.Tensor:
    """``log`` of the probability on the allowed labels at each step of ``sets`` (in step order)."""
    steps = sorted(sets)
    if not steps:
        return log_probs.new_zeros(0)
    return torch.stack([torch.logsumexp(log_probs[step, torch.as_tensor(sets[step], device=log_probs.device)], dim=0)
                        for step in steps])


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sets: dict[int, Sequence[int]],
) -> tuple[torch.Tensor, int]:
    """Mean cross entropy of the targets under the softmax restricted to the DFA-allowed labels at each step.

    This is the distribution a DFA-masked decoder samples from, so it trains exactly what decoding uses. A step
    without an allowed set, or whose target is outside it (the label disagrees with the DFA), keeps the ordinary
    full-softmax cross entropy; the second value counts those disagreements.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    full = -log_probs.gather(1, targets.reshape(-1, 1)).squeeze(1)
    losses = list(full.unbind(0))
    misses = 0
    for step, allowed in sets.items():
        if int(targets[step]) not in set(int(label) for label in allowed):
            misses += 1
            continue
        index = torch.as_tensor(list(allowed), device=log_probs.device)
        losses[step] = full[step] + torch.logsumexp(log_probs[step, index], dim=0)
    return torch.stack(losses).mean(), misses

