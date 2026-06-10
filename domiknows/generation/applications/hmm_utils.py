"""Shared HMM utility helpers for application-level generation flows."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HMMRuntimeView:
    """Runtime matrix view used by strict HMM-backed generation flows."""

    model: Any
    initial_belief: torch.Tensor
    static_transition: torch.Tensor | None
    static_emission: torch.Tensor | None
    prompt_ids: torch.Tensor | None = None
    transition_potential: Any = None

    def emission_for(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        """Return emission matrix for current prefix, static or dynamic."""
        if self.static_emission is not None:
            return self.static_emission
        return self.model.step_emission_probs(self.prompt_ids, prefix_labels)

    def transition_after(
        self,
        prefix_labels: Sequence[int],
        label: int,
        posterior: torch.Tensor,
    ) -> torch.Tensor:
        """Return transition matrix used after observing one label."""
        if self.static_transition is not None:
            return self.static_transition
        if hasattr(self.model, "step_transition_probs"):
            return self.model.step_transition_probs(
                self.prompt_ids,
                prefix_labels,
                transition_potential=self.transition_potential,
            )
        if hasattr(self.model, "_transition_for_prefix"):
            return self.model._transition_for_prefix(
                step=len(prefix_labels),
                prefix=tuple(prefix_labels) + (int(label),),
                belief=posterior,
            )
        raise ValueError("HMM runtime view cannot resolve a transition matrix")

    def forward_update(self, belief: torch.Tensor, prefix_labels: Sequence[int], label: int) -> torch.Tensor:
        """Apply one observation + transition update in the runtime HMM view."""
        emission = self.emission_for(prefix_labels).to(device=belief.device, dtype=belief.dtype)
        posterior = hmm_observation_posterior(belief, emission, label)
        transition = self.transition_after(prefix_labels, label, posterior).to(device=belief.device, dtype=belief.dtype)
        next_belief = torch.matmul(posterior, transition)
        return next_belief / next_belief.sum().clamp_min(torch.finfo(next_belief.dtype).eps)


def has_hmm_matrices(model) -> bool:
    """Check whether model exposes static HMM initial/transition/emission tensors."""
    return all(hasattr(model, name) for name in ("initial_probs", "transition_probs", "emission_probs"))


def normalise_hmm_vector(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Return a normalized 1D probability vector from raw HMM weights."""
    vector = torch.as_tensor(values, dtype=torch.float32)
    if vector.dim() != 1:
        raise ValueError(f"expected HMM belief vector to be 1D, got shape {tuple(vector.shape)}")
    total = vector.sum().clamp_min(torch.finfo(vector.dtype).eps)
    return vector / total


def hmm_next_label_logits(belief: torch.Tensor, emission: torch.Tensor) -> torch.Tensor:
    """Project HMM belief through emission matrix and return log probabilities."""
    if belief.dim() != 1:
        raise ValueError(f"expected HMM belief vector to be 1D, got shape {tuple(belief.shape)}")
    if emission.dim() != 2:
        raise ValueError(f"expected HMM emission matrix to be 2D, got shape {tuple(emission.shape)}")
    probs = torch.matmul(belief, emission)
    return torch.log(probs.clamp_min(torch.finfo(probs.dtype).eps))


def hmm_forward_update(
    belief: torch.Tensor,
    emission: torch.Tensor,
    transition: torch.Tensor,
    label: int,
) -> torch.Tensor:
    """Apply one HMM observation/transition step and return normalized next belief."""
    if transition.dim() != 2:
        raise ValueError(f"expected HMM transition matrix to be 2D, got shape {tuple(transition.shape)}")
    label = int(label)
    if label < 0 or label >= emission.shape[-1]:
        raise ValueError(f"label {label} is outside HMM emission label count {emission.shape[-1]}")
    eps = torch.finfo(belief.dtype).eps
    posterior = belief * emission[:, label]
    posterior = posterior / posterior.sum().clamp_min(eps)
    next_belief = torch.matmul(posterior, transition)
    return next_belief / next_belief.sum().clamp_min(eps)


def hmm_observation_posterior(
    belief: torch.Tensor,
    emission: torch.Tensor,
    label: int,
) -> torch.Tensor:
    """Return posterior belief after conditioning on one observed label."""
    label = int(label)
    if label < 0 or label >= emission.shape[-1]:
        raise ValueError(f"label {label} is outside HMM emission label count {emission.shape[-1]}")
    eps = torch.finfo(belief.dtype).eps
    posterior = belief * emission[:, label]
    return posterior / posterior.sum().clamp_min(eps)


def static_hmm_teacher_forced_log_probs(model, labels: torch.Tensor) -> torch.Tensor:
    """Compute label log probabilities for static-matrix HMM models."""
    state = normalise_hmm_vector(model.initial_probs)
    transition = torch.as_tensor(model.transition_probs, dtype=state.dtype, device=state.device)
    emission = torch.as_tensor(model.emission_probs, dtype=state.dtype, device=state.device)
    rows = []
    for raw_label in labels.long().reshape(-1).tolist():
        rows.append(hmm_next_label_logits(state, emission))
        state = hmm_forward_update(state, emission, transition, int(raw_label))
    if not rows:
        return torch.empty((0, int(emission.shape[-1])), dtype=emission.dtype, device=emission.device)
    return torch.stack(rows, dim=0)


def lookahead_remaining_after_label(
    remaining_steps: int | None,
    lookahead_max_steps: int | None,
) -> int | None:
    """Compute remaining depth budget for recursive lookahead after one step."""
    if remaining_steps is None:
        return 8 if lookahead_max_steps is None else lookahead_max_steps
    after_label = max(0, int(remaining_steps) - 1)
    if lookahead_max_steps is None:
        return after_label
    return min(after_label, int(lookahead_max_steps))


def resolve_hmm_snapshot(
    scorer_head,
    prompt_ids: torch.Tensor,
    *,
    transition_potential=None,
) -> HMMRuntimeView:
    """Build a unified runtime HMM view for strict HMM-backed decoding."""
    try:
        from ..learners.hmm.discrete.discreteHMM import DiscreteHMM
        from ..learners.hmm.discrete.discreteHMMLearner import HMMGenerationHead
        from ..learners.hmm.graph.graphAwareHMMLearner import GraphHMMGenerationHead
    except Exception as exc:
        raise ValueError("product_hmm_dfa requires HMMGenerationHead, GraphHMMGenerationHead, or DiscreteHMM support") from exc

    if isinstance(scorer_head, DiscreteHMM):
        return HMMRuntimeView(
            model=scorer_head,
            initial_belief=normalise_hmm_vector(scorer_head.initial_probs),
            static_transition=torch.as_tensor(scorer_head.transition_probs),
            static_emission=torch.as_tensor(scorer_head.emission_probs),
        )

    if isinstance(scorer_head, GraphHMMGenerationHead):
        has_dynamic_transition = getattr(scorer_head, "dynamic_transition", None) is not None
        has_transition_energy = getattr(scorer_head, "transition_energy", None) is not None
        return HMMRuntimeView(
            model=scorer_head,
            initial_belief=normalise_hmm_vector(
                scorer_head._initial_probs_for_prompt(
                    prompt_ids if getattr(scorer_head, "prompt_conditioning", "none") != "none" else None,
                    1,
                )[0]
            ),
            static_transition=None if (has_dynamic_transition or has_transition_energy) else torch.as_tensor(scorer_head.transition_probs),
            static_emission=torch.as_tensor(scorer_head.emission_probs),
            prompt_ids=prompt_ids,
            transition_potential=transition_potential,
        )

    if not isinstance(scorer_head, HMMGenerationHead):
        raise ValueError("product_hmm_dfa requires scorer_head to be HMMGenerationHead, GraphHMMGenerationHead, or DiscreteHMM")

    prompt_conditioning = getattr(scorer_head, "prompt_conditioning", "none")
    step_conditioning = getattr(scorer_head, "step_dynamics_conditioning", "none")
    if prompt_conditioning != "none" and step_conditioning != "none":
        return HMMRuntimeView(
            model=scorer_head,
            initial_belief=normalise_hmm_vector(scorer_head.prompt_initial_probs(prompt_ids)),
            static_transition=None,
            static_emission=None,
            prompt_ids=prompt_ids,
            transition_potential=transition_potential,
        )

    kwargs = {}
    if prompt_conditioning != "none":
        kwargs["instruction_tokens"] = prompt_ids
    hmm = scorer_head.production_hmm(transition_potential=transition_potential, **kwargs)
    return HMMRuntimeView(
        model=scorer_head,
        initial_belief=normalise_hmm_vector(hmm.initial_probs),
        static_transition=torch.as_tensor(hmm.transition_probs),
        static_emission=torch.as_tensor(hmm.emission_probs),
        prompt_ids=prompt_ids if prompt_conditioning != "none" else None,
        transition_potential=transition_potential,
    )


__all__ = [
    "HMMRuntimeView",
    "has_hmm_matrices",
    "hmm_forward_update",
    "hmm_next_label_logits",
    "hmm_observation_posterior",
    "lookahead_remaining_after_label",
    "normalise_hmm_vector",
    "resolve_hmm_snapshot",
    "static_hmm_teacher_forced_log_probs",
]