"""Discrete HMM compact-label learner head."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .discreteHMM import DiscreteHMM
from ....latent import LatentTransitionPotential, apply_hmm_transition_potential
from ...common.base import CompactLabelGenerationHead
from ...common.utils import (
    TransitionPotentialInput,
    _coerce_label_to_token_id,
    _empty_or_prompt,
    _first_generated_index,
    _invert_label_to_token_id,
    _labels_from_input_ids,
    _normalise_flat_ids,
    _positive_int,
    _random_hmm_parameters,
    _resolve_label_count,
    _resolve_state_count,
    _safe_log,
    _target_label_batch,
    _target_labels,
    _validate_hmm_shapes,
    _validate_label,
)

__all__ = ["HMMGenerationHead"]

class HMMGenerationHead(CompactLabelGenerationHead):
    """Compact-label generation head backed by a discrete HMM.

    The HMM is parameterized with trainable logits when ``trainable=True`` and
    frozen tensors otherwise.  Emissions are labels in the compact
    ``GenerationEncoder`` vocabulary, so the output can be attached directly to
    ``token[generated_token]`` via ``ModuleLearner``.
    """

    def __init__(
        self,
        model: DiscreteHMM | None = None,
        *,
        label_count: int | None = None,
        state_count: int | None = None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int | None = 0,
    ):
        # Resolve static shape/mapping configuration for compact labels and HMM states.
        super().__init__()
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_count = _resolve_label_count(model, label_count)
        self.state_count = _resolve_state_count(model, state_count)
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)

        if model is None:
            # Initialize a random, normalized HMM when no prebuilt model is supplied.
            initial, transition, emission = _random_hmm_parameters(
                self.state_count,
                self.label_count,
                random_seed,
            )
        else:
            # Reuse provided model parameters after shape and dtype normalization.
            initial = torch.as_tensor(model.initial, dtype=torch.float32)
            transition = torch.as_tensor(model.transition, dtype=torch.float32)
            emission = torch.as_tensor(model.emission, dtype=torch.float32)
            _validate_hmm_shapes(initial, transition, emission, self.state_count, self.label_count)

        # initial_logits: unnormalized scores for the initial hidden-state distribution P(z0).
        # transition_logits: unnormalized scores for state transition probabilities P(zt|zt-1).
        # emission_logits: unnormalized scores for label emission probabilities P(yt|zt).
        # We optimize logits and convert with softmax to keep each distribution valid.
        self.initial_logits = torch.nn.Parameter(_safe_log(initial), requires_grad=trainable)
        self.transition_logits = torch.nn.Parameter(_safe_log(transition), requires_grad=trainable)
        self.emission_logits = torch.nn.Parameter(_safe_log(emission), requires_grad=trainable)

    @property
    def initial_probs(self) -> torch.Tensor:
        # Initial hidden-state distribution.
        return torch.softmax(self.initial_logits, dim=-1)

    @property
    def transition_probs(self) -> torch.Tensor:
        # State-to-state transition matrix.
        return torch.softmax(self.transition_logits, dim=-1)

    @property
    def emission_probs(self) -> torch.Tensor:
        # State-to-label emission matrix.
        return torch.softmax(self.emission_logits, dim=-1)

    def transition_probs_with_potential(self, transition_potential: TransitionPotentialInput = None) -> torch.Tensor:
        # Apply optional latent transition potential without mutating base transition logits.
        """Return HMM transition probabilities after optional latent-potential reweighting."""
        return apply_hmm_transition_potential(self.transition_probs, transition_potential)

    def token_id_for_label(self, label: int) -> int:
        # Map one compact label to the tokenizer id appended during generation.
        """Return the raw tokenizer id used when appending a compact label."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        # Convert prompt/generated token ids into compact labels recognized by this head.
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _next_logits_from_prefix_labels(
        self,
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        # Run one forward filtering pass over prefix labels to predict next-label logits.
        # state is the current belief distribution over hidden HMM states.
        state = self.initial_probs
        transition = self.transition_probs_with_potential(transition_potential)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps

        for raw_label in prefix_labels:
            # Bayesian update: condition current state on observed label emission.
            label = _validate_label(raw_label, self.label_count)
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            # Propagate filtered state through transition dynamics.
            state = torch.matmul(posterior, transition)

        # Predict distribution of the next label from the current filtered state.
        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def next_label_logits(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        transition_potential: TransitionPotentialInput = None,
        **kwargs,
    ) -> torch.Tensor:
        # Public decoding API: tokenize prefix to labels, then compute next-label logits.
        """Return next-step logits over compact generation labels."""
        if transition_potential is None:
            transition_potential = kwargs.get("transition_potential")
        return self._next_logits_from_prefix_labels(
            self._labels_from_input_ids(input_ids),
            transition_potential=transition_potential,
        )

    def production_hmm(self, transition_potential: TransitionPotentialInput = None) -> DiscreteHMM:
        # Materialize an immutable HMM view suitable for inference helpers/serialization.
        """Return a Torch-backed HMM view of the current head parameters."""
        return DiscreteHMM(
            self.transition_probs_with_potential(transition_potential),
            self.emission_probs,
            self.initial_probs,
            tuple(range(self.label_count)),
            normalize=False,
        )

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        transition_potential: TransitionPotentialInput = None,
        **kwargs,
    ) -> torch.Tensor:
        # Teacher-forced rollout: return per-step log-probs for supervised training loss.
        """Return teacher-forced log-probs shaped ``[batch, seq, label_count]``."""
        if transition_potential is None:
            transition_potential = kwargs.get("transition_potential")
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.initial_logits.device,
            lengths=lengths,
        )
        batch, seq_len = labels.shape
        # state[batch_idx] stores that sequence's belief over hidden states at this step.
        state = self.initial_probs.expand(batch, -1)
        transition = self.transition_probs_with_potential(transition_potential)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        outputs = []
        for step in range(seq_len):
            # Predict label probabilities from current state before consuming target label.
            next_probs = torch.matmul(state, emission)
            outputs.append(torch.log(next_probs.clamp_min(eps)))
            # Condition state on gold label for this step (teacher forcing).
            posterior = state * emission[:, labels[:, step]].transpose(0, 1)
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
            next_state = torch.matmul(posterior, transition)
            # Keep finished/padded positions unchanged.
            active = (step < lengths_t).unsqueeze(-1)
            state = torch.where(active, next_state, state)
        # Ensure logits are normalized in log-space for downstream losses.
        result = torch.log_softmax(torch.stack(outputs, dim=1), dim=-1)
        # Block gradient flow through padded positions so the loss only trains
        # the model on real generated tokens (zeroing the log-probs makes every
        # label entry at a padded position equal, giving zero gradient there).
        mask = (
            torch.arange(result.shape[1], device=result.device).unsqueeze(0)
            < lengths_t.unsqueeze(1)
        ).unsqueeze(-1)
        result = result * mask.to(result.dtype)
        return result[0] if squeeze else result

    def forward(
        self,
        _contains,
        _instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput = None,
        **kwargs,
    ):
        """Teacher-forced log-probs for DomiKnowS generation concept learning.

        This is the **unconditional** HMM head: *instruction_tokens* is part of
        the sensor signature for compatibility with the prompt-conditioned head
        and is intentionally ignored.  Use
        :class:`~.promptConditionedDiscreteHMMLearner.PromptConditionedHMMGenerationHead`
        when prompt conditioning is required.
        """
        return self.sequence_log_probs(
            target_labels,
            transition_potential=transition_potential,
            **kwargs,
        )

    def trainable_parameter_names(self) -> list[str]:
        # Expose only parameters participating in gradient updates.
        """Return names of parameters optimized by a normal Torch optimizer."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]
