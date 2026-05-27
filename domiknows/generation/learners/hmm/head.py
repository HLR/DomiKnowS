"""Discrete HMM compact-label learner head."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .core import DiscreteHMM
from ...latent_potentials import LatentTransitionPotential, apply_hmm_transition_potential
from ..common.base import CompactLabelGenerationHead
from ..common.utils import (
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
        trainable: bool = False,
        random_seed: int | None = 0,
    ):
        super().__init__()
        self.pad_size = _positive_int(pad_size, "pad_size")
        self.label_count = _resolve_label_count(model, label_count)
        self.state_count = _resolve_state_count(model, state_count)
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)

        if model is None:
            initial, transition, emission = _random_hmm_parameters(
                self.state_count,
                self.label_count,
                random_seed,
            )
        else:
            initial = torch.as_tensor(model.initial, dtype=torch.float32)
            transition = torch.as_tensor(model.transition, dtype=torch.float32)
            emission = torch.as_tensor(model.emission, dtype=torch.float32)
            _validate_hmm_shapes(initial, transition, emission, self.state_count, self.label_count)

        self.initial_logits = torch.nn.Parameter(_safe_log(initial), requires_grad=trainable)
        self.transition_logits = torch.nn.Parameter(_safe_log(transition), requires_grad=trainable)
        self.emission_logits = torch.nn.Parameter(_safe_log(emission), requires_grad=trainable)

    @property
    def initial_probs(self) -> torch.Tensor:
        return torch.softmax(self.initial_logits, dim=-1)

    @property
    def transition_probs(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits, dim=-1)

    @property
    def emission_probs(self) -> torch.Tensor:
        return torch.softmax(self.emission_logits, dim=-1)

    def transition_probs_with_potential(self, transition_potential: TransitionPotentialInput = None) -> torch.Tensor:
        """Return HMM transition probabilities after optional latent-potential reweighting."""
        return apply_hmm_transition_potential(self.transition_probs, transition_potential)

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id used when appending a compact label."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _next_logits_from_prefix_labels(
        self,
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        state = self.initial_probs
        transition = self.transition_probs_with_potential(transition_potential)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps

        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            posterior = state * emission[:, label]
            posterior = posterior / posterior.sum().clamp_min(eps)
            state = torch.matmul(posterior, transition)

        next_probs = torch.matmul(state, emission)
        return torch.log(next_probs.clamp_min(eps))

    def next_label_logits(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        transition_potential: TransitionPotentialInput = None,
    ) -> torch.Tensor:
        """Return next-step logits over compact generation labels."""
        return self._next_logits_from_prefix_labels(
            self._labels_from_input_ids(input_ids),
            transition_potential=transition_potential,
        )

    def production_hmm(self, transition_potential: TransitionPotentialInput = None) -> DiscreteHMM:
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
    ) -> torch.Tensor:
        """Return teacher-forced log-probs shaped ``[batch, seq, label_count]``."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.initial_logits.device,
            lengths=lengths,
        )
        batch, seq_len = labels.shape
        state = self.initial_probs.expand(batch, -1)
        transition = self.transition_probs_with_potential(transition_potential)
        emission = self.emission_probs
        eps = torch.finfo(emission.dtype).eps
        outputs = []
        for step in range(seq_len):
            next_probs = torch.matmul(state, emission)
            outputs.append(torch.log(next_probs.clamp_min(eps)))
            posterior = state * emission[:, labels[:, step]].transpose(0, 1)
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
            next_state = torch.matmul(posterior, transition)
            active = (step < lengths_t).unsqueeze(-1)
            state = torch.where(active, next_state, state)
        result = torch.log_softmax(torch.stack(outputs, dim=1), dim=-1)
        return result[0] if squeeze else result

    def forward(
        self,
        _contains,
        instruction_tokens: torch.Tensor,
        target_labels: torch.Tensor,
        transition_potential: TransitionPotentialInput = None,
    ):
        """Teacher-forced log-probs for DomiKnowS generation concept learning."""
        return self.sequence_log_probs(target_labels, transition_potential=transition_potential)

    def trainable_parameter_names(self) -> list[str]:
        """Return names of parameters optimized by a normal Torch optimizer."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]
