"""Spectral WFA compact-label learner head."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .hankel import WeightedFiniteAutomaton
from ..common.base import CompactLabelGenerationHead
from ...latent_potentials import LatentTransitionPotential, apply_wfa_transition_potential
from ..common.utils import (
    TransitionPotentialInput,
    _coerce_label_to_token_id,
    _empty_or_prompt,
    _first_generated_index,
    _invert_label_to_token_id,
    _labels_from_input_ids,
    _normalise_flat_ids,
    _positive_int,
    _random_wfa_parameters,
    _resolve_wfa_label_count,
    _resolve_wfa_state_count,
    _safe_log,
    _target_label_batch,
    _target_labels,
    _validate_label,
    _validate_wfa_shapes,
)

__all__ = ["SpectralWFAGenerationHead"]

class SpectralWFAGenerationHead(CompactLabelGenerationHead):
    """Compact-label generation head backed by a signed weighted automaton.

    WFA sequence scores may be signed, so next-symbol scores are treated as
    logits rather than probabilities.  This keeps spectral outputs source
    compatible with DomiKnowS concept losses without forcing stochastic
    normalization.
    """

    def __init__(
        self,
        model: WeightedFiniteAutomaton | None = None,
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
        self.label_count = _resolve_wfa_label_count(model, label_count)
        self.state_count = _resolve_wfa_state_count(model, state_count)
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)

        if model is None:
            initial, transitions, final = _random_wfa_parameters(
                self.state_count,
                self.label_count,
                random_seed,
            )
        else:
            initial = torch.as_tensor(model.initial, dtype=torch.float32)
            final = torch.as_tensor(model.final, dtype=torch.float32)
            transitions = torch.stack(
                [torch.as_tensor(model.transitions[symbol], dtype=torch.float32) for symbol in model.symbols],
                dim=0,
            )
            _validate_wfa_shapes(initial, transitions, final, self.state_count, self.label_count)

        self.initial = torch.nn.Parameter(initial, requires_grad=trainable)
        self.transitions = torch.nn.Parameter(transitions, requires_grad=trainable)
        self.final = torch.nn.Parameter(final, requires_grad=trainable)

    def transitions_with_potential(
        self,
        transition_potential: TransitionPotentialInput = None,
        *,
        mode: str = "multiply",
    ) -> torch.Tensor:
        """Return signed WFA transitions after optional latent-potential reweighting."""
        return apply_wfa_transition_potential(self.transitions, transition_potential, mode=mode)

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id used when appending a compact label."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def _labels_from_input_ids(self, input_ids: torch.Tensor | Sequence[int]) -> list[int]:
        return _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)

    def _prefix_state(
        self,
        prefix_labels: Sequence[int],
        *,
        transitions: torch.Tensor | None = None,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        state = self.initial
        transitions = (
            self.transitions_with_potential(transition_potential, mode=transition_potential_mode)
            if transitions is None
            else transitions
        )
        for raw_label in prefix_labels:
            label = _validate_label(raw_label, self.label_count)
            state = torch.matmul(state, transitions[label])
        return state

    def _next_logits_from_prefix_labels(
        self,
        prefix_labels: Sequence[int],
        transition_potential: TransitionPotentialInput = None,
        *,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        transitions = self.transitions_with_potential(transition_potential, mode=transition_potential_mode)
        state = self._prefix_state(prefix_labels, transitions=transitions)
        next_states = torch.einsum("s,lsd->ld", state, transitions)
        return torch.matmul(next_states, self.final)

    def next_label_logits(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        """Return next-step signed WFA scores over compact labels."""
        return self._next_logits_from_prefix_labels(
            self._labels_from_input_ids(input_ids),
            transition_potential=transition_potential,
            transition_potential_mode=transition_potential_mode,
        )

    def production_wfa(
        self,
        transition_potential: TransitionPotentialInput = None,
        *,
        transition_potential_mode: str = "multiply",
    ) -> WeightedFiniteAutomaton:
        """Return a Torch-backed WFA view of the current head parameters."""
        return WeightedFiniteAutomaton(
            self.initial,
            self.transitions_with_potential(transition_potential, mode=transition_potential_mode),
            self.final,
            tuple(range(self.label_count)),
        )

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        transition_potential: TransitionPotentialInput = None,
        transition_potential_mode: str = "multiply",
    ) -> torch.Tensor:
        """Return teacher-forced log-probs shaped ``[batch, seq, label_count]``."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.initial.device,
            lengths=lengths,
        )
        batch, seq_len = labels.shape
        state = self.initial.expand(batch, -1)
        transitions = self.transitions_with_potential(transition_potential, mode=transition_potential_mode)
        outputs = []
        for step in range(seq_len):
            next_states = torch.einsum("bs,lsd->bld", state, transitions)
            outputs.append(torch.matmul(next_states, self.final))
            chosen = transitions.index_select(0, labels[:, step])
            next_state = torch.bmm(state.unsqueeze(1), chosen).squeeze(1)
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
