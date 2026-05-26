"""Graph spectral-WFA compact-label learner head."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import torch

from ..compact import CompactLabelGenerationHead
from ...graph_hmm.constraints import validate_mask
from .utils import (
    _coerce_label_to_token_id,
    _flat_input_ids,
    _invert_label_to_token_id,
    _labels_from_input_ids,
    _normalize_vector,
    _target_label_batch,
    _validate_label,
    _validate_wfa_shapes,
)

__all__ = ["GraphSpectralGenerationHead"]

class GraphSpectralGenerationHead(CompactLabelGenerationHead):
    """Compact-label signed WFA head initialized from graph spectral learning."""

    def __init__(
        self,
        *,
        label_count: int,
        state_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        trainable: bool = True,
        random_seed: int = 0,
        initial=None,
        final=None,
        operators: Sequence[Any] | Mapping[Any, Any] | None = None,
        symbols: Sequence[Any] | None = None,
    ):
        """Initialize signed-WFA Torch head for compact-label decoding/training."""
        super().__init__()
        if label_count < 1 or state_count < 1:
            raise ValueError("label_count and state_count must be at least 1")
        self.label_count = int(label_count)
        self.state_count = int(state_count)
        self.pad_size = int(pad_size)
        self.symbols = tuple(symbols) if symbols is not None else tuple(range(self.label_count))
        if len(self.symbols) != self.label_count:
            raise ValueError("symbols length must match label_count")
        self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
        self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)
        generator = torch.Generator().manual_seed(int(random_seed))
        if initial is None:
            # Bias state 0 so random init starts with a mild anchor.
            initial_t = torch.randn(self.state_count, generator=generator) * 0.1
            initial_t[0] += 1.0
        else:
            initial_t = torch.as_tensor(initial, dtype=torch.float32)
        if final is None:
            final_t = torch.randn(self.state_count, generator=generator) * 0.1
        else:
            final_t = torch.as_tensor(final, dtype=torch.float32)
        if operators is None:
            operators_t = torch.randn(self.label_count, self.state_count, self.state_count, generator=generator) * 0.1
        else:
            if isinstance(operators, Mapping):
                # Preserve explicit symbol ordering when operators are keyed.
                operators_t = torch.stack([torch.as_tensor(operators[symbol], dtype=torch.float32) for symbol in self.symbols], dim=0)
            else:
                operators_t = torch.as_tensor(operators, dtype=torch.float32)
        _validate_wfa_shapes(initial_t, final_t, operators_t, self.label_count, self.state_count)
        self.initial = torch.nn.Parameter(initial_t, requires_grad=trainable)
        self.final = torch.nn.Parameter(final_t, requires_grad=trainable)
        self.operators = torch.nn.Parameter(operators_t, requires_grad=trainable)

    @classmethod
    def from_graph_spectral(
        cls,
        automaton,
        *,
        trainable: bool = True,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
    ) -> "GraphSpectralGenerationHead":
        automaton._require_fitted()
        return cls(
            label_count=len(automaton.id_to_symbol),
            state_count=int(automaton.initial.numel()),
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            trainable=trainable,
            initial=automaton.initial,
            final=automaton.final,
            operators=automaton.operators,
            symbols=automaton.id_to_symbol,
        )

    def token_id_for_label(self, label: int) -> int:
        """Resolve tokenizer id for a label; fail for non 1-to-1 mappings."""
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def trainable_parameter_names(self) -> list[str]:
        """List parameter names currently participating in gradient updates."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def sequence_log_probs(self, target_labels: torch.Tensor | Sequence[int], *, lengths=None) -> torch.Tensor:
        """Compute per-step log-probabilities for target label sequences."""
        labels, _lengths_t, step_mask, squeeze = _target_label_batch(target_labels, self.pad_size, lengths=lengths)
        outputs = []
        for row in labels:
            state = self.initial
            row_outputs = []
            for label_t in row:
                # Score each possible next label before advancing with target label.
                logits = torch.stack([state @ self.operators[label] @ self.final for label in range(self.label_count)])
                row_outputs.append(torch.log_softmax(logits, dim=-1))
                state = state @ self.operators[int(label_t.item())]
            outputs.append(torch.stack(row_outputs, dim=0))
        result = torch.stack(outputs, dim=0)
        # Zero padded timesteps so downstream losses can ignore them safely.
        result = result * step_mask.to(dtype=result.dtype).unsqueeze(-1)
        return result[0] if squeeze else result

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return unnormalized next-label scores for a token-id prefix."""
        prefix_labels = _labels_from_input_ids(input_ids, self._token_id_to_label, self.label_count)
        state = self.initial
        for label in prefix_labels:
            state = state @ self.operators[label]
        return torch.stack([state @ self.operators[label] @ self.final for label in range(self.label_count)])

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        """PMD module interface: returns sequence log-probabilities."""
        return self.sequence_log_probs(target_labels)
