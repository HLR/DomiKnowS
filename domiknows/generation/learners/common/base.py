"""Base compact-label learner contracts."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from .utils import _coerce_label_to_token_id, _invert_label_to_token_id, _positive_int, _validate_label


@runtime_checkable
class CompactLabelSequenceModel(Protocol):
    """Structural interface shared by compact-label generation/scoring heads."""

    label_count: int

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        """Return teacher-forced compact-label log-probabilities."""

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int], **kwargs) -> torch.Tensor:
        """Return next-step logits over compact labels."""

    def greedy_label_inference(self, vocabulary, input_ids: torch.Tensor | Sequence[int], **kwargs):
        """Run unconstrained greedy compact-label inference."""

    def beam_label_inference(self, vocabulary, input_ids: torch.Tensor | Sequence[int], **kwargs):
        """Run unconstrained beam compact-label inference."""

    def sample_label_inference(self, vocabulary, input_ids: torch.Tensor | Sequence[int], **kwargs):
        """Run unconstrained stochastic compact-label inference."""

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return teacher-forced compact-label log-probabilities."""

    def token_id_for_label(self, label: int) -> int:
        """Map one compact label back to the concrete token id appended by decoding."""

    def trainable_parameter_names(self) -> list[str]:
        """Return names of parameters currently optimized by Torch."""


class CompactLabelGenerationHead(torch.nn.Module):
    """Base class documenting the PMD-compatible compact-label head contract."""

    def __init__(
        self,
        *,
        label_count: int | None = None,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
    ):
        super().__init__()
        self.label_count = _positive_int(label_count, "label_count") if label_count is not None else None
        self.pad_size = _positive_int(pad_size, "pad_size")
        if label_count is not None:
            self.label_to_token_id = _coerce_label_to_token_id(label_to_token_id, self.label_count)
            self._token_id_to_label = _invert_label_to_token_id(self.label_to_token_id)

    def token_id_for_label(self, label: int) -> int:
        if self.label_count is None:
            raise ValueError("label_count is not configured")
        label = _validate_label(label, self.label_count)
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return int(token_id)

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int], **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def greedy_label_inference(self, vocabulary, input_ids: torch.Tensor | Sequence[int], **kwargs):
        """Run unconstrained greedy compact-label inference through this head."""
        from ...inference import greedy_label_inference

        return greedy_label_inference(self, vocabulary, input_ids, **kwargs)

    def beam_label_inference(self, vocabulary, input_ids: torch.Tensor | Sequence[int], **kwargs):
        """Run unconstrained beam compact-label inference through this head."""
        from ...inference import beam_label_inference

        return beam_label_inference(self, vocabulary, input_ids, **kwargs)

    def sample_label_inference(self, vocabulary, input_ids: torch.Tensor | Sequence[int], **kwargs):
        """Run unconstrained stochastic compact-label inference through this head."""
        from ...inference import sample_label_inference

        return sample_label_inference(self, vocabulary, input_ids, **kwargs)

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.sequence_log_probs(target_labels, instruction_tokens=instruction_tokens)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]
