"""Prompt encoders used by prompt-conditioned automata learners."""
from __future__ import annotations

import torch

__all__ = ["FrozenBackbonePromptEncoder", "PromptEmbeddingEncoder"]

class PromptEmbeddingEncoder(torch.nn.Module):
    """Small trainable prompt encoder for offline prompt-conditioned heads."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            _positive_int(vocab_size, "vocab_size"),
            _positive_int(hidden_size, "hidden_size"),
        )

    @property
    def output_size(self) -> int:
        return int(self.embedding.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.embedding(input_ids.long()).mean(dim=1)


class FrozenBackbonePromptEncoder(torch.nn.Module):
    """Frozen-backbone prompt encoder mirroring the HF compact-label learner."""

    def __init__(self, backbone: torch.nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self._output_size = int(hidden_size) if hidden_size is not None else _infer_backbone_hidden_size(backbone)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            try:
                output = self.backbone(input_ids.long(), output_hidden_states=True)
            except TypeError:
                output = self.backbone(input_ids.long())
            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, "last_hidden_state"):
                features = output.last_hidden_state
            elif hasattr(output, "hidden_states") and output.hidden_states:
                features = output.hidden_states[-1]
            elif hasattr(output, "logits"):
                features = output.logits
            else:
                raise ValueError("backbone output must expose tensor features, hidden states, or logits")
        if features.shape[-1] != self._output_size:
            raise ValueError(
                "backbone feature size does not match prompt encoder output_size; "
                "pass a backbone that exposes hidden states or set backbone_hidden_size"
            )
        if features.dim() == 2:
            return features.detach()
        return features[:, -1, :].detach()
