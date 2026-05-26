"""Transformer compact-label learner head."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import CompactLabelGenerationHead
from .utils import (
    _coerce_label_to_token_id,
    _empty_prompt,
    _first_generated_index,
    _invert_label_to_token_id,
    _normalise_flat_ids,
    _normalise_prompt_ids,
    _positive_int,
    _resolve_vocab_size,
    _target_label_batch,
    _validate_label,
    _validate_labels,
    _validate_token_ids,
)

class TransformerCompactLabelGenerationHead(CompactLabelGenerationHead):
    """Small causal Transformer head over compact generation labels."""

    def __init__(
        self,
        *,
        label_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        vocab_size: int | None = None,
        embedding_dim: int = 32,
        hidden_size: int | None = None,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.0,
        trainable: bool = True,
        random_seed: int | None = 0,
    ):
        if random_seed is None:
            super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
            self._init_modules(vocab_size, embedding_dim, hidden_size, num_layers, num_heads, dropout)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(random_seed))
                super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
                self._init_modules(vocab_size, embedding_dim, hidden_size, num_layers, num_heads, dropout)
        self._set_trainable(trainable)

    def _init_modules(
        self,
        vocab_size: int | None,
        embedding_dim: int,
        hidden_size: int | None,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        self.vocab_size = _resolve_vocab_size(vocab_size, self.label_to_token_id, self.label_count)
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        feedforward = _positive_int(hidden_size or self.embedding_dim * 4, "hidden_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.num_heads = _positive_int(num_heads, "num_heads")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        self.prompt_embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.label_embedding = torch.nn.Embedding(self.label_count + 1, self.embedding_dim)
        self.position_embedding = torch.nn.Embedding(self.pad_size + 1, self.embedding_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=feedforward,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.prompt_projector = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.output = torch.nn.Linear(self.embedding_dim, self.label_count)

    def _set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self.output.weight.device,
            lengths=lengths,
        )
        _validate_labels(labels, self.label_count)
        batch, seq_len = labels.shape
        prompt = _empty_prompt(instruction_tokens, batch, device=self.output.weight.device)
        previous = torch.full((batch, 1), self.label_count, dtype=torch.long, device=labels.device)
        if seq_len > 1:
            previous = torch.cat([previous, labels[:, :-1]], dim=1)
        logits = self._logits_for_previous(prompt, previous)
        log_probs = torch.log_softmax(logits, dim=-1)
        mask = (torch.arange(seq_len, device=labels.device).unsqueeze(0) < lengths_t.unsqueeze(1)).unsqueeze(-1)
        log_probs = log_probs * mask.to(log_probs.dtype)
        return log_probs[0] if squeeze else log_probs

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int], **_kwargs) -> torch.Tensor:
        prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        previous = [self.label_count] + [_validate_label(label, self.label_count) for label in prefix_labels]
        if len(previous) > self.pad_size:
            previous = previous[-self.pad_size :]
        previous_t = torch.tensor([previous], dtype=torch.long, device=self.output.weight.device)
        logits = self._logits_for_previous(prompt_ids, previous_t)
        return logits[0, -1]

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.sequence_log_probs(target_labels, instruction_tokens=instruction_tokens)

    def _logits_for_previous(self, prompt: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        _validate_token_ids(prompt, self.vocab_size, "instruction_tokens")
        prompt_features = self.prompt_embedding(prompt).mean(dim=1)
        if prompt_features.shape[0] == 1 and previous.shape[0] > 1:
            prompt_features = prompt_features.expand(previous.shape[0], -1)
        positions = torch.arange(previous.shape[1], device=previous.device).clamp_max(self.pad_size)
        hidden = self.label_embedding(previous) + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + self.prompt_projector(prompt_features).unsqueeze(1)
        causal_mask = torch.triu(
            torch.full((previous.shape[1], previous.shape[1]), float("-inf"), device=previous.device),
            diagonal=1,
        )
        encoded = self.transformer(hidden, mask=causal_mask)
        return self.output(encoded)

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        ids, _device = _normalise_flat_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or [0]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.output.weight.device)
        _validate_token_ids(prompt, self.vocab_size, "input_ids")
        return prompt, labels
