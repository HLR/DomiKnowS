"""GRU compact-label learner head."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from ..common.base import CompactLabelGenerationHead
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

class GRUCompactLabelGenerationHead(CompactLabelGenerationHead):
    """Small autoregressive GRU head over compact generation labels.

    The prompt is summarized with a learned token embedding and used to
    initialize the GRU hidden state.  Generated prefixes are represented in the
    compact label space, so the head can be decoded with ``constrained_label_*``.
    """

    def __init__(
        self,
        *,
        label_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        vocab_size: int | None = None,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        trainable: bool = True,
        random_seed: int | None = 0,
    ):
        if random_seed is None:
            super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
            self._init_modules(vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(random_seed))
                super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
                self._init_modules(vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        self._set_trainable(trainable)

    def _init_modules(
        self,
        vocab_size: int | None,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        self.vocab_size = _resolve_vocab_size(vocab_size, self.label_to_token_id, self.label_count)
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.num_layers = _positive_int(num_layers, "num_layers")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        self.prompt_embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.label_embedding = torch.nn.Embedding(self.label_count + 1, self.embedding_dim)
        self.prompt_to_hidden = torch.nn.Linear(self.embedding_dim, self.hidden_size * self.num_layers)
        self.gru = torch.nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = torch.nn.Linear(self.hidden_size, self.label_count)

    def _set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)

    def _initial_hidden(self, instruction_tokens: torch.Tensor | Sequence[int], batch_size: int | None = None) -> torch.Tensor:
        prompt = _normalise_prompt_ids(instruction_tokens, device=self.output.weight.device)
        _validate_token_ids(prompt, self.vocab_size, "instruction_tokens")
        features = self.prompt_embedding(prompt).mean(dim=1)
        if batch_size is not None and features.shape[0] == 1 and batch_size > 1:
            features = features.expand(batch_size, -1)
        hidden = self.prompt_to_hidden(features)
        return hidden.view(features.shape[0], self.num_layers, self.hidden_size).transpose(0, 1).contiguous()

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
        hidden = self._initial_hidden(prompt, batch_size=batch)
        start_id = self.label_count
        previous = torch.full((batch, 1), start_id, dtype=torch.long, device=labels.device)
        if seq_len > 1:
            previous = torch.cat([previous, labels[:, :-1]], dim=1)
        embedded = self.label_embedding(previous)
        output, _hidden = self.gru(embedded, hidden)
        log_probs = torch.log_softmax(self.output(output), dim=-1)
        mask = (torch.arange(seq_len, device=labels.device).unsqueeze(0) < lengths_t.unsqueeze(1)).unsqueeze(-1)
        log_probs = log_probs * mask.to(log_probs.dtype)
        return log_probs[0] if squeeze else log_probs

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int], **_kwargs) -> torch.Tensor:
        prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        hidden = self._initial_hidden(prompt_ids, batch_size=1)
        current = torch.tensor([[self.label_count]], dtype=torch.long, device=self.output.weight.device)
        output = None
        for label in prefix_labels:
            output, hidden = self.gru(self.label_embedding(current), hidden)
            current = torch.tensor([[_validate_label(label, self.label_count)]], dtype=torch.long, device=self.output.weight.device)
        output, _hidden = self.gru(self.label_embedding(current), hidden)
        return self.output(output[0, -1])

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.sequence_log_probs(target_labels, instruction_tokens=instruction_tokens)

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        ids, _device = _normalise_flat_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or [0]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.output.weight.device)
        _validate_token_ids(prompt, self.vocab_size, "input_ids")
        return prompt, labels
