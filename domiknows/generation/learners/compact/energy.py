"""Energy compact-label learner head.

This module implements an autoregressive local energy model in compact label
space. For each decoding step, it computes an energy value per candidate label:

- lower energy means the label is more compatible with prompt and prefix,
- higher energy means the label is less preferred.

The public next-step interface exposes logits as negative energies so it can be
used by standard decoders.

How this is used as a generative model with a DFA
--------------------------------------------------
The model itself is an unconstrained scorer. A DFA-based decoder enforces
structural validity externally:

1. Keep DFA state q and generated label prefix.
2. Call next_label_logits(input_ids_with_prefix) to get -energy logits.
3. Ask the DFA for labels reachable from state q.
4. Mask disallowed labels in logits (for example set to -inf).
5. Sample or argmax from masked logits.
6. Transition DFA state with the chosen label and append to prefix.
7. Repeat until acceptance or stopping criterion.

This cleanly separates neural preference modeling from symbolic constraints.
"""
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

class EnergyCompactLabelGenerationHead(CompactLabelGenerationHead):
    """Local energy-based compact-label scorer with next-step logits.

    Lower energy means a more preferred next compact label.  The current
    compact decoders consume ``-energy`` through ``next_label_logits`` and then
    apply their normal DFA mask.

    The context is finite-width (context_size) and left-padded with sentinel
    id label_count when the prefix is shorter than context_size.
    """

    def __init__(
        self,
        *,
        label_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        vocab_size: int | None = None,
        context_size: int = 3,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        trainable: bool = True,
        random_seed: int | None = 0,
    ):
        if random_seed is None:
            super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
            self._init_modules(vocab_size, context_size, embedding_dim, hidden_size)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(random_seed))
                super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
                self._init_modules(vocab_size, context_size, embedding_dim, hidden_size)
        self._set_trainable(trainable)

    def _init_modules(
        self,
        vocab_size: int | None,
        context_size: int,
        embedding_dim: int,
        hidden_size: int,
    ) -> None:
        self.vocab_size = _resolve_vocab_size(vocab_size, self.label_to_token_id, self.label_count)
        self.context_size = _positive_int(context_size, "context_size")
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.prompt_embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.label_embedding = torch.nn.Embedding(self.label_count + 1, self.embedding_dim)
        self.energy_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * (self.context_size + 2), self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, 1),
        )

    def _set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)

    def step_energy(
        self,
        instruction_tokens: torch.Tensor | Sequence[int],
        prefix_labels: torch.Tensor | Sequence[int],
        next_label: int,
    ) -> torch.Tensor:
        """Return local energy for one prompt, prefix, and candidate label.

        Lower values indicate higher preference under the model.
        """
        label = _validate_label(int(next_label), self.label_count)
        prompt = _normalise_prompt_ids(instruction_tokens, device=self._device)
        if isinstance(prefix_labels, torch.Tensor):
            prefix = [int(value) for value in prefix_labels.detach().long().reshape(-1).tolist()]
        else:
            prefix = [int(value) for value in prefix_labels]
        context = self._context_from_prefix(prefix).reshape(1, 1, self.context_size)
        return self._energies_for_contexts(prompt, context)[0, 0, label]

    def sequence_energy(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return summed teacher-forced local energies for target labels.

        For each position, context is built from prior gold labels. The method
        gathers the energy of each gold next label and sums over valid positions.
        """
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self._device,
            lengths=lengths,
            preserve_input_lengths=True,
        )
        _validate_labels(labels, self.label_count)
        batch, seq_len = labels.shape
        prompt = _empty_prompt(instruction_tokens, batch, device=self._device)
        contexts = []
        for step in range(seq_len):
            contexts.append(self._context_tensor(labels[:, :step]))
        context_tensor = torch.stack(contexts, dim=1)
        energies = self._energies_for_contexts(prompt, context_tensor)
        gold_energies = energies.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) < lengths_t.unsqueeze(1)
        totals = (gold_energies * mask.to(gold_energies.dtype)).sum(dim=1)
        return totals[0] if squeeze else totals

    def sequence_score(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return a high-is-better sequence score, equal to negative energy."""
        return -self.sequence_energy(
            target_labels,
            lengths=lengths,
            instruction_tokens=instruction_tokens,
        )

    def sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Return token-level log probabilities induced by local energies.

        Logits are computed as negative energies and normalized with softmax,
        giving a locally normalized autoregressive distribution.
        """
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self._device,
            lengths=lengths,
        )
        _validate_labels(labels, self.label_count)
        batch, seq_len = labels.shape
        prompt = _empty_prompt(instruction_tokens, batch, device=self._device)
        contexts = []
        for step in range(seq_len):
            contexts.append(self._context_tensor(labels[:, :step]))
        context_tensor = torch.stack(contexts, dim=1)
        log_probs = torch.log_softmax(-self._energies_for_contexts(prompt, context_tensor), dim=-1)
        mask = (torch.arange(seq_len, device=labels.device).unsqueeze(0) < lengths_t.unsqueeze(1)).unsqueeze(-1)
        log_probs = log_probs * mask.to(log_probs.dtype)
        return log_probs[0] if squeeze else log_probs

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int], **_kwargs) -> torch.Tensor:
        """Compute one-step logits for autoregressive constrained decoding.

        The returned vector is -energy over candidate labels in [0, label_count).
        This is the expected entrypoint for DFA-constrained decoding: obtain
        logits here, apply DFA-state masking externally, then sample or argmax.
        """
        prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        context = self._context_from_prefix(prefix_labels).reshape(1, 1, self.context_size)
        return -self._energies_for_contexts(prompt_ids, context)[0, 0]

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.sequence_log_probs(target_labels, instruction_tokens=instruction_tokens)

    @property
    def _device(self) -> torch.device:
        return self.energy_mlp[0].weight.device

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        """Split flat ids into prompt-token and generated-label segments."""
        ids, _device = _normalise_flat_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or [0]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        _validate_token_ids(prompt, self.vocab_size, "input_ids")
        return prompt, labels

    def _context_tensor(self, prefix_labels: torch.Tensor) -> torch.Tensor:
        """Build batched fixed-width contexts from batched label prefixes."""
        batch = prefix_labels.shape[0]
        context = torch.full((batch, self.context_size), self.label_count, dtype=torch.long, device=prefix_labels.device)
        if prefix_labels.shape[1] > 0:
            tail = prefix_labels[:, -self.context_size :]
            context[:, -tail.shape[1] :] = tail
        return context

    def _context_from_prefix(self, prefix_labels: Sequence[int]) -> torch.Tensor:
        """Build one fixed-width context from an in-memory label prefix."""
        values = [_validate_label(label, self.label_count) for label in prefix_labels[-self.context_size :]]
        padded = [self.label_count] * (self.context_size - len(values)) + values
        return torch.tensor(padded, dtype=torch.long, device=self._device)

    def _energies_for_contexts(self, prompt: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """Return per-label energies for each (prompt, context) row."""
        _validate_token_ids(prompt, self.vocab_size, "instruction_tokens")
        prompt_features = self.prompt_embedding(prompt).mean(dim=1)
        if prompt_features.shape[0] == 1 and contexts.shape[0] > 1:
            prompt_features = prompt_features.expand(contexts.shape[0], -1)
        if prompt_features.shape[0] != contexts.shape[0]:
            raise ValueError("instruction_tokens batch size must be 1 or match target_labels")
        context_features = self.label_embedding(contexts).flatten(start_dim=-2)
        prompt_features = prompt_features.unsqueeze(1).expand(-1, contexts.shape[1], -1)
        label_ids = torch.arange(self.label_count, device=contexts.device)
        label_features = self.label_embedding(label_ids).reshape(1, 1, self.label_count, self.embedding_dim)
        prompt_features = prompt_features.unsqueeze(2).expand(-1, -1, self.label_count, -1)
        context_features = context_features.unsqueeze(2).expand(-1, -1, self.label_count, -1)
        features = torch.cat([prompt_features, context_features, label_features.expand(contexts.shape[0], contexts.shape[1], -1, -1)], dim=-1)
        return self.energy_mlp(features).squeeze(-1)
