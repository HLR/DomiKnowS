"""CRF compact-label scorer."""
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

class CRFCompactLabelScorer(CompactLabelGenerationHead):
    """Compact linear-chain CRF scorer over compact labels.

    Exact global training uses the linear-chain CRF partition function.
    ``next_label_logits`` remains a local proposal interface for the existing
    compact DFA decoders; exact constrained CRF decoding requires a product
    Viterbi search over CRF and DFA states.
    """

    def __init__(
        self,
        *,
        label_count: int,
        pad_size: int = 4,
        label_to_token_id: Sequence[int | None] | None = None,
        vocab_size: int | None = None,
        embedding_dim: int = 32,
        use_end_logits: bool = True,
        trainable: bool = True,
        random_seed: int | None = 0,
    ):
        if random_seed is None:
            super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
            self._init_modules(vocab_size, embedding_dim, use_end_logits)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(random_seed))
                super().__init__(label_count=label_count, pad_size=pad_size, label_to_token_id=label_to_token_id)
                self._init_modules(vocab_size, embedding_dim, use_end_logits)
        self._set_trainable(trainable)

    def _init_modules(self, vocab_size: int | None, embedding_dim: int, use_end_logits: bool) -> None:
        self.vocab_size = _resolve_vocab_size(vocab_size, self.label_to_token_id, self.label_count)
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.use_end_logits = bool(use_end_logits)
        self.prompt_embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.unary_projector = torch.nn.Linear(self.embedding_dim, self.label_count)
        self.start_logits = torch.nn.Parameter(torch.zeros(self.label_count))
        self.transition_logits = torch.nn.Parameter(torch.zeros(self.label_count, self.label_count))
        if self.use_end_logits:
            self.end_logits = torch.nn.Parameter(torch.zeros(self.label_count))
        else:
            self.register_parameter("end_logits", None)

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
        """Return exact CRF marginal log-probs for PMD/DataNode probabilities."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self._device,
            lengths=lengths,
            preserve_input_lengths=True,
        )
        _validate_labels(labels, self.label_count)
        log_probs = self.marginal_log_probs(
            labels,
            lengths=lengths_t,
            instruction_tokens=instruction_tokens,
        )
        return log_probs[0] if squeeze else log_probs

    def local_sequence_log_probs(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return locally normalized teacher-forced log-probs.

        This preserves the original CRF-like scorer behavior for debugging and
        local proposal diagnostics. Use ``crf_nll`` for exact CRF training.
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
        logits = self._local_logits_for_labels(prompt, labels)
        log_probs = torch.log_softmax(logits, dim=-1)
        mask = (torch.arange(seq_len, device=labels.device).unsqueeze(0) < lengths_t.unsqueeze(1)).unsqueeze(-1)
        log_probs = log_probs * mask.to(log_probs.dtype)
        return log_probs[0] if squeeze else log_probs

    def next_label_logits(self, input_ids: torch.Tensor | Sequence[int], **_kwargs) -> torch.Tensor:
        prompt_ids, prefix_labels = self._split_prompt_and_prefix(input_ids)
        unary = self._prompt_unary(prompt_ids)[0]
        if prefix_labels:
            previous = _validate_label(prefix_labels[-1], self.label_count)
            return unary + self.transition_logits[previous]
        return unary + self.start_logits

    def sequence_score(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return the unnormalized gold path score for target labels."""
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
        unary = self._prompt_unary(prompt)
        scores = torch.zeros(batch, dtype=unary.dtype, device=unary.device)
        for step in range(seq_len):
            active = step < lengths_t
            current = labels[:, step]
            local = unary.gather(1, current.unsqueeze(1)).squeeze(1)
            if step == 0:
                local = local + self.start_logits[current]
            else:
                previous = labels[:, step - 1]
                local = local + self.transition_logits[previous, current]
            scores = torch.where(active, scores + local, scores)
        if self.end_logits is not None:
            last_indices = (lengths_t - 1).clamp_min(0)
            last_labels = labels.gather(1, last_indices.unsqueeze(1)).squeeze(1)
            scores = scores + self.end_logits[last_labels]
        return scores[0] if squeeze else scores

    def sequence_energy(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Source-compatible alias for ``sequence_score``."""
        return self.sequence_score(
            target_labels,
            lengths=lengths,
            instruction_tokens=instruction_tokens,
        )

    def log_partition(
        self,
        lengths: torch.Tensor | Sequence[int] | int,
        *,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Compute exact CRF log partition values with forward DP."""
        lengths_t, max_len, squeeze = self._normalise_lengths(lengths, max_length=max_length)
        prompt = _empty_prompt(instruction_tokens, int(lengths_t.numel()), device=self._device)
        unary = self._prompt_unary(prompt)
        if unary.shape[0] == 1 and lengths_t.numel() > 1:
            unary = unary.expand(int(lengths_t.numel()), -1)
        if unary.shape[0] != lengths_t.numel():
            raise ValueError("instruction_tokens batch size must be 1 or match lengths")
        alpha, log_z = self._forward_alg(unary, lengths_t, max_len)
        del alpha
        return log_z[0] if squeeze else log_z

    def crf_nll(
        self,
        target_labels: torch.Tensor | Sequence[int],
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return exact negative log-likelihood under the linear-chain CRF."""
        labels, lengths_t, squeeze = _target_label_batch(
            target_labels,
            self.pad_size,
            device=self._device,
            lengths=lengths,
            preserve_input_lengths=True,
        )
        _validate_labels(labels, self.label_count)
        score = self.sequence_score(labels, lengths=lengths_t, instruction_tokens=instruction_tokens)
        log_z = self.log_partition(lengths_t, instruction_tokens=instruction_tokens, max_length=labels.shape[1])
        losses = log_z.reshape(-1) - score.reshape(-1)
        if reduction == "none":
            return losses[0] if squeeze else losses
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.mean()
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")

    def marginal_log_probs(
        self,
        labels_or_lengths: torch.Tensor | Sequence[int] | int,
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
        instruction_tokens: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return exact token marginal log-probs shaped ``[batch, seq, labels]``."""
        lengths_t, max_len, squeeze = self._lengths_from_labels_or_lengths(labels_or_lengths, lengths=lengths)
        prompt = _empty_prompt(instruction_tokens, int(lengths_t.numel()), device=self._device)
        unary = self._prompt_unary(prompt)
        if unary.shape[0] == 1 and lengths_t.numel() > 1:
            unary = unary.expand(int(lengths_t.numel()), -1)
        if unary.shape[0] != lengths_t.numel():
            raise ValueError("instruction_tokens batch size must be 1 or match target_labels")
        alpha, log_z = self._forward_alg(unary, lengths_t, max_len)
        beta = self._backward_alg(unary, lengths_t, max_len)
        result = alpha + beta - log_z.reshape(-1, 1, 1)
        mask = torch.arange(max_len, device=self._device).unsqueeze(0) < lengths_t.unsqueeze(1)
        result = result * mask.unsqueeze(-1).to(result.dtype)
        return result[0] if squeeze else result

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        return self.sequence_log_probs(target_labels, instruction_tokens=instruction_tokens)

    @property
    def _device(self) -> torch.device:
        return self.start_logits.device

    def _split_prompt_and_prefix(self, input_ids: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, list[int]]:
        ids, _device = _normalise_flat_ids(input_ids)
        split = _first_generated_index(ids, self._token_id_to_label)
        prompt_ids = ids[:split] or [0]
        prefix_ids = ids[split:]
        labels = [self._token_id_to_label[int(token_id)] for token_id in prefix_ids if int(token_id) in self._token_id_to_label]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        _validate_token_ids(prompt, self.vocab_size, "input_ids")
        return prompt, labels

    def _prompt_unary(self, instruction_tokens: torch.Tensor | Sequence[int]) -> torch.Tensor:
        prompt = _normalise_prompt_ids(instruction_tokens, device=self._device)
        _validate_token_ids(prompt, self.vocab_size, "instruction_tokens")
        return self.unary_projector(self.prompt_embedding(prompt).mean(dim=1))

    def _local_logits_for_labels(self, instruction_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        unary = self._prompt_unary(instruction_tokens)
        if unary.shape[0] == 1 and labels.shape[0] > 1:
            unary = unary.expand(labels.shape[0], -1)
        if unary.shape[0] != labels.shape[0]:
            raise ValueError("instruction_tokens batch size must be 1 or match target_labels")
        rows = []
        for step in range(labels.shape[1]):
            if step == 0:
                rows.append(unary + self.start_logits)
            else:
                previous = labels[:, step - 1]
                rows.append(unary + self.transition_logits.index_select(0, previous))
        return torch.stack(rows, dim=1)

    def _forward_alg(self, unary: torch.Tensor, lengths: torch.Tensor, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = unary + self.start_logits
        history = [alpha]
        for _step in range(1, max_len):
            scores = alpha.unsqueeze(2) + self.transition_logits.unsqueeze(0)
            alpha = unary + torch.logsumexp(scores, dim=1)
            history.append(alpha)
        stacked = torch.stack(history, dim=1)
        last = stacked.gather(1, (lengths - 1).reshape(-1, 1, 1).expand(-1, 1, self.label_count)).squeeze(1)
        if self.end_logits is not None:
            last = last + self.end_logits
        return stacked, torch.logsumexp(last, dim=-1)

    def _backward_alg(self, unary: torch.Tensor, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        end = self.end_logits if self.end_logits is not None else torch.zeros(self.label_count, dtype=unary.dtype, device=unary.device)
        beta = end.reshape(1, -1).expand(unary.shape[0], -1)
        history = [None for _ in range(max_len)]
        history[-1] = beta
        for step in range(max_len - 2, -1, -1):
            scores = self.transition_logits.unsqueeze(0) + unary.unsqueeze(1) + beta.unsqueeze(1)
            candidate = torch.logsumexp(scores, dim=2)
            is_terminal = step == (lengths - 1)
            beta = torch.where(is_terminal.unsqueeze(1), end.reshape(1, -1), candidate)
            history[step] = beta
        return torch.stack(history, dim=1)

    def _normalise_lengths(
        self,
        lengths: torch.Tensor | Sequence[int] | int,
        *,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, int, bool]:
        if isinstance(lengths, int):
            lengths_t = torch.tensor([lengths], dtype=torch.long, device=self._device)
            squeeze = True
        else:
            lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=self._device).reshape(-1)
            squeeze = lengths_t.numel() == 1
        if lengths_t.numel() < 1:
            raise ValueError("lengths must contain at least one value")
        if torch.any(lengths_t < 1):
            raise ValueError("CRF sequence lengths must be positive")
        max_len = int(max_length or int(lengths_t.max().item()))
        if max_len < int(lengths_t.max().item()):
            raise ValueError("max_length must be at least max(lengths)")
        return lengths_t, max_len, squeeze

    def _lengths_from_labels_or_lengths(
        self,
        labels_or_lengths: torch.Tensor | Sequence[int] | int,
        *,
        lengths: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, int, bool]:
        if lengths is not None:
            labels = torch.as_tensor(labels_or_lengths, dtype=torch.long, device=self._device)
            squeeze = labels.dim() <= 1
            if labels.dim() == 0:
                max_len = int(lengths if isinstance(lengths, int) else torch.as_tensor(lengths).reshape(-1)[0].item())
            elif labels.dim() == 1:
                max_len = int(labels.numel())
            else:
                max_len = int(labels.shape[1])
            lengths_t, _max_len, _squeeze_lengths = self._normalise_lengths(lengths, max_length=max_len)
            return lengths_t, max_len, squeeze
        if isinstance(labels_or_lengths, int):
            return self._normalise_lengths(labels_or_lengths)
        values = torch.as_tensor(labels_or_lengths, dtype=torch.long, device=self._device)
        if values.dim() == 0:
            return self._normalise_lengths(int(values.item()))
        if values.dim() == 1:
            # Treat a 1D tensor/sequence as one label sequence, matching
            # sequence_log_probs and other compact heads.
            return torch.tensor([values.numel()], dtype=torch.long, device=self._device), int(values.numel()), True
        if values.dim() == 2:
            return torch.full((values.shape[0],), values.shape[1], dtype=torch.long, device=self._device), int(values.shape[1]), False
        raise ValueError("labels_or_lengths must be a length, [seq], or [batch, seq]")
