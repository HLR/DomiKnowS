"""Shared helpers for compact-label learner heads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

__all__ = [
    '_positive_int',
    '_coerce_label_to_token_id',
    '_invert_label_to_token_id',
    '_resolve_vocab_size',
    '_validate_label',
    '_validate_labels',
    '_validate_token_ids',
    '_normalise_flat_ids',
    '_normalise_prompt_ids',
    '_empty_prompt',
    '_first_generated_index',
    '_target_label_batch',
]

def _positive_int(value: int | None, name: str) -> int:
    value = int(value) if value is not None else 0
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value

def _coerce_label_to_token_id(label_to_token_id: Sequence[int | None] | None, label_count: int) -> tuple[int | None, ...]:
    if label_to_token_id is None:
        return tuple(range(label_count))
    if len(label_to_token_id) != label_count:
        raise ValueError("label_to_token_id must contain one entry per compact label")
    return tuple(None if token_id is None else int(token_id) for token_id in label_to_token_id)

def _invert_label_to_token_id(label_to_token_id: Sequence[int | None]) -> Mapping[int, int]:
    return {int(token_id): label for label, token_id in enumerate(label_to_token_id) if token_id is not None}

def _resolve_vocab_size(vocab_size: int | None, label_to_token_id: Sequence[int | None], label_count: int) -> int:
    max_token_id = max((int(token_id) for token_id in label_to_token_id if token_id is not None), default=-1)
    inferred = max(label_count + 16, max_token_id + 1, 1)
    return inferred if vocab_size is None else _positive_int(vocab_size, "vocab_size")

def _validate_label(label: int, label_count: int) -> int:
    label = int(label)
    if label < 0 or label >= label_count:
        raise ValueError(f"label {label} is out of range for {label_count} labels")
    return label

def _validate_labels(labels: torch.Tensor, label_count: int) -> None:
    if torch.any((labels < 0) | (labels >= label_count)):
        raise ValueError(f"target_labels must be in [0, {label_count})")

def _validate_token_ids(ids: torch.Tensor, vocab_size: int, name: str) -> None:
    if torch.any(ids < 0) or torch.any(ids >= vocab_size):
        raise ValueError(f"{name} contains token ids outside configured vocab_size={vocab_size}")

def _normalise_flat_ids(input_ids: torch.Tensor | Sequence[int]) -> tuple[list[int], torch.device]:
    if isinstance(input_ids, torch.Tensor):
        device = input_ids.device
        if input_ids.dim() == 0:
            flat = [int(input_ids.item())]
        elif input_ids.dim() == 1:
            flat = [int(value) for value in input_ids.detach().long().tolist()]
        elif input_ids.dim() == 2 and input_ids.shape[0] == 1:
            flat = [int(value) for value in input_ids[0].detach().long().tolist()]
        else:
            raise ValueError("input_ids must describe a single sequence")
        return flat, device
    return [int(value) for value in input_ids], torch.device("cpu")

def _normalise_prompt_ids(input_ids: torch.Tensor | Sequence[int], *, device: torch.device) -> torch.Tensor:
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.detach().long().to(device)
    else:
        ids = torch.tensor(list(input_ids), dtype=torch.long, device=device)
    if ids.dim() == 0:
        ids = ids.reshape(1, 1)
    elif ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError("instruction_tokens must have shape [seq] or [batch, seq]")
    return ids

def _empty_prompt(
    instruction_tokens: torch.Tensor | Sequence[int] | None,
    batch_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if instruction_tokens is None:
        return torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    prompt = _normalise_prompt_ids(instruction_tokens, device=device)
    if prompt.shape[0] == 1 and batch_size > 1:
        return prompt.expand(batch_size, -1)
    if prompt.shape[0] != batch_size:
        raise ValueError("instruction_tokens batch size must be 1 or match target_labels")
    return prompt

def _first_generated_index(ids: Sequence[int], token_id_to_label: Mapping[int, int]) -> int:
    for index, token_id in enumerate(ids):
        if int(token_id) in token_id_to_label:
            return index
    return len(ids)

def _target_label_batch(
    target_labels: torch.Tensor | Sequence[int],
    pad_size: int,
    *,
    device: torch.device,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int = 0,
    preserve_input_lengths: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if isinstance(target_labels, torch.Tensor):
        labels = target_labels.detach().long().to(device)
    else:
        labels = torch.tensor(target_labels, dtype=torch.long, device=device)
    squeeze = labels.dim() == 1
    if squeeze:
        labels = labels.unsqueeze(0)
    if labels.dim() != 2:
        raise ValueError("target_labels must have shape [seq] or [batch, seq]")
    input_seq_len = labels.shape[1]
    if labels.shape[1] >= pad_size:
        labels = labels[:, :pad_size]
    else:
        padding = torch.full((labels.shape[0], pad_size - labels.shape[1]), int(eos_label), dtype=torch.long, device=device)
        labels = torch.cat([labels, padding], dim=1)
    if lengths is None:
        inferred = min(int(input_seq_len), int(labels.shape[1])) if preserve_input_lengths else int(labels.shape[1])
        lengths_t = torch.full((labels.shape[0],), inferred, dtype=torch.long, device=device)
    else:
        lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device).reshape(-1)
        if lengths_t.numel() != labels.shape[0]:
            raise ValueError("lengths must contain one value per batch item")
        lengths_t = torch.clamp(lengths_t, min=1, max=labels.shape[1])
    return labels, lengths_t, squeeze
