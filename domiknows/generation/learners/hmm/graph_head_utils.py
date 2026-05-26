"""Shared helpers for graph-HMM learner heads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .constraints import project_matrix_rows

__all__ = [
    '_random_hmm_parameters',
    '_safe_log',
    '_normalize_vector',
    '_validate_hmm_shapes',
    '_validate_wfa_shapes',
    '_target_label_batch',
    '_coerce_label_to_token_id',
    '_invert_label_to_token_id',
    '_labels_from_input_ids',
    '_flat_input_ids',
    '_normalise_prompt_ids',
    '_first_generated_index',
    '_validate_label',
]

def _random_hmm_parameters(
    state_count: int,
    label_count: int,
    random_seed: int,
    transition_mask: torch.Tensor,
    emission_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random positive HMM parameters and project through legal masks."""
    generator = torch.Generator().manual_seed(int(random_seed))
    initial = torch.rand(state_count, generator=generator) + 0.1
    transition = torch.rand(state_count, state_count, generator=generator) + 0.1
    emission = torch.rand(state_count, label_count, generator=generator) + 0.1
    return (
        _normalize_vector(initial),
        project_matrix_rows(transition, transition_mask).to(dtype=torch.float32),
        project_matrix_rows(emission, emission_mask).to(dtype=torch.float32),
    )

def _safe_log(tensor: torch.Tensor) -> torch.Tensor:
    """Numerically safe log with dtype-aware epsilon floor."""
    return torch.log(tensor.clamp_min(torch.finfo(tensor.dtype).eps))

def _normalize_vector(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a non-negative 1D vector with uniform fallback on zero sum."""
    if tensor.ndim != 1:
        raise ValueError("initial must be a rank-1 tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("initial must contain only finite values")
    tensor = tensor.clamp_min(0)
    total = tensor.sum()
    if total <= 0:
        return torch.full_like(tensor, 1.0 / tensor.numel())
    return tensor / total

def _validate_hmm_shapes(initial: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor, state_count: int, label_count: int) -> None:
    """Validate HMM parameter tensor shapes."""
    if tuple(initial.shape) != (state_count,):
        raise ValueError("initial shape must be [state_count]")
    if tuple(transition.shape) != (state_count, state_count):
        raise ValueError("transition shape must be [state_count, state_count]")
    if tuple(emission.shape) != (state_count, label_count):
        raise ValueError("emission shape must be [state_count, label_count]")

def _validate_wfa_shapes(initial: torch.Tensor, final: torch.Tensor, operators: torch.Tensor, label_count: int, state_count: int) -> None:
    """Validate signed-WFA parameter shapes and finiteness."""
    if tuple(initial.shape) != (state_count,):
        raise ValueError("initial shape must be [state_count]")
    if tuple(final.shape) != (state_count,):
        raise ValueError("final shape must be [state_count]")
    if tuple(operators.shape) != (label_count, state_count, state_count):
        raise ValueError("operators shape must be [label_count, state_count, state_count]")
    for name, tensor in (("initial", initial), ("final", final), ("operators", operators)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")

def _target_label_batch(target_labels: torch.Tensor | Sequence[int], pad_size: int, *, lengths=None):
    """Coerce labels to a 2D long batch and pad/truncate to ``pad_size``."""
    labels = torch.as_tensor(target_labels, dtype=torch.long)
    squeeze = labels.ndim == 1
    if squeeze:
        labels = labels.unsqueeze(0)
    if labels.ndim != 2:
        raise ValueError("target_labels must be rank 1 or 2")
    if labels.shape[1] > pad_size:
        labels = labels[:, :pad_size]
    if labels.shape[1] < pad_size:
        pad = torch.zeros((labels.shape[0], pad_size - labels.shape[1]), dtype=torch.long, device=labels.device)
        labels = torch.cat([labels, pad], dim=1)
    if (labels < 0).any():
        raise ValueError("target_labels must be non-negative")
    lengths_t = (
        torch.as_tensor(lengths, dtype=torch.long, device=labels.device)
        if lengths is not None
        else torch.full((labels.shape[0],), labels.shape[1], dtype=torch.long, device=labels.device)
    )
    if lengths_t.ndim != 1 or lengths_t.shape[0] != labels.shape[0]:
        raise ValueError("lengths must be a rank-1 tensor/sequence with one value per batch item")
    if (lengths_t < 0).any():
        raise ValueError("lengths must be non-negative")
    lengths_t = lengths_t.clamp(max=labels.shape[1])
    # step_mask[b, t] == True iff timestep t is valid for batch item b.
    step_mask = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0) < lengths_t.unsqueeze(1)
    return labels, lengths_t, step_mask, squeeze

def _coerce_label_to_token_id(label_to_token_id: Sequence[int | None] | None, label_count: int) -> tuple[int | None, ...]:
    """Normalize optional label->token mapping and validate length."""
    if label_to_token_id is None:
        return tuple(range(label_count))
    values = tuple(None if value is None else int(value) for value in label_to_token_id)
    if len(values) != label_count:
        raise ValueError("label_to_token_id length must match label_count")
    return values

def _invert_label_to_token_id(label_to_token_id: Sequence[int | None]) -> dict[int, int]:
    """Build token->label lookup for decoding from tokenizer space."""
    return {int(token_id): label for label, token_id in enumerate(label_to_token_id) if token_id is not None}

def _labels_from_input_ids(input_ids: torch.Tensor | Sequence[int], token_id_to_label: Mapping[int, int], label_count: int) -> list[int]:
    """Map token ids to compact labels, validating label range."""
    ids = _flat_input_ids(input_ids)
    labels = []
    for token_id in ids:
        label = token_id_to_label.get(int(token_id), int(token_id))
        labels.append(_validate_label(label, label_count))
    return labels

def _flat_input_ids(input_ids: torch.Tensor | Sequence[int]) -> list[int]:
    """Normalize one prompt/generated token-id sequence to a Python list."""
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 0:
            return [int(input_ids.item())]
        if input_ids.dim() == 1:
            return [int(value) for value in input_ids.detach().long().tolist()]
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            return [int(value) for value in input_ids[0].detach().long().tolist()]
        raise ValueError("input_ids must describe a single sequence")
    return [int(value) for value in input_ids]

def _normalise_prompt_ids(input_ids: torch.Tensor | Sequence[int], *, device: torch.device) -> torch.Tensor:
    """Normalize prompt ids to ``[batch, prompt_len]`` long tensor."""
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

def _first_generated_index(ids: Sequence[int], token_id_to_label: Mapping[int, int]) -> int:
    """Return index where compact generated token ids begin."""
    for index, token_id in enumerate(ids):
        if int(token_id) in token_id_to_label:
            return index
    return len(ids)

def _validate_label(label: int, label_count: int) -> int:
    """Ensure label index is in ``[0, label_count)``."""
    label = int(label)
    if label < 0 or label >= label_count:
        raise ValueError(f"label {label} is out of range")
    return label
