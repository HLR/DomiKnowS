"""Shared helpers for automata-backed learner heads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ...latent_potentials import LatentTransitionPotential
from .prompt_encoders import FrozenBackbonePromptEncoder, PromptEmbeddingEncoder

TransitionPotentialInput = LatentTransitionPotential | torch.Tensor | Sequence[Sequence[float]] | None

__all__ = [
    '_positive_int',
    '_normalise_dynamics_conditioning',
    '_normalise_step_dynamics_conditioning',
    '_stack_base_and_optional_experts',
    '_infer_backbone_hidden_size',
    '_build_prompt_encoder',
    '_configure_prompt_encoder_trainability',
    '_normalise_prompt_ids',
    '_normalise_flat_ids',
    '_first_generated_index',
    '_empty_or_prompt',
    '_resolve_label_count',
    '_resolve_state_count',
    '_resolve_wfa_label_count',
    '_resolve_wfa_state_count',
    '_coerce_label_to_token_id',
    '_invert_label_to_token_id',
    '_labels_from_input_ids',
    '_target_labels',
    '_target_label_batch',
    '_validate_label',
    '_safe_log',
    '_random_hmm_parameters',
    '_random_wfa_parameters',
    '_random_hmm_dynamics_experts',
    '_random_wfa_dynamics_experts',
    '_validate_hmm_shapes',
    '_validate_wfa_shapes',
    'TransitionPotentialInput',
]

def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value

def _normalise_dynamics_conditioning(value: str) -> str:
    value = str(value).lower().replace("-", "_")
    if value not in {"none", "gated"}:
        raise ValueError("dynamics_conditioning must be 'none' or 'gated'")
    return value

def _normalise_step_dynamics_conditioning(value: str) -> str:
    value = str(value).lower().replace("-", "_")
    if value not in {"none", "prefix_gated"}:
        raise ValueError("step_dynamics_conditioning must be 'none' or 'prefix_gated'")
    return value

def _stack_base_and_optional_experts(base: torch.Tensor, experts: torch.Tensor | None) -> torch.Tensor:
    if experts is None or experts.numel() == 0:
        return base.unsqueeze(0)
    return torch.cat([base.unsqueeze(0), experts], dim=0)

def _infer_backbone_hidden_size(backbone: torch.nn.Module) -> int:
    """Infer the final feature dimension exposed by a frozen prompt backbone."""
    embedding = getattr(backbone, "embedding", None)
    if embedding is not None and hasattr(embedding, "embedding_dim"):
        return int(embedding.embedding_dim)

    config = getattr(backbone, "config", None)
    if config is not None:
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)

    for attr in ("hidden_size", "n_embd", "output_size"):
        value = getattr(backbone, attr, None)
        if value is not None:
            return int(value)

    raise ValueError("backbone hidden size could not be inferred; pass backbone_hidden_size")

def _build_prompt_encoder(
    *,
    prompt_encoder: torch.nn.Module | None,
    prompt_encoder_type: str,
    prompt_vocab_size: int,
    prompt_hidden_size: int,
    backbone: torch.nn.Module | None,
    backbone_hidden_size: int | None,
) -> torch.nn.Module:
    if prompt_encoder is not None:
        if not hasattr(prompt_encoder, "output_size"):
            raise ValueError("prompt_encoder must expose an output_size property")
        return prompt_encoder

    encoder_type = prompt_encoder_type.lower().replace("-", "_")
    if encoder_type == "embedding":
        return PromptEmbeddingEncoder(prompt_vocab_size, prompt_hidden_size)
    if encoder_type in {"frozen_backbone", "backbone"}:
        if backbone is None:
            raise ValueError("backbone is required when prompt_encoder_type='frozen_backbone'")
        return FrozenBackbonePromptEncoder(backbone, hidden_size=backbone_hidden_size)
    raise ValueError("prompt_encoder_type must be 'embedding' or 'frozen_backbone'")

def _configure_prompt_encoder_trainability(prompt_encoder: torch.nn.Module, trainable: bool) -> None:
    if isinstance(prompt_encoder, FrozenBackbonePromptEncoder):
        for parameter in prompt_encoder.backbone.parameters():
            parameter.requires_grad_(False)
        return
    for parameter in prompt_encoder.parameters():
        parameter.requires_grad_(trainable)

def _normalise_prompt_ids(
    input_ids: torch.Tensor | Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(input_ids, torch.Tensor):
        prompt = input_ids.detach().long()
        if prompt.dim() == 0:
            prompt = prompt.reshape(1, 1)
        elif prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        elif prompt.dim() != 2:
            raise ValueError("instruction_tokens must have shape [seq] or [batch, seq]")
        if prompt.numel() == 0:
            prompt = torch.zeros((1, 1), dtype=torch.long, device=prompt.device)
        return prompt.to(device)

    ids = [int(token_id) for token_id in input_ids]
    if not ids:
        ids = [0]
    return torch.tensor([ids], dtype=torch.long, device=device)

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

def _first_generated_index(ids: Sequence[int], token_id_to_label: Mapping[int, int]) -> int:
    for index, token_id in enumerate(ids):
        if int(token_id) in token_id_to_label:
            return index
    return len(ids)

def _empty_or_prompt(
    instruction_tokens: torch.Tensor | Sequence[int] | None,
    device: torch.device,
) -> torch.Tensor:
    if instruction_tokens is None:
        return torch.zeros((1, 1), dtype=torch.long, device=device)
    return _normalise_prompt_ids(instruction_tokens, device=device)

def _resolve_label_count(model: object | None, label_count: int | None) -> int:
    if model is not None:
        inferred = len(model.symbols)
        if label_count is not None and int(label_count) != inferred:
            raise ValueError("label_count does not match HMM symbol count")
        return inferred
    if label_count is None:
        raise ValueError("label_count is required when no HMM model is supplied")
    return _positive_int(label_count, "label_count")

def _resolve_state_count(model: object | None, state_count: int | None) -> int:
    if model is not None:
        inferred = model.state_count
        if state_count is not None and int(state_count) != inferred:
            raise ValueError("state_count does not match HMM state count")
        return inferred
    if state_count is None:
        raise ValueError("state_count is required when no HMM model is supplied")
    return _positive_int(state_count, "state_count")

def _resolve_wfa_label_count(model: WeightedFiniteAutomaton | None, label_count: int | None) -> int:
    if model is not None:
        inferred = len(model.symbols)
        if label_count is not None and int(label_count) != inferred:
            raise ValueError("label_count does not match WFA symbol count")
        return inferred
    if label_count is None:
        raise ValueError("label_count is required when no WFA model is supplied")
    return _positive_int(label_count, "label_count")

def _resolve_wfa_state_count(model: WeightedFiniteAutomaton | None, state_count: int | None) -> int:
    if model is not None:
        inferred = model.state_count
        if state_count is not None and int(state_count) != inferred:
            raise ValueError("state_count does not match WFA state count")
        return inferred
    if state_count is None:
        raise ValueError("state_count is required when no WFA model is supplied")
    return _positive_int(state_count, "state_count")

def _coerce_label_to_token_id(
    label_to_token_id: Sequence[int | None] | None,
    label_count: int,
) -> tuple[int | None, ...]:
    if label_to_token_id is None:
        return tuple(range(label_count))
    if len(label_to_token_id) != label_count:
        raise ValueError("label_to_token_id must contain one entry per compact label")
    return tuple(None if token_id is None else int(token_id) for token_id in label_to_token_id)

def _invert_label_to_token_id(label_to_token_id: Sequence[int | None]) -> Mapping[int, int]:
    return {int(token_id): label for label, token_id in enumerate(label_to_token_id) if token_id is not None}

def _labels_from_input_ids(
    input_ids: torch.Tensor | Sequence[int],
    token_id_to_label: Mapping[int, int],
    label_count: int,
) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        flat = input_ids.detach().reshape(-1).tolist()
    else:
        flat = list(input_ids)
    labels = []
    for token_id in flat:
        label = token_id_to_label.get(int(token_id))
        if label is not None and 0 <= label < label_count:
            labels.append(int(label))
    return labels

def _target_labels(
    target_labels: torch.Tensor | Sequence[int],
    pad_size: int,
    *,
    device: torch.device,
    eos_label: int = 0,
) -> torch.Tensor:
    if isinstance(target_labels, torch.Tensor):
        labels = target_labels.detach().long().reshape(-1).to(device)
    else:
        labels = torch.tensor(list(target_labels), dtype=torch.long, device=device)
    if labels.numel() >= pad_size:
        return labels[:pad_size]
    padding = torch.full((pad_size - labels.numel(),), int(eos_label), dtype=torch.long, device=device)
    return torch.cat([labels, padding], dim=0)

def _target_label_batch(
    target_labels: torch.Tensor | Sequence[int],
    pad_size: int,
    *,
    device: torch.device,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int = 0,
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
    if labels.shape[1] >= pad_size:
        labels = labels[:, :pad_size]
    else:
        padding = torch.full(
            (labels.shape[0], pad_size - labels.shape[1]),
            int(eos_label),
            dtype=torch.long,
            device=device,
        )
        labels = torch.cat([labels, padding], dim=1)
    if torch.any(labels < 0):
        raise ValueError("target_labels must be non-negative")
    if lengths is None:
        lengths_t = torch.full((labels.shape[0],), labels.shape[1], dtype=torch.long, device=device)
    else:
        lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device).reshape(-1)
    if lengths_t.numel() != labels.shape[0]:
        raise ValueError("lengths must contain one value per batch item")
    lengths_t = torch.clamp(lengths_t, min=1, max=labels.shape[1])
    return labels, lengths_t, squeeze

def _validate_label(label: int, label_count: int) -> int:
    label = int(label)
    if label < 0 or label >= label_count:
        raise ValueError(f"label {label} is out of range for {label_count} labels")
    return label

def _safe_log(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values.float().clamp_min(torch.finfo(torch.float32).eps))

def _random_hmm_parameters(state_count: int, label_count: int, random_seed: int):
    generator = torch.Generator().manual_seed(int(random_seed))
    initial = torch.rand(state_count, generator=generator) + 0.1
    transition = torch.rand(state_count, state_count, generator=generator) + 0.1
    emission = torch.rand(state_count, label_count, generator=generator) + 0.1
    initial = initial / initial.sum()
    transition = transition / transition.sum(dim=-1, keepdim=True)
    emission = emission / emission.sum(dim=-1, keepdim=True)
    return initial, transition, emission

def _random_wfa_parameters(state_count: int, label_count: int, random_seed: int):
    generator = torch.Generator().manual_seed(int(random_seed))
    initial = torch.randn(state_count, generator=generator) * 0.1
    transitions = torch.randn(label_count, state_count, state_count, generator=generator) * 0.1
    final = torch.randn(state_count, generator=generator) * 0.1
    return initial, transitions, final

def _random_hmm_dynamics_experts(
    expert_count: int,
    state_count: int,
    label_count: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_count < 1:
        return (
            torch.empty((0, state_count, state_count), dtype=torch.float32),
            torch.empty((0, state_count, label_count), dtype=torch.float32),
        )
    transition_experts = []
    emission_experts = []
    for offset in range(expert_count):
        _initial, transition, emission = _random_hmm_parameters(
            state_count,
            label_count,
            random_seed + offset,
        )
        transition_experts.append(_safe_log(transition))
        emission_experts.append(_safe_log(emission))
    return torch.stack(transition_experts, dim=0), torch.stack(emission_experts, dim=0)

def _random_wfa_dynamics_experts(
    expert_count: int,
    state_count: int,
    label_count: int,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_count < 1:
        return (
            torch.empty((0, label_count, state_count, state_count), dtype=torch.float32),
            torch.empty((0, state_count), dtype=torch.float32),
        )
    transition_experts = []
    final_experts = []
    for offset in range(expert_count):
        _initial, transitions, final = _random_wfa_parameters(
            state_count,
            label_count,
            random_seed + offset,
        )
        transition_experts.append(transitions)
        final_experts.append(final)
    return torch.stack(transition_experts, dim=0), torch.stack(final_experts, dim=0)

def _validate_hmm_shapes(initial, transition, emission, state_count: int, label_count: int) -> None:
    if initial.shape != (state_count,):
        raise ValueError("HMM initial vector has the wrong shape")
    if transition.shape != (state_count, state_count):
        raise ValueError("HMM transition matrix has the wrong shape")
    if emission.shape != (state_count, label_count):
        raise ValueError("HMM emission matrix has the wrong shape")

def _validate_wfa_shapes(initial, transitions, final, state_count: int, label_count: int) -> None:
    if initial.shape != (state_count,):
        raise ValueError("WFA initial vector has the wrong shape")
    if transitions.shape != (label_count, state_count, state_count):
        raise ValueError("WFA transition tensor has the wrong shape")
    if final.shape != (state_count,):
        raise ValueError("WFA final vector has the wrong shape")
