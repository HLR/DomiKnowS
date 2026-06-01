"""Compact-label inference helpers that return constrained decoder results."""
from __future__ import annotations

from typing import Sequence

import torch

from ..dfa.core import DFA
from ..dfa._constraints import accept_all_dfa
from ..dfa.decoder import (
    ConstrainedGenerationResult,
    constrained_label_beam_search_decode,
    constrained_label_greedy_decode,
    constrained_label_sample_decode,
)
from ..dfa.vocabulary import TokenVocabulary


def greedy_label_inference(
    model,
    vocabulary: TokenVocabulary,
    prompt_ids: torch.Tensor | Sequence[int],
    *,
    dfa: DFA | None = None,
    max_new_tokens: int,
    eos_label: int | None = None,
    next_label_kwargs: dict | None = None,
    allow_empty_input: bool = False,
) -> ConstrainedGenerationResult:
    # Decode by repeatedly taking the highest-scoring next compact label.
    """Run compact-label greedy inference and return a constrained decode result."""
    _validate_common(max_new_tokens)
    _normalise_input_ids(prompt_ids, allow_empty_input=allow_empty_input)
    active_dfa = dfa if dfa is not None else accept_all_dfa(vocabulary)
    return constrained_label_greedy_decode(
        model,
        prompt_ids,
        vocabulary,
        active_dfa,
        max_new_tokens=max_new_tokens,
        eos_label=eos_label,
        next_label_kwargs=next_label_kwargs,
    )


def beam_label_inference(
    model,
    vocabulary: TokenVocabulary,
    prompt_ids: torch.Tensor | Sequence[int],
    *,
    dfa: DFA | None = None,
    max_new_tokens: int,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    num_return_sequences: int = 1,
    eos_label: int | None = None,
    next_label_kwargs: dict | None = None,
    allow_empty_input: bool = False,
) -> ConstrainedGenerationResult:
    # Keep top-k partial hypotheses, expanding each with high-probability next labels.
    """Run compact-label beam search and return a constrained decode result."""
    _validate_common(max_new_tokens)
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    if length_penalty <= 0.0:
        raise ValueError("length_penalty must be positive")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be at least 1")
    _normalise_input_ids(prompt_ids, allow_empty_input=allow_empty_input)
    active_dfa = dfa if dfa is not None else accept_all_dfa(vocabulary)
    return constrained_label_beam_search_decode(
        model,
        prompt_ids,
        vocabulary,
        active_dfa,
        max_new_tokens=max_new_tokens,
        eos_label=eos_label,
        beam_size=beam_size,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        num_return_sequences=num_return_sequences,
        next_label_kwargs=next_label_kwargs,
    )


def sample_label_inference(
    model,
    vocabulary: TokenVocabulary,
    prompt_ids: torch.Tensor | Sequence[int],
    *,
    dfa: DFA | None = None,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    eos_label: int | None = None,
    next_label_kwargs: dict | None = None,
    allow_empty_input: bool = False,
) -> ConstrainedGenerationResult:
    # Decode stochastically from filtered next-label distributions.
    """Run compact-label sampling and return a constrained decode result."""
    _validate_common(max_new_tokens)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    _normalise_input_ids(prompt_ids, allow_empty_input=allow_empty_input)
    active_dfa = dfa if dfa is not None else accept_all_dfa(vocabulary)
    return constrained_label_sample_decode(
        model,
        prompt_ids,
        vocabulary,
        active_dfa,
        max_new_tokens=max_new_tokens,
        eos_label=eos_label,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
        next_label_kwargs=next_label_kwargs,
    )


def _validate_common(max_new_tokens: int) -> None:
    # Shared guard for generation length arguments.
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")


def _normalise_input_ids(
    input_ids: torch.Tensor | Sequence[int],
    *,
    allow_empty_input: bool,
) -> tuple[list[int], torch.device]:
    # Accept 1D/2D tensors or plain sequences and return a flat list plus device.
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2:
            ids = input_ids[0].tolist()
        elif input_ids.dim() == 1:
            ids = input_ids.tolist()
        else:
            raise ValueError(f"expected input_ids to be 1D or 2D, got shape {tuple(input_ids.shape)}")
        ids = [int(token_id) for token_id in ids]
        if not ids and not allow_empty_input:
            raise ValueError("input_ids is empty; pass allow_empty_input=True to decode without a prompt")
        return ids, input_ids.device
    ids = [int(token_id) for token_id in input_ids]
    if not ids and not allow_empty_input:
        raise ValueError("input_ids is empty; pass allow_empty_input=True to decode without a prompt")
    return ids, torch.device("cpu")
