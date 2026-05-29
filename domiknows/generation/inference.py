"""Unconstrained compact-label inference helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .dfa.vocabulary import TokenVocabulary


@dataclass(frozen=True)
class LabelInferenceResult:
    """Result returned by compact-label inference helpers."""

    labels: tuple[int, ...]
    symbols: tuple[str, ...]
    score: float
    token_ids: tuple[int, ...] = ()
    scores: tuple[float, ...] = ()
    candidates: tuple["LabelInferenceResult", ...] = ()
    finished: bool = False

    def normalized_score(self, length_penalty: float = 1.0) -> float:
        # Normalize accumulated log-score so shorter/longer hypotheses are comparable.
        """Return length-normalized score for ranking beam candidates."""
        if length_penalty <= 0.0:
            raise ValueError("length_penalty must be positive")
        length = max(1, len(self.labels))
        return self.score / (length**length_penalty)


def greedy_label_inference(
    model,
    vocabulary: TokenVocabulary,
    input_ids: torch.Tensor | Sequence[int],
    *,
    max_new_tokens: int,
    eos_label: int | None = None,
    next_label_kwargs: dict | None = None,
    allow_empty_input: bool = False,
) -> LabelInferenceResult:
    # Decode by repeatedly taking the highest-scoring next compact label.
    """Run unconstrained greedy inference for a compact-label model."""
    _validate_common(max_new_tokens)
    # ids is the mutable token-id prefix for decoding steps; device is where model tensors run.
    ids, device = _normalise_input_ids(input_ids, allow_empty_input=allow_empty_input)
    eos_label = int(vocabulary.eos_label if eos_label is None else eos_label)
    emittable = _emittable_labels(model, vocabulary)
    labels: list[int] = []
    scores: list[float] = []
    total_score = 0.0

    for _step in range(max_new_tokens):
        # Score one-step continuation conditioned on current token prefix.
        logits = _next_label_logits(model, ids, device, next_label_kwargs=next_label_kwargs)
        masked = _mask_to_emittable(logits, emittable)
        log_probs = torch.log_softmax(masked, dim=-1)
        next_label = int(torch.argmax(masked).item())
        score_delta = _log_prob_at(log_probs, next_label)

        labels.append(next_label)
        scores.append(score_delta)
        total_score += score_delta
        ids.append(_token_id_for_label(model, vocabulary, next_label))
        if next_label == eos_label:
            break

    # Convert raw decoded labels to rich result payload (symbols, ids, metadata).
    return _result_from_labels(
        model,
        vocabulary,
        ids,
        labels,
        score=total_score,
        scores=scores,
        eos_label=eos_label,
    )


def beam_label_inference(
    model,
    vocabulary: TokenVocabulary,
    input_ids: torch.Tensor | Sequence[int],
    *,
    max_new_tokens: int,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    num_return_sequences: int = 1,
    eos_label: int | None = None,
    next_label_kwargs: dict | None = None,
    allow_empty_input: bool = False,
) -> LabelInferenceResult:
    # Keep top-k partial hypotheses, expanding each with high-probability next labels.
    """Run unconstrained beam search for a compact-label model."""
    _validate_common(max_new_tokens)
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    if length_penalty <= 0.0:
        raise ValueError("length_penalty must be positive")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be at least 1")

    ids, device = _normalise_input_ids(input_ids, allow_empty_input=allow_empty_input)
    eos_label = int(vocabulary.eos_label if eos_label is None else eos_label)
    emittable = _emittable_labels(model, vocabulary)
    beams = (
        _result_from_labels(
            model,
            vocabulary,
            ids,
            [],
            score=0.0,
            scores=[],
            eos_label=eos_label,
        ),
    )
    finished: list[LabelInferenceResult] = []

    for _step in range(max_new_tokens):
        expanded: list[LabelInferenceResult] = []

        for candidate in beams:
            # Completed beams are carried over unchanged.
            if candidate.finished:
                expanded.append(candidate)
                continue

            candidate_ids = list(candidate.token_ids)
            logits = _next_label_logits(model, candidate_ids, device, next_label_kwargs=next_label_kwargs)
            masked = _mask_to_emittable(logits, emittable)
            log_probs = torch.log_softmax(masked, dim=-1)
            valid_labels = torch.nonzero(masked > _MASK_FILL_VALUE / 2, as_tuple=False).flatten()
            local_beam = min(beam_size, int(valid_labels.numel()))
            local_scores, local_positions = torch.topk(log_probs[valid_labels], local_beam)

            # Spawn local beam continuations from this candidate.
            for score_delta, position in zip(local_scores.tolist(), local_positions.tolist()):
                next_label = int(valid_labels[int(position)].item())
                next_id = _token_id_for_label(model, vocabulary, next_label)
                next_candidate = _result_from_labels(
                    model,
                    vocabulary,
                    candidate_ids + [next_id],
                    [*candidate.labels, next_label],
                    score=float(candidate.score + score_delta),
                    scores=[*candidate.scores, float(score_delta)],
                    eos_label=eos_label,
                )
                expanded.append(next_candidate)
                if next_candidate.finished:
                    finished.append(next_candidate)

        if not expanded:
            break

        # Re-rank globally after expansion using length-normalized scores.
        ranked = sorted(
            expanded,
            key=lambda item: item.normalized_score(length_penalty),
            reverse=True,
        )
        beams = tuple(ranked[:beam_size])
        if early_stopping and len(finished) >= num_return_sequences:
            break

    ranked_candidates = sorted(
        finished + list(beams),
        key=lambda item: item.normalized_score(length_penalty),
        reverse=True,
    )
    if not ranked_candidates:
        ranked_candidates = [
            _result_from_labels(
                model,
                vocabulary,
                ids,
                [],
                score=0.0,
                scores=[],
                eos_label=eos_label,
            )
        ]
    # Return best candidate plus optional n-best list.
    returned = tuple(ranked_candidates[:num_return_sequences])
    best = returned[0]
    return LabelInferenceResult(
        labels=best.labels,
        symbols=best.symbols,
        score=best.score,
        token_ids=best.token_ids,
        scores=best.scores,
        candidates=returned,
        finished=best.finished,
    )


def sample_label_inference(
    model,
    vocabulary: TokenVocabulary,
    input_ids: torch.Tensor | Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    eos_label: int | None = None,
    next_label_kwargs: dict | None = None,
    allow_empty_input: bool = False,
) -> LabelInferenceResult:
    # Decode stochastically from filtered next-label distributions.
    """Run unconstrained stochastic sampling for a compact-label model."""
    _validate_common(max_new_tokens)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    ids, device = _normalise_input_ids(input_ids, allow_empty_input=allow_empty_input)
    eos_label = int(vocabulary.eos_label if eos_label is None else eos_label)
    emittable = _emittable_labels(model, vocabulary)
    labels: list[int] = []
    scores: list[float] = []
    total_score = 0.0

    for _step in range(max_new_tokens):
        # Apply temperature and optional top-k/top-p constraints before sampling.
        logits = _next_label_logits(model, ids, device, next_label_kwargs=next_label_kwargs)
        masked = _mask_to_emittable(logits, emittable)
        constrained_logits = masked / float(temperature)
        filtered = _filter_sampling_logits(constrained_logits, top_k=top_k, top_p=top_p)
        if torch.all(filtered <= _MASK_FILL_VALUE / 2):
            # Fallback to unfiltered constrained logits if filtering pruned everything.
            filtered = constrained_logits

        probs = torch.softmax(filtered, dim=-1)
        next_label = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        score_delta = float(torch.log(probs[next_label].clamp_min(torch.finfo(probs.dtype).tiny)).item())

        labels.append(next_label)
        scores.append(score_delta)
        total_score += score_delta
        ids.append(_token_id_for_label(model, vocabulary, next_label))
        if next_label == eos_label:
            break

    # Package sampled trajectory and bookkeeping fields.
    return _result_from_labels(
        model,
        vocabulary,
        ids,
        labels,
        score=total_score,
        scores=scores,
        eos_label=eos_label,
    )


_MASK_FILL_VALUE = -1e9


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


def _next_label_logits(
    model,
    ids: Sequence[int],
    device: torch.device,
    *,
    next_label_kwargs: dict | None = None,
) -> torch.Tensor:
    # Query model one-step compact-label logits for the current prefix.
    if not hasattr(model, "next_label_logits"):
        raise ValueError("compact-label inference requires model.next_label_logits(input_ids)")
    model_input = torch.tensor([list(map(int, ids))], dtype=torch.long, device=device)
    logits = model.next_label_logits(model_input, **(next_label_kwargs or {})).reshape(-1)
    if logits.dim() != 1:
        raise ValueError(f"expected compact label logits to be 1D, got shape {tuple(logits.shape)}")
    return logits


def _token_id_for_label(model, vocabulary: TokenVocabulary, label: int) -> int:
    # Resolve label->token mapping, preferring model-specific overrides.
    if hasattr(model, "token_id_for_label"):
        return int(model.token_id_for_label(int(label)))
    return int(vocabulary.token_id_for_label(int(label)))


def _emittable_labels(model, vocabulary: TokenVocabulary) -> tuple[int, ...]:
    # Enumerate labels that can be converted to actual token ids.
    labels: list[int] = []
    for label in range(int(vocabulary.label_count)):
        try:
            _token_id_for_label(model, vocabulary, label)
        except ValueError:
            continue
        labels.append(label)
    if not labels:
        raise ValueError("no compact labels can be emitted by the model")
    return tuple(labels)


def _mask_to_emittable(logits: torch.Tensor, emittable_labels: Sequence[int]) -> torch.Tensor:
    # Restrict logits to labels that are valid/emittable for this model-vocabulary pair.
    if logits.dim() != 1:
        raise ValueError(f"expected 1D label logits for one inference step, got shape {tuple(logits.shape)}")
    masked = torch.full_like(logits, _MASK_FILL_VALUE)
    for label in emittable_labels:
        label = int(label)
        if 0 <= label < masked.numel():
            masked[label] = logits[label]
    if torch.all(masked <= _MASK_FILL_VALUE / 2):
        raise ValueError("emittable-label masking removed every label from the logits")
    return masked


def _filter_sampling_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    # Apply top-k and/or nucleus (top-p) filtering in logit space.
    filtered = logits.clone()

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1 when provided")
        if top_k < filtered.numel():
            # Keep only the largest top_k logits.
            values, _indices = torch.topk(filtered, top_k)
            threshold = values[-1]
            filtered = filtered.masked_fill(filtered < threshold, _MASK_FILL_VALUE)

    if top_p is not None:
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if top_p < 1.0:
            # Keep the smallest prefix whose cumulative probability mass exceeds top_p.
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered[sorted_indices[remove]] = _MASK_FILL_VALUE

    return filtered


def _log_prob_at(log_probs: torch.Tensor, label: int) -> float:
    # Convert one selected label's log-prob to a plain Python float.
    return float(log_probs[int(label)].detach().cpu().item())


def _result_from_labels(
    model,
    vocabulary: TokenVocabulary,
    token_ids: Sequence[int],
    labels: Sequence[int],
    *,
    score: float,
    scores: Sequence[float],
    eos_label: int,
) -> LabelInferenceResult:
    # Build the standardized inference result object used by all decoding paths.
    del model
    label_tuple = tuple(int(label) for label in labels)
    return LabelInferenceResult(
        labels=label_tuple,
        symbols=tuple(vocabulary.token_for_label(label) for label in label_tuple),
        score=float(score),
        token_ids=tuple(int(token_id) for token_id in token_ids),
        scores=tuple(float(item) for item in scores),
        finished=bool(label_tuple and label_tuple[-1] == int(eos_label)),
    )
