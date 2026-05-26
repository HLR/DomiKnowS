"""DFA-constrained decoders for autoregressive language models.

This module implements token-by-token constrained decoding: at every
autoregressive step the model's full-vocabulary logit vector is masked so that
only tokens *permitted* by the current DFA state have non-fill scores.

Public API
----------
:class:`ConstrainedGenerationResult`
    Dataclass returned by constrained decoding functions.

:func:`mask_logits_for_dfa`
    Low-level utility that applies a set of allowed DFA label IDs to a
    1-D logit tensor.  Can be used independently for custom decoding loops.

:func:`mask_label_logits_for_dfa`
    Low-level utility for compact-label models that already emit logits over
    the DomiKnowS generation label space.

:func:`constrained_greedy_decode`
    High-level entry point: runs the full greedy decoding loop with
    DFA-guided masking and returns a :class:`ConstrainedGenerationResult`.

:func:`constrained_beam_search_decode`
    Beam search with one DFA state per beam.

:func:`constrained_sample_decode`
    Temperature / top-k / top-p sampling after DFA masking.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch

from .learners.dfa.core import DFA
from .vocabulary import TokenVocabulary


@dataclass
class BeamCandidate:
    """Internal beam-search candidate state.

    Attributes:
        token_ids: Full token ID sequence produced so far (prompt + generated).
        labels: DomiKnowS generation label IDs for the generated portion only.
        state: Current DFA state after consuming all labels in *labels*.
        score: Cumulative sum of per-step log-probabilities for this candidate.
        finished: ``True`` once the candidate has emitted EOS in an accepting
            DFA state and will not be expanded further.
        cache_state: Optional autoregressive KV-cache carried alongside this
            candidate so each beam can be decoded independently.
    """

    token_ids: list[int]
    labels: list[int]
    state: object
    score: float
    finished: bool = False
    cache_state: "CachedDecoderState | None" = None

    def normalized_score(self, length_penalty: float = 1.0) -> float:
        """Return length-normalized score used for final ranking."""
        length = max(1, len(self.labels))
        return self.score / (length ** length_penalty)


@dataclass
class CachedDecoderState:
    """Internal autoregressive state for cache-aware decoding.

    Wraps the token sequence together with HuggingFace-style
    ``past_key_values`` so the model does not have to re-process prompt tokens
    on every step.  Falls back to full-sequence re-encoding when the model
    does not support caching (``cache_supported = False``).

    Attributes:
        token_ids: Complete token ID history for this generation branch.
        pending_input_ids: Tokens that have not yet been fed to the model;
            typically just the last appended token when caching is active.
        past_key_values: HuggingFace ``past_key_values`` tensor structure
            returned by the model on the previous forward pass, or ``None``
            when the model does not support / has not yet returned a cache.
        device: PyTorch device on which model tensors are placed.
        cache_supported: Set to ``False`` when a cached forward pass fails,
            causing subsequent steps to fall back to full re-encoding.
    """

    token_ids: list[int]
    pending_input_ids: list[int]
    past_key_values: object | None
    device: torch.device
    cache_supported: bool = True


@dataclass
class ConstrainedGenerationResult:
    """Result of a DFA-constrained decoding pass.

    Attributes:
        token_ids: Full list of token IDs produced by the decoder, including
            the prompt tokens followed by all generated tokens.
        labels: List of DomiKnowS vocabulary label IDs for the *generated*
            tokens only (does not include prompt tokens).
        final_state: The DFA state reached after the last generated token.
            Useful for resuming or inspecting the constraint automaton.
        accepted: Whether *final_state* is an accepting state of the DFA,
            i.e. whether the generated sequence satisfies the constraint.
        score: Optional cumulative log-probability score for beam/sampling
            modes.  Greedy decoding leaves it as ``None``.
        scores: Optional per-token or per-candidate scores depending on the
            decoding mode.
        candidates: Optional beam candidates retained for inspection.
    """

    token_ids: list[int]
    labels: list[int]
    final_state: object
    accepted: bool
    score: float | None = None
    scores: list[float] | None = None
    candidates: list[BeamCandidate] | None = None


def mask_logits_for_dfa(
    logits: torch.Tensor,
    allowed_labels: set[int],
    vocabulary: TokenVocabulary,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Mask a logit vector so only DFA-allowed tokens have non-fill scores.

    The masking strategy depends on whether the ``other_label`` (the catch-all
    bucket for unknown tokens) is allowed:

    * **``other_label`` allowed** — start from a *clone* of the original
      logits (everything allowed) and mask out the known tokens that are
      *not* in ``allowed_labels``.  This correctly preserves the logit
      values for all unknown tokens mapped to ``other_label``.
    * **``other_label`` not allowed** — start from a tensor filled with
      ``fill_value`` and copy in only the known tokens whose label is in
      ``allowed_labels``.

    Args:
        logits: 1-D float tensor of shape ``(vocab_size,)`` for a single
            decoding step.
        allowed_labels: Set of DomiKnowS label IDs that the DFA currently
            permits.
        vocabulary: :class:`~.vocabulary.TokenVocabulary` providing the
            label ↔ token-ID mapping and the ``other_label`` value.
        fill_value: Score assigned to masked-out positions.  Should be
            sufficiently negative to be ignored by ``argmax`` / softmax.
            Defaults to ``-1e9``.

    Returns:
        A 1-D tensor of the same shape and device as *logits* where
        disallowed token positions are set to ``fill_value``.

    Raises:
        ValueError: If *logits* is not 1-D.
        ValueError: If ``vocabulary.tokenizer`` is ``None`` (required to
            enumerate known token IDs).
        ValueError: If the masking leaves every position at or below
            ``fill_value / 2`` (i.e. no token is permitted at this step).
    """

    if logits.dim() != 1:
        raise ValueError(f"expected 1D logits for one decoding step, got shape {tuple(logits.shape)}")
    if vocabulary.tokenizer is None:
        raise ValueError("TokenVocabulary.tokenizer is required for full-logit masking")

    masked = torch.full_like(logits, fill_value)
    known_ids = vocabulary.known_token_ids

    if vocabulary.other_label in allowed_labels:
        # "other" is allowed: clone all logits and selectively block known
        # tokens whose label is explicitly excluded.
        masked = logits.clone()
        blocked_known = [
            token_id
            for label, token_id in enumerate(known_ids)
            if label not in allowed_labels and 0 <= token_id < masked.numel()
        ]
        if blocked_known:
            masked[torch.tensor(blocked_known, device=masked.device)] = fill_value
    else:
        # "other" is not allowed: start with all-fill and enable only
        # explicitly allowed known tokens.
        for label in allowed_labels:
            token_id = vocabulary.token_id_for_label(label)
            if 0 <= token_id < masked.numel():
                masked[token_id] = logits[token_id]

    # Safety check: at least one token must survive masking.
    if torch.all(masked <= fill_value / 2):
        raise ValueError("DFA masking removed every token from the logits")
    return masked


def mask_label_logits_for_dfa(
    logits: torch.Tensor,
    allowed_labels: set[int],
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Mask compact-label logits so only DFA-allowed labels remain.

    Args:
        logits: 1-D float tensor of shape ``(label_count,)``.
        allowed_labels: Set of compact DomiKnowS generation labels currently
            permitted by the DFA.
        fill_value: Score assigned to masked-out labels.

    Returns:
        A clone of *logits* with disallowed label positions filled.

    Raises:
        ValueError: If *logits* is not 1-D or no valid label survives.
    """
    if logits.dim() != 1:
        raise ValueError(f"expected 1D label logits for one decoding step, got shape {tuple(logits.shape)}")

    masked = torch.full_like(logits, fill_value)
    for label in allowed_labels:
        label = int(label)
        if 0 <= label < masked.numel():
            masked[label] = logits[label]

    if torch.all(masked <= fill_value / 2):
        raise ValueError("DFA masking removed every label from the logits")
    return masked


def _normalise_input_ids(input_ids: torch.Tensor | Sequence[int]) -> tuple[list[int], torch.device]:
    """Return flat token ids plus the device to use for model inputs."""
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2:
            ids = input_ids[0].tolist()
        elif input_ids.dim() == 1:
            ids = input_ids.tolist()
        else:
            raise ValueError(f"expected input_ids to be 1D or 2D, got shape {tuple(input_ids.shape)}")
        return [int(token_id) for token_id in ids], input_ids.device
    return [int(token_id) for token_id in input_ids], torch.device("cpu")


def _next_logits(model, ids: list[int], device: torch.device) -> torch.Tensor:
    """Run *model* on a single sequence and return logits for the next token."""
    model_input = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return model(model_input).logits[0, -1, :]


def _cached_state_from_ids(ids: list[int], device: torch.device) -> CachedDecoderState:
    """Create a fresh cache state whose first pending segment is the prompt."""
    return CachedDecoderState(
        token_ids=list(ids),
        pending_input_ids=list(ids),
        past_key_values=None,
        device=device,
        cache_supported=True,
    )


def _cache_output_past(output) -> object | None:
    """Extract HuggingFace-style ``past_key_values`` from a model output."""
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    if isinstance(output, dict):
        return output.get("past_key_values")
    return None


def _cache_output_logits(output) -> torch.Tensor:
    """Extract logits from a HuggingFace-style model output."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    raise ValueError("model output must expose logits")


def _next_logits_cached(
    model,
    cache_state: CachedDecoderState,
    *,
    use_cache: bool = True,
) -> tuple[torch.Tensor, CachedDecoderState]:
    """Return next logits, using ``past_key_values`` when the model supports it."""
    if not use_cache or not cache_state.cache_supported:
        return _next_logits(model, cache_state.token_ids, cache_state.device), CachedDecoderState(
            token_ids=list(cache_state.token_ids),
            pending_input_ids=[],
            past_key_values=None,
            device=cache_state.device,
            cache_supported=False,
        )

    pending = cache_state.pending_input_ids or cache_state.token_ids
    model_input = torch.tensor(pending, dtype=torch.long, device=cache_state.device).unsqueeze(0)
    kwargs = {"use_cache": True}
    if cache_state.past_key_values is not None:
        kwargs["past_key_values"] = cache_state.past_key_values

    try:
        output = model(model_input, **kwargs)
    except TypeError:
        return _next_logits(model, cache_state.token_ids, cache_state.device), CachedDecoderState(
            token_ids=list(cache_state.token_ids),
            pending_input_ids=[],
            past_key_values=None,
            device=cache_state.device,
            cache_supported=False,
        )

    past = _cache_output_past(output)
    logits = _cache_output_logits(output)[0, -1, :]
    if past is None:
        return logits, CachedDecoderState(
            token_ids=list(cache_state.token_ids),
            pending_input_ids=[],
            past_key_values=None,
            device=cache_state.device,
            cache_supported=False,
        )

    return logits, CachedDecoderState(
        token_ids=list(cache_state.token_ids),
        pending_input_ids=[],
        past_key_values=past,
        device=cache_state.device,
        cache_supported=True,
    )


def _append_cached_token(cache_state: CachedDecoderState, token_id: int) -> CachedDecoderState:
    """Append a selected token and mark it pending for the next cached call."""
    return CachedDecoderState(
        token_ids=cache_state.token_ids + [int(token_id)],
        pending_input_ids=[int(token_id)] if cache_state.cache_supported else [],
        past_key_values=cache_state.past_key_values,
        device=cache_state.device,
        cache_supported=cache_state.cache_supported,
    )


def _clone_cache_value(value):
    """Clone a cache object enough for independent beam branches."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_cache_value(item) for key, item in value.items()}
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "to_legacy_cache"):
        legacy = _clone_cache_value(value.to_legacy_cache())
        from_legacy = getattr(type(value), "from_legacy_cache", None)
        if from_legacy is not None:
            return from_legacy(legacy)
        return legacy
    if hasattr(value, "__dict__"):
        try:
            cloned = type(value).__new__(type(value))
            for key, item in vars(value).items():
                setattr(cloned, key, _clone_cache_value(item))
            return cloned
        except Exception:
            pass
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise ValueError("could not clone past_key_values for beam cache branching") from exc


def _clone_cached_state_for_branch(cache_state: CachedDecoderState, token_id: int) -> CachedDecoderState:
    """Create an independent child cache state after beam expansion."""
    if not cache_state.cache_supported:
        return _append_cached_token(cache_state, token_id)
    return CachedDecoderState(
        token_ids=cache_state.token_ids + [int(token_id)],
        pending_input_ids=[int(token_id)],
        past_key_values=_clone_cache_value(cache_state.past_key_values),
        device=cache_state.device,
        cache_supported=True,
    )


def _next_label_logits(
    model,
    ids: list[int],
    device: torch.device,
    *,
    model_kwargs: dict | None = None,
    next_label_kwargs: dict | None = None,
) -> torch.Tensor:
    """Run a compact-label model and return logits for the next label."""
    model_input = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    if hasattr(model, "next_label_logits"):
        logits = model.next_label_logits(model_input, **(next_label_kwargs or {}))
    else:
        output = model(model_input, **(model_kwargs or {}))
        logits = output.logits[0, -1, :] if hasattr(output, "logits") else output
    if logits.dim() != 1:
        raise ValueError(f"expected compact label logits to be 1D, got shape {tuple(logits.shape)}")
    return logits


def _token_id_for_generated_label(model, vocabulary: TokenVocabulary, label: int) -> int:
    """Return the raw tokenizer id used after a compact label is selected."""
    if hasattr(model, "token_id_for_label"):
        return int(model.token_id_for_label(label))
    return vocabulary.token_id_for_label(label)


def _emittable_labels(model, vocabulary: TokenVocabulary) -> set[int]:
    """Return compact labels that can be converted into concrete token ids."""
    labels = set()
    for label in range(vocabulary.label_count):
        try:
            _token_id_for_generated_label(model, vocabulary, label)
        except ValueError:
            continue
        labels.add(label)
    return labels


def _resolve_eos_token_id(vocabulary: TokenVocabulary, eos_token_id: int | None) -> int:
    """Resolve an optional EOS token id against the generation vocabulary."""
    return eos_token_id if eos_token_id is not None else vocabulary.token_id_for_token(vocabulary.eos_token)


def _advance_dfa(dfa: DFA, state, label: int):
    """Advance *dfa* or raise a clear error when the transition is missing."""
    next_state = dfa.step(state, label)
    if next_state is None:
        raise RuntimeError(f"DFA has no transition from {state!r} on label {label!r}")
    return next_state


def _validate_common(max_new_tokens: int) -> None:
    """Validate arguments common to every decoding strategy."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")


def _candidate_rank(candidate: BeamCandidate, length_penalty: float) -> float:
    """Sort key for beam candidates."""
    return candidate.normalized_score(length_penalty)


def _filter_sampling_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to an already constrained logit vector."""
    filtered = logits.clone()

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1 when provided")
        if top_k < filtered.numel():
            values, _indices = torch.topk(filtered, top_k)
            threshold = values[-1]
            filtered = filtered.masked_fill(filtered < threshold, fill_value)

    if top_p is not None:
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered[sorted_indices[remove]] = fill_value

    return filtered


def constrained_greedy_decode(
    model,
    input_ids: torch.Tensor | Sequence[int],
    vocabulary: TokenVocabulary,
    dfa: DFA,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    use_cache: bool = True,
) -> ConstrainedGenerationResult:
    """Run DFA-constrained greedy decoding from a prompt.

    At each step:
    1. Forward-pass the model on the current token sequence.
    2. Query the DFA for allowed labels at the current state.
    3. Mask logits to remove disallowed tokens via :func:`mask_logits_for_dfa`.
    4. Select the highest-logit permitted token (greedy ``argmax``).
    5. Advance the DFA state and append the token to the sequence.
    6. Stop early if the selected token is EOS *and* the DFA is in an
       accepting state (valid complete sequence found).

    Args:
        model: A HuggingFace-compatible causal language model whose forward
            pass returns an object with a ``.logits`` attribute of shape
            ``(batch, seq_len, vocab_size)``.
        input_ids: Prompt token IDs as either a ``(1, T)`` / ``(T,)``
            :class:`torch.Tensor` or a plain Python sequence of ints.
        vocabulary: :class:`~.vocabulary.TokenVocabulary` mapping token IDs
            to DomiKnowS label IDs used by the DFA.
        dfa: Constraint :class:`~.learners.DFA` whose alphabet must cover
            the vocabulary's label set.
        max_new_tokens: Hard upper bound on the number of tokens that may be
            generated (not counting the prompt).
        eos_token_id: Token ID treated as the end-of-sequence sentinel.  When
            ``None``, falls back to
            ``vocabulary.token_id_for_token(vocabulary.eos_token)``.

    Returns:
        A :class:`ConstrainedGenerationResult` containing the full token ID
        list (prompt + generated), the generated label sequence, the final
        DFA state, and whether that state is accepting.

    Raises:
        RuntimeError: If the DFA has no transition from the current state on
            the selected label (indicates a vocabulary / DFA mismatch).
        ValueError: Propagated from :func:`mask_logits_for_dfa` if masking
            removes all tokens at any step.
    """
    _validate_common(max_new_tokens)
    ids, device = _normalise_input_ids(input_ids)

    state = dfa.start_state
    labels: list[int] = []
    cache_state = _cached_state_from_ids(ids, device)
    # Resolve EOS token ID once before the loop.
    eos_token_id = _resolve_eos_token_id(vocabulary, eos_token_id)

    for step_idx in range(max_new_tokens):
        # Take logits for the last position only.
        logits, cache_state = _next_logits_cached(model, cache_state, use_cache=use_cache)
        # Ask the DFA which labels are reachable within the remaining budget.
        remaining_steps = max_new_tokens - step_idx
        allowed = {int(label) for label in dfa.allowed_tokens(state, remaining_steps=remaining_steps)}
        # Mask out disallowed tokens in the full vocabulary logit space.
        masked = mask_logits_for_dfa(logits, allowed, vocabulary)
        # Greedy selection: highest-logit permitted token.
        next_id = int(torch.argmax(masked).item())
        next_label = vocabulary.label_for_token_id(next_id)

        # Advance the DFA; a missing transition signals a vocabulary mismatch.
        next_state = _advance_dfa(dfa, state, next_label)

        ids.append(next_id)
        labels.append(next_label)
        state = next_state
        cache_state = _append_cached_token(cache_state, next_id)

        # Stop early when EOS is produced in a valid (accepting) DFA state.
        if next_id == eos_token_id and dfa.is_accepting(state):
            break

    return ConstrainedGenerationResult(
        token_ids=ids,
        labels=labels,
        final_state=state,
        accepted=dfa.is_accepting(state),
    )


def constrained_beam_search_decode(
    model,
    input_ids: torch.Tensor | Sequence[int],
    vocabulary: TokenVocabulary,
    dfa: DFA,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    num_return_sequences: int = 1,
    use_cache: bool = True,
) -> ConstrainedGenerationResult:
    """Run DFA-constrained beam search from a prompt.

    Each beam carries its own DFA state and optional KV-cache.  At every
    expansion step the logits are masked by the active DFA state before
    log-probabilities are computed, so no returned candidate can contain a
    token that was disallowed by the DFA.  Beams that exhaust all valid tokens
    at any step are silently dropped rather than producing an error.

    Args:
        model: A HuggingFace-compatible causal language model.
        input_ids: Prompt token IDs as a ``(1, T)`` / ``(T,)``
            :class:`torch.Tensor` or a plain Python sequence of ints.
        vocabulary: :class:`~.vocabulary.TokenVocabulary` mapping token IDs
            to DomiKnowS label IDs.
        dfa: Constraint :class:`~.learners.DFA` whose alphabet must cover
            the vocabulary's label set.
        max_new_tokens: Hard upper bound on the number of tokens that may be
            generated per beam.
        eos_token_id: Token ID treated as EOS.  Defaults to
            ``vocabulary.token_id_for_token(vocabulary.eos_token)``.
        beam_size: Number of active beams kept at each step.  Must be ≥ 1.
        length_penalty: Exponent applied to sequence length when ranking
            finished candidates.  Values > 1 favour longer sequences; < 1
            favour shorter ones.  Must be positive.
        early_stopping: When ``True``, stop as soon as *num_return_sequences*
            finished (accepted) candidates have been found.
        num_return_sequences: How many top-ranked candidates to include in the
            result.  Must be ≥ 1.
        use_cache: Whether to use ``past_key_values`` caching.  Set to
            ``False`` to force full re-encoding at every step.

    Returns:
        A :class:`ConstrainedGenerationResult` for the best-ranked candidate,
        with ``scores`` listing the cumulative scores of all *num_return_sequences*
        returned candidates and ``candidates`` holding the full
        :class:`BeamCandidate` objects for inspection.

    Raises:
        ValueError: If *beam_size*, *length_penalty*, or *num_return_sequences*
            are out of range.
        RuntimeError: If the DFA has no transition from a beam's current state
            on the selected label (indicates a vocabulary / DFA mismatch).
    """
    _validate_common(max_new_tokens)
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    if length_penalty <= 0.0:
        raise ValueError("length_penalty must be positive")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be at least 1")

    ids, device = _normalise_input_ids(input_ids)
    eos_token_id = _resolve_eos_token_id(vocabulary, eos_token_id)
    initial_cache_state = _cached_state_from_ids(ids, device)
    beams = [
        BeamCandidate(
            token_ids=ids,
            labels=[],
            state=dfa.start_state,
            score=0.0,
            cache_state=initial_cache_state,
        )
    ]
    finished: list[BeamCandidate] = []

    for step_idx in range(max_new_tokens):
        expanded: list[BeamCandidate] = []
        remaining_steps = max_new_tokens - step_idx

        for candidate in beams:
            if candidate.finished:
                expanded.append(candidate)
                continue

            candidate_cache_state = candidate.cache_state or _cached_state_from_ids(candidate.token_ids, device)
            logits, parent_cache_state = _next_logits_cached(
                model,
                candidate_cache_state,
                use_cache=use_cache,
            )
            allowed = {int(label) for label in dfa.allowed_tokens(candidate.state, remaining_steps=remaining_steps)}
            try:
                masked = mask_logits_for_dfa(logits, allowed, vocabulary)
            except ValueError:
                continue

            log_probs = torch.log_softmax(masked, dim=-1)
            valid_token_ids = torch.nonzero(masked > -5e8, as_tuple=False).flatten()
            if valid_token_ids.numel() == 0:
                continue
            local_beam = min(beam_size, int(valid_token_ids.numel()))
            local_scores, local_positions = torch.topk(log_probs[valid_token_ids], local_beam)

            for score_delta, position in zip(local_scores.tolist(), local_positions.tolist()):
                next_id = int(valid_token_ids[int(position)].item())
                next_label = vocabulary.label_for_token_id(next_id)
                next_state = _advance_dfa(dfa, candidate.state, next_label)
                next_labels = candidate.labels + [next_label]
                next_ids = candidate.token_ids + [next_id]
                is_finished = next_id == eos_token_id and dfa.is_accepting(next_state)
                try:
                    next_cache_state = _clone_cached_state_for_branch(parent_cache_state, next_id)
                except ValueError:
                    if use_cache:
                        raise
                    next_cache_state = _cached_state_from_ids(next_ids, device)
                    next_cache_state.cache_supported = False
                next_candidate = BeamCandidate(
                    token_ids=next_ids,
                    labels=next_labels,
                    state=next_state,
                    score=float(candidate.score + score_delta),
                    finished=is_finished,
                    cache_state=next_cache_state,
                )
                expanded.append(next_candidate)
                if is_finished:
                    finished.append(next_candidate)

        if not expanded:
            break

        expanded.sort(key=lambda item: _candidate_rank(item, length_penalty), reverse=True)
        beams = expanded[:beam_size]
        if early_stopping and len(finished) >= num_return_sequences:
            break

    accepted_candidates = [candidate for candidate in finished + beams if dfa.is_accepting(candidate.state)]
    ranked = accepted_candidates if accepted_candidates else beams
    ranked = sorted(ranked, key=lambda item: _candidate_rank(item, length_penalty), reverse=True)
    if not ranked:
        ranked = [BeamCandidate(token_ids=ids, labels=[], state=dfa.start_state, score=0.0)]
    returned = ranked[:num_return_sequences]
    best = returned[0]

    return ConstrainedGenerationResult(
        token_ids=best.token_ids,
        labels=best.labels,
        final_state=best.state,
        accepted=dfa.is_accepting(best.state),
        score=best.score,
        scores=[candidate.score for candidate in returned],
        candidates=returned,
    )


def constrained_sample_decode(
    model,
    input_ids: torch.Tensor | Sequence[int],
    vocabulary: TokenVocabulary,
    dfa: DFA,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    use_cache: bool = True,
) -> ConstrainedGenerationResult:
    """Run DFA-constrained stochastic decoding from a prompt.

    At each step:

    1. The DFA mask is applied to remove tokens disallowed by the current
       constraint state.
    2. Temperature scaling divides the surviving logits.
    3. Optional top-k and/or nucleus (top-p) filtering further narrows the
       candidate set inside the constrained space.
    4. A single token is drawn from the resulting probability distribution
       via ``torch.multinomial``.
    5. The DFA state is advanced and the token is appended.
    6. Decoding stops early when EOS is sampled in an accepting DFA state.

    Args:
        model: A HuggingFace-compatible causal language model.
        input_ids: Prompt token IDs as a ``(1, T)`` / ``(T,)``
            :class:`torch.Tensor` or a plain Python sequence of ints.
        vocabulary: :class:`~.vocabulary.TokenVocabulary` mapping token IDs
            to DomiKnowS label IDs.
        dfa: Constraint :class:`~.learners.DFA` whose alphabet must cover
            the vocabulary's label set.
        max_new_tokens: Hard upper bound on the number of tokens that may be
            generated.
        eos_token_id: Token ID treated as EOS.  Defaults to
            ``vocabulary.token_id_for_token(vocabulary.eos_token)``.
        temperature: Softmax temperature.  Values < 1 sharpen the
            distribution (more deterministic); values > 1 flatten it (more
            random).  Must be positive.
        top_k: If set, only the *top_k* highest-logit tokens (after DFA
            masking) are kept before sampling.  Must be ≥ 1 when provided.
        top_p: Nucleus-sampling threshold in ``(0, 1]``.  The smallest set
            of tokens whose cumulative probability exceeds *top_p* is kept
            after DFA masking.  Applied after *top_k* when both are given.
        generator: Optional :class:`torch.Generator` for reproducible
            sampling.
        use_cache: Whether to use ``past_key_values`` caching.  Set to
            ``False`` to force full re-encoding at every step.

    Returns:
        A :class:`ConstrainedGenerationResult` with the full token sequence,
        label sequence, final DFA state, acceptance flag, cumulative
        log-probability score, and per-step log-probabilities.

    Raises:
        ValueError: If *temperature* is non-positive, or if *top_k* / *top_p*
            are out of range.
        RuntimeError: If the DFA has no transition from the current state on
            the sampled label.
        ValueError: Propagated from :func:`mask_logits_for_dfa` if masking
            removes all tokens at any step.
    """
    _validate_common(max_new_tokens)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    ids, device = _normalise_input_ids(input_ids)
    eos_token_id = _resolve_eos_token_id(vocabulary, eos_token_id)
    state = dfa.start_state
    labels: list[int] = []
    token_scores: list[float] = []
    total_score = 0.0
    cache_state = _cached_state_from_ids(ids, device)

    for step_idx in range(max_new_tokens):
        logits, cache_state = _next_logits_cached(model, cache_state, use_cache=use_cache)
        remaining_steps = max_new_tokens - step_idx
        allowed = {int(label) for label in dfa.allowed_tokens(state, remaining_steps=remaining_steps)}
        masked = mask_logits_for_dfa(logits, allowed, vocabulary)
        constrained_logits = masked / float(temperature)
        filtered = _filter_sampling_logits(constrained_logits, top_k=top_k, top_p=top_p)
        if torch.all(filtered <= -5e8):
            filtered = constrained_logits

        probs = torch.softmax(filtered, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        next_label = vocabulary.label_for_token_id(next_id)
        next_state = _advance_dfa(dfa, state, next_label)
        log_prob = float(torch.log(probs[next_id].clamp_min(torch.finfo(probs.dtype).tiny)).item())

        ids.append(next_id)
        labels.append(next_label)
        token_scores.append(log_prob)
        total_score += log_prob
        state = next_state
        cache_state = _append_cached_token(cache_state, next_id)

        if next_id == eos_token_id and dfa.is_accepting(state):
            break

    return ConstrainedGenerationResult(
        token_ids=ids,
        labels=labels,
        final_state=state,
        accepted=dfa.is_accepting(state),
        score=total_score,
        scores=token_scores,
    )


def constrained_label_greedy_decode(
    model,
    input_ids: torch.Tensor | Sequence[int],
    vocabulary: TokenVocabulary,
    dfa: DFA,
    max_new_tokens: int,
    eos_label: int | None = None,
    model_kwargs: dict | None = None,
    next_label_kwargs: dict | None = None,
) -> ConstrainedGenerationResult:
    """Run DFA-constrained greedy decoding for compact-label generation heads.

    Unlike :func:`constrained_greedy_decode`, *model* is expected to emit logits
    directly over ``vocabulary.label_count`` labels rather than over the full
    tokenizer vocabulary.  The chosen label is mapped back to a concrete
    tokenizer ID and appended to the model input for the next autoregressive step.

    At each step:

    1. Query the DFA for allowed labels at the current state.
    2. Intersect with the set of *emittable* labels (those that have a
       corresponding concrete token ID).
    3. Call the model's compact-label head to get per-label logits.
    4. Mask and select the highest-logit permitted label (greedy ``argmax``).
    5. Advance the DFA state and append the resolved token ID.
    6. Stop early when the EOS label is selected in an accepting DFA state.

    Args:
        model: A model that exposes either a ``next_label_logits(input_ids)``
            method or a standard HuggingFace forward pass returning logits of
            shape ``(1, seq_len, label_count)``.
        input_ids: Prompt token IDs as a ``(1, T)`` / ``(T,)``
            :class:`torch.Tensor` or a plain Python sequence of ints.
        vocabulary: :class:`~.vocabulary.TokenVocabulary` providing the
            label ↔ token-ID mapping.
        dfa: Constraint :class:`~.learners.DFA` whose alphabet must cover
            the vocabulary's label set.
        max_new_tokens: Hard upper bound on the number of tokens that may be
            generated (not counting the prompt).
        eos_label: Label ID treated as end-of-sequence.  Defaults to
            ``vocabulary.eos_label``.

    Returns:
        A :class:`ConstrainedGenerationResult` containing the full token ID
        list, the generated label sequence, the final DFA state, whether that
        state is accepting, the cumulative log-probability score, and per-step
        log-probabilities.

    Raises:
        RuntimeError: If the DFA has no transition from the current state on
            the selected label.
        ValueError: Propagated from :func:`mask_label_logits_for_dfa` if
            masking removes all labels at any step.
    """
    _validate_common(max_new_tokens)
    ids, device = _normalise_input_ids(input_ids)
    eos_label = vocabulary.eos_label if eos_label is None else int(eos_label)
    state = dfa.start_state
    labels: list[int] = []
    token_scores: list[float] = []
    total_score = 0.0
    emittable = _emittable_labels(model, vocabulary)

    for step_idx in range(max_new_tokens):
        remaining_steps = max_new_tokens - step_idx
        allowed = {int(label) for label in dfa.allowed_tokens(state, remaining_steps=remaining_steps)}
        allowed &= emittable
        logits = _next_label_logits(
            model,
            ids,
            device,
            model_kwargs=model_kwargs,
            next_label_kwargs=next_label_kwargs,
        )
        masked = mask_label_logits_for_dfa(logits, allowed)
        log_probs = torch.log_softmax(masked, dim=-1)
        next_label = int(torch.argmax(masked).item())
        next_state = _advance_dfa(dfa, state, next_label)
        next_id = _token_id_for_generated_label(model, vocabulary, next_label)
        log_prob = float(log_probs[next_label].item())

        ids.append(next_id)
        labels.append(next_label)
        token_scores.append(log_prob)
        total_score += log_prob
        state = next_state

        if next_label == eos_label and dfa.is_accepting(state):
            break

    return ConstrainedGenerationResult(
        token_ids=ids,
        labels=labels,
        final_state=state,
        accepted=dfa.is_accepting(state),
        score=total_score,
        scores=token_scores,
    )


def constrained_label_beam_search_decode(
    model,
    input_ids: torch.Tensor | Sequence[int],
    vocabulary: TokenVocabulary,
    dfa: DFA,
    max_new_tokens: int,
    eos_label: int | None = None,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    num_return_sequences: int = 1,
    model_kwargs: dict | None = None,
    next_label_kwargs: dict | None = None,
) -> ConstrainedGenerationResult:
    """Run DFA-constrained beam search for compact-label generation heads.

    Each beam carries its own generated compact labels, concrete tokenizer IDs,
    DFA state, and cumulative log-probability score.  Logits are produced over
    ``vocabulary.label_count`` labels, masked by the active DFA state, and only
    then expanded by beam search.  Selected labels are converted back to raw
    tokenizer IDs through ``model.token_id_for_label(...)`` when available, or
    :class:`TokenVocabulary` otherwise.
    """
    _validate_common(max_new_tokens)
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    if length_penalty <= 0.0:
        raise ValueError("length_penalty must be positive")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be at least 1")

    ids, device = _normalise_input_ids(input_ids)
    eos_label = vocabulary.eos_label if eos_label is None else int(eos_label)
    emittable = _emittable_labels(model, vocabulary)
    beams = [BeamCandidate(token_ids=ids, labels=[], state=dfa.start_state, score=0.0)]
    finished: list[BeamCandidate] = []

    for step_idx in range(max_new_tokens):
        expanded: list[BeamCandidate] = []
        remaining_steps = max_new_tokens - step_idx

        for candidate in beams:
            if candidate.finished:
                expanded.append(candidate)
                continue

            allowed = {int(label) for label in dfa.allowed_tokens(candidate.state, remaining_steps=remaining_steps)}
            allowed &= emittable
            try:
                logits = _next_label_logits(
                    model,
                    candidate.token_ids,
                    device,
                    model_kwargs=model_kwargs,
                    next_label_kwargs=next_label_kwargs,
                )
                masked = mask_label_logits_for_dfa(logits, allowed)
            except ValueError:
                continue

            log_probs = torch.log_softmax(masked, dim=-1)
            valid_labels = torch.nonzero(masked > -5e8, as_tuple=False).flatten()
            if valid_labels.numel() == 0:
                continue
            local_beam = min(beam_size, int(valid_labels.numel()))
            local_scores, local_positions = torch.topk(log_probs[valid_labels], local_beam)

            for score_delta, position in zip(local_scores.tolist(), local_positions.tolist()):
                next_label = int(valid_labels[int(position)].item())
                next_state = _advance_dfa(dfa, candidate.state, next_label)
                next_id = _token_id_for_generated_label(model, vocabulary, next_label)
                next_candidate = BeamCandidate(
                    token_ids=candidate.token_ids + [next_id],
                    labels=candidate.labels + [next_label],
                    state=next_state,
                    score=float(candidate.score + score_delta),
                    finished=next_label == eos_label and dfa.is_accepting(next_state),
                )
                expanded.append(next_candidate)
                if next_candidate.finished:
                    finished.append(next_candidate)

        if not expanded:
            break

        expanded.sort(key=lambda item: _candidate_rank(item, length_penalty), reverse=True)
        beams = expanded[:beam_size]
        if early_stopping and len(finished) >= num_return_sequences:
            break

    accepted_candidates = [candidate for candidate in finished + beams if dfa.is_accepting(candidate.state)]
    ranked = accepted_candidates if accepted_candidates else beams
    ranked = sorted(ranked, key=lambda item: _candidate_rank(item, length_penalty), reverse=True)
    if not ranked:
        ranked = [BeamCandidate(token_ids=ids, labels=[], state=dfa.start_state, score=0.0)]
    returned = ranked[:num_return_sequences]
    best = returned[0]

    return ConstrainedGenerationResult(
        token_ids=best.token_ids,
        labels=best.labels,
        final_state=best.state,
        accepted=dfa.is_accepting(best.state),
        score=best.score,
        scores=[candidate.score for candidate in returned],
        candidates=returned,
    )


def constrained_label_sample_decode(
    model,
    input_ids: torch.Tensor | Sequence[int],
    vocabulary: TokenVocabulary,
    dfa: DFA,
    max_new_tokens: int,
    eos_label: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    model_kwargs: dict | None = None,
    next_label_kwargs: dict | None = None,
) -> ConstrainedGenerationResult:
    """Run DFA-constrained sampling for compact-label generation heads.

    DFA masking is applied before temperature, top-k, and top-p filtering.  If
    filtering removes every currently constrained label, sampling falls back to
    the unfiltered DFA-masked logits so an invalid label is never emitted just
    because stochastic filtering was too aggressive.
    """
    _validate_common(max_new_tokens)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    ids, device = _normalise_input_ids(input_ids)
    eos_label = vocabulary.eos_label if eos_label is None else int(eos_label)
    state = dfa.start_state
    labels: list[int] = []
    token_scores: list[float] = []
    total_score = 0.0
    emittable = _emittable_labels(model, vocabulary)

    for step_idx in range(max_new_tokens):
        remaining_steps = max_new_tokens - step_idx
        allowed = {int(label) for label in dfa.allowed_tokens(state, remaining_steps=remaining_steps)}
        allowed &= emittable
        logits = _next_label_logits(
            model,
            ids,
            device,
            model_kwargs=model_kwargs,
            next_label_kwargs=next_label_kwargs,
        )
        masked = mask_label_logits_for_dfa(logits, allowed)
        constrained_logits = masked / float(temperature)
        filtered = _filter_sampling_logits(constrained_logits, top_k=top_k, top_p=top_p)
        if torch.all(filtered <= -5e8):
            filtered = constrained_logits

        probs = torch.softmax(filtered, dim=-1)
        next_label = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        next_state = _advance_dfa(dfa, state, next_label)
        next_id = _token_id_for_generated_label(model, vocabulary, next_label)
        log_prob = float(torch.log(probs[next_label].clamp_min(torch.finfo(probs.dtype).tiny)).item())

        ids.append(next_id)
        labels.append(next_label)
        token_scores.append(log_prob)
        total_score += log_prob
        state = next_state

        if next_label == eos_label and dfa.is_accepting(state):
            break

    return ConstrainedGenerationResult(
        token_ids=ids,
        labels=labels,
        final_state=state,
        accepted=dfa.is_accepting(state),
        score=total_score,
        scores=token_scores,
    )


