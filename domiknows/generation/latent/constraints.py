"""Soft (product-logic) loss functions for latent generation constraints.

This module provides differentiable approximations to hard logical constraints
over autoregressive token probability sequences.  All functions operate on
probability tensors of shape

- ``[seq_len, label_count]`` — single sequence (unbatched), or
- ``[batch, seq_len, label_count]`` — batched sequences.

The implemented semantics follow the **product t-norm** (Łukasiewicz product
logic):

- NOT(p)      = 1 - p
- AND(p, q)   = p * q
- OR(p, q)    = 1 - (1-p) * (1-q)  (De Morgan dual of AND)
- IMPLICATION(p, q) = p * (1 - q)  (penalty for p=1, q=0)

The loss functions penalise violations of constraints of the form
``IF token_t == if_label THEN formula(window after t)``.

Public API
----------
:func:`soft_not`, :func:`soft_and`, :func:`soft_or`
    Element-wise product-logic connectives.

:func:`soft_exists`
    Soft existence of a label in a probability window.

:func:`implication_loss`
    Soft implication penalty tensor.

:func:`chain_exists_loss`
    Loss for ``IF label_t THEN EXISTS other_label in [t+1, t+window]``.

:func:`window_all_loss`, :func:`window_any_loss`, :func:`window_formula_loss`
    Generalised window formula losses.

Formula DSL
-----------
A :data:`Formula` is either an integer label id, or a nested tuple
``(op, operand, ...)`` where *op* is ``"and"`` or ``"or"``.
Example: ``("and", 3, ("or", 5, 7))`` means
"label 3 **and** (label 5 **or** label 7)".
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

@dataclass(frozen=True)
class LabelRef:
    """Reference to a label in a named probability concept.

    Plain integer formula leaves continue to refer to the default concept,
    usually ``"generated_token"``.  Use ``LabelRef("latent_state", 2)`` when
    evaluating formulas over a dictionary of probability tensors.
    """

    concept: str
    label: int


@dataclass(frozen=True)
class LatentLossItem:
    """Diagnostics for one latent window-loss term."""

    name: str
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    weight: float
    top_violations: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class LatentLossBreakdown:
    """Total latent loss plus per-spec diagnostics."""

    total: torch.Tensor
    items: tuple[LatentLossItem, ...]


#: A recursive formula over label ids.  Leaves are either bare integer labels
#: in the default concept or :class:`LabelRef` values for cross-concept rules.
Formula: TypeAlias = int | LabelRef | tuple[object, ...]


def soft_not(value: torch.Tensor) -> torch.Tensor:
    """Element-wise product-logic negation: NOT(p) = 1 - p.

    Args:
        value: Probability tensor with values in [0, 1].

    Returns:
        Tensor of the same shape with negated probabilities.
    """
    return 1.0 - value


def soft_and(*values: torch.Tensor) -> torch.Tensor:
    """Element-wise product t-norm conjunction: AND(p, q) = p * q.

    Reduces left-to-right across all supplied tensors so that
    ``soft_and(a, b, c) == a * b * c``.

    Args:
        *values: One or more probability tensors of broadcastable shapes.

    Returns:
        Element-wise product of all inputs.

    Raises:
        ValueError: If called with no arguments.
    """
    if not values:
        raise ValueError("soft_and requires at least one value")
    result = values[0]
    # Accumulate the product left-to-right.
    for value in values[1:]:
        result = result * value
    return result


def soft_or(*values: torch.Tensor) -> torch.Tensor:
    """Element-wise product t-conorm disjunction via De Morgan duality.

    OR(p, q) = 1 - (1-p)(1-q), generalised to N operands as
    ``1 - AND(NOT(p1), NOT(p2), ...)``.  This is the exact dual of
    :func:`soft_and` under the product t-norm.

    Args:
        *values: One or more probability tensors of broadcastable shapes.

    Returns:
        Element-wise soft-OR of all inputs.

    Raises:
        ValueError: If called with no arguments.
    """
    if not values:
        raise ValueError("soft_or requires at least one value")
    # De Morgan: OR = NOT(AND(NOT(v) for v in values))
    return soft_not(soft_and(*(soft_not(value) for value in values)))


def soft_exists(probs: torch.Tensor, label: int, start: int, end: int) -> torch.Tensor:
    """Soft probability that *label* appears at least once in ``probs[start:end]``.

    Implements EXISTS using the product t-norm:
    ``EXISTS = 1 - PROD_{t=start}^{end-1}(1 - p_t)``.

    ``probs`` may be shaped ``[seq_len, label_count]`` or
    ``[batch, seq_len, label_count]``.
    Index bounds are clamped to ``[0, seq_len]``; an empty window returns zero.

    Args:
        probs: Token probability tensor (unbatched or batched).
        label: Integer label id to test for existence.
        start: Inclusive start index of the window (clamped).
        end: Exclusive end index of the window (clamped).

    Returns:
        Scalar (unbatched) or 1-D batch tensor with the soft existence score
        for the window.

    Raises:
        ValueError: If *label* is out of range.
    """
    probs, was_batched = _as_batched_probs(probs)
    _validate_label(label, probs.shape[-1])
    seq_len = probs.shape[1]
    # Clamp bounds to the valid sequence range.
    start = max(0, min(int(start), seq_len))
    end = max(start, min(int(end), seq_len))
    if start == end:
        # Empty window — no position can satisfy the constraint.
        result = probs.new_zeros((probs.shape[0],))
    else:
        # EXISTS = 1 - PROD(1 - p_t) over the window.
        window = probs[:, start:end, int(label)]
        result = soft_not(torch.prod(soft_not(window), dim=1))
    return result if was_batched else result.squeeze(0)


def evaluate_latent_loss(
    specs: Sequence[Any],
    probs: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int | LabelRef | None = None,
    top_k: int = 3,
) -> LatentLossBreakdown:
    """Evaluate latent window specs and return weighted diagnostics.

    ``specs`` are intentionally duck-typed so callers can pass
    :class:`domiknows.generation.latent.enforcement.LatentWindowSpec` without creating
    an import cycle.  Each spec should expose ``if_label``, ``formula``,
    ``window``, ``weight``, ``reduction``, and optionally ``name`` /
    ``concept`` / ``empty_window_policy``.
    """
    reference = _reference_tensor(probs)
    if not specs:
        return LatentLossBreakdown(reference.new_zeros(()), ())

    items: list[LatentLossItem] = []
    total = reference.new_zeros(())
    for index, spec in enumerate(specs):
        concept = getattr(spec, "concept", "generated_token")
        empty_window_policy = getattr(spec, "empty_window_policy", "penalize")
        per_position = window_formula_penalty_tensor(
            probs,
            getattr(spec, "if_label"),
            getattr(spec, "formula"),
            getattr(spec, "window"),
            concept=concept,
            mask=mask,
            lengths=lengths,
            eos_label=eos_label,
            empty_window_policy=empty_window_policy,
        )
        raw_loss = _reduce_loss(
            per_position.loss,
            getattr(spec, "reduction", "mean"),
            per_position.was_batched,
            position_mask=per_position.position_mask,
        )
        weighted = raw_loss * float(getattr(spec, "weight", 1.0))
        if weighted.dim() > 0:
            total = total + weighted.mean()
        else:
            total = total + weighted
        items.append(
            LatentLossItem(
                name=getattr(spec, "name", None) or f"latent_{index}",
                raw_loss=raw_loss,
                weighted_loss=weighted,
                weight=float(getattr(spec, "weight", 1.0)),
                top_violations=_top_violations(per_position.loss, per_position.position_mask, top_k),
            )
        )
    return LatentLossBreakdown(total, tuple(items))


def implication_loss(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Element-wise product-logic implication penalty: p * (1 - q).

    A high loss occurs when *lhs* is close to 1 (antecedent holds) and
    *rhs* is close to 0 (consequent fails).  The penalty is zero whenever
    either the antecedent is false or the consequent is true.

    Args:
        lhs: Soft truth value of the antecedent; shape broadcastable with *rhs*.
        rhs: Soft truth value of the consequent; same shape as *lhs*.

    Returns:
        Per-element implication penalty tensor of the same shape.
    """
    return lhs * soft_not(rhs)


def chain_exists_loss(
    probs: torch.Tensor,
    if_label: int,
    then_label: int,
    window: int,
    *,
    reduction: str = "mean",
    mask: torch.Tensor | None = None,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int | None = None,
    empty_window_policy: str = "penalize",
) -> torch.Tensor:
    """Soft loss for: IF token_t == if_label THEN EXISTS then_label in [t+1, t+window].

    For every position *t*, the penalty is
    ``p(if_label)_t * (1 - EXISTS(then_label, t+1, t+1+window))``.
    The per-position penalties are averaged across the sequence, then
    reduced across the batch according to *reduction*.

    Args:
        probs: Token probability tensor shaped ``[seq_len, num_labels]``
            or ``[batch, seq_len, num_labels]``.
        if_label: Integer label id for the antecedent token.
        then_label: Integer label id that must exist in the future window.
        window: Number of future positions to search (must be ≥ 1).
        reduction: How to aggregate across the batch dimension.
            ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (``"mean"`` or ``"sum"``), or a 1-D batch tensor
        (``"none"`` or unbatched input).

    Raises:
        ValueError: If *window* < 1 or either label is out of range.
    """
    probs, was_batched = _as_batched_probs(probs)
    _validate_window(window)
    _validate_label(if_label, probs.shape[-1])
    _validate_label(then_label, probs.shape[-1])

    position_mask = _build_position_mask(
        probs,
        mask=mask,
        lengths=lengths,
        eos_label=eos_label,
        was_batched=was_batched,
    )
    lhs = probs[:, :, int(if_label)]
    rhs = _window_exists(probs, int(then_label), window, position_mask=position_mask)
    losses = implication_loss(lhs, rhs)
    position_mask = _apply_empty_window_policy(position_mask, window, empty_window_policy)
    return _reduce_loss(losses, reduction, was_batched, position_mask=position_mask)


def window_all_loss(
    probs: torch.Tensor,
    if_label: int,
    required_labels: Sequence[int],
    window: int,
    *,
    reduction: str = "mean",
    mask: torch.Tensor | None = None,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int | None = None,
    empty_window_policy: str = "penalize",
) -> torch.Tensor:
    """Soft loss for: IF token_t == if_label THEN ALL required_labels appear in [t+1, t+window].

    A shorthand for :func:`window_formula_loss` with an ``"and"`` formula over
    all *required_labels*.

    Args:
        probs: Token probability tensor shaped ``[seq_len, num_labels]``
            or ``[batch, seq_len, num_labels]``.
        if_label: Integer label id for the antecedent token.
        required_labels: Non-empty sequence of label ids that must all exist
            in the window.
        window: Number of future positions to search (must be ≥ 1).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar or per-batch loss tensor (same rules as :func:`chain_exists_loss`).

    Raises:
        ValueError: If *required_labels* is empty or *window* < 1.
    """
    if not required_labels:
        raise ValueError("required_labels must not be empty")
    # Build an AND formula over all required labels.
    formula: Formula = ("and", *[int(label) for label in required_labels])
    return window_formula_loss(
        probs,
        if_label,
        formula,
        window,
        reduction=reduction,
        mask=mask,
        lengths=lengths,
        eos_label=eos_label,
        empty_window_policy=empty_window_policy,
    )


def window_any_loss(
    probs: torch.Tensor,
    if_label: int,
    candidate_labels: Sequence[int],
    window: int,
    *,
    reduction: str = "mean",
    mask: torch.Tensor | None = None,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int | None = None,
    empty_window_policy: str = "penalize",
) -> torch.Tensor:
    """Soft loss for: IF token_t == if_label THEN ANY candidate_label appears in [t+1, t+window].

    A shorthand for :func:`window_formula_loss` with an ``"or"`` formula over
    all *candidate_labels*.

    Args:
        probs: Token probability tensor shaped ``[seq_len, num_labels]``
            or ``[batch, seq_len, num_labels]``.
        if_label: Integer label id for the antecedent token.
        candidate_labels: Non-empty sequence of label ids; at least one must
            appear in the window.
        window: Number of future positions to search (must be ≥ 1).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar or per-batch loss tensor.

    Raises:
        ValueError: If *candidate_labels* is empty or *window* < 1.
    """
    if not candidate_labels:
        raise ValueError("candidate_labels must not be empty")
    # Build an OR formula over all candidate labels.
    formula: Formula = ("or", *[int(label) for label in candidate_labels])
    return window_formula_loss(
        probs,
        if_label,
        formula,
        window,
        reduction=reduction,
        mask=mask,
        lengths=lengths,
        eos_label=eos_label,
        empty_window_policy=empty_window_policy,
    )


def window_formula_loss(
    probs: torch.Tensor | Mapping[str, torch.Tensor],
    if_label: int | LabelRef,
    formula: Formula,
    window: int,
    *,
    reduction: str = "mean",
    concept: str = "generated_token",
    mask: torch.Tensor | None = None,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int | LabelRef | None = None,
    empty_window_policy: str = "penalize",
) -> torch.Tensor:
    """Soft loss for: IF token_t == if_label THEN formula is satisfied in [t+1, t+window].

    The most general windowed implication loss.  *formula* is evaluated
    recursively via :func:`_evaluate_window_formula` using the
    :data:`Formula` mini-DSL::

        formula := label_id                  # EXISTS label in window
                 | ("and", formula, ...)     # soft AND
                 | ("or",  formula, ...)     # soft OR

    Args:
        probs: Token probability tensor shaped ``[seq_len, num_labels]``
            or ``[batch, seq_len, num_labels]``.
        if_label: Integer label id for the antecedent token.
        formula: A :data:`Formula` expression tree.
        window: Number of future positions to search (must be ≥ 1).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (``"mean"``/``"sum"``) or per-batch 1-D tensor
        (``"none"`` or unbatched input).

    Raises:
        ValueError: If *window* < 1, *if_label* is out of range, or
            *formula* contains an unknown operator.
    """
    per_position = window_formula_penalty_tensor(
        probs,
        if_label,
        formula,
        window,
        concept=concept,
        mask=mask,
        lengths=lengths,
        eos_label=eos_label,
        empty_window_policy=empty_window_policy,
    )
    return _reduce_loss(
        per_position.loss,
        reduction,
        per_position.was_batched,
        position_mask=per_position.position_mask,
    )


@dataclass(frozen=True)
class _PerPositionLoss:
    loss: torch.Tensor
    position_mask: torch.Tensor | None
    was_batched: bool


def window_formula_penalty_tensor(
    probs: torch.Tensor | Mapping[str, torch.Tensor],
    if_label: int | LabelRef,
    formula: Formula,
    window: int,
    *,
    concept: str = "generated_token",
    mask: torch.Tensor | None = None,
    lengths: torch.Tensor | Sequence[int] | None = None,
    eos_label: int | LabelRef | None = None,
    empty_window_policy: str = "penalize",
) -> _PerPositionLoss:
    """Return per-position implication penalties for a latent window formula."""
    tensors, was_batched = _normalise_prob_mapping(probs, default_concept=concept)
    _validate_window(window)
    default_probs = tensors[str(concept)]
    position_mask = _build_position_mask(
        default_probs,
        mask=mask,
        lengths=lengths,
        eos_label=eos_label,
        was_batched=was_batched,
        probs_by_concept=tensors,
        default_concept=concept,
    )
    lhs_ref = _coerce_label_ref(if_label, concept)
    lhs_probs = _probs_for_ref(tensors, lhs_ref)
    lhs = lhs_probs[:, :, lhs_ref.label]
    rhs = _evaluate_window_formula(
        tensors,
        formula,
        window,
        default_concept=concept,
        position_mask=position_mask,
    )
    losses = implication_loss(lhs, rhs)
    position_mask = _apply_empty_window_policy(position_mask, window, empty_window_policy)
    return _PerPositionLoss(losses, position_mask, was_batched)


def _as_batched_probs(probs: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return probs as [batch, seq_len, label_count] and whether it was batched.

    Args:
        probs: Tensor-like probabilities shaped ``[seq_len, label_count]`` or
            ``[batch, seq_len, label_count]``.

    Returns:
        ``(batched_probs, was_batched)`` where ``batched_probs`` is always 3-D.

    Raises:
        ValueError: If the input is not 2-D or 3-D.
    """
    if not isinstance(probs, torch.Tensor):
        probs = torch.as_tensor(probs, dtype=torch.float32)
    if probs.dim() == 2:
        return probs.unsqueeze(0), False
    if probs.dim() == 3:
        return probs, True
    raise ValueError("probs must have shape [seq_len, label_count] or [batch, seq_len, label_count]")


def _reference_tensor(probs: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(probs, Mapping):
        if not probs:
            raise ValueError("probs mapping must not be empty")
        first = next(iter(probs.values()))
        return first if isinstance(first, torch.Tensor) else torch.as_tensor(first, dtype=torch.float32)
    return probs if isinstance(probs, torch.Tensor) else torch.as_tensor(probs, dtype=torch.float32)


def _normalise_prob_mapping(
    probs: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    default_concept: str,
) -> tuple[dict[str, torch.Tensor], bool]:
    if isinstance(probs, Mapping):
        tensors: dict[str, torch.Tensor] = {}
        was_batched: bool | None = None
        shape: tuple[int, int] | None = None
        for name, value in probs.items():
            batched, current_was_batched = _as_batched_probs(value)
            if was_batched is None:
                was_batched = current_was_batched
                shape = (batched.shape[0], batched.shape[1])
            elif shape != (batched.shape[0], batched.shape[1]):
                raise ValueError("all probability concepts must share batch and sequence dimensions")
            tensors[str(name)] = batched
        if default_concept not in tensors:
            raise ValueError(f"probability mapping is missing default concept {default_concept!r}")
        return tensors, bool(was_batched)
    batched, was_batched = _as_batched_probs(probs)
    return {str(default_concept): batched}, was_batched


def _coerce_label_ref(label: int | LabelRef, default_concept: str) -> LabelRef:
    if isinstance(label, LabelRef):
        return label
    return LabelRef(str(default_concept), int(label))


def _probs_for_ref(tensors: Mapping[str, torch.Tensor], ref: LabelRef) -> torch.Tensor:
    if ref.concept not in tensors:
        raise ValueError(f"probability mapping is missing concept {ref.concept!r}")
    probs = tensors[ref.concept]
    _validate_label(int(ref.label), probs.shape[-1])
    return probs


def _validate_label(label: int, label_count: int) -> None:
    """Raise if *label* is not a valid index into the label dimension.

    Args:
        label: Candidate label id to validate.
        label_count: Total number of labels (size of the last tensor dimension).

    Raises:
        TypeError: If *label* is not an ``int``.
        ValueError: If *label* is negative or ≥ *label_count*.
    """
    if not isinstance(label, int):
        raise TypeError("labels must be integer ids")
    if label < 0 or label >= label_count:
        raise ValueError(f"label {label} is outside the probability label dimension")


def _validate_window(window: int) -> None:
    """Raise if *window* is less than 1.

    Args:
        window: Number of future positions in the look-ahead window.

    Raises:
        ValueError: If *window* < 1.
    """
    if int(window) < 1:
        raise ValueError("window must be at least 1")


def _build_position_mask(
    probs: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    lengths: torch.Tensor | Sequence[int] | None,
    eos_label: int | LabelRef | None,
    was_batched: bool,
    probs_by_concept: Mapping[str, torch.Tensor] | None = None,
    default_concept: str = "generated_token",
) -> torch.Tensor | None:
    batch_size, seq_len, _label_count = probs.shape
    result = None
    if lengths is not None:
        lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=probs.device)
        if lengths_t.dim() == 0:
            lengths_t = lengths_t.view(1)
        if not was_batched and lengths_t.numel() == 1:
            pass
        elif lengths_t.numel() != batch_size:
            raise ValueError("lengths must have one value per batch item")
        positions = torch.arange(seq_len, device=probs.device).unsqueeze(0)
        result = positions < lengths_t.view(-1, 1).clamp_min(0).clamp_max(seq_len)
    if mask is not None:
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=probs.device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
        if mask_t.shape != (batch_size, seq_len):
            raise ValueError("mask must have shape [seq_len] or [batch, seq_len]")
        result = mask_t if result is None else (result & mask_t)
    if eos_label is not None:
        ref = _coerce_label_ref(eos_label, default_concept)
        source = _probs_for_ref(probs_by_concept or {default_concept: probs}, ref)
        eos_positions = torch.argmax(source, dim=-1) == ref.label
        eos_seen = torch.cumsum(eos_positions.to(torch.int64), dim=1) > 0
        before_or_at_first_eos = ~torch.roll(eos_seen, shifts=1, dims=1)
        before_or_at_first_eos[:, 0] = True
        eos_mask = before_or_at_first_eos | eos_positions
        result = eos_mask if result is None else (result & eos_mask)
    return result


def _apply_empty_window_policy(
    position_mask: torch.Tensor | None,
    window: int,
    empty_window_policy: str,
) -> torch.Tensor | None:
    if empty_window_policy not in {"penalize", "ignore"}:
        raise ValueError("empty_window_policy must be 'penalize' or 'ignore'")
    if empty_window_policy == "penalize":
        return position_mask
    if position_mask is None:
        return None
    batch_size, seq_len = position_mask.shape
    has_future = torch.zeros_like(position_mask)
    for pos in range(seq_len):
        start = pos + 1
        end = min(seq_len, pos + 1 + int(window))
        if start < end:
            has_future[:, pos] = position_mask[:, start:end].any(dim=1)
    return position_mask & has_future


def _window_exists(
    probs: torch.Tensor,
    label: int,
    window: int,
    *,
    position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the soft existence of *label* in each position's forward window.

    For every position *t* in ``[0, seq_len)``, fills
    ``output[:, t] = soft_exists(probs, label, t+1, t+1+window)``.  Positions
    where the window would extend beyond the sequence boundary are clamped.

    Args:
        probs: Batched probability tensor shaped ``[batch, seq_len, label_count]``.
        label: Integer label id to test for existence.
        window: Number of forward positions to include in the window.

    Returns:
        Soft existence tensor shaped ``[batch, seq_len]``.
    """
    batch_size, seq_len, label_count = probs.shape
    _validate_label(label, label_count)
    # Initialise to zero; positions with an empty window stay zero.
    values = probs.new_zeros((batch_size, seq_len))
    for pos in range(seq_len):
        start = pos + 1
        end = min(seq_len, pos + 1 + int(window))
        # Only fill positions where at least one future step exists.
        if start < end:
            window_probs = probs[:, start:end, int(label)]
            if position_mask is not None:
                window_probs = window_probs * position_mask[:, start:end].to(dtype=probs.dtype)
            values[:, pos] = soft_not(torch.prod(soft_not(window_probs), dim=1))
    return values


def _evaluate_window_formula(
    probs_by_concept: Mapping[str, torch.Tensor],
    formula: Formula,
    window: int,
    *,
    default_concept: str,
    position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recursively evaluate a :data:`Formula` over each position's forward window.

    Base case: an integer label id maps to :func:`_window_exists`.
    Recursive cases: ``("and", ...)`` maps to :func:`soft_and`,
    ``("or", ...)`` maps to :func:`soft_or`.

    Args:
        probs: Batched probability tensor shaped ``[batch, seq_len, label_count]``.
        formula: A :data:`Formula` expression tree.
        window: Forward look-ahead window size, forwarded to
            :func:`_window_exists` at every leaf.

    Returns:
        Satisfaction tensor shaped ``[batch, seq_len]``.

    Raises:
        ValueError: If *formula* has an unrecognised operator or is malformed.
    """
    if isinstance(formula, (int, LabelRef)):
        # Leaf node: soft existence of this label in the window.
        ref = _coerce_label_ref(formula, default_concept)
        probs = _probs_for_ref(probs_by_concept, ref)
        return _window_exists(probs, ref.label, window, position_mask=position_mask)
    if not isinstance(formula, tuple) or len(formula) < 2:
        raise ValueError("formula must be a label id or a non-empty ('and'/'or', ...) tuple")

    op = formula[0]
    if op not in {"and", "or"}:
        raise ValueError("formula operator must be 'and' or 'or'")
    # Recurse into each child operand.
    values = [
        _evaluate_window_formula(
            probs_by_concept,
            child,
            window,
            default_concept=default_concept,
            position_mask=position_mask,
        )
        for child in formula[1:]
    ]
    return soft_and(*values) if op == "and" else soft_or(*values)


def _reduce_loss(
    per_position_loss: torch.Tensor,
    reduction: str,
    was_batched: bool,
    *,
    position_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average per-position losses to per-sequence, then apply batch reduction.

    Reduction is always applied first over the sequence dimension (``dim=1``)
    so that sequences of different lengths are weighted equally by default.
    The batch dimension is then reduced according to *reduction*.

    Args:
        per_position_loss: Per-step loss tensor shaped ``[batch, seq_len]``.
        reduction: ``"mean"`` averages across the batch;
            ``"sum"`` sums; ``"none"`` returns the per-sequence tensor.
        was_batched: If ``False`` (original input was 2-D), the batch
            dimension is squeezed from the ``"none"`` result.

    Returns:
        Scalar tensor (``"mean"``/``"sum"``) or 1-D / scalar tensor
        (``"none"`` depending on *was_batched*).

    Raises:
        ValueError: If *reduction* is not one of the accepted values.
    """
    if position_mask is not None:
        weighted = per_position_loss * position_mask.to(dtype=per_position_loss.dtype)
        denom = position_mask.to(dtype=per_position_loss.dtype).sum(dim=1).clamp_min(1.0)
        per_sequence = weighted.sum(dim=1) / denom
    else:
        # Average per-position penalties into a single value per sequence.
        per_sequence = per_position_loss.mean(dim=1)
    if reduction == "none":
        # Return per-sequence losses; squeeze batch dim for unbatched inputs.
        return per_sequence if was_batched else per_sequence.squeeze(0)
    if reduction == "mean":
        return per_sequence.mean()
    if reduction == "sum":
        return per_sequence.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _top_violations(
    per_position_loss: torch.Tensor,
    position_mask: torch.Tensor | None,
    top_k: int,
) -> tuple[tuple[int, int, float], ...]:
    if top_k <= 0:
        return ()
    values = per_position_loss.detach()
    if position_mask is not None:
        values = values.masked_fill(~position_mask, 0.0)
    flat = values.flatten()
    if flat.numel() == 0:
        return ()
    count = min(int(top_k), flat.numel())
    top = torch.topk(flat, k=count)
    seq_len = values.shape[1]
    out = []
    for flat_index, score in zip(top.indices.tolist(), top.values.tolist(), strict=False):
        if score <= 0:
            continue
        out.append((int(flat_index // seq_len), int(flat_index % seq_len), float(score)))
    return tuple(out)
