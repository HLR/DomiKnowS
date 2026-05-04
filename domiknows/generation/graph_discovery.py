"""Introspection utilities that extract DFA constraints from a DomiKnowS graph.

After a generation graph is built with :class:`~.encoder.GenerationEncoder`
and constraints are applied (either via :func:`~.generation_constraints.apply_all_constraints`
or :func:`~.enforcement.mark_for_dfa`), this module walks the graph's logical
constraint nodes and reconstructs the corresponding
:class:`~.constraints.GenerationConstraint` objects so they can be compiled
into a single combined DFA for token-level hard decoding.

Recognised DomiKnowS constraint shapes
----------------------------------------
- ``ifL(is_before_rel, ifL(eos, eos))``          → :class:`~.constraints.EosClosureConstraint`
- ``atMostAL(notL(eos), N)``                      → :class:`~.constraints.MaxNonEosConstraint`
- ``atLeastAL(token_value, N)``                   → :class:`~.constraints.RequiredTokenConstraint`
- ``atMostAL(token_value, 0)``                    → :class:`~.constraints.ForbiddenTokenConstraint`
- ``ifL(existsAL(token), atMostAL(notL(eos), N))``→ :class:`~.constraints.ConditionalMaxNonEosConstraint`
- ``andL(...)`` over supported leaves             → intersection/conjunction
- ``orL(...)`` over supported leaves              → union/disjunction

Any other shape that still references generation concepts is considered
*unsupported* and handled according to the ``on_unsupported`` policy.

Public API
----------
:func:`discover_generation_constraints`
    Primary entry point — returns a tuple of discovered constraints.

:func:`constraints_to_dfa_from_graph`
    Convenience wrapper that discovers constraints and immediately compiles
    them into a single DFA.
"""
from __future__ import annotations

import warnings
from typing import Iterable

from .constraints import (
    AllOfGenerationConstraint,
    AnyOfGenerationConstraint,
    ConditionalMaxNonEosConstraint,
    EosClosureConstraint,
    ForbiddenTokenConstraint,
    GenerationConstraint,
    MaxNonEosConstraint,
    RequiredTokenConstraint,
    all_of_constraints,
    any_of_constraints,
    constraints_to_dfa,
    forbidden_token,
    if_token_present_then_at_most_non_eos,
    max_non_eos,
    no_token_after_eos,
    required_token,
)


def discover_generation_constraints(
    graph,
    bundle,
    *,
    on_unsupported: str = "warn",
) -> tuple[GenerationConstraint, ...]:
    """Discover DFA-enforceable generation constraints from a DomiKnowS graph.

    Discovery proceeds in two passes:

    1. **Bundle pass** — constraints already stored in ``bundle.constraints``
       (placed there by :meth:`~.encoder.GenerationEncoder.build_graph`) are
       collected first.
    2. **Graph pass** — every head logical constraint in the graph is examined:

       a. If it has a ``_generation_dfa_constraint`` marker (set by
          :func:`~.enforcement.mark_for_dfa`), that marker is used directly
          or triggers pattern matching.
       b. Otherwise, structural pattern matching (:func:`_match_lc`) is
          attempted.
       c. Latent-only constraints (``_generation_latent_specs``) are silently
          skipped.
       d. Unrecognised constraints that reference generation concepts are
          handled according to *on_unsupported*.

    Duplicate constraints (by semantic key) are deduplicated automatically.

    Args:
        graph: A built DomiKnowS :class:`~domiknows.graph.Graph` with a
            ``logicalConstrains`` attribute.
        bundle: The :class:`~.encoder.GenerationBundle` returned alongside
            *graph*.
        on_unsupported: Policy for unrecognised generation-relevant constraints.
            ``"warn"`` (default) emits a :class:`RuntimeWarning`;
            ``"error"`` raises a :class:`ValueError`;
            ``"ignore"`` silently skips.

    Returns:
        Tuple of unique :class:`~.constraints.GenerationConstraint` objects
        discovered from the graph, in encounter order.

    Raises:
        ValueError: If *on_unsupported* is not one of the accepted values, or
            if *on_unsupported* is ``"error"`` and an unsupported constraint
            is encountered.
    """
    if on_unsupported not in {"ignore", "warn", "error"}:
        raise ValueError("on_unsupported must be 'ignore', 'warn', or 'error'")

    constraints: list[GenerationConstraint] = []
    seen: set = set()

    # Pass 1: seed from constraints already attached to the bundle.
    for constraint in getattr(bundle, "constraints", ()):
        _append_unique(constraints, seen, constraint)

    # Pass 2: walk head logical constraints in the graph.
    for lc_name, lc in graph.logicalConstrains.items():
        # Skip non-head (child/sub-expression) constraints.
        if not getattr(lc, "headLC", True):
            continue
        # Try the explicit DFA marker first; fall back to structural matching.
        discovered = _marked_dfa_constraints(lc, bundle)
        if discovered is False:
            # No marker — attempt pattern-based recognition.
            discovered = _match_lc_many(lc, bundle)
        if not discovered:
            # Not recognised as a DFA constraint; check if it needs a warning.
            if _is_latent_marked(lc):
                # Latent-only constraint; not applicable to hard decoding.
                continue
            if _is_generation_relevant(lc, bundle):
                _handle_unsupported(lc_name, lc, on_unsupported)
            continue
        for constraint in discovered:
            _append_unique(constraints, seen, constraint)

    return tuple(constraints)


def constraints_to_dfa_from_graph(graph, bundle, *, on_unsupported: str = "warn"):
    """Discover constraints from *graph* and compile them into a single DFA.

    Convenience wrapper around :func:`discover_generation_constraints` +
    :func:`~.constraints.constraints_to_dfa`.  Equivalent to::

        constraints_to_dfa(
            discover_generation_constraints(graph, bundle, on_unsupported=...),
            bundle.vocabulary,
        )

    Args:
        graph: A built DomiKnowS graph.
        bundle: The :class:`~.encoder.GenerationBundle` returned alongside
            *graph*.
        on_unsupported: Forwarded to :func:`discover_generation_constraints`.

    Returns:
        A :class:`~.automata.DFA` accepting all sequences that satisfy every
        discovered constraint.
    """
    return constraints_to_dfa(
        discover_generation_constraints(graph, bundle, on_unsupported=on_unsupported),
        bundle.vocabulary,
    )


def _append_unique(constraints: list[GenerationConstraint], seen: set, constraint: GenerationConstraint) -> None:
    """Append *constraint* to *constraints* if its semantic key is not in *seen*.

    Uses :func:`_constraint_key` to deduplicate constraints that are
    semantically identical even if they are separate objects.

    Args:
        constraints: Accumulator list being built.
        seen: Set of already-seen keys; mutated in place.
        constraint: Candidate constraint to add.
    """
    key = _constraint_key(constraint)
    if key not in seen:
        seen.add(key)
        constraints.append(constraint)


def _constraint_key(constraint: GenerationConstraint):
    """Return a hashable semantic identity key for *constraint*.

    Used by :func:`_append_unique` to detect duplicate constraints.
    The key encodes the constraint type and its defining parameters so
    that two independently constructed but equivalent instances are
    treated as the same constraint.

    Args:
        constraint: Any :class:`~.constraints.GenerationConstraint`.

    Returns:
        A tuple usable as a dict key / set member.
    """
    if isinstance(constraint, EosClosureConstraint):
        return ("eos_closure",)
    if isinstance(constraint, MaxNonEosConstraint):
        return ("max_non_eos", constraint.max_count)
    if isinstance(constraint, RequiredTokenConstraint):
        return ("required", constraint.token, constraint.min_count)
    if isinstance(constraint, ForbiddenTokenConstraint):
        return ("forbidden", constraint.token)
    if isinstance(constraint, ConditionalMaxNonEosConstraint):
        return ("conditional_max_non_eos", constraint.token, constraint.max_count)
    if isinstance(constraint, AllOfGenerationConstraint):
        return ("all_of", tuple(sorted((_constraint_key(child) for child in constraint.children), key=repr)))
    if isinstance(constraint, AnyOfGenerationConstraint):
        return ("any_of", tuple(sorted((_constraint_key(child) for child in constraint.children), key=repr)))
    # Fallback for unknown subtypes: use class + name attribute.
    return (constraint.__class__, getattr(constraint, "name", None))


def _match_lc_many(lc, bundle) -> tuple[GenerationConstraint, ...] | None:
    """Attempt to match *lc* against the known generation constraint shapes.

    Dispatches on the DomiKnowS logical-constraint class name and delegates to
    the appropriate ``_match_*`` helper.  Boolean ``andL`` / ``orL`` nodes are
    matched recursively.  Returns ``None`` when no pattern matches; returns an
    empty tuple when a boolean expression contains no generation-relevant
    supported children.

    Args:
        lc: A DomiKnowS logical constraint node.
        bundle: The :class:`~.encoder.GenerationBundle` providing concept
            references for structural matching.

    Returns:
        Tuple of :class:`~.constraints.GenerationConstraint` objects if the
        shape is recognised, else ``None``.
    """
    cls_name = lc.__class__.__name__
    if cls_name == "andL":
        return _match_and_lc(lc, bundle)
    if cls_name == "orL":
        return _match_or_lc(lc, bundle)
    if cls_name == "ifL":
        return _as_constraint_tuple(_match_if_lc(lc, bundle))
    if cls_name == "atMostAL":
        return _as_constraint_tuple(_match_at_most_lc(lc, bundle))
    if cls_name == "atLeastAL":
        return _as_constraint_tuple(_match_at_least_lc(lc, bundle))
    if cls_name == "existsAL":
        return _as_constraint_tuple(_match_exists_lc(lc, bundle))
    return None


def _as_constraint_tuple(constraint: GenerationConstraint | None) -> tuple[GenerationConstraint, ...] | None:
    """Normalize an optional single constraint to the discovery tuple shape."""
    if constraint is None:
        return None
    return (constraint,)


def _marked_dfa_constraints(lc, bundle) -> tuple[GenerationConstraint, ...] | None | bool:
    """Resolve the ``_generation_dfa_constraint`` marker on *lc*.

    Three return conventions are used to distinguish three cases, allowing
    the caller to decide whether to proceed with structural matching:

    - ``False``  — no marker present; caller should try ``_match_lc``.
    - ``None``   — marker present but could not be resolved to a constraint
                   (unrecognised marker type).
    - A tuple of :class:`~.constraints.GenerationConstraint` instances —
      resolved constraints ready to be appended.

    Args:
        lc: A DomiKnowS logical constraint node.
        bundle: The :class:`~.encoder.GenerationBundle` for structural
            fallback matching when the marker is ``True``.

    Returns:
        ``False``, ``None``, or a :class:`~.constraints.GenerationConstraint`.
    """
    marker = getattr(lc, "_generation_dfa_constraint", False)
    if marker is False:
        # No marker — signal the caller to try structural pattern matching.
        return False
    if isinstance(marker, GenerationConstraint):
        # Explicit constraint object stored directly in the marker.
        return (marker,)
    if marker is True:
        # Marker present but no explicit constraint; fall back to structural matching.
        return _match_lc_many(lc, bundle)
    # Unknown marker value — cannot resolve.
    return None


def _is_latent_marked(lc) -> bool:
    """Return ``True`` if *lc* has any latent window specs attached.

    Used to suppress the "unsupported" warning for constraints that were
    intentionally marked for soft (latent) enforcement only and should not
    be compiled into a hard DFA.

    Args:
        lc: A DomiKnowS logical constraint node.
    """
    return bool(getattr(lc, "_generation_latent_specs", ()))


def _match_and_lc(lc, bundle) -> tuple[GenerationConstraint, ...] | None:
    """Match an ``andL`` as a conjunction of supported generation children.

    Generation-relevant unsupported children make the whole ``andL``
    unsupported.  Non-generation children are ignored for hard decoding and
    remain available to normal DomiKnowS loss/verification.
    """
    constraints: list[GenerationConstraint] = []
    for child in getattr(lc, "e", ()):
        child_match = _match_lc_many(child, bundle) if hasattr(child, "e") else None
        if child_match is None:
            if _is_generation_relevant(child, bundle):
                return None
            continue
        constraints.extend(child_match)
    return tuple(constraints)


def _match_or_lc(lc, bundle) -> tuple[GenerationConstraint, ...] | None:
    """Match an ``orL`` as a union of fully supported generation branches."""
    branches: list[GenerationConstraint] = []
    for child in getattr(lc, "e", ()):
        if not _is_generation_relevant(child, bundle):
            return None
        child_match = _match_lc_many(child, bundle) if hasattr(child, "e") else None
        if not child_match:
            return None
        if len(child_match) == 1:
            branches.append(child_match[0])
        else:
            branches.append(all_of_constraints(child_match))
    if not branches:
        return None
    return (any_of_constraints(branches),)


def _match_if_lc(lc, bundle) -> GenerationConstraint | None:
    """Try to match an ``ifL`` constraint to a known generation shape.

    Recognised patterns:
    - EOS-closure: ``ifL(is_before_rel, ifL(eos_x, eos_y))``
    - Conditional max non-EOS: ``ifL(existsAL(token), atMostAL(notL(eos), N))``

    Args:
        lc: An ``ifL`` DomiKnowS logical constraint node.
        bundle: :class:`~.encoder.GenerationBundle` for concept references.

    Returns:
        A :class:`~.constraints.GenerationConstraint` or ``None``.
    """
    # Check for the EOS-closure pattern first.
    if _is_eos_closure(lc, bundle):
        return no_token_after_eos()
    # Check for: ifL(existsAL(trigger_token), atMostAL(notL(eos), N))
    if len(lc.e) == 2:
        token = _exists_token(lc.e[0], bundle)
        max_count = _non_eos_at_most_count(lc.e[1], bundle)
        if token is not None and max_count is not None:
            return if_token_present_then_at_most_non_eos(token, max_count)
    return None


def _match_at_most_lc(lc, bundle) -> GenerationConstraint | None:
    """Try to match an ``atMostAL`` constraint to a known generation shape.

    Recognised patterns:
    - Max non-EOS: ``atMostAL(notL(eos), N)``
    - Forbidden token: ``atMostAL(token_value, 0)``

    Args:
        lc: An ``atMostAL`` DomiKnowS logical constraint node.
        bundle: :class:`~.encoder.GenerationBundle` for concept references.

    Returns:
        A :class:`~.constraints.GenerationConstraint` or ``None``.
    """
    # Try: atMostAL(notL(eos), N) → MaxNonEosConstraint
    max_count = _non_eos_at_most_count(lc, bundle)
    if max_count is not None:
        return max_non_eos(max_count)
    # Try: atMostAL(token_value, 0) → ForbiddenTokenConstraint
    token = _direct_token(lc.e, bundle)
    limit = _last_int(lc.e)
    if token is not None and limit == 0:
        return forbidden_token(token)
    return None


def _match_at_least_lc(lc, bundle) -> GenerationConstraint | None:
    """Try to match an ``atLeastAL`` constraint to a known generation shape.

    Recognised pattern:
    - Required token: ``atLeastAL(token_value, N)`` where N ≥ 1.

    Args:
        lc: An ``atLeastAL`` DomiKnowS logical constraint node.
        bundle: :class:`~.encoder.GenerationBundle` for concept references.

    Returns:
        A :class:`~.constraints.RequiredTokenConstraint` or ``None``.
    """
    token = _direct_token(lc.e, bundle)
    min_count = _last_int(lc.e)
    if token is not None and min_count is not None and min_count >= 1:
        return required_token(token, min_count)
    return None


def _match_exists_lc(lc, bundle) -> GenerationConstraint | None:
    """Try to match an ``existsAL`` constraint to a known generation shape.

    Recognised pattern:
    - Required token (min 1): ``existsAL(token_value)``.

    Args:
        lc: An ``existsAL`` DomiKnowS logical constraint node.
        bundle: :class:`~.encoder.GenerationBundle` for concept references.

    Returns:
        A :class:`~.constraints.RequiredTokenConstraint` with ``min_count=1``
        or ``None``.
    """
    token = _exists_token(lc, bundle)
    if token is not None:
        return required_token(token, 1)
    return None


def _is_eos_closure(lc, bundle) -> bool:
    """Return ``True`` if *lc* matches the EOS-closure ``ifL`` pattern.

    Expected shape::

        ifL(
            is_before_rel,          # pairwise ordering concept
            ...,
            ifL(
                eos_token_value(x), # EOS on first token
                eos_token_value(y), # EOS on second token
            ),
        )

    Args:
        lc: An ``ifL`` node to inspect.
        bundle: :class:`~.encoder.GenerationBundle` supplying
            ``is_before_rel`` and the EOS token string.

    Returns:
        ``True`` if the shape matches, ``False`` otherwise.
    """
    if len(lc.e) < 3:
        return False
    # First element must be the is_before_rel concept tuple.
    rel = _concept_tuple(lc.e[0])
    if rel is None or rel[0] is not bundle.is_before_rel:
        return False
    # Find a nested ifL sub-expression.
    nested = next((item for item in lc.e if getattr(item, "__class__", None).__name__ == "ifL"), None)
    if nested is None:
        return False
    # Both children of the nested ifL must resolve to the EOS token.
    tokens = [_token_from_tuple(item, bundle) for item in nested.e if _concept_tuple(item) is not None]
    return tokens == [bundle.vocabulary.eos_token, bundle.vocabulary.eos_token]


def _non_eos_at_most_count(lc, bundle) -> int | None:
    """Extract the count from an ``atMostAL(notL(eos), N)`` sub-expression.

    Returns ``None`` when *lc* does not match this pattern, so callers can
    use it as a boolean-style check.

    Args:
        lc: A DomiKnowS logical constraint node to inspect.
        bundle: :class:`~.encoder.GenerationBundle` for EOS token lookup.

    Returns:
        The integer cap *N* if the pattern matches, else ``None``.
    """
    if getattr(lc, "__class__", None).__name__ != "atMostAL":
        return None
    limit = _last_int(lc.e)
    if limit is None:
        return None
    # The first element must be notL wrapping the EOS token value.
    if not lc.e or getattr(lc.e[0], "__class__", None).__name__ != "notL":
        return None
    token = _direct_token(lc.e[0].e, bundle)
    if token == bundle.vocabulary.eos_token:
        return limit
    return None


def _exists_token(lc, bundle) -> str | None:
    """Extract the token from an ``existsAL(token_value)`` sub-expression.

    Args:
        lc: A DomiKnowS logical constraint node.
        bundle: :class:`~.encoder.GenerationBundle` for token lookup.

    Returns:
        The surface-form token string if *lc* is an ``existsAL`` containing
        exactly one token value, else ``None``.
    """
    if getattr(lc, "__class__", None).__name__ != "existsAL":
        return None
    return _direct_token(lc.e, bundle)


def _direct_token(elements: Iterable, bundle) -> str | None:
    """Return the single token referenced by *elements*, or ``None``.

    Extracts all concept tuples from *elements*, resolves them to token
    strings via :func:`_token_from_tuple`, and returns the unique result.
    Returns ``None`` if zero or more than one distinct token is found.

    Args:
        elements: Iterable of sub-expressions from a logical constraint node.
        bundle: :class:`~.encoder.GenerationBundle` for token lookup.

    Returns:
        A surface-form token string, or ``None``.
    """
    tokens = [_token_from_tuple(item, bundle) for item in elements if _concept_tuple(item) is not None]
    # Filter out items that didn't resolve to a known token.
    tokens = [token for token in tokens if token is not None]
    if len(tokens) != 1:
        return None
    return tokens[0]


def _token_from_tuple(item, bundle) -> str | None:
    """Resolve a concept 4-tuple to its surface-form token string.

    A concept tuple has the shape ``(concept, name, label, cardinality)``.
    The function checks that *concept* is the ``generated_token`` enum concept
    and then maps *label* (an integer index stored as a string) back to the
    surface-form token via the vocabulary.

    Args:
        item: Candidate 4-tuple from a logical constraint expression.
        bundle: :class:`~.encoder.GenerationBundle` providing
            ``generated_token`` and the vocabulary.

    Returns:
        The surface-form token string, or ``None`` if *item* is not a valid
        concept tuple for ``generated_token``.
    """
    concept_tuple = _concept_tuple(item)
    if concept_tuple is None:
        return None
    concept, _name, label, _cardinality = concept_tuple
    # Only enum values belonging to generated_token carry vocabulary labels.
    if concept is not bundle.generated_token or label is None:
        return None
    try:
        return bundle.vocabulary.token_for_label(int(label))
    except (ValueError, IndexError):
        return None


def _concept_tuple(item):
    """Return *item* if it is a valid concept 4-tuple, else ``None``.

    DomiKnowS concept references inside logical constraints are stored as
    plain tuples of the form ``(concept, name, label, cardinality)``.
    This guard ensures downstream helpers only operate on that shape.

    Args:
        item: Any object from a logical constraint expression.

    Returns:
        *item* if it is a 4-tuple, else ``None``.
    """
    if isinstance(item, tuple) and len(item) == 4:
        return item
    return None


def _last_int(elements: Iterable) -> int | None:
    """Return the last integer found in *elements*, or ``None``.

    DomiKnowS ``atMostAL`` / ``atLeastAL`` nodes store their numeric
    bound as a plain Python ``int`` appended to their ``e`` list.  This
    helper extracts it without caring about position.

    Args:
        elements: Iterable of sub-expressions from a logical constraint node.

    Returns:
        The last ``int`` in *elements*, or ``None`` if none are present.
    """
    ints = [item for item in elements if isinstance(item, int)]
    return int(ints[-1]) if ints else None


def _is_generation_relevant(lc, bundle) -> bool:
    """Return ``True`` if *lc* references any generation-specific concept.

    Used to determine whether an unrecognised logical constraint should
    trigger the ``on_unsupported`` policy.  Constraints that do not
    reference ``generated_token`` or ``is_before_rel`` are silently ignored
    because they belong to unrelated parts of the graph.

    Args:
        lc: A DomiKnowS logical constraint node.
        bundle: :class:`~.encoder.GenerationBundle` providing the concept
            references to look for.

    Returns:
        ``True`` if at least one concept tuple in the constraint tree
        references a generation concept.
    """
    for item in _walk_lc(lc):
        concept_tuple = _concept_tuple(item)
        if concept_tuple is None:
            continue
        concept = concept_tuple[0]
        # Any reference to generated_token or is_before_rel counts.
        if concept is bundle.generated_token or concept is bundle.is_before_rel:
            return True
    return False


def _walk_lc(lc):
    """Yield *lc* and all of its descendants in depth-first order.

    DomiKnowS logical constraints store their sub-expressions in an ``e``
    attribute.  This recursive generator traverses the full expression tree
    so that :func:`_is_generation_relevant` can inspect every node.

    Args:
        lc: Root logical constraint node to walk.

    Yields:
        The root node followed by each descendant node.
    """
    yield lc
    for item in getattr(lc, "e", ()):
        yield item
        # Recurse into sub-expressions that themselves have children.
        if hasattr(item, "e"):
            yield from _walk_lc(item)


def _handle_unsupported(lc_name: str, lc, on_unsupported: str) -> None:
    """Emit a warning or raise an error for an unrecognised generation constraint.

    Args:
        lc_name: The string key of the constraint in ``graph.logicalConstrains``.
        lc: The logical constraint node.
        on_unsupported: One of ``"error"``, ``"warn"``, or ``"ignore"``.
            ``"error"`` raises :class:`ValueError`;
            ``"warn"`` emits a :class:`RuntimeWarning`;
            ``"ignore"`` does nothing.
    """
    message = (
        f"DomiKnowS logical constraint {lc_name} ({lc.__class__.__name__}) references "
        "generation concepts but is not supported by generation DFA discovery"
    )
    if on_unsupported == "error":
        raise ValueError(message)
    if on_unsupported == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
