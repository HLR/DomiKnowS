"""
Reusable DomiKnowS constraint library for generation graphs built with
GenerationEncoder.

The functions mirror domiknows.graph.visual.visual_constraints: they are
generic, parameterized by a graph context, and avoid task-local token
boilerplate. Each helper also returns the corresponding GenerationConstraint
object so the same specification can be compiled to a DFA when supported.

Typical usage:

    encoder = GenerationEncoder(vocab, eos_token="<eos>", tokenizer=tokenizer)
    graph, bundle = encoder.build_graph()
    with graph:
        constraints = apply_all_constraints(
            bundle.context,
            max_non_eos_count=8,
            required_tokens=["A"],
            forbidden_tokens=["B"],
        )
    dfa = constraints_to_dfa(constraints, bundle.vocabulary)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .constraints import (
    ConditionalMaxNonEosConstraint,
    EosClosureConstraint,
    ForbiddenTokenConstraint,
    GenerationConstraint,
    MaxNonEosConstraint,
    RequiredTokenConstraint,
    forbidden_token,
    if_token_present_then_at_most_non_eos,
    max_non_eos,
    no_token_after_eos,
    required_token,
)


def _apply(ctx, constraint: GenerationConstraint) -> GenerationConstraint:
    """Compile *constraint* into *ctx* and return it.

    Validates that the constraint supports DomiKnowS compilation before
    calling :meth:`~.constraints.GenerationConstraint.apply_domiknows`.
    This keeps all public helper functions free of the same boilerplate check.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` (or any object
            implementing the :class:`~.constraints.DomiKnowSGenerationContext`
            protocol) that the constraint writes logical expressions into.
        constraint: The :class:`~.constraints.GenerationConstraint` to
            compile.  Must have ``supports_domiknows = True``.

    Returns:
        *constraint* unchanged (for transparent pass-through in callers).

    Raises:
        ValueError: If ``constraint.supports_domiknows`` is ``False``.
    """
    if not constraint.supports_domiknows:
        raise ValueError(f"{constraint.__class__.__name__} does not support DomiKnowS constraints")
    constraint.apply_domiknows(ctx)
    return constraint


def _required_items(required_tokens: Mapping[str, int] | Iterable[str]) -> tuple[tuple[str, int], ...]:
    """Normalise the *required_tokens* argument to a sequence of (token, min_count) pairs.

    Accepts two input shapes:
    - A :class:`~collections.abc.Mapping` from token string to minimum count
      (e.g. ``{"A": 2, "B": 1}``).
    - An iterable of plain token strings, in which case each token's minimum
      count defaults to ``1``.

    Args:
        required_tokens: Either a mapping of ``{token: min_count}`` or an
            iterable of token strings.

    Returns:
        A tuple of ``(token, min_count)`` pairs ready for iteration.
    """
    if isinstance(required_tokens, Mapping):
        # Mapping path: extract explicit minimum counts.
        return tuple((token, int(min_count)) for token, min_count in required_tokens.items())
    # Iterable path: default every token to a minimum count of 1.
    return tuple((token, 1) for token in required_tokens)


def apply_eos_closure_constraint(ctx) -> EosClosureConstraint:
    """Compile an EOS-closure constraint into *ctx*.

    Once an EOS token is produced, every subsequent token must also be EOS.
    This is the standard sequence-termination invariant for autoregressive
    generation.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write the
            DomiKnowS logical expression into.

    Returns:
        The compiled :class:`~.constraints.EosClosureConstraint` instance.
    """
    return _apply(ctx, no_token_after_eos())


def apply_max_non_eos_constraint(ctx, max_count: int) -> MaxNonEosConstraint:
    """Compile a max-non-EOS-count constraint into *ctx*.

    Limits the total number of non-EOS tokens that may be generated.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write the
            DomiKnowS logical expression into.
        max_count: Maximum number of non-EOS tokens allowed (≥ 0).

    Returns:
        The compiled :class:`~.constraints.MaxNonEosConstraint` instance.
    """
    return _apply(ctx, max_non_eos(max_count))


def apply_required_token_constraint(ctx, token: str, min_count: int = 1) -> RequiredTokenConstraint:
    """Compile a required-token constraint into *ctx*.

    Requires *token* to appear at least *min_count* times in the output.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write the
            DomiKnowS logical expression into.
        token: Surface-form token that must appear.
        min_count: Minimum required occurrences (≥ 1).  Defaults to ``1``.

    Returns:
        The compiled :class:`~.constraints.RequiredTokenConstraint` instance.
    """
    return _apply(ctx, required_token(token, min_count=min_count))


def apply_required_token_constraints(
    ctx,
    required_tokens: Mapping[str, int] | Iterable[str],
) -> tuple[RequiredTokenConstraint, ...]:
    """Compile multiple required-token constraints into *ctx*.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write DomiKnowS
            expressions into.
        required_tokens: Either a mapping ``{token: min_count}`` or an
            iterable of token strings (each defaulting to ``min_count=1``).

    Returns:
        Tuple of compiled :class:`~.constraints.RequiredTokenConstraint`
        instances, one per token.
    """
    return tuple(apply_required_token_constraint(ctx, token, min_count) for token, min_count in _required_items(required_tokens))


def apply_forbidden_token_constraint(ctx, token: str) -> ForbiddenTokenConstraint:
    """Compile a forbidden-token constraint into *ctx*.

    Prevents *token* from appearing anywhere in the generated output.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write the
            DomiKnowS logical expression into.
        token: Surface-form token to forbid.

    Returns:
        The compiled :class:`~.constraints.ForbiddenTokenConstraint` instance.
    """
    return _apply(ctx, forbidden_token(token))


def apply_forbidden_token_constraints(ctx, tokens: Iterable[str]) -> tuple[ForbiddenTokenConstraint, ...]:
    """Compile multiple forbidden-token constraints into *ctx*.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write DomiKnowS
            expressions into.
        tokens: Iterable of surface-form tokens to forbid.

    Returns:
        Tuple of compiled :class:`~.constraints.ForbiddenTokenConstraint`
        instances, one per token.
    """
    return tuple(apply_forbidden_token_constraint(ctx, token) for token in tokens)


def apply_conditional_max_non_eos_constraint(
    ctx,
    token: str,
    max_count: int,
) -> ConditionalMaxNonEosConstraint:
    """Compile a conditional max-non-EOS constraint into *ctx*.

    If *token* appears anywhere in the output, limits the total number of
    non-EOS tokens to *max_count*.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write the
            DomiKnowS logical expression into.
        token: Trigger token whose presence activates the cap.
        max_count: Maximum non-EOS tokens allowed once *token* appears.

    Returns:
        The compiled
        :class:`~.constraints.ConditionalMaxNonEosConstraint` instance.
    """
    return _apply(ctx, if_token_present_then_at_most_non_eos(token, max_count))


def apply_conditional_max_non_eos_constraints(
    ctx,
    rules: Mapping[str, int] | Iterable[tuple[str, int]],
) -> tuple[ConditionalMaxNonEosConstraint, ...]:
    """Compile multiple token-triggered max-length constraints into *ctx*.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write DomiKnowS
            expressions into.
        rules: Either a mapping ``{trigger_token: max_count}`` or an iterable
            of ``(trigger_token, max_count)`` pairs.

    Returns:
        Tuple of compiled
        :class:`~.constraints.ConditionalMaxNonEosConstraint` instances.
    """
    # Normalise to an iterable of (token, max_count) pairs.
    items = rules.items() if isinstance(rules, Mapping) else rules
    return tuple(apply_conditional_max_non_eos_constraint(ctx, token, int(max_count)) for token, max_count in items)


def default_generation_constraints(
    *,
    eos_closure: bool = True,
    max_non_eos_count: int | None = None,
    required_tokens: Mapping[str, int] | Iterable[str] = (),
    forbidden_tokens: Iterable[str] = (),
    conditional_max_non_eos: Mapping[str, int] | Iterable[tuple[str, int]] = (),
) -> tuple[GenerationConstraint, ...]:
    """Build a collection of generation constraints without applying them to a graph.

    This is the *graph-free* counterpart of :func:`apply_all_constraints`.
    The returned constraints can be compiled to a DFA via
    :func:`~.constraints.constraints_to_dfa` or applied later with
    :func:`apply_constraint_objects`.

    Args:
        eos_closure: If ``True`` (default), include an
            :class:`~.constraints.EosClosureConstraint` (no non-EOS after
            first EOS).
        max_non_eos_count: If not ``None``, include a
            :class:`~.constraints.MaxNonEosConstraint` with this cap.
        required_tokens: Tokens that must appear; either a mapping
            ``{token: min_count}`` or a plain iterable (``min_count=1``).
        forbidden_tokens: Tokens that must not appear.
        conditional_max_non_eos: Token-triggered max-length rules; either a
            mapping ``{trigger: max_count}`` or an iterable of
            ``(trigger, max_count)`` pairs.

    Returns:
        Tuple of :class:`~.constraints.GenerationConstraint` objects in
        declaration order.
    """
    constraints: list[GenerationConstraint] = []
    if eos_closure:
        constraints.append(no_token_after_eos())
    if max_non_eos_count is not None:
        constraints.append(max_non_eos(max_non_eos_count))
    # Add one RequiredTokenConstraint per (token, min_count) pair.
    constraints.extend(required_token(token, min_count) for token, min_count in _required_items(required_tokens))
    # Add one ForbiddenTokenConstraint per forbidden token.
    constraints.extend(forbidden_token(token) for token in forbidden_tokens)
    # Normalise conditional rules and add one ConditionalMaxNonEosConstraint each.
    items = conditional_max_non_eos.items() if isinstance(conditional_max_non_eos, Mapping) else conditional_max_non_eos
    constraints.extend(if_token_present_then_at_most_non_eos(token, int(max_count)) for token, max_count in items)
    return tuple(constraints)


def apply_constraint_objects(ctx, constraints: Sequence[GenerationConstraint]) -> tuple[GenerationConstraint, ...]:
    """Apply pre-built :class:`~.constraints.GenerationConstraint` objects to *ctx*.

    Filters out constraints that do not support DomiKnowS compilation
    (``supports_domiknows = False``) and compiles the rest via :func:`_apply`.
    Useful when constraints were constructed separately (e.g. by
    :func:`default_generation_constraints`) and need to be registered into
    a graph context after the fact.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write DomiKnowS
            expressions into.
        constraints: Sequence of :class:`~.constraints.GenerationConstraint`
            objects to apply.  Constraints with ``supports_domiknows = False``
            are silently skipped.

    Returns:
        Tuple of the constraints that were successfully compiled (i.e. those
        with ``supports_domiknows = True``).
    """
    return tuple(_apply(ctx, constraint) for constraint in constraints if constraint.supports_domiknows)


def apply_all_constraints(
    ctx,
    *,
    eos_closure: bool = True,
    max_non_eos_count: int | None = None,
    required_tokens: Mapping[str, int] | Iterable[str] = (),
    forbidden_tokens: Iterable[str] = (),
    conditional_max_non_eos: Mapping[str, int] | Iterable[tuple[str, int]] = (),
) -> tuple[GenerationConstraint, ...]:
    """Build and compile a full generation-constraint bundle into *ctx*.

    Combines :func:`default_generation_constraints` (which builds the
    constraint objects) with :func:`apply_constraint_objects` (which compiles
    them into the graph context).  This is the primary high-level entry point
    for registering all constraints in one call inside a ``with Graph(...)``
    block.

    Args:
        ctx: A :class:`~.encoder.GenerationGraphContext` to write DomiKnowS
            expressions into.
        eos_closure: If ``True`` (default), apply an EOS-closure constraint.
        max_non_eos_count: If not ``None``, apply a max-non-EOS-count
            constraint with this cap.
        required_tokens: Tokens that must appear; either a mapping
            ``{token: min_count}`` or a plain iterable (``min_count=1``).
        forbidden_tokens: Tokens that must not appear.
        conditional_max_non_eos: Token-triggered max-length rules; either a
            mapping ``{trigger: max_count}`` or an iterable of
            ``(trigger, max_count)`` pairs.

    Returns:
        Tuple of :class:`~.constraints.GenerationConstraint` objects that were
        compiled into *ctx* (``supports_domiknows = True`` ones only).
    """
    # Build constraint objects first, then apply them to the context.
    constraints = default_generation_constraints(
        eos_closure=eos_closure,
        max_non_eos_count=max_non_eos_count,
        required_tokens=required_tokens,
        forbidden_tokens=forbidden_tokens,
        conditional_max_non_eos=conditional_max_non_eos,
    )
    return apply_constraint_objects(ctx, constraints)
