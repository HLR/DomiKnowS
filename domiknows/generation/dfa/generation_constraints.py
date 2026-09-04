"""Reusable DomiKnowS graph helpers for DFA-enforceable generation constraints.

These helpers write DomiKnowS logical constraints into a
:class:`~.dfa.encoder.GenerationGraphContext`.  DFA compilation is graph-only:
call :func:`domiknows.generation.dfa.constraints_to_dfa_from_graph` after the
graph is built to compile the supported graph constraints into a DFA.

The functions in this module emit the exact LC shapes that
:mod:`.graph_discovery` recognises, so a constraint written here will round-trip
through DFA compilation without additional pattern hints.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def _required_items(required_tokens: Mapping[str, int] | Iterable[str]) -> tuple[tuple[str, int], ...]:
    """Normalise required-token input into ``(token, min_count)`` pairs."""
    if isinstance(required_tokens, Mapping):
        return tuple((token, int(min_count)) for token, min_count in required_tokens.items())
    return tuple((token, 1) for token in required_tokens)


def _conditional_items(conditional_max_non_eos: Mapping[str, int] | Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    items = conditional_max_non_eos.items() if isinstance(conditional_max_non_eos, Mapping) else conditional_max_non_eos
    return tuple((token, int(max_count)) for token, max_count in items)


def apply_eos_closure_constraint(ctx):
    """Apply the standard no-token-after-EOS graph constraint.

    Emits ``ifL(is_before_rel, ifL(eos(x), eos(y)))`` so that once EOS appears
    at position *x*, every later position *y* must also be EOS.
    """
    from domiknows.graph.logicalConstrain import ifL

    before = ctx.is_before_rel("before")
    return ifL(
        before,
        ifL(
            ctx.token_value(
                ctx.vocabulary.eos_token,
                "x",
                path=("before", ctx.first_token),
            ),
            ctx.token_value(
                ctx.vocabulary.eos_token,
                "y",
                path=("before", ctx.second_token),
            ),
        ),
    )


def apply_max_non_eos_constraint(ctx, max_count: int):
    """Apply a graph constraint limiting the number of non-EOS tokens.

    Emits ``atMostAL(non_eos(x), max_count)``.
    """
    from domiknows.graph.logicalConstrain import atMostAL

    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    return atMostAL(ctx.non_eos("x"), max_count)


def apply_required_token_constraint(ctx, token: str, min_count: int = 1):
    """Apply a graph constraint requiring *token* at least *min_count* times.

    Emits ``atLeastAL(token_value(token, x), min_count)``.
    """
    from domiknows.graph.logicalConstrain import atLeastAL

    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    return atLeastAL(ctx.token_value(token, "x"), min_count)


def apply_required_token_constraints(
    ctx,
    required_tokens: Mapping[str, int] | Iterable[str],
) -> tuple[object, ...]:
    """Apply one required-token graph constraint per requested token."""
    return tuple(apply_required_token_constraint(ctx, token, min_count) for token, min_count in _required_items(required_tokens))


def apply_forbidden_token_constraint(ctx, token: str):
    """Apply a graph constraint forbidding *token*.

    Emits ``atMostAL(token_value(token, x), 0)``.
    """
    from domiknows.graph.logicalConstrain import atMostAL

    return atMostAL(ctx.token_value(token, "x"), 0)


def apply_forbidden_token_constraints(ctx, tokens: Iterable[str]) -> tuple[object, ...]:
    """Apply one forbidden-token graph constraint per token."""
    return tuple(apply_forbidden_token_constraint(ctx, token) for token in tokens)


def apply_conditional_max_non_eos_constraint(ctx, token: str, max_count: int):
    """Apply ``if token appears then at most max_count non-EOS``.

    Emits ``ifL(existsAL(token_value(token, x)), atMostAL(non_eos(y), max_count))``.
    """
    from domiknows.graph.logicalConstrain import atMostAL, existsAL, ifL

    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    return ifL(
        existsAL(ctx.token_value(token, "x")),
        atMostAL(ctx.non_eos("y"), max_count),
    )


def apply_conditional_max_non_eos_constraints(
    ctx,
    conditional_max_non_eos: Mapping[str, int] | Iterable[tuple[str, int]],
) -> tuple[object, ...]:
    """Apply one conditional max-length graph constraint per trigger token."""
    return tuple(apply_conditional_max_non_eos_constraint(ctx, token, max_count) for token, max_count in _conditional_items(conditional_max_non_eos))


def apply_all_constraints(
    ctx,
    *,
    include_eos_closure: bool = True,
    max_non_eos_count: int | None = None,
    required_tokens: Mapping[str, int] | Iterable[str] = (),
    forbidden_tokens: Iterable[str] = (),
    conditional_max_non_eos: Mapping[str, int] | Iterable[tuple[str, int]] = (),
) -> tuple[object, ...]:
    """Apply the common generation graph constraints and return created LCs."""
    constraints: list[object] = []
    if include_eos_closure:
        constraints.append(apply_eos_closure_constraint(ctx))
    if max_non_eos_count is not None:
        constraints.append(apply_max_non_eos_constraint(ctx, max_non_eos_count))
    constraints.extend(apply_required_token_constraints(ctx, required_tokens))
    constraints.extend(apply_forbidden_token_constraints(ctx, forbidden_tokens))
    constraints.extend(apply_conditional_max_non_eos_constraints(ctx, conditional_max_non_eos))
    return tuple(constraints)
