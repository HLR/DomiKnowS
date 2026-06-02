"""Three head logical constraints exercising the LC -> DFA pipeline.

Each constraint targets a different facet of
:mod:`domiknows.generation.dfa._lc_normalize`:

LC #1 -- "Polite conversation"
    Deeply nested ``andL`` containing a path-aware EOS-closure (``ifL`` over
    ``is_before_rel`` with ``first_token`` / ``second_token`` traversal), a
    plain ``atMostAL`` count cap, a ``notL(andL(existsAL, existsAL))`` clause
    that the normalizer rewrites via De Morgan's into an ``orL`` of
    ``_ForbiddenLeaf`` nodes, and a plain ``orL`` over ``existsAL`` clauses.

LC #2 -- "Heterogeneous andL salvage"
    ``andL`` of a regular leaf and an inner ``andL`` whose direct children are
    raw concept tuples (no LC class).  The normalizer drops the irregular
    sibling and surfaces it via ``NormalForm.irregular_children`` so the
    matcher's ``on_unsupported`` policy still warns / errors, while the
    regular leaf still contributes its DFA.

LC #3 -- "Double-negation sanity"
    ``notL(notL(atMostAL(B, 1)))`` -- exercised purely to show the
    double-negation rewrite produces the same DFA as the bare atom.
"""
from __future__ import annotations

from domiknows.graph.logicalConstrain import (
    andL,
    atMostAL,
    existsAL,
    ifL,
    notL,
    orL,
)


def apply_constraints(graph, bundle) -> None:
    """Register the three head LCs on *graph*.

    Reads context helpers (``token_value``, ``is_before_rel``, ``first_token``,
    ``second_token``) from the bundle's :class:`GenerationGraphContext`.
    """
    ctx = bundle.context
    with graph:
        # LC #1 -- Polite conversation (the nested + path one).
        andL(
            # EOS-closure: once the END token is emitted at any position p1, every
            # position p2 that follows p1 (via ``is_before_rel``) must also emit
            # END.  This is the canonical path-aware shape recognised by
            # ``_match_if_lc`` as an EOS-closure DFA.
            ifL(
                ctx.is_before_rel("before"),
                ifL(
                    ctx.token_value(
                        "END", "x", path=("before", ctx.first_token)
                    ),
                    ctx.token_value(
                        "END", "y", path=("before", ctx.second_token)
                    ),
                ),
            ),
            # At most one B in the whole sequence.
            atMostAL(ctx.token_value("B", "x"), 1),
            # not(existsA and existsC): cannot contain both A and C.  The normalizer applies
            # De Morgan's to push the ``notL`` into the children, then
            # collapses ``notL(existsAL(t))`` into a ``_ForbiddenLeaf(t)``,
            # producing a smaller DFA than the literal complement-of-product.
            notL(
                andL(
                    existsAL(ctx.token_value("A", "x")),
                    existsAL(ctx.token_value("C", "y")),
                ),
            ),
            # existsA or existsC: at least one of A or C must appear.  Combined with the
            # previous clause, the constraint becomes "exactly one of A or C".
            orL(
                existsAL(ctx.token_value("A", "x")),
                existsAL(ctx.token_value("C", "y")),
            ),
        )

        # LC #2 -- Heterogeneous andL salvage.  ``D`` is forbidden via a
        # regular leaf; the inner ``andL`` of raw concept tuples is structurally
        # irregular (no LC class) and is dropped by the normalizer's
        # ``_split_regular_irregular_andL`` with a warning.
        andL(
            atMostAL(ctx.token_value("D", "x"), 0),
            andL(
                ctx.token_value("A", "y"),
                ctx.token_value("B", "y"),
            ),
        )

        # LC #3 -- Double-negation sanity.  After normalization, this is
        # indistinguishable from a bare ``atMostAL(B, 1)`` (which LC #1 already
        # registers); intersecting two identical-language DFAs is a no-op.
        notL(notL(atMostAL(ctx.token_value("B", "x"), 1)))


CONSTRAINT_DESCRIPTIONS = (
    (
        "polite_conversation",
        "andL of EOS-closure (path-aware ifL), atMostAL(B, 1), notL(andL(exists A, exists C)), and orL(exists A, exists C).",
    ),
    (
        "heterogeneous_salvage",
        "andL(atMostAL(D, 0), andL(token_value(A), token_value(B))) -- the inner andL of concept tuples is salvaged off as irregular.",
    ),
    (
        "double_negation_sanity",
        "notL(notL(atMostAL(B, 1))) -- collapses to the bare atom after normalize.",
    ),
)
