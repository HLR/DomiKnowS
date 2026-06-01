"""LC AST (Abstract Syntax Tree) normalization for the LogicalConstraintsToDFAPipeline.

Recursively normalizing and flattening the logical structure before pattern matching:

* push ``notL`` to leaves via De Morgan's laws
* eliminate double negations
* flatten same-op chains (``andL(andL(a, b), c) → andL(a, b, c)``)
* deduplicate identical atoms
* constant-fold trivial contradictions and tautologies
* identify the set of basic constraint atoms

This module produces a small mirror AST (``_AndNode`` / ``_OrNode`` /
``_NotNode``) for the boolean operators while leaving the original DomiKnowS
LC objects intact at the leaves.  Mirror nodes carry a ``_kind`` attribute
that the matcher in :mod:`graph_discovery` reads via :func:`kind`, so the
existing per-class ``_match_*_lc`` helpers keep working without changes to
their bodies.

Two synthetic node types fall outside the boolean operators:

* ``_ForbiddenLeaf(token)`` is produced by the ``notL(existsAL(t))`` →
  forbidden-token specialization.  It is the smallest-DFA encoding of that
  shape and is dispatched by the matcher via its ``_kind`` ``"_forbidden_token"``.
* ``_TopNode`` / ``_BottomNode`` represent the constant-folded language
  (compiled to :func:`~.accept_all_dfa` / :func:`~.empty_dfa` by the matcher).

Heterogeneous-``andL`` salvage is communicated through
``NormalForm.irregular_children``: the caller (matcher) can still raise / warn
per its ``on_unsupported`` policy on those LCs, but the regular siblings stay
compiled into the final product DFA instead of being dropped wholesale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _AndNode:
    e: tuple
    _kind: str = "andL"


@dataclass(frozen=True)
class _OrNode:
    e: tuple
    _kind: str = "orL"


@dataclass(frozen=True)
class _NotNode:
    e: tuple  # exactly one child
    _kind: str = "notL"


@dataclass(frozen=True)
class _TopNode:
    """Constant-true subtree (compiled to ``accept_all_dfa``)."""

    e: tuple = ()
    _kind: str = "_top"


@dataclass(frozen=True)
class _BottomNode:
    """Constant-false subtree (compiled to ``empty_dfa``)."""

    e: tuple = ()
    _kind: str = "_bottom"


@dataclass(frozen=True)
class _ForbiddenLeaf:
    """Specialized leaf produced by ``notL(existsAL(token))``.

    Encodes the same language as ``atMostAL(token_value, 0)`` and is compiled
    via :func:`~.forbidden_token_dfa` — strictly smaller than
    ``complement_dfa(required_token_dfa(token, 1))``.
    """

    token: str
    e: tuple = ()
    _kind: str = "_forbidden_token"


@dataclass(frozen=True)
class NormalForm:
    """Result of :func:`normalize_lc`."""

    tree: Any
    atoms: tuple
    is_constant: str | None
    irregular_children: tuple


def kind(lc) -> str:
    """Return the dispatch class name for *lc*.

    Mirror nodes expose ``_kind``; original LC nodes fall back to
    ``lc.__class__.__name__``.  This is the single hook that lets
    :func:`graph_discovery._match_lc_many` consume both kinds.
    """
    # Mirror nodes use _kind; original nodes use their class name.
    return getattr(lc, "_kind", None) or lc.__class__.__name__


def normalize_lc(lc, *, bundle) -> NormalForm:
    """Normalize *lc* and return a :class:`NormalForm`.

    The original LC node is not mutated; the returned tree is a fresh mirror
    structure suitable for pattern matching by the downstream matcher.
    """
    # Produce normalized tree plus metadata in one recursive pass.
    tree, atoms, irregular = _normalize(lc, bundle)
    is_constant = None
    # Track whether normalization collapsed to a language constant.
    if isinstance(tree, _TopNode):
        is_constant = "top"
    elif isinstance(tree, _BottomNode):
        is_constant = "bottom"
    return NormalForm(
        tree=tree,
        atoms=tuple(sorted(atoms, key=repr)),
        is_constant=is_constant,
        irregular_children=tuple(irregular),
    )


# --------------------------------------------------------------------------- #
# Recursive normalizer                                                        #
# --------------------------------------------------------------------------- #


def _normalize(lc, bundle):
    """Return ``(tree, atoms, irregular_children)`` for *lc*."""
    op = kind(lc)
    # Dispatch boolean operators to specialized normalizers.
    if op == "andL":
        return _normalize_and(lc, bundle)
    if op == "orL":
        return _normalize_or(lc, bundle)
    if op == "notL":
        return _normalize_not(lc, bundle)
    if op == "nandL":
        return _normalize_negated_aggregate(lc, bundle, base_op="andL")
    if op == "norL":
        return _normalize_negated_aggregate(lc, bundle, base_op="orL")
    # Other LC types (atMostAL, existsAL, ifL, …) and original leaves pass
    # through unchanged.  Build a single-element atom set so the caller can
    # dedup later.
    key = _canonical_key(lc, bundle)
    atoms = frozenset({key}) if key is not None else frozenset()
    return lc, atoms, ()


def _normalize_and(lc, bundle):
    children: list = []
    atoms: set = set()
    irregular: list = []
    # Normalize children, then apply and-specific simplifications.
    for child in getattr(lc, "e", ()):
        norm_child, child_atoms, child_irregular = _normalize(child, bundle)
        irregular.extend(child_irregular)
        atoms.update(child_atoms)
        if isinstance(norm_child, _TopNode):
            # AND identity: drop a True child.
            continue
        if isinstance(norm_child, _BottomNode):
            # Short-circuit to bottom; later atoms are unreachable.
            return _BottomNode(), atoms, irregular
        if isinstance(norm_child, _AndNode):
            # Flatten nested andL.
            children.extend(norm_child.e)
            continue
        children.append(norm_child)
    # Remove duplicates and detect X ∧ ¬X contradictions.
    children, contradiction = _dedup_with_contradiction(children, bundle)
    if contradiction:
        return _BottomNode(), atoms, irregular
    # Keep compilable children in-tree, report unsupported generation-relevant ones.
    children, irregular_subset = _split_regular_irregular_andL(children, bundle)
    irregular.extend(irregular_subset)
    # Canonical final shape for conjunctions.
    if not children:
        return _TopNode(), atoms, irregular
    if len(children) == 1:
        return children[0], atoms, irregular
    return _AndNode(tuple(children)), atoms, irregular


def _normalize_or(lc, bundle):
    children: list = []
    atoms: set = set()
    irregular: list = []
    # Normalize children, then apply or-specific simplifications.
    for child in getattr(lc, "e", ()):
        norm_child, child_atoms, child_irregular = _normalize(child, bundle)
        irregular.extend(child_irregular)
        atoms.update(child_atoms)
        if isinstance(norm_child, _BottomNode):
            # OR identity: drop a False child.
            continue
        if isinstance(norm_child, _TopNode):
            return _TopNode(), atoms, irregular
        if isinstance(norm_child, _OrNode):
            children.extend(norm_child.e)
            continue
        children.append(norm_child)
    # Remove duplicates and detect X ∨ ¬X tautologies.
    children, tautology = _dedup_with_tautology(children, bundle)
    if tautology:
        return _TopNode(), atoms, irregular
    # Canonical final shape for disjunctions.
    if not children:
        return _BottomNode(), atoms, irregular
    if len(children) == 1:
        return children[0], atoms, irregular
    return _OrNode(tuple(children)), atoms, irregular


def _normalize_not(lc, bundle):
    # notL is unary; malformed nodes are deferred to matcher diagnostics.
    children = list(getattr(lc, "e", ()))
    if len(children) != 1:
        # Malformed notL — leave it as-is so the matcher can report it.
        return lc, frozenset(), ()
    # Normalize inner subtree first, then push negation structurally.
    inner, inner_atoms, inner_irregular = _normalize(children[0], bundle)
    return _push_negation(inner, bundle, inner_atoms, inner_irregular)


def _normalize_negated_aggregate(lc, bundle, *, base_op: str):
    """Normalize ``nandL`` (``¬(A ∧ B)``) and ``norL`` (``¬(A ∨ B)``)."""
    # Re-express as base aggregate first, then push a single outer negation.
    synthetic = _AndNode(tuple(getattr(lc, "e", ()))) if base_op == "andL" else _OrNode(tuple(getattr(lc, "e", ())))
    base_tree, atoms, irregular = _normalize(synthetic, bundle)
    return _push_negation(base_tree, bundle, atoms, irregular)


def _push_negation(inner, bundle, inner_atoms, inner_irregular):
    """Return the negation of *inner* with De Morgan / double-neg elim applied."""
    # Constant negation cases.
    if isinstance(inner, _TopNode):
        return _BottomNode(), inner_atoms, inner_irregular
    if isinstance(inner, _BottomNode):
        return _TopNode(), inner_atoms, inner_irregular
    if isinstance(inner, _NotNode):
        # ¬¬X = X
        return inner.e[0], inner_atoms, inner_irregular
    if isinstance(inner, _AndNode):
        # ¬(A ∧ B) = ¬A ∨ ¬B — flip and re-normalize via _normalize_or so
        # the resulting OR is itself flattened/deduped.
        flipped = tuple(_push_negation(child, bundle, frozenset(), ())[0] for child in inner.e)
        return _normalize_or(_OrNode(flipped), bundle)
    if isinstance(inner, _OrNode):
        # ¬(A ∨ B) = ¬A ∧ ¬B
        flipped = tuple(_push_negation(child, bundle, frozenset(), ())[0] for child in inner.e)
        return _normalize_and(_AndNode(flipped), bundle)
    # Specialization: notL(existsAL(t)) → forbidden(t)
    forbidden = _try_not_exists_to_forbidden(inner, bundle)
    if forbidden is not None:
        key = _canonical_key(forbidden, bundle)
        return forbidden, frozenset({key}) if key is not None else frozenset(), inner_irregular
    # Default: wrap as _NotNode for the matcher to complement at the DFA level.
    return _NotNode((inner,)), inner_atoms, inner_irregular


# --------------------------------------------------------------------------- #
# Dedup / contradiction detection                                             #
# --------------------------------------------------------------------------- #


def _dedup_with_contradiction(children, bundle):
    """Drop duplicate atoms; return ``(deduped, True)`` if ``X ∧ ¬X`` is found."""
    out: list = []
    seen_keys: dict = {}
    for child in children:
        # Canonical key drives dedup and contradiction checks.
        key = _canonical_key(child, bundle)
        if key is None:
            out.append(child)
            continue
        if key in seen_keys:
            # Exact duplicate atom.
            continue
        negation = _negation_canonical_key(child, bundle)
        if negation is not None and negation in seen_keys:
            return out, True
        # The inverse check: any previously-seen ¬X paired with this X?
        if _matches_existing_negation(key, seen_keys):
            return out, True
        seen_keys[key] = child
        out.append(child)
    return out, False


def _dedup_with_tautology(children, bundle):
    """Drop duplicates; return ``(deduped, True)`` if ``X ∨ ¬X`` is found."""
    out: list = []
    seen_keys: dict = {}
    for child in children:
        # Canonical key drives dedup and tautology checks.
        key = _canonical_key(child, bundle)
        if key is None:
            out.append(child)
            continue
        if key in seen_keys:
            # Exact duplicate atom.
            continue
        negation = _negation_canonical_key(child, bundle)
        if negation is not None and negation in seen_keys:
            return out, True
        if _matches_existing_negation(key, seen_keys):
            return out, True
        seen_keys[key] = child
        out.append(child)
    return out, False


def _negation_canonical_key(child, bundle):
    """If *child* is a ``_NotNode``, return the canonical key of the inner node."""
    if isinstance(child, _NotNode) and child.e:
        return _canonical_key(child.e[0], bundle)
    if isinstance(child, _ForbiddenLeaf):
        return ("existsAL", child.token)
    return None


def _matches_existing_negation(child_key, seen_keys) -> bool:
    """Detect ``X`` matching a previously-seen ``notL(X)``."""
    for previous_key in seen_keys:
        # Generic notL(X) key shape.
        if isinstance(previous_key, tuple) and len(previous_key) >= 2 and previous_key[0] == "notL":
            inner = previous_key[1]
            if isinstance(inner, tuple) and len(inner) == 1 and inner[0] == child_key:
                return True
        # Specialized forbidden-token key shape.
        if isinstance(previous_key, tuple) and len(previous_key) == 2 and previous_key[0] == "_forbidden_token":
            if isinstance(child_key, tuple) and child_key[:2] == ("existsAL", previous_key[1]):
                return True
    return False


# --------------------------------------------------------------------------- #
# Heterogeneous-andL salvage                                                  #
# --------------------------------------------------------------------------- #


def _split_regular_irregular_andL(children, bundle):
    """Drop generation-relevant-but-irregular children from an andL list.

    Returns ``(regular_children, irregular_children)`` so the caller can keep
    the regular conjunction and surface the irregular ones through
    ``NormalForm.irregular_children``.  Non-generation-relevant children are
    left in place — the matcher already ignores them harmlessly.

    An ``andL`` child is treated as irregular when it is generation-relevant
    (so dropping it changes semantics) but is not in a class the matcher can
    compile (so leaving it would make the whole ``andL`` fail).  The salvage
    drops such children from the regular conjunction and surfaces them through
    the second tuple so the caller can warn / error per policy.
    """
    from .graph_discovery import _is_generation_relevant  # late import: avoid cycles

    regular: list = []
    irregular: list = []
    for child in children:
        # Mirror nodes are always regular and compilable.
        if _kind_is_mirror(child):
            regular.append(child)
            continue
        # Generation-relevant but unsupported nodes are surfaced as irregular.
        if _is_generation_relevant(child, bundle) and not _looks_supported(child):
            irregular.append(child)
            continue  # drop from the regular conjunction
        # Otherwise keep the node in the conjunction.
        regular.append(child)
    return regular, irregular


def _kind_is_mirror(node) -> bool:
    # Mirror node kinds produced by this module.
    return kind(node) in {"andL", "orL", "notL", "_top", "_bottom", "_forbidden_token"}


def _looks_supported(lc) -> bool:
    """Cheap precheck: is this LC class one the matcher can try to compile?"""
    return kind(lc) in {
        "andL", "orL", "notL", "ifL", "nandL", "norL", "xorL", "equivalenceL", "iffL",
        "atMostAL", "atLeastAL", "exactAL", "existsAL",
    }


# --------------------------------------------------------------------------- #
# Canonical keys                                                              #
# --------------------------------------------------------------------------- #


def _canonical_key(node, bundle):
    """Return a hashable identity key for dedup, or ``None`` when not derivable."""
    op = kind(node)
    # Synthetic constants and specialized leaves.
    if op == "_top":
        return ("_top",)
    if op == "_bottom":
        return ("_bottom",)
    if op == "_forbidden_token":
        return ("_forbidden_token", node.token)
    if op in {"andL", "orL"}:
        # Aggregate keys are built recursively from child keys.
        child_keys = tuple(_canonical_key(child, bundle) for child in getattr(node, "e", ()))
        if any(key is None for key in child_keys):
            return None
        # Sort child keys so commutativity gives identical keys for permutations.
        return (op, tuple(sorted(child_keys, key=repr)))
    if op == "notL":
        # notL is unary but represented as a tuple for key uniformity.
        children = tuple(_canonical_key(child, bundle) for child in getattr(node, "e", ()))
        if any(key is None for key in children):
            return None
        return ("notL", children)
    if op == "ifL":
        children = tuple(_canonical_key(child, bundle) for child in getattr(node, "e", ()))
        if any(key is None for key in children):
            return None
        return ("ifL", children)
    # Original LC leaves.  Late import keeps the module dependency one-way.
    from .graph_discovery import (
        _direct_token,
        _exists_token,
        _last_int,
        _non_eos_at_most_count,
    )
    if op == "atMostAL":
        return (
            "atMostAL",
            _direct_token(node.e, bundle),
            _last_int(node.e),
            _non_eos_at_most_count(node, bundle),
        )
    if op == "atLeastAL":
        return ("atLeastAL", _direct_token(node.e, bundle), _last_int(node.e))
    if op == "exactAL":
        return ("exactAL", _direct_token(node.e, bundle), _last_int(node.e))
    if op == "existsAL":
        return ("existsAL", _exists_token(node, bundle))
    # Anything else (with paths, V instances, eqL filters, …): use a
    # structural fallback via repr.  Not perfect, but conservative — if two
    # nodes have different reprs they are simply not deduped.
    try:
        return ("_lc", op, repr(getattr(node, "e", ())))
    except Exception:  # pragma: no cover - very defensive
        return None


# --------------------------------------------------------------------------- #
# Specialised rewrites                                                        #
# --------------------------------------------------------------------------- #


def _try_not_exists_to_forbidden(node, bundle):
    """Rewrite ``existsAL(t)`` (being negated) to a ``_ForbiddenLeaf(t)``."""
    # Only applies to notL(existsAL(token)).
    if kind(node) != "existsAL":
        return None
    from .graph_discovery import _exists_token  # late import
    token = _exists_token(node, bundle)
    if token is None:
        return None
    return _ForbiddenLeaf(token=token)
