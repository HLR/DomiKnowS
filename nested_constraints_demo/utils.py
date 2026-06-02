"""Demo-local helpers: mirror-tree printer + re-exports of shared utilities.

The mirror-tree printer is the demo-specific piece -- it walks the normalized
``NormalForm.tree`` returned by
:func:`domiknows.generation.dfa._lc_normalize.normalize_lc` and renders an
indented Python-source-like view that makes the rewrites (De Morgan,
``_ForbiddenLeaf`` collapse, double-neg elim) visually obvious.

Snapshot / training helpers are re-exported from
:mod:`Tasks.real_hmm_pmd_learning.utils` so the demo doesn't fork them.  The
shim imports them once at module load.
"""
from __future__ import annotations

from typing import Any

# Re-exports of shared training / logging helpers from the existing demo.  No
# cross-task code lives here; we just expose the same names so the new demo's
# ``run_demo.py`` can write ``from .utils import print_demo_header`` etc.
try:
    from Tasks.real_hmm_pmd_learning.utils import (
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        _enable_remote_debug,
        capture_parameter_snapshot,
        print_constrained_greedy_inference,
        print_gradient_snapshot,
        print_greedy_inference,
        print_inference_header,
        print_no_training_requested,
        print_parameter_update_snapshot,
        print_trained_batch,
        print_training_header,
        reset_optimizer_grad_snapshot,
    )
except ImportError:  # pragma: no cover - direct-script fallback
    import sys
    import pathlib

    _real_hmm = pathlib.Path(__file__).resolve().parent.parent / "real_hmm_pmd_learning"
    sys.path.insert(0, str(_real_hmm.parent))
    from Tasks.real_hmm_pmd_learning.utils import (  # type: ignore[no-redef]
        AdamWithGradSnapshot,
        _enable_domiknows_production_logging,
        _enable_remote_debug,
        capture_parameter_snapshot,
        print_constrained_greedy_inference,
        print_gradient_snapshot,
        print_greedy_inference,
        print_inference_header,
        print_no_training_requested,
        print_parameter_update_snapshot,
        print_trained_batch,
        print_training_header,
        reset_optimizer_grad_snapshot,
    )


__all__ = [
    "AdamWithGradSnapshot",
    "_enable_domiknows_production_logging",
    "_enable_remote_debug",
    "capture_parameter_snapshot",
    "format_lc_source",
    "format_mirror_tree",
    "print_constrained_greedy_inference",
    "print_gradient_snapshot",
    "print_greedy_inference",
    "print_inference_header",
    "print_no_training_requested",
    "print_parameter_update_snapshot",
    "print_trained_batch",
    "print_training_header",
    "reset_optimizer_grad_snapshot",
]


# --------------------------------------------------------------------------- #
# LC + mirror tree pretty printers                                            #
# --------------------------------------------------------------------------- #


def _kind(node) -> str:
    return getattr(node, "_kind", None) or type(node).__name__


def format_lc_source(lc, *, indent: int = 0) -> str:
    """Render an original DomiKnowS LC as compact Python-source-like text.

    Walks ``lc.e`` for boolean nodes and falls back to ``repr`` for leaves.
    Used by the demo to print the input LC tree before normalization so the
    reader can compare it side-by-side with the rewritten mirror tree.
    """
    pad = "  " * indent
    if _kind(lc) in {"andL", "orL", "notL", "ifL", "nandL", "norL", "xorL", "iffL", "equivalenceL"}:
        op = _kind(lc)
        children = list(getattr(lc, "e", ()))
        if not children:
            return f"{pad}{op}()"
        body = ",\n".join(format_lc_source(child, indent=indent + 1) for child in children)
        return f"{pad}{op}(\n{body},\n{pad})"
    if _kind(lc) in {"atMostAL", "atLeastAL", "exactAL", "existsAL"}:
        op = _kind(lc)
        children = list(getattr(lc, "e", ()))
        body = ", ".join(_compact_leaf_arg(child) for child in children)
        return f"{pad}{op}({body})"
    return f"{pad}{_compact_leaf_arg(lc)}"


def _compact_leaf_arg(item: Any) -> str:
    """Render a non-LC argument compactly (concept tuples, V-instances, ints)."""
    if isinstance(item, tuple) and len(item) == 4:
        concept, name, label, _cardinality = item
        concept_name = getattr(concept, "name", None) or repr(concept)
        if label is not None:
            return f"{concept_name}.{label}({name!r})"
        return f"{concept_name}({name!r})"
    if hasattr(item, "v"):  # V-instances expose their path as ``.v``
        return f"V(path={item.v!r})"
    if isinstance(item, int):
        return str(item)
    return repr(item)


def format_mirror_tree(tree, *, indent: int = 0) -> str:
    """Render a normalized mirror AST returned by ``normalize_lc(...).tree``.

    Mirror nodes (``_AndNode`` / ``_OrNode`` / ``_NotNode`` / ``_ForbiddenLeaf`` /
    ``_TopNode`` / ``_BottomNode``) are displayed with their ``_kind`` label so
    the rewrites are explicit; original LC leaves that survive normalization
    are delegated to :func:`format_lc_source`.
    """
    pad = "  " * indent
    op = _kind(tree)
    if op == "_top":
        return f"{pad}_TopNode  # constant-true (accept_all_dfa)"
    if op == "_bottom":
        return f"{pad}_BottomNode  # constant-false (empty_dfa)"
    if op == "_forbidden_token":
        token = getattr(tree, "token", "?")
        return f"{pad}_ForbiddenLeaf(token={token!r})  # forbidden_token_dfa"
    if op in {"andL", "orL", "notL"}:
        children = list(getattr(tree, "e", ()))
        if not children:
            return f"{pad}{op}()"
        body = ",\n".join(format_mirror_tree(child, indent=indent + 1) for child in children)
        return f"{pad}{op}(\n{body},\n{pad})"
    # Fall through: an original LC leaf survived normalization unchanged.
    return format_lc_source(tree, indent=indent)
