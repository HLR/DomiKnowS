"""
visual_constraints.py

Reusable, *generic* commonsense constraint library for DomiKnowS
visual-reasoning graphs built with build_visual_reasoning_graph().

All constraints are parameterised by concept references from ctx —
they never hard-code specific attribute values so the library works
for any scene-understanding task.

Usage:
    from visual_reasoning_graph import build_visual_reasoning_graph
    from visual_constraints import apply_all_constraints

    graph, ctx = build_visual_reasoning_graph(...)
    with graph:
        apply_all_constraints(ctx)
"""

from domiknows.graph import V, ifL, andL, nandL, notL, equivalenceL


# ======================================================================
# Configuration tables
# ======================================================================

# (relation_name, inverse_name)
INVERSE_PAIRS = [
    ("left_of",     "right_of"),
    ("above",       "below"),
    ("in_front_of", "behind"),
]

# Cannot both be true for same ordered pair (A,B)
MUTEX_PAIRS = [
    ("left_of",  "right_of"),
    ("above",    "below"),
    ("in_front_of", "behind"),
]


# ======================================================================
# §1  Spatial: inverse relations
# ======================================================================

def apply_inverse_constraints(ctx, *, p: int = 95):
    """R(A,B) ↔ R_inv(B,A) for every inverse pair present in ctx."""
    for r1_name, r2_name in INVERSE_PAIRS:
        r1 = ctx.get(r1_name)
        r2 = ctx.get(r2_name)
        if r1 is None or r2 is None:
            continue
        equivalenceL(
            r1(V.pair),
            r2(V.pair),
            p=p,
            name=f"inverse_{r1_name}_{r2_name}",
        )


# ======================================================================
# §2  Spatial: mutual exclusion
# ======================================================================

def apply_mutex_constraints(ctx, *, p: int = 100):
    """¬(R(A,B) ∧ R_opp(A,B)) for opposite relation pairs."""
    for r1_name, r2_name in MUTEX_PAIRS:
        r1 = ctx.get(r1_name)
        r2 = ctx.get(r2_name)
        if r1 is None or r2 is None:
            continue
        nandL(
            r1(V.pair),
            r2(V.pair),
            p=p,
            name=f"mutex_{r1_name}_{r2_name}",
        )


# ======================================================================
# §3  Spatial: transitivity (soft)
# ======================================================================

def apply_transitive_constraints(ctx, *, relations=None, p: int = 70):
    """
    R(A,B) ∧ R(B,C) ⇒ R(A,C)  — soft, lower priority.

    Parameters
    ----------
    relations : list[str] | None
        Which spatial relation names to make transitive.
        Defaults to left_of, right_of, above, below.
    """
    relations = relations or ["left_of", "right_of", "above", "below"]
    for rel_name in relations:
        rel = ctx.get(rel_name)
        if rel is None:
            continue
        ifL(
            andL(rel(V.ab), rel(V.bc)),
            rel(V.ac),
            p=p,
            name=f"transitive_{rel_name}",
        )


# ======================================================================
# §4  Cross-attribute plausibility (soft world knowledge)
# ======================================================================

def apply_nand_combos(
    concept_dict_a: dict,
    concept_dict_b: dict,
    implausible: list[tuple[str, str]],
    *,
    p: int = 60,
    tag: str = "implausible",
):
    """
    For each (val_a, val_b) in `implausible`,
    create nandL(concept_a(x), concept_b(x)).

    Works for any two attribute dictionaries, e.g.
      apply_nand_combos(shapes, sizes, [("sphere","small")])
    """
    for val_a, val_b in implausible:
        ca = concept_dict_a.get(val_a)
        cb = concept_dict_b.get(val_b)
        if ca is None or cb is None:
            continue
        nandL(
            ca(V.x),
            cb(V.x),
            p=p,
            name=f"{tag}_{val_a}_{val_b}",
        )


# ======================================================================
# §5  Convenience: apply all generic constraints
# ======================================================================

def apply_all_constraints(
    ctx: dict,
    *,
    implausible_shape_size: list[tuple[str, str]] | None = None,
    implausible_color_material: list[tuple[str, str]] | None = None,
):
    """
    Apply the full generic constraint set to a ctx dict
    returned by `build_visual_reasoning_graph`.
    """
    # --- Spatial ---
    apply_inverse_constraints(ctx)
    apply_mutex_constraints(ctx)
    apply_transitive_constraints(ctx)

    # --- Optional domain-specific plausibility ---
    if implausible_shape_size:
        apply_nand_combos(
            ctx["shapes"], ctx["sizes"],
            implausible_shape_size,
            tag="implausible_shape_size",
        )
    if implausible_color_material:
        # material is EnumConcept, so build a dict from its values
        mat = ctx["material"]
        mat_dict = {v: getattr(mat, v) for v in mat.enum}
        apply_nand_combos(
            ctx["colors"], mat_dict,
            implausible_color_material,
            tag="implausible_color_mat",
        )