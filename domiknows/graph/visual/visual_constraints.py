"""
visual_constraints.py

Reusable, *generic* commonsense constraint library for DomiKnowS
visual-reasoning graphs built with build_visual_reasoning_graph().

All constraints are parameterised by concept references from ctx —
they never hard-code specific attribute values so the library works
for any scene-understanding task.

Uses the simplified string-variable syntax (not V-based paths).

Usage:
    from visual_reasoning_graph import build_visual_reasoning_graph
    from visual_constraints import apply_all_constraints

    graph, ctx = build_visual_reasoning_graph(...)
    with graph:
        apply_all_constraints(ctx)
"""

from domiknows.graph import ifL, andL, nandL, notL, equivalenceL

# ======================================================================
# §0  Spatial: opposite (reverse) relations  A ↔ ¬B
# In English we say "A is left of B" ↔ "B is not left of A" (i.e. "B is right of A").
# ======================================================================

OPPOSITE_PAIRS = [
    ("left_of",     "right_of"),
    ("above",       "below"),
    ("in_front_of", "behind"),
]

def apply_opposite_constraints(ctx, *, pairs=None):
    """R(A,B) ↔ ¬R_opp(A,B) for every opposite pair present in ctx."""
    pairs = pairs or [
        (ctx.get(r1), ctx.get(r2)) for r1, r2 in OPPOSITE_PAIRS
    ]
    for r1, r2 in pairs:
        if r1 is None or r2 is None:
            continue
        equivalenceL(
            r1('a', 'b'),
            notL(r2('a', 'b')),
            name=f"rev_{r1.name}_{r2.name}",
        )


# ======================================================================
# §1  Spatial: inverse relations
# In English we say "A is left of B" ↔ "B is right of A".
# ======================================================================

def apply_inverse_constraints(ctx, *, pairs=None):
    """R(A,B) ↔ R_inv(B,A) for every inverse pair present in ctx."""
    pairs = pairs or [
        (ctx.get(r1), ctx.get(r2)) for r1, r2 in OPPOSITE_PAIRS
    ]
    for r1, r2 in pairs:
        if r1 is None or r2 is None:
            continue
        # r1(a,b) ↔ r2(b,a): same variables, reversed argument order
        equivalenceL(
            r1('a', 'b'),
            r2('b', 'a'),
            name=f"inverse_{r1.name}_{r2.name}",
        )

# ======================================================================
# §2  Spatial: transitivity (soft)
# In English we say "A is left of B" ∧ "B is left of C" ⇒ "A is left of C".
# ======================================================================

def apply_transitive_constraints(ctx, *, relations=None):
    """
    R(A,B) ∧ R(B,C) ⇒ R(A,C).

    Parameters
    ----------
    relations : list | None
        Which spatial relations to make transitive.
        Defaults to left_of, right_of, above, below.
    """
    relations = relations or [
        ctx["left_of"], ctx["right_of"], ctx["above"], ctx["below"]
    ]
    for rel in relations:
        if rel is None:
            continue
        ifL(
            andL(
                rel('a', 'b'),
                rel('b', 'c'),
            ),
            rel('a', 'c'),
            name=f"transitive_{rel.name}",
        )

# ======================================================================
# §3  Cross-attribute plausibility (soft world knowledge)
# In English we say "A is a small and elephant" is implausible, so "small(x) ∧ elephant(x)" ⇒ False.
# ======================================================================

def apply_nand_combos(
    concept_dict_a: dict,
    concept_dict_b: dict,
    implausible: list[tuple[str, str]],
    *,
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
            ca('x'),
            cb('x'),
            name=f"{tag}_{val_a}_{val_b}",
        )


# ======================================================================
# §4  Convenience: apply all generic constraints
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
    apply_opposite_constraints(ctx)
    apply_inverse_constraints(ctx)
    apply_transitive_constraints(ctx)

    # --- Optional domain-specific plausibility ---
    if implausible_shape_size:
        apply_nand_combos(
            ctx["shapes"], ctx["sizes"],
            implausible_shape_size,
            tag="implausible_shape_size",
        )
    if implausible_color_material:
        # material is EnumConcept — build a dict from its enum values
        mat = ctx["material"]
        mat_dict = {v: getattr(mat, v) for v in mat.enum}
        apply_nand_combos(
            ctx["colors"], mat_dict,
            implausible_color_material,
            tag="implausible_color_mat",
        )