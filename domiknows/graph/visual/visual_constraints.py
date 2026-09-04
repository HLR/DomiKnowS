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

from domiknows.graph import ifL, andL, nandL, notL, equivalenceL

# ======================================================================
# §0  Spatial: opposite constraints (mutual exclusion)
#
#     nandL(R1(a,b), R2(a,b))
#
# Both concepts live on pair_forward, so this expresses:
#   "A cannot be both left of B and right of B simultaneously"
#
# This is mutual exclusion, NOT exhaustive dichotomy.  Two objects
# may be neither left nor right of each other (e.g. directly
# above/below, or at the same x-coordinate).
# ======================================================================

OPPOSITE_PAIRS = [
    ("left_of",     "right_of"),
    ("above",       "below"),
    ("in_front_of", "behind"),
]

def apply_opposite_constraints(ctx, *, pairs=None):
    """nandL(R1(a,b), R2(a,b))  — mutual exclusion on pair_forward."""
    if pairs is None:
        pairs = [(ctx.get(r1), ctx.get(r2)) for r1, r2 in OPPOSITE_PAIRS]
    for r1, r2 in pairs:
        if r1 is None or r2 is None:
            continue
        nandL(
            r1('a', 'b'),
            r2('a', 'b'),
            name=f"opp_{r1.name}_{r2.name}",
        )


# ======================================================================
# §1  Spatial: inverse constraints   R_fwd(a,b) ↔ R_inv_rev(a,b)
#
# The forward concept and the reverse concept live on DIFFERENT nodes
# (pair_forward vs pair_reverse), so their ILP variables are distinct
# even though both use the slot names ('a','b').
#
# This encodes: "A is left of B"  ↔  "B is right of A"
# without any contradiction with §0.
#
# equivalenceL is correct here because this is a true semantic
# identity — left_fwd(a,b) and right_rev(a,b) refer to the exact
# same real-world fact, just represented on different graph nodes.
# ======================================================================

INVERSE_PAIRS = [
    ("left_of",     "right_of_rev"),
    ("right_of",    "left_of_rev"),
    ("above",       "below_rev"),
    ("below",       "above_rev"),
    ("in_front_of", "behind_rev"),
    ("behind",      "in_front_of_rev"),
]

def apply_inverse_constraints(ctx, *, pairs=None):
    """R_fwd(a,b) ↔ R_inv_rev(a,b)  — across pair_forward and pair_reverse."""
    if pairs is None:
        pairs = [(ctx.get(r1), ctx.get(r2)) for r1, r2 in INVERSE_PAIRS]
    for r1, r2 in pairs:
        if r1 is None or r2 is None:
            continue
        equivalenceL(
            r1('a', 'b'),
            r2('a', 'b'),
            name=f"inverse_{r1.name}_{r2.name}",
        )


# ======================================================================
# §2  Spatial: transitivity   R(a,b) ∧ R(b,c) ⇒ R(a,c)
# ======================================================================

def apply_transitive_constraints(ctx, *, relations=None):
    """R(a,b) ∧ R(b,c) ⇒ R(a,c)  — applied on pair_forward only."""
    relations = relations or [
        ctx.get("left_of"), ctx.get("right_of"),
        ctx.get("above"),   ctx.get("below"),
    ]
    for rel in relations:
        if rel is None:
            continue
        ifL(
            andL(rel('a', 'b'), rel('b', 'c')),
            rel('a', 'c'),
            name=f"transitive_{rel.name}",
        )


# ======================================================================
# §3  Cross-attribute plausibility
# ======================================================================

def apply_nand_combos(
    concept_dict_a: dict,
    concept_dict_b: dict,
    implausible: list[tuple[str, str]],
    *,
    tag: str = "implausible",
):
    for val_a, val_b in implausible:
        ca = concept_dict_a.get(val_a)
        cb = concept_dict_b.get(val_b)
        if ca is None or cb is None:
            continue
        nandL(ca('x'), cb('x'), name=f"{tag}_{val_a}_{val_b}")


# ======================================================================
# §4  Apply all constraints
#
# All three spatial constraint types are safe to use simultaneously
# because pair_forward and pair_reverse are distinct concept nodes:
#   §0 opposite   — pair_forward only (nandL)    → no conflict
#   §1 inverse    — pair_forward ↔ pair_reverse   → no conflict with §0
#   §2 transitive — pair_forward only              → no conflict
# ======================================================================

def apply_all_constraints(
    ctx: dict,
    *,
    implausible_shape_size: list[tuple[str, str]] | None = None,
    implausible_color_material: list[tuple[str, str]] | None = None,
):
    apply_opposite_constraints(ctx)
    apply_inverse_constraints(ctx)
    apply_transitive_constraints(ctx)

    if implausible_shape_size:
        apply_nand_combos(
            ctx["shapes"], ctx["sizes"],
            implausible_shape_size,
            tag="implausible_shape_size",
        )
    if implausible_color_material:
        mat = ctx["material"]
        mat_dict = {v: getattr(mat, v) for v in mat.enum}
        apply_nand_combos(
            ctx["colors"], mat_dict,
            implausible_color_material,
            tag="implausible_color_mat",
        )