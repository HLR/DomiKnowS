# Multi-variable executable formulas were not grounded jointly

**Status:** fixed on `develop` (commits `8327b963`, `17b0e234`)
**Files:** `domiknows/solver/logicalConstraintConstructor.py`, `domiknows/solver/compiled/formula.py`, `domiknows/graph/dataNode.py`
**Tests:** `test_regr/fixes/test_multivar_executable_grounding.py` (standalone, synthetic, brute-force truth)

## Symptom

Any executable formula whose relations form a chain over 3 or more variables was evaluated wrongly:

```python
existsL(andL(A('a'), B('b'), C('c'), left('a', 'b'), left('b', 'c')))
```

- **Loss / training:** raised a tensor size mismatch.
- **Verification / `evaluate_condition`:** returned False even when the scene satisfies the formula. There was no error or warning.

Two-variable formulas (`existsL(andL(A('a'), B('b'), left('a', 'b')))`, all CLEVR one-relation questions) were fine, which is why this went unnoticed. Every 3D-FORCE question has 4–6 variables.

## Root cause

Each operand is enumerated over its own variable tuple: `A` over `(a)`, `left('a','b')` over `(a, b)`, `left('b','c')` over `(b, c)`. Before they are combined:

- `reduceToCommonGrounding` only handles operands that **share a variable common to all of them**. It quantifies the other variables away.
- A chain has no variable common to all operands, so the reduction declined. The operands were then combined **row by row** at different lengths. Loss mode raised; verify mode silently produced an empty or False result.

A related trap: quantifying each relation separately is not exact either. In scene `s2` of the test, `exists b. left(a,b)` and `exists b. left(b,c)` are both true with *different* `b`, so the per-relation approximation says True while the correct answer is False.

## Fix

New `LogicalConstraintConstructor.expandToJointGrounding`, called by both the interpreter and the compiled evaluator in loss, verify and circuit mode:

1. **Join.** Build the exact joint table over the union of the operands' variables (one row per variable assignment) and gather every operand onto those rows.
2. **Gate.** Applies only when at least two operands are bound to *different* variable sets, no variable is common to all of them, and at least one operand is relation-bound. Unchanged cases:
   - co-grounded formulas (all CLEVR one-relation questions);
   - single-shared-variable formulas, which still use `reduceToCommonGrounding`;
   - plain per-entity comparisons such as `sameL(color, 'x', 'y')`, where row-wise pairing is the intended semantics.
3. **Pruning.**
   - Verify mode restricts each variable's candidates using the hard unary values. This is exact.
   - Loss mode prunes only when the table exceeds `DOMIKNOWS_JOINT_SOFT_PRUNE_ROWS` (default 300k). It keeps the top `DOMIKNOWS_JOINT_SOFT_PRUNE_TOPK` (default 4) candidates per variable, ranked by unary evidence. This is approximate.
   - Pruning never touches an enclosing `iotaL`/`miotaL` answer variable.
   - Tables above `JOINT_GROUNDING_MAX_ROWS` (3M) are declined and fall back to the old path.
4. **Supporting changes.**
   - A second unary on the same variable (`A('a'), A('a')`) inherits the variable's grounding.
   - Nested results carry their joint or common binding, so `miotaL` can reduce a joined table to its answer variable (one value per object).
   - `getActiveExecutableConstraintNames` no longer reports the constraint concept's own `label` sensor as a phantom constraint named `label`, which crashed the ILP answer solver.

## Reproduce

```bash
cd test_regr/fixes
python multivar_executable_repro.py verify s1/q3        # 3-variable chain, True
python multivar_executable_repro.py verify s2/q3        # False; the per-relation approximation says True
python multivar_executable_repro.py train  s1/q3,s2/q3  # loss path (raised before the fix)
python -m pytest -q test_multivar_executable_grounding.py
```

## Points worth reviewing

- **Gate choice.** Is excluding plain-variable formulas like `sameL` the right boundary for the library in general?
- **Soft pruning.** Top-k soft pruning in loss mode is approximate. It changes gradients for candidates the model currently ranks low. The two environment variables are how we tuned it; a proper option on the program may be preferable.
- **Cost.** Joint tables grow as n^k. 3D-FORCE scenes have up to 12 objects and 5 variables, so pruning is what keeps training usable (about 1 h → 1 min per hundred items).
