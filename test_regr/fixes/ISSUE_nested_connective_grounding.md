# Relations inside a nested `orL` / `andL` evaluated as False

**Status:** fixed in the working tree on `develop` (not yet committed)
**Files:** `domiknows/graph/candidates.py`, `domiknows/solver/logicalConstraintConstructor.py`, `domiknows/solver/compiled/formula.py`
**Tests:** `test_regr/fixes/test_multivar_executable_grounding.py::test_relation_inside_nested_connective_*` (19 cases)

## Symptom

A formula with a relation inside a nested connective is verified **False** on scenes where it is true. The equivalent flat formula is verified correctly. There is no error or warning.

```python
existsL(andL(A('a'), B('b'), left('a', 'b')))                          # correct
existsL(andL(A('a'), B('b'), orL(left('a', 'b'), left('a', 'b'))))     # always False
existsL(andL(A('a'), B('b'), C('c'), left('a', 'b'),
             andL(C('c'), left('b', 'c'))))                            # always False
```

**Real impact.** On 3D-FORCE we express "object-perspective left" as a disjunction over an object's heading:
`orL(andL(hd0('b'), dir270('a','b')), andL(hd15('b'), dir285('a','b')), ...)`.
With ground-truth predictions (oracle mode), question accuracy was **79.3%**, and every failure was a true question answered False. After the fixes it is **99.7%**. The remaining errors come from bin quantisation, not the library.

## Root causes and fixes

These are four separate gaps; each one alone keeps the result False.

### 1. Forward declaration lookup did not search nested constraints

**Where:** `candidates.py`, `getDatanoteForVariable`

In `andL(A('a'), orL(left('a','b'), ...))` the compiler rewrites `A('a')` as the path `(left_3, arg1)`. The relation `left_3` is declared **inside** the `orL`. The on-demand lookup for a variable's declaration scanned only the top-level `lc.e`. So `A('a')` got `[None]` candidates and its values were `None`.

**Fix:** search nested `LcElement`s recursively, and resolve the declaration with its owning constraint.

### 2. Path operands had no binding when their relation was grounded in a nested constraint

**Where:** `logicalConstraintConstructor.py`, new `_recordRelationArguments` and `fillNestedPathBindings`

`fillPathBindings` resolves a path operand from the binding of its source relation. That binding lives only in the nested constraint's local table. The outer operand stayed unbound, and `expandToJointGrounding` declines when an unbound operand is not a scalar.

**Fix:**
- Record each relation variable's argument variables whenever it is grounded. Relation variable names such as `left_3` are globally unique.
- At join time, bind an unbound path operand `(left_3, argK)` to the variable at position K.
- Bind only to the **projected** variable (`('a',)` with repeated keys), not to the full `(a, b)` grid. The operand's value depends only on that argument, and the over-specified grid blocks alignment with a relation on `(b, c)` (see 4).

### 3. Nested constraints did not inherit the enclosing constraint's bindings

**Where:** both evaluators, the re-binding branch plus the nested call site

Every `constructLogicalConstrains` / `constructCompiled` call starts with an empty `lcVariableBindings`. In `andL(..., C('c'), ..., andL(C('c'), left('b','c')))`, the inner `C('c')` re-binds the outer variable `c` (as `_x1`), but the outer binding of `c` is not visible there. So `_x1` is unbound.

**Fix:** keep a `ChainMap` of enclosing bindings (`self._outer_bindings`) around each nested call. The re-binding branch falls back to it.

### 4. Operands sharing a variable were never aligned in verify mode

**Where:** `expandToJointGrounding`

Inside the nested `andL(C('c'), left('b','c'))` the operands are bound to `(c)` (3 rows) and `(b, c)` (9 rows). They share `c`, so the join gate declined. `reduceToCommonGrounding` runs only in loss mode, so verify mode combined 3 rows with 9 rows and produced an empty result. At the top level this never happens: relation expansion re-grounds earlier operands onto the relation's rows, but the expansion does not reach the enclosing constraint's variables.

**Fix:** when one operand spans every variable of the others, align them onto **that operand's rows**. Its row order is kept, no rows are added and no pruning is applied, so sibling `orL` branches stay row-compatible. The star case `{z, x}` with `{z, y}` has no spanning operand and still uses `reduceToCommonGrounding`.

**Behaviour change to review:** in loss mode, formulas of this spanning shape are now joined exactly instead of being reduced to the shared variable. That is more correct for nested use, because the result keeps both variables. It can change losses for existing formulas of that shape.

### Also fixed

`expandToJointGrounding` (added in the earlier fix) assumed all operand columns are on one device. Verify-mode values of a nested result are stacked on CPU while predicate columns are on CUDA, so it raised a device error. Columns are now moved to a common device.

## Reproduce

```bash
cd test_regr/fixes
python multivar_executable_repro.py verify s1/q2or      # causes 1 + 2
python multivar_executable_repro.py verify s1/q3and     # cause 3
python multivar_executable_repro.py verify s1/q3orandb  # cause 4 + projection
python multivar_executable_repro.py verify s4/q4orand   # 4 variables, heading-composition shape
python -m pytest -q test_multivar_executable_grounding.py
```

## Verification

- **Before the fixes:** `s1/q2or` and `s1/q2and` (true) were verified False on the committed code.
- **Regression file:** all 28 tests in the file pass, including the earlier joint-grounding and `miotaL` selection tests.
- **Whole `test_regr/fixes` directory:** 20,225 passed, 4 skipped.
  - One failure is unrelated: `test_stress_train_epoch_dispatch` hits `'InferenceProgram' object has no attribute 'use_gumbel'`, in untouched code.
  - Two files cannot be collected at all (`No module named 'main'`).
- **Real data:** on 66 real 4–5 variable 3D-FORCE questions, the framework verdict now matches brute-force evaluation of the same logic string on all 66 (before: 24 disagreed).

## Points worth reviewing

- **Registry.** Is a registry keyed by globally unique relation variable names acceptable, or should nested scopes pass bindings explicitly?
- **Binding scope.** Should bindings be scoped per head constraint rather than kept on the evaluator instance? Today they are stored on `self` and restored after each nested call; this is not exception-safe.
- **Silent failures.** The broader problem is that unaligned operands fail silently (row-wise combination at different lengths). A warning, or an error in verify mode, would have exposed all four gaps immediately.
