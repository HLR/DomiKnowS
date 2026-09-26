# Graph-level (global) rules over relation pairs and triples

Found while adding 3D-FORCE spatial rules (xor, inverse, transitivity) as
graph-level constraints trained with `include_global_constraint_loss`.
Reproduction: `global_rules_repro.py` (synthetic scenes, hand-set 0/1 values,
brute-force truth per grounding, loss and head-level verify); tests:
`test_global_rule_grounding.py` (fails on the pre-fix tree for 1-7, passes after).

## Fixed

1. **Swapped arguments are ignored.**
   `equivalenceL(left('a','b'), right('b','a'))` was evaluated as `right(a,b)`.
   Also inside executables: `existsL(andL(A('a'), B('b'), left('a','b'), right('b','a')))`
   returned True where the answer is False.
   Cause: `expandToJointGrounding` compared each operand's variables as *sets*,
   so `('a','b')` and `('b','a')` looked co-grounded and rows were paired one-to-one.
   Fix: compare the ordered tuples; swapped operands then use the existing
   alignment (rows looked up by variable name on the spanning operand).
   Not covered (these paths never call `expandToJointGrounding`):
   `sample=True`, ILP (head-level verify: see 6).

2. **Head `andL` / `orL` / `nandL` with 3+ operands returned truth instead of loss.**
   `createLogicalConstrains` guessed from `inspect.signature` whether
   `onlyConstrains` was passed positionally; for `andVar(self, _, *var, onlyConstrains=False)`
   its index is 2, so with 3+ operands it was dropped. Loss paths (all t-norms,
   compiled or not, sample loss) trained such rules inverted; in the Gurobi
   path the head constraint was not enforced at all (a violating assignment
   was OPTIMAL). Executables were unaffected (head is existsL / miotaL).
   Introduced in a3b13d2f. Fix: never take the positional shortcut for
   functions with `*var` (onlyConstrains is keyword-only there; no caller
   passes it positionally). Behaviour change: such rules now train with the
   right sign and ILP enforces them.

3. **`ifL(andL(r(a,b), r(b,c)), r(a,c))` crashed (size mismatch).**
   The nested andL's operands share `b` with no spanning operand, so
   `reduceToCommonGrounding` quantified a and c away (one row per b); the
   existential reading is exact only under an existential parent. Fix: a
   constraint nested in a plain connective (not a count / accumulated count /
   iotaL / miotaL / queryL / sumL) keeps the joint table (`keep_joint`).
   Under existential parents nothing changes. Row counts / loss size change
   for e.g. `notL(andL(L(a,b), L(b,c)))` (4 -> 64 rows, now correct).
   `_CompareCountsBaseL` (greaterL, ...) counts its operands like a count and
   is now in the exclusion list too (`EXISTENTIAL_PARENTS` in
   `LogicalConstraintConstructor`, shared with the compiled constructor).

4. **Implication (`ifVar`) returned satisfaction as loss for rows holding a 0,
   and Goedel counted `a == b` as violated.**
   The Goedel / Product vector paths fell back to per-element calls of the
   singleton variant `ifVarS` when a 0 was present (G: in the consequent,
   P: in the premise), passing `onlyConstrains` down and inverting again on
   return; `ifVarBatched` (compiled fast path) copied that on purpose. The
   Goedel vector path used strict `var2 > var1` for the fully-true branch
   (`ifVarS` had `>=`), so an `ifL(distinct, ...)` diagonal with equal ~0
   values counted as violated. Also `ifVarS` Product gave `0 -> 0` truth 0
   (`0 / 1e-4`); Goguen implication is 1 when `a <= b`. Fix: `ifVar` is one
   element-wise implementation for scalars, vectors and matrices (G
   `b >= a -> 1 else b`, P `a <= b -> 1 else b / a`, L unchanged) with one
   inversion; `ifVarS` is removed and `ifVarBatched` calls `ifVar`.
   Behaviour change: such G / P implication rows now train with the right sign.

5. **Head-level rules with a shared variable were read existentially.**
   `ifL(L(a,b), L(b,c))` at the head was reduced by
   `reduceToCommonGrounding` to `(exists a. L(a,b)) -> (exists c. L(b,c))`,
   one row per b. A graph rule means "for all a, b, c": head-level
   constraints other than `EXISTENTIAL_PARENTS` now keep the joint table
   (`keepJointFor`), like nested ones under a plain connective (fix 3); the
   compiled constructor now does the same (it passed no `keep_joint` at all).
   Behaviour change: n^k groundings per such rule (the joint-table row
   guards `JOINT_GROUNDING_MAX_ROWS` / soft pruning still apply);
   `existsL(...)` and other counts are unchanged.

6. **Head-level verify skipped grounding alignment.**
   The uncompiled `verify and headLC` branch returned before
   `expandToJointGrounding`: `inverse` reported the swapped relation,
   `trans_nand` 16 of 64 rows, `trans_if` nothing. Both constructors now
   align head-level verify like the loss and hand the verifier the aligned
   operands (its `ifSatisfied` reads the premise row by row; the compiled
   path raised IndexError on joined rows). Verify-mode pruning by unary
   evidence (exact only for existential / conjunctive readings) is now
   disabled under `keep_joint`, where rows with a false premise count.

7. **A rebuilt graph in the same process reused the first graph's solver.**
   The `ilpOntSolverFactory` cache key (solver class, ontology, config key
   names, kwarg names) did not include the graph, so any rebuilt graph got
   the solver, and rules, of the first; `Graph.clear()` did not reset it.
   Fix: the graphs (by identity) and kwarg values are part of the key, and
   `Graph.clear()` also clears the factory. Rule sweeps / cross-validation
   that rebuild the graph without `Graph.clear()` create one cached solver
   per graph.

## Added options (`InferenceModel` / `InferenceProgram`)

- `global_constraint_tnorm`: t-norm of the graph-global loss (default: `tnorm`).
- `global_constraint_reduction`: `'sum'` (default) or `'mean'` over each rule's
  groundings (a rule over triples has n^3 groundings).

## Notes

- Goedel implication is still discontinuous: a premise slightly above the
  consequent (`0.0004 -> 0.0003`) is a full violation. The repro's hand-set
  logits give equal values, so G is exact there; with learned values
  `global_constraint_tnorm='P'` or `'L'` is the safer choice.
- `sample=True` and ILP still do not use `expandToJointGrounding`.
