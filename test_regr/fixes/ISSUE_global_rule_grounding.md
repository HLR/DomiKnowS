# Graph-level (global) rules over relation pairs and triples

Found while adding 3D-FORCE spatial rules (xor, inverse, transitivity) as
graph-level constraints trained with `include_global_constraint_loss`.
Reproduction: `global_rules_repro.py` (synthetic scenes, hand-set 0/1 values,
brute-force truth per grounding); tests: `test_global_rule_grounding.py`
(fails on the pre-fix tree for 1-3, passes after).

## Fixed

1. **Swapped arguments are ignored.**
   `equivalenceL(left('a','b'), right('b','a'))` was evaluated as `right(a,b)`.
   Also inside executables: `existsL(andL(A('a'), B('b'), left('a','b'), right('b','a')))`
   returned True where the answer is False.
   Cause: `expandToJointGrounding` compared each operand's variables as *sets*,
   so `('a','b')` and `('b','a')` looked co-grounded and rows were paired one-to-one.
   Fix: compare the ordered tuples; swapped operands then use the existing
   alignment (rows looked up by variable name on the spanning operand).
   Not covered (these paths never call `expandToJointGrounding`): uncompiled
   head-level verify (`verify and headLC` early return), `sample=True`, ILP.

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
   Open: `_CompareCountsBaseL` may belong in the exclusion list; uncompiled
   head-level verify of this rule returns an empty result.

## Added options (`InferenceModel` / `InferenceProgram`)

- `global_constraint_tnorm`: t-norm of the graph-global loss (default: `tnorm`).
- `global_constraint_reduction`: `'sum'` (default) or `'mean'` over each rule's
  groundings (a rule over triples has n^3 groundings).

## Open questions

4. Goedel implication: `ifVar` uses strict `var2 > var1` for the "fully true"
   branch while `ifVarS` / `ifVarBatched` use `>=`; with near-0 premises
   (`ifL(distinct, ...)` on the diagonal) rules count as violated. Should
   `ifVar` use `>=`?
5. Head-level rules with a shared variable (`ifL(L(a,b), L(b,c))`) get the
   existential reading (one row per b); is that intended for graph rules,
   which usually mean "for all"?
6. (Low severity, usability.) A second graph built in the same process
   reuses the first graph's solver: `ilpOntSolverFactory` caches solvers by
   (solver class, ontology, config), not by graph, and `Graph.clear()` /
   `Concept.clear()` / `Relation.clear()` do not reset it;
   `ilpOntSolverFactory.clear()` does (test_regr/conftest.py calls it for
   every test, so the suite is unaffected). Should `Graph.clear()` also clear
   the solver cache?  The repro runs one rule per process for this reason.
