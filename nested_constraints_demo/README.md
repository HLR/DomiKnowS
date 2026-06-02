# Nested Logical Constraints with Paths — DomiKnowS LC → DFA demo

A full end-to-end showcase of the LC → DFA pipeline on a graph whose three
head logical constraints are deeply nested, path-aware, and mixed-regularity.
The demo pretty-prints every step of the diagram in
[`domiknows/generation/dfa/LogicalConstraintsToDFAPipeline.png`](../../domiknows/generation/dfa/LogicalConstraintsToDFAPipeline.png),
runs PMD training under the compiled DFA, and verifies that greedy +
DFA-constrained inference both produce constraint-satisfying outputs.

## Vocabulary

```
A, B, C, D, END  + _other (padding label)
```

EOS = `END`. `_other` is a non-emittable padding label used internally by the
compact-label head.

## The three head constraints

1. **Polite conversation** (`andL` with EOS-closure + count cap + De Morgan +
   `orL`):

   ```
   andL(
     ifL(
       is_before_rel("before"),
       ifL(
         token_value("END", "x", path=("before", first_token)),
         token_value("END", "y", path=("before", second_token)),
       ),
     ),                                           # EOS-closure (path-aware)
     atMostAL(token_value("B", "x"), 1),          # at most one B
     notL(andL(existsAL("A"), existsAL("C"))),    # not both A and C
     orL(existsAL("A"), existsAL("C")),           # at least one of A or C
   )
   ```

   Net effect: contains exactly one of `A` xor `C`, at most one `B`, and after
   `END` nothing else may follow.

2. **Heterogeneous andL salvage** — exercises the normalizer's
   `_split_regular_irregular_andL` path:

   ```
   andL(
     atMostAL(token_value("D", "x"), 0),           # regular: D is forbidden
     andL(token_value("A", "y"), token_value("B", "y")),  # irregular sibling
   )
   ```

   The inner `andL` of raw concept tuples has no LC-class children — the
   matcher cannot compile it.  The normalizer drops it from the regular
   conjunction and surfaces it via `NormalForm.irregular_children` so the
   `on_unsupported` policy still fires a `RuntimeWarning`, while the regular
   `atMostAL(D, 0)` still contributes its DFA fragment.

3. **Double-negation sanity** — exercises double-negation elimination:

   ```
   notL(notL(atMostAL(token_value("B", "x"), 1)))
   ```

   The normalizer collapses the double `notL`; the resulting DFA is identical
   to a bare `atMostAL(B, 1)` (which constraint #1 already registers, so the
   final intersection is a no-op on language but the demo shows the rewrite
   happened).

## Files

| File | Purpose |
|---|---|
| `graph.py` | `build_graph()` + `build_bundle()` (registers the three LCs after wiring). |
| `constraints.py` | `apply_constraints(graph, bundle)` — the three head LCs in §The three head constraints. |
| `corpus.py` | Labelled corpus buckets + `verify_acceptance(dfa, bundle, buckets)`. |
| `stream_generator.py` | Mock generator producing 50/50 valid + invalid sequences across two prompts (`with_A`, `with_C`). |
| `learning_program.py` | PMD program + compact-head builder. |
| `learned_model_interface.py` | Greedy inference helpers. |
| `utils.py` | Demo-local pretty printers (`format_lc_source`, `format_mirror_tree`) + re-exports of training helpers from `Tasks/real_hmm_pmd_learning/utils.py`. |
| `run_demo.py` | End-to-end runner (pipeline trace → corpus acceptance → training → inference). |
| `run_performance.py` | Side-by-side learner comparison across `discrete-hmm`, `graph-hmm`, `energy`. |

## CLI

End-to-end demo:

```
uv run --project Tasks/nested_constraints_demo \
    python Tasks/nested_constraints_demo/run_demo.py
```

Useful flags:

* `--learner {discrete-hmm,hmm,graph-hmm,energy}` (default `discrete-hmm`)
* `--inference-prompt {with_A,with_C}` (default `with_A`)
* `--steps 15 --lr 0.1 --beta 0.3` — `--steps 15` is the stable upper bound
  for this constraint set; values ≥ 20 occasionally drive the constraint
  loss into NaN under the nested-constraint DFA.

Side-by-side learner benchmark:

```
uv run --project Tasks/nested_constraints_demo \
    python Tasks/nested_constraints_demo/run_performance.py
```

## What you should see

* The **pipeline trace** prints every head LC in source form, then its
  normalized mirror tree.  Constraint #1's `notL(andL(existsAL, existsAL))`
  collapses into an `orL(_ForbiddenLeaf("A"), _ForbiddenLeaf("C"))` (De
  Morgan + leaf-pattern rewrite).  Constraint #2's pretty-print shows one
  irregular sibling salvaged off.  Constraint #3 collapses to its inner
  atom.
* The **DFA size comparison** shows that `minimize_dfa` (the literal step-5
  of the diagram) removes a non-trivial number of states from the raw
  product.
* The **corpus acceptance** table verifies every labelled bucket matches the
  DFA's verdict — valid sequences are accepted, every invalid bucket is
  rejected by the rule it violates.
* The **training loop** prints per-batch gradient + parameter-update
  snapshots so it is visible that the compact head is actually being trained
  under the compiled DFA.
* The **inference** section reports both unconstrained greedy and DFA-masked
  greedy outputs, and verifies that each one is accepted by the DFA.
