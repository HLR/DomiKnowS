# DomiKnowS ILP reproducers

This folder lives at `test_regr/ILPtests` of a DomiKnowS checkout (run the commands
below from the checkout root; the runner finds the checkout by looking for the
`domiknows` package in a parent directory). It contains
tests and diagnostics, **not library repairs**. Tests assert desired behavior
and deliberately remain red when a reported defect is present. There are no
`xfail` markers that disguise failures as success.

## Revision and results

- Repository: https://github.com/HLR/DomiKnowS
- Branch: `develop`
- Commit: `c3002b1a97696846d59b254fb12bde5388ee62ab`
- Source commit date: 2026-09-27T21:39:06-05:00
- Final run: `results/20260928T200637Z.json` (also JUnit XML and text log).
- **13 failed, 14 passed, 11 skipped**, in approximately 1.8 seconds on this host.
- Native opt-in was enabled; all native skips report **expired license**.
- Earlier complete run `20260928T200544Z` has the same failures; the final run
  adds one license-gated auxiliary-construction diagnostic.
- No tracked library changes: the run's `tracked_diff` is empty.

The original user's checkout was deliberately left untouched because its Git
index reports hundreds of deletions alongside untracked copies. This separate
checkout is on develop; the original branch was NOT switched or reset.

## Reproduce

Use a compatible DomiKnowS environment. Tested versions: Python 3.12.4,
PyTorch 2.8.0, pytest 7.4.4, gurobipy 13.0.0; other imports are the checkout's
normal dependencies. No image model, dataset, checkpoint, GPU, API key, or
experiment server is needed. Gurobi installation is needed for imports;
only `test_native.py` needs a working optimizer license.

```sh
python test_regr/ILPtests/run_suite.py
DOMIKNOWS_REPRO_NATIVE=1 python test_regr/ILPtests/run_suite.py
# Focus on one report (including its matching native tests):
python test_regr/ILPtests/run_suite.py -k D03
```

Optional `REPRO_GIT=/path/to/git` selects a usable Git executable for provenance.
The runner writes timestamped JSON/XML/log reports. Exit 1 is expected while
repro assertions fail. No scheduled jobs or remote experiments are launched.

Use a physical runner script. Direct `python -m pytest` failed on this host
because `_default_log_dir()` chose pytest's installation directory and tried
to create `site-packages/pytest/logs`. This import-time logging/permission
issue is distinct from test assertions; it was worked around, not repaired.

## Issue-to-test map

All library paths below are relative to the checkout. A component/backend-spy
test is not a claim that a native Gurobi solve ran.

| ID | Test file and name suffix | Library location | Evidence / required behavior |
|---|---|---|---|
| D01 | `test_components.py`: `direct_decode_uses_winner_and_restores_snapshot` | `solver/answerModule/answerSolver.py`, `solve_active_constraints` | Real orchestration plus solver/decoder spies sees local key and old world. Require winner population before decoding and snapshot restoration for no-populate. Native counterpart included. |
| D02 | `test_components.py`: `binary_ilp_leaf_returns_scalar` | `solver/logicalConstraintConstructor.py`, `getMLResult` | Calls real leaf reader with binary element descriptor `(flag,1,0)` and `[0]`/`[1]` tensors; must return truth, not None. |
| D03 | `test_symbolic_bindings.py`: `compiled_inverse_bindings_fixed_truth` | `solver/compiled/formula.py`, `constructCompiled` | Real compiled binding with ILP flags and numeric truth backend; swapped arguments fail, same-order passes. Native inverse fixture included. |
| D04 | `test_symbolic_bindings.py`: `compiled_join_retains_known_existential_witness` | compiled binding plus `graph/logicalConstrain.py` | Directed two-object chain has a known alternating witness. Two/three-hop formulas return None and no backend calls; one-hop control passes. Native test compares six tiny cases with all 256 atomic worlds. |
| D05 | `test_components.py`: `grounding_mismatch_must_not_silently_drop_constraint` | `graph/logicalConstrain.py`, `createLogicalConstrains` | Direct 4-row/2-row inputs must raise instead of returning an empty constraint list. |
| D06 | `test_components.py`: `hard_witness_survives_joint_expansion` | `solver/logicalConstraintConstructor.py`, `expandToJointGrounding` | Real default expansion loses a certified hard witness at 11^6; 3^6 control passes. No repair flag or modified limit. This is expansion-level, not native selector end-to-end. |
| D07 | `test_components.py`: `same_script_workers_do_not_share_solution_path` | `utils.py`, `_default_log_dir`; solver fixed output basenames | Runs real resolver for two working directories and a common script. Confirms path aliasing, not an actual concurrent overwrite. |
| D08 | `test_components.py`: `clear_cache_visits_linked_nodes_and_preserves_predictions`; native repeat test | `AnswerSolver._clear_ilp_cache` | Helper passes; native repeated model construction is blocked. Do not report a freshly reproduced cache bug from this passing control. |
| D09 | `test_components.py`: `collect_populated_binary_child_results` | `graph/dataNode.py`, `collectInferredResults` | Minimal real graph with populated child returns an empty tensor. |
| D10 | `test_performance.py`, `test_native.py` | logical row collection, `gurobiILPBooleanMethods.py`, `gurobiILPOntSolver.py` | Bounded assembly/name probes; exact expression/name checks; IIS dispatch test with status double. TIME_LIMIT incorrectly calls IIS. Native name/synchronization/repeated-conjunction probes are license-blocked. |
| T01 | `test_components.py`: `product_implication_has_finite_gradient_for_tiny_satisfied_antecedent` | `solver/lcLossBooleanMethods.py`, `ifVar` | Actual FP32 backward produces NaN for a satisfied tiny antecedent. Training-only finding. |
| T02 | `test_components.py`: `three_operand_head_is_violation_not_truth` | `graph/logicalConstrain.py`, variadic head dispatch | All 8 Boolean assignments for AND/OR/NAND pass. Historical loss-direction defect not reproduced. |

## Interpretation and limits

- D03/D04 do not monkeypatch the compiler. Their small backend computes Boolean
  truth on fixed inputs to expose binding errors before optimization. Input
  `<concept>/xP` entries match the native binary index (0).
- D01 spies replace only the external solved-world response and decoder for
  observing orchestration. D10 uses external status/expression doubles, with
  actual library control flow. Tests explicitly label these boundaries.
- D04 and D05 are related root cause and unsafe error handling, not independent
  accuracy penalties. D01/D02/D06 can also interact; fix and test separately.
- D10 timing probes are small, warm-up-sensitive measurements. They do not
  prove a production speedup, a universal complexity bound, or full SAG support.
  Do not enlarge 3D datasets or disable grounding guards just to force a pass.
- Native fixture collection passed, but license-gated bodies could not execute;
  they are prepared reproductions requiring first licensed validation, not
  already certified native results. No license contents are in this bundle.
- Gurobi's expired license is not a DomiKnowS bug. Use a legitimately renewed
  license; these tests do not replace the optimizer or bypass its license.
- Prior generated-rule wiring, metric mismatch, empty-MAP-world semantics and
  zero-rule TemporalQA experiments were application/protocol issues and are not
  presented as core library bugs here.
- No repair, commit, push, pull request, experiment restart, or cleanup was done.

Recommended fix order: identity-correct grounding and fail-closed errors;
winning-world/leaf/exact decoding; output isolation and solver status handling;
then independent performance work. After fixes, all correctness assertions
should pass and native end-to-end coverage should run with a valid license.
