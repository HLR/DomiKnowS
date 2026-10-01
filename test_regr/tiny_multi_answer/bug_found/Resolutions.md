# Resolutions: status of fixes for the reported ILP issues

Companion to `HANDOFF.txt` and `README.md` in this folder. Those documents
describe the defects as found (13 failed, 14 passed, 11 skipped on develop
`c3002b1a`). This one records what has been done about each.

Status date: 2026-09-30. Branch: `develop-ILP-bug`. The fixes are committed,
one commit per fix (see "Commits"), and are not pushed.

## Commits

In order, on top of `cf3bd575`:

| Commit | Change |
|---|---|
| `53827cf5` | D09: `collectInferredResults` leaf-concept check |
| `9b8cbd62` | T01: finite gradient for the product implication |
| `669d1c74` | D02: binary ILP leaf in the loss-mode reader |
| `9b50930e` | D07: per-worker default log directory |
| `361718df` | D03, D04: join operands over different variable tuples for ILP |
| `bf7e674d` | D05: raise on a grounding mismatch |
| `779b9aa1` | D06: exact joint grounding for hard decoding |
| `44e593e3` | D01: populate the winning world before direct decoding |
| `36c63067` | miota decode returns the selected positions |
| `15a1abe2` | D10: IIS only for proven-infeasible models, with a switch |
| `dde976c1` | queryL hypothesis grounding in ILP mode |
| `e94084bc` | D10 performance: quadratic membership tests in DataNode |
| `1c378fa1` | `pysdd` as a default dependency |
| `0bd49bad` | This document |
| `5eeb56da` | Ignore the suite's `results/` folder |
| `95bcd023` | Regression checks section |
| `a1115c7f` | Fixes for the `test_regr/fixes` failures: path variables of a selector condition bind to the enclosing answer variable |
| `44f3f0a2` | ... batched loss variables are realigned by index after a relation expansion |
| `dcbed231` | ... `queryVar` reads the batched subclass data (it only used the first entity) |
| `9815f7b4` | ... dispatch stress test updated for `use_gumbel` |
| `df280bfb` | ... multivar grounding subprocess retried once on timeout |
| `1b81f1cb` | Regression checks after the `fixes` round |
| `5ceb875e` | D05: an operand with no groundings skips the constraint instead of raising |
| `61b4a485` | `_follow_once` returned three values for an empty node list |
| `4a899b97` | `dummy_datanode` tests load their graphs by file path |
| `19f7da50` | ConllQA: default data file chosen by the requested portion |
| `e754b004` | ConllQA: edge sensor declared after `word['offset']` and `match_phrase` |
| `7620db78` | ConllQA: `--device auto` |
| `fc995655` | ConllQA: per-tensor debug trace opt-in |
| `410990a6` | ConllQA: `result.txt` closed before the accuracy assertion |
| `8086c976` | ConllQA tests: one process per test |

Intermediate commits are not all green. D01's native test needs the miota
commit after it, and the D10 IIS test needs the D10 commit. The suite passes
from the IIS commit onward (47 of 47 there, 56 of 56 at the end).

## Result

| Run | Before | After |
|---|---|---|
| Reproducer suite, default | 13 failed, 14 passed, 11 skipped | 41 passed, 17 skipped (the 6 query tests are native, so they skip) |
| Reproducer suite, `DOMIKNOWS_REPRO_NATIVE=1` | native tests not run | **58 passed, 0 skipped** (38 original, 6 in `test_query_hypothesis.py`, 12 new D10 and IIS tests, 2 for the `dummy_datanode` fixes) |

The native tests were never blocked by the license on this machine. Gurobi
13.0.3 solves a test model here; they skipped only because the opt-in variable
was not set. Run them with:

```sh
cd test_regr/tiny_multi_answer/bug_found
DOMIKNOWS_REPRO_NATIVE=1 ../../../.venv/Scripts/python.exe run_suite.py
```

No reproducer assertion was edited, no `xfail` added, and no grounding limit
raised to force a pass.

## Status per issue

| ID | Status | Fix | Location |
|---|---|---|---|
| D01 | Fixed, native-verified | The winning world is populated before any direct decode (miota and multi-answer query), and the decoder reads the `ILP` key. With `populate=False` the previous world is snapshotted and restored in a `finally`. | `solver/answerModule/answerSolver.py` |
| D02 | Fixed | The loss-mode leaf reader handles a one-element binary ILP tensor instead of indexing `[1]` past its end. The `except IndexError` was not widened. | `solver/logicalConstraintConstructor.py` (`getMLResult`) |
| D03 | Fixed, native-verified | Joint-grounding alignment now also runs when building the ILP model, so `right(y,x)` is no longer paired row-wise with `left(x,y)`. | `logicalConstraintConstructor.py`, `compiled/formula.py` |
| D04 | Fixed, native-verified | Same mechanism: operands over different variable tuples (two- and three-hop chains) are joined, so the known witness survives. The one-hop control still passes. | same as D03 |
| D05 | Fixed | A row-count mismatch now raises `ValueError` naming the constraint and set sizes, instead of returning an empty constraint. An operand with no groundings at all (a nested constraint that found nothing to ground) is skipped with a warning, because that is an empty constraint, not a misalignment. | `graph/logicalConstrain.py` (`createLogicalConstrains`) |
| D06 | Fixed | The top-k soft pruning is now opt-in (`allow_soft_prune`, default off). Training keeps it on, so training behavior is unchanged. `_decode_miota` sets `_exact_grounding`, so hard decoding is exact. | `logicalConstraintConstructor.py`, `compiled/formula.py`, `answerSolver.py` |
| D07 | Fixed for different working directories | `DOMIKNOWS_LOG_DIR` overrides the log directory. A run whose working directory differs from the script's directory gets a `logs/<dirname>_<hash>` subfolder. A script inside `site-packages` (for example `python -m pytest`) no longer puts logs there. | `utils.py` (`_default_log_dir`) |
| D08 | Not a defect | The cache-clearing helper passes its test, and the native repeat test passes. Nothing changed. | none |
| D09 | Fixed | `if not rootConcept` treated a leaf `Concept` (no contained concepts) as missing, because `Concept` is falsy through `__len__`. Now `is None`. | `graph/dataNode.py` (`collectInferredResults`) |
| D10 | Fixed (status) and two measured quadratic hotspots fixed (performance) | `computeIIS()` now runs only for `INFEASIBLE` and `INF_OR_UNBD`, not after `TIME_LIMIT`, and can be switched off with `domiknows.setComputeIIS(False)` or `DOMIKNOWS_COMPUTE_IIS=0`. `findDatanodes` and DataNode link construction no longer use list-membership tests (see "D10 performance work"). | `solver/gurobiILPOntSolver.py`, `utils.py`, `graph/dataNode.py` |
| T01 | Fixed | The product implication uses a unit denominator on the branch `torch.where` discards, so a tiny satisfied antecedent (`1e-30`) no longer overflows to a NaN gradient. | `solver/lcLossBooleanMethods.py` (`ifVar`) |
| T02 | Not a defect | The three-operand head truth tables already pass. Nothing changed. | none |

## IIS diagnostic flag

`computeIIS()` can be as expensive as the solve, and a hypothesis search
(`solve_active_constraints`) only needs to know a model is infeasible. The
diagnostic is now switchable:

- `domiknows.setComputeIIS(False)` / `domiknows.getComputeIIS()`. `setComputeIIS`
  takes a bool and raises `TypeError` otherwise.
- `DOMIKNOWS_COMPUTE_IIS` (`0`, `false`, `no`, `off` disable it) sets the
  initial value when the library is imported.
- The default is on, so existing behavior is unchanged: a proven-infeasible
  model still gets its IIS and `GurobiInfeasible.ilp`.
- When off, a proven-infeasible model skips `computeIIS` and the file write, and
  the solver logs that it skipped. A stale `GurobiInfeasible.ilp` from an
  earlier run is still removed.
- The flag never turns the diagnostic on for `TIME_LIMIT` or any other status;
  only `INFEASIBLE` and `INF_OR_UNBD` are eligible.
- It is a process-wide setting, not per solver.

**Hypothesis search never computes an IIS.** While `solve_active_constraints`
tries joint hypotheses, an infeasible one is an expected outcome that the
search handles itself (`raiseOnInfeasible=False`), so an IIS for it is wasted
work. `_calculateILPSelection` and `processILPModelForP` take an optional
`computeIIS` argument: `None` follows the process-wide flag, and an explicit
`True` or `False` wins for that call. The search passes `False`, so it skips
the IIS even when the flag is on. Direct ILP use (`calculateILPSelection`,
`inferILPResults`) still follows the flag. If every hypothesis is infeasible,
`solve_active_constraints` still raises its `RuntimeError` as before, but no
IIS file is written to explain which constraints conflict.

Tests: `test_performance.py` (`test_D10_iis_*`, including the per-call
override), `test_components.py` (the search passes `computeIIS=False`) and
`test_query_hypothesis.py` (real Gurobi with a forbidden class, so one
hypothesis is infeasible: no `GurobiInfeasible.ilp` is written with the flag
on or off). The last two fail if the search stops passing `computeIIS=False`.

## D10 performance work

Approach: measure first, change only what a measurement supports, keep it apart
from the semantic fixes.

**What was measured.**

- `_collectVariableSetups` (singleton-row assembly) is already linear: about
  20 microseconds per row for 16 operands, up to 65,536 rows (1.3 s).
  Unchanged.
- A cProfile of a full `inferILPResults` on a synthetic graph (unary rules plus
  a pair rule) showed that about 60% of the run was `findDatanodes`, almost all
  of it 2.56 million calls to `DataNode.__eq__` for a 460-node graph.
- Timing graph construction showed the same shape: 2.2x the nodes cost 5.4x the
  time.

**What changed (both in `graph/dataNode.py`).**

1. `findDatanodes` tested `dn in returnDns` on a list, which calls the
   Python-level `__eq__` against every element. `DataNode` equality is by `id`,
   so it now tracks a set of ids beside the list. Results and order are
   identical.
2. `addRelationLink` and `addChildDataNode` tested `dn in <link list>` the
   same way, so the root's `contains` list made construction quadratic. A new
   `_hasLink` keeps a per-relation id set. It is trusted only while it still
   describes the same list object at the same length, so code that replaces or
   shortens the lists directly (`removeRelationLink`, `resetChildDataNode`, the
   list swaps in `inferILPResults`, `executableInference`) triggers a rebuild
   rather than a stale answer.

**Measured effect** (this machine; absolute numbers will differ elsewhere):

| Operation | Before | After |
|---|---|---|
| `findDatanodes` over 6,480 nodes | 2.20 s | 0.04 s |
| Build a graph of 14,520 nodes | 12.5 s | 0.37 s |
| Growth for 4x the nodes (construction) | 19x | 4-6x |
| Growth for 4x the nodes (`findDatanodes`) | 16.5x | linear |

**Probes** (`test_performance.py`): two shape checks compare 4x growth in node
count and assert the ratio stays under 10 (linear is about 4, quadratic about
16), with GC paused so its superlinear cost does not blur the result. A third
test checks the link cache keeps exact list semantics across duplicates,
removal, re-adding and a replaced list. On the baseline library the two shape
checks fail (about 19x and 16x); with the changes they pass. They are shape
checks, not speed limits. On the baseline the construction check passed once
in a repeat run (a timing outlier), so a pass there is weaker evidence than a
fail.

**What was not done or not measurable.**

- End-to-end ILP timings could not be measured at scale: this machine's Gurobi
  license is size-limited (about 2,000 variables and constraints), so a model
  with 30 pair rows on each axis is refused. The improvements above are
  measured on the datanode operations the solver uses, not on a full solve.
- Small end-to-end ILP runs are dominated by a fixed 0.3 s `time.sleep` in
  `move_existing_logfile_with_timestamp`. It is a retry backoff when a locked
  log file cannot be renamed (Windows). It is per solver instance, not
  scaling, and log handling was left alone.
- After the fix the remaining profile is flat (`createILPVariables`, repeated
  `findDatanodes` calls for the same concept, `OrderedSet` visits). Caching
  concept lookups across calls is a possible next step, but no measurement
  shows it matters at the sizes that fit the license.
- `candidates.py` (`dn not in relDns` in the `instanceID` path) has the same
  pattern but was not shown to be hot, so it was left unchanged.

## Behavior changes to be aware of

- **Miota answer format.** `_decode_miota` now returns the positions of the
  candidates at or above the threshold. An empty list means nothing was
  selected. It used to return a 0/1 indicator per candidate. A forced-false
  selector therefore gives `[]` where it used to give `[0]`. The only consumer
  found in `test_regr/Clever` reads `selectionDistribution` from the loss
  path, so it is unaffected. Other consumers were not audited.
- **D05 can surface dropped constraints.** A grounding mismatch now raises. It
  used to be logged and the constraint silently dropped. Any constraint that was
  being dropped will now fail loudly. One such case is described next.
- **ILP joint grounding.** `expandToJointGrounding` gained an `objects` mode
  for ILP values (Gurobi variables, numbers or None). They are gathered in
  Python, unpruned, and rows a relation does not ground read as constant 0. A
  table over `JOINT_GROUNDING_MAX_ROWS` is declined, and the mismatch then
  raises (D05).

## Additional defect found and fixed along the way

Making D05 strict exposed a masked bug. In ILP mode the `queryL` hypothesis
`andL(class(a), iotaL(...))` was silently dropped: `a` had 36 expanded rows and
the selector returned one group of 6 per-candidate variables. Every class
hypothesis was therefore unconstrained, and the answer was chosen by the
objective alone. Broadcasting the scalar was tried and rejected, because it
enforced all 36 rows and made every class infeasible.

The fix has three parts:

1. `selectorResultBinding` records which candidate each selection variable
   belongs to, so a parent connective can align it.
2. ILP joint grounding also joins operands that share a variable tuple but not
   identical row lists.
3. The hypothesis is now `existsL(andL(class(a), iotaL(...)))`, meaning the
   selected object has this class (`answerSolver.py`, `build_query`).

The clevr ad-hoc ILP test
(`test_regr/fixes/test_clevr_inference_vs_gumbel_task.py`) passes again. It
only checks that hypotheses no longer error, so `test_query_hypothesis.py` was
added to check the behavior (next section).

### Verification that the hypothesis changes the answer

`test_query_hypothesis.py` runs real Gurobi. Items have fixed colors (item 0
red, item 1 blue) and the selector is made to pick item `s`. The answer must be
the color of the selected item: `red` for `s=0`, `blue` for `s=1`. A dropped
constraint leaves the class hypotheses tied, so the first one (`red`) always
wins. Two selector shapes are covered, each with `s` in {0, 1}:

- a plain entity selector `iotaL(target('x'))`;
- a relational selector `iotaL(andL(target('a'), rel('a','b'), mark('b')))`,
  where the class variable is expanded over 3x3 pair rows.

| Library | `s=0` | `s=1` |
|---|---|---|
| Baseline (stashed) | pass | **fail**: answers `red`, expected `blue` (both selector shapes) |
| With these fixes | pass | pass |

The first version of these tests also failed after the fix. Even the plain
selector raised the D05 mismatch: the selector result (one group of 3 values)
and the class variable (3 groups of 1) share the same variable and row keys, so
the join was skipped as co-grounded. The join gate now also treats a
single-group-per-row-values operand as needing alignment
(`expandToJointGrounding`, `spreads`).

## Regression checks

Run on the committed code (`df280bfb`, all fixes in place) with the repo
virtualenv (Python 3.12, `pysdd` installed). Baseline means the same tests on
`cf3bd575` with the library changes stashed.

| Suite | Result at HEAD |
|---|---|
| Reproducer suite, `DOMIKNOWS_REPRO_NATIVE=1` | 56 passed, 0 skipped |
| `test_regr/solver`, `graph_errors` and `simple_regression` | 342 passed, 7 skipped, 0 failed (1 m 40 s) |
| `test_regr/fixes` (run alone) | **20,307 passed, 4 skipped, 0 failed** (39 m 31 s) |

The `fixes` run and the working tree it ran on are identical to `df280bfb`:
it started before the last fixes were split into commits, with the same file
contents.

**No test that passed on baseline fails at HEAD, and every failure seen earlier
is gone.** Two earlier rounds had failures; both are resolved.

- `test_regr/solver`: 31 failures on baseline, all `pysdd` tests. Gone now that
  the module is a default dependency.
- `test_regr/fixes`: 5 failures in the first full run, all fixed (next
  section). The 4 real ones also failed on baseline, so they were old defects.

### The five `test_regr/fixes` failures

| Test | Cause | Fix |
|---|---|---|
| `test_inference_program_stress.py::test_stress_train_epoch_dispatch` | Stale test, also failing on baseline. `train_epoch` picks `GumbelPrimalDualProgram` or `PrimalDualProgram` by `use_gumbel`, but the hand-built program had no `use_gumbel` and the test only knew the Gumbel route. | The test sets the attribute and covers both routes (`9815f7b4`). |
| `test_queryl_inference_multiclass.py::test_query_l_executable_returns_query_distribution` and `::test_inference_model_backprops_direct_query_label` | Library bug, also failing on baseline: a path variable in a selector condition could not find its source variable (it belongs to the enclosing constraint), and the realignment after a relation expansion copied a whole batched tensor into every row, so operands of 36 and 6 rows were combined row by row. | `fillPathBindings` also looks at the outer bindings (`a1115c7f`); the new `expandBatchedGroup` realigns batched variables by index (`44f3f0a2`). |
| `test_queryl_inference_multiclass.py::test_godel_query_distribution_is_hard_with_gradient` | Library bug, also failing on baseline: `queryVar` read one value from each per-entity class vector, so the class matrix covered only the first entity and the selection weights were cut to it. With a hard Godel selection on any other entity the gradient was exactly zero. | `queryVar` builds the [entities, subclasses] matrix from the batched columns (`dcbed231`). |
| `test_multivar_executable_grounding.py::test_two_variable_formulas_verify_exactly[s2/q2]` | Not a defect. A `subprocess.TimeoutExpired` (900 s limit on a case that takes about 10 s) in a long run that shared the machine with another heavy suite. Alone the file passes, and the case's runtime is the same as on baseline (about 10.5 s against 10.7 to 12.5 s). | The helper retries once on timeout, so a real hang still fails (`df280bfb`). |

The queryL fixes change loss-mode results for queries that run through
relation-expanded selectors: `queryVar` now weighs all candidate entities, where
it used to see only the first, so the query distribution can differ from what
those paths produced before.

### `test_regr/dummy_datanode`

Run alone it collected but failed 4 of 11 tests on baseline; at `df280bfb` it
failed 5 of 11 (one of them a regression from D05). Run in the same session as
the `Tasks/clevr_inference_vs_gumbel` tests it could not be collected at all.
All 11 pass now (`4a899b97` and the two commits before it), alone and together
with those tests (18 passed with `test_queryl_inference_multiclass.py`).

| Problem | Cause | Fix |
|---|---|---|
| `test_satisfaction_report_execution` failed (new at `df280bfb`, passed on baseline) | D05 also raised when an operand had no groundings at all (`_lc1 has 1 elements and _lc2 as 0 elements`), which used to be skipped. | An operand with no groundings skips the constraint with a warning; operands that all have groundings but in different numbers still raise (`5ceb875e`). |
| 4 tests failed on baseline: `inferILPResults` raised `too many values to unpack (expected 2)` | `_follow_once` returned `[], [], []` for an empty node list while its caller unpacks two values. | Return `[], []` (`61b4a485`). |
| Collection failure next to the clevr tests: `cannot import name 'graph' from 'graph'` | The tests imported `graph` and `graph_multi` through `sys.path`; another test puts `Tasks/clevr_inference_vs_gumbel` first, whose `graph.py` then wins. | Load the two modules beside the test by file path under unique names (`4a899b97`). |

New tests in `test_components.py` cover the first two. They fail on the library
without the fixes, and the native reproducer suite is 58 of 58.

These changes came after the full `test_regr/fixes` run above. After them this
was run again: `solver`, `graph_errors`, `simple_regression`, `dummy_datanode`
and the `fixes` files for queryL, clevr, global rule grounding, LC error
reporting, the stress tests and `existsL` scope (437 passed, 8 skipped, 0
failed). The full `fixes` directory was not repeated.

Not covered: the other `test_regr` directories (`Clever`,
`EmbodiedAgentInterface`, `GraphQA`, `InferenceAPI`, `JointEmbodiedAgentInterface`,
`Reinforcement`, `TemporalRelation`, `VLABenchAgentInterface`, `examples`,
`generation`, `namedTree`, `sensor`, `tiny_dynamic_graph`, `vizual`, plus
`test_common_backbone.py`) were not run, and whether they need data, models or
a GPU was not checked. The intermediate commits were also not run in full; see
"Commits" for what was checked there.

## `test_regr/ConllQA`

Not part of the original ILP issues; run because it exercises the loss path
that several fixes above touch. It could not run at all at the start of this
work (9 of 9 failed), for reasons unrelated to the library.

| Problem | Cause | Fix |
|---|---|---|
| `OSError: Can't find model 'en_core_web_sm'` | The spaCy model the README asks for was not installed. | Installed `en_core_web_sm` 3.8.0 into `.venv` (environment, not repo). |
| `ImportError: cannot import name 'BertModel'` | The venv had `torch 2.10.0+cpu` next to `torchvision 0.25.0+cu128`; that CUDA build cannot load against CPU torch, and `transformers` imports it. | Uninstalled `torchvision` from `.venv` (environment, not repo). Reinstall `torchvision 0.25.0+cpu` if vision code is needed. |
| `FileNotFoundError` for the data file | `--data_path` defaulted to a hardcoded absolute path to a directory that does not exist, pointing at `conllQA2.json`, which holds only `entities_with_relation`; the tests need portions from `conllQA.json`. | Default `None`; use the shipped file that contains the requested portion (`19f7da50`). |
| `KeyError: 'offset'` | The `phrase` EdgeSensor was declared before the word `JointSensor` that defines `'offset'` and before `match_phrase`. | Declared after both, as in `main_og.py` (`e754b004`). |
| `Torch not compiled with CUDA enabled` | `--device` defaulted to a hard `cuda`. | Default `auto`: CUDA if available, else CPU (`7620db78`). |
| One test stalled for over an hour | `logging.basicConfig(level=DEBUG)` at import logged every tensor on every step; pytest captured megabytes. | Default `WARNING`; `CONLLQA_LOG_LEVEL=DEBUG` re-enables it (`fc995655`). |
| Results lost on a failed accuracy check | `result.txt` was never closed. | Closed before the assertion (`410990a6`). |
| Accuracy 0.0 for every test after the first, and `test_general_run` failing | Each test builds its own model and DomiKnowS sensor assignments stack within one process, so later in-process tests see stale sensors. Run in its own process the same test scores normally. | Run each case in its own process by default, with `sys.executable` (`8086c976`). `USE_SUBPROCESS=false` keeps the in-process mode for debugging. |

Evidence for the isolation diagnosis: Godel zero-counting gave 97% as the first
test in a process and 0.0 for Lukasiewicz and product after it;
`over_counting_lukas` run in its own process gave 99.02%; `entities_with_relation`
run standalone gave 93.33% (60 samples), 100% (60 samples, 2 epochs), 93.75%
(400 samples), 99.35% (all 922, 1 epoch) and 99.13% (all 922, 5 epochs, the exact
`test_general_run` configuration).

What was verified:

- One full in-process run: 8 of 9 passed. Those tests only assert exit code 0,
  and after the first test the accuracies in that run are the polluted 0.0 values,
  so it shows the pipeline runs, not that the models learn.
- In subprocess mode, the default now: `over_counting_lukas` (99.02%) and
  `test_general_run` (passed, 36 min 40 s; 99.13% when run standalone).
- Not repeated: the whole file in subprocess mode (about 4 hours on CPU).

The failing `test_general_run` in the earlier subprocess-mode attempt was in fact
executing in-process (its traceback goes through `result = main(args)`); why
`USE_SUBPROCESS` was not `true` in that run was not found. It resolves to `true`
now. On the baseline library the short `entities_with_relation` run scores 0.0
and at HEAD 93.33%, so these library changes help this case.

## Remaining and open

1. **D07, same working directory.** Two workers running the same script from
   the same working directory still share `GurobiSolution.sol` and the other
   solver output files. Set `DOMIKNOWS_LOG_DIR` per worker. The D07 test
   demonstrates path aliasing, not a live write race.
2. **D10 performance, remaining.** Profile a full solve at production scale on
   a machine with an unrestricted Gurobi license before doing more. See the
   list above for what was left unchanged and why.
3. **Miota consumers.** Audit other code that reads the decoded miota list,
   given the format change above.
4. **Line endings.** The repo stores LF. The editing tools wrote CRLF several
   times. Each touched file was converted back, and the working-tree diff shows
   no whole-file changes.

The suite output folder `test_regr/tiny_multi_answer/bug_found/results/` (JSON,
XML and logs written by `run_suite.py`) is listed in the root `.gitignore`.
