# Resolution summary for the HANDOFF.txt issues

One page. The issues are those reported in `HANDOFF.txt` (found on develop
`c3002b1a`, 2026-09-28); `Resolutions.md` has the details, evidence and every
commit.

**Result.** At the report, 13 of the 38 reproducer tests failed and 11 could not
run (expired license). All 38 pass now with real Gurobi, as do 36 tests added
since (74 in total, `DOMIKNOWS_REPRO_NATIVE=1 python test_regr/ILPtests/run_suite.py`).
The full `test_regr/fixes` directory passes (20,307 passed, 0 failed), and so
does `test_regr/solver`. No reproducer assertion was weakened and none was marked
`xfail`.

| ID | Issue as reported | Resolution | Commit |
|---|---|---|---|
| D01 | Direct miota decoding read the local softmax and the previous ILP world before the winner was populated | Populate the winning world first, decode a miota from it (when it covers every candidate, otherwise from the scores), restore the old world when `populate=False`. The miota answer stays the documented 0/1 list per candidate (a forced-false single candidate gives `[0]`). | `44e593e3`, `36c63067` |
| D02 | Loss-form ILP leaf reader returned `None` for a binary `[0]`/`[1]` | Read the single element of the binary ILP tensor | `669d1c74` |
| D03 | Compiled ILP binding paired `left(x,y)` with `right(x,y)` instead of `right(y,x)` | Joint-grounding alignment now also runs when building the ILP | `361718df` |
| D04 | Two- and three-hop existential conjunctions lost a known witness | Same fix as D03 | `361718df` |
| D05 | A grounding mismatch logged an error and silently dropped the constraint | Raise `ValueError` on a real mismatch; an operand with no groundings at all is skipped with a warning | `bf7e674d`, `5ceb875e` |
| D06 | Default top-four pruning dropped a certified witness at 11^6 rows | The pruning is opt-in (training keeps it); hard decoding is exact | `779b9aa1` |
| D07 | Same script from different working directories shared one solution path | `DOMIKNOWS_LOG_DIR` override; logs namespaced by working directory; never inside `site-packages` | `9b50930e` |
| D08 | Cache clearing | Not a defect (the helper passes its test); no change | none |
| D09 | `collectInferredResults` returned an empty tensor for a populated child | A leaf `Concept` is falsy (`__len__`); compare with `None` | `53827cf5` |
| D10 | `computeIIS` ran after `TIME_LIMIT`; model-building speed unmeasured | IIS only for proven-infeasible models, with `setComputeIIS` / `DOMIKNOWS_COMPUTE_IIS` and skipped in the hypothesis search (`15a1abe2`); performance below | see below |
| T01 | Product implication gave a NaN gradient for a tiny satisfied antecedent | Unit denominator on the branch `torch.where` discards | `9b8cbd62` |
| T02 | Three-operand head loss direction | Not a defect (already correct); no change | none |

**D10 performance** (profiled with a full Gurobi license; Gurobi's own solve is
under 2% of the time, the rest is model construction):

- Quadratic list-membership tests in the datanode graph (`e94084bc`): building
  14,520 nodes 12.5 s to 0.37 s; `findDatanodes` over 6,480 nodes 2.2 s to 0.04 s.
- ILP build (`38e9057c`, `c2a2a36f`, `5a81ee5f`, `9945fba0`): a full solve on an
  80 x 80 pair graph 14.1 s to 5.4 s, with the model checked identical to the
  old one.
- Hypothesis search builds the shared model once and adds/removes each
  hypothesis (`75c1b37a`): 24 hypotheses over 60 items 8.25 s to 1.20 s, same
  answers (12 equivalence cases).

**Found while fixing, not in the report.** With D05 strict, the ILP `queryL`
hypothesis `andL(class(a), iotaL(...))` turned out to have been silently dropped,
so class hypotheses were never constrained: the answer ignored which object was
selected (`dde976c1`). A native test now requires the answer to follow the
selected object.

**Still open.**

- D07: two workers started from the same directory still share the solver output
  files; set `DOMIKNOWS_LOG_DIR` per worker.
- D10: profiled on a synthetic graph; a real workload (for example the clevr
  tasks on a large scene) has not been profiled.
- The native tests need a valid Gurobi license; the one for the gpu2 server was
  expired when last checked.
- Readers of the miota answer outside this repository were not audited; those
  inside were (an earlier "selected positions" format was reverted after it broke
  six `tiny_multi_answer` tests; see `Resolutions.md`).
