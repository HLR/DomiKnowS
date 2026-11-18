# LIBERO object-transfer diagnostics

Implementation provenance correction (17 November 2026): the historical
`contract`, `raw-contract` and `aligned-contract` measurements below used a
handwritten Python contract. They are preserved historical measurements, not
results of the DomiKnowS-compiled replacement in `executor_graph.py`. The new
implementation passed 76 regression checks and exact replay of 2,100 archived
token masks on 300 episodes (zero mask or greedy-choice differences). This was
CPU replay, not fresh neural inference or physical execution; never relabel
these archived runs as having executed the replacement. Evidence is in
`/home/auszok/experiments/joint_executor_lc_validation_20261117/replay/summary.json`.

Frozen continuation epoch-2 snapshot; no training updates. Same 100 previously inspected episodes. Baseline generation and scores reproduced; baseline physical execution is the explicitly reused archive. Five intervention executions are fresh, using the unchanged 96/100-calibrated executor and 1,000-step horizon.

| Condition | Native /100 | Object /50 | Spatial /50 | Valid /100 | Grounder source Object /50 | Planner source Object /50 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 23 | 4 | 19 | 47 | 1 | 12 |
| raw | 24 | 5 | 19 | 44 | 6 | 16 |
| contract | 34 | 12 | 22 | 100 | 1 | 12 |
| raw-contract | 38 | 16 | 22 | 100 | 6 | 16 |
| aligned | 28 | 3 | 25 | 40 | 31 | 27 |
| aligned-contract | 57 | 27 | 30 | 100 | 31 | 27 |

`raw`: learned head queries original public instruction. `contract`: executor pick/place schemas, range/static-fixture/holding preconditions, empty-hand completion; no task goal or answer template. `aligned`: instruction role phrases ranked against RGB-only object regions and context with frozen paired SigLIP features, all visible candidates; no category lookup or hidden reference inputs.

Per-cohort native confidence intervals and paired task-cluster bootstrap intervals are in results.json. Episode Wilson intervals do not account for within-task scene correlation. Nearest competing objects and margins are recorded, but no new hard-negative training has been performed. This is a diagnostic regression on inspected scenes; a further unseen-scene evaluation is required for a generalization claim.

Validation and provenance: 36 distinct checks covered (34 current interface/reporting tests plus two unchanged native-DFA contract tests). All 15 recorded jobs exited zero. Baseline predictions, rendered inputs, validity and abstract scores reproduced 100/100 archived episodes. Public observations matched across all conditions; rendered planner prompts matched within each constraint pair. The default raw-query API integration was separately tested with read-only overlays; see postflight/interface_manifest.json and validation.json. Frozen generation source and original artifacts were preserved.

Paired native improvement for aligned-contract versus baseline: +34 percentage points, 95% task-cluster bootstrap interval +18 to +50 (4,000 resamples, seed 42, 20 task clusters). It gained 37 episode successes and lost three, all Spatial; Object gained 23 and lost none. This does not establish a new untouched-test generalization result.

Artifact root: `/home/auszok/experiments/joint_object_transfer_fixes_20261113_attempt2`. Exact PIDs, physical GPU UUIDs, container images, commands, logs and exit codes are in `COMMANDS.md` and `jobs/*.json`. Inference used completely free physical GPUs 3 and 6 on gpu2.ihmc.us; native execution used CPU software rendering. Source checkpoint SHA256: `24642fd2af0d9395eeb5547f3e3c89e33328bfafc743d368df5e02d989492986`.
