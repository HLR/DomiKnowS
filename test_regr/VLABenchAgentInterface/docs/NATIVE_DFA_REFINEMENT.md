# Native planning DFA refinement — 14 October 2026

This revision corrects the native frozen-Qwen output protocol. Historical
eight-operation compact-grammar results in the Word report remain historical;
these code changes do not revise those measurements. Graph checkpoint decoding,
training rewards, and checkpoint selection are unchanged.

## Protocol

The default native `benchmark` profile uses the official benchmark subtask
patterns pinned to OpenMOSS/VLABench revision
`cf588fe60c0c7282174fe979f5913170cfe69017`. Its `push` operation requires a
`target_container_name`, matching the planning references. The default native
operation bound is 16 rather than eight, and native IDs span the complete
supplied scene table. The graph planner retains its existing vocabulary and
eight-operation limit. The explicit `legacy` profile retains its former schema
and compact syntax; select an eight-operation limit when reproducing that language.

The byte-level tokenizer projection accepts whitespace outside string literals,
either `name`/`params` member order, and Unicode escapes for ASCII string content.
It rejects split integers, duplicate or extra keys, malformed JSON, invalid IDs,
incomplete subtasks, and EOS before a complete plan. Multi-character tokens can
cross JSON and escape-sequence boundaries.

Source: [official benchmark patterns](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/evaluation/utils.py).

## Instruction object filters

The default `instruction` filter uses explicit object names in supported source
clauses, or instruction object types matched to scene classes, to restrict `pick`.
Explicit destination container names restrict `place`, `insert`, `pour`, and
`push`. Multiple matching objects remain candidates. Unsupported wording,
negation/exclusion, ordinals, spatial descriptions, and source relations cause
abstention. No world-coordinate convention, chemical-color knowledge, or
commonsense answer is guessed.

Only component identities/classes and the instruction enter these rules.
Reference operations, condition/goal fields, and target labels are not read by
the grounding function. Independent evaluation entity tables also exclude
condition/goal fields. The grounded variant therefore explicitly consumes scene
class metadata in addition to the VLM's image/instruction/name table; it is a
separate ablation from formatting and schema correction alone.

Per-example records preserve candidate sets and rule evidence. `native_valid`
checks the native protocol; `grounded_valid` also checks candidate restrictions.
The original `valid` field remains the graph-schema diagnostic, which rejects
benchmark-style `push` arguments. Official planning metrics remain independent
of all these diagnostics. Resume identities distinguish the language revision,
profile, candidate-filter setting, and operation budget. HMM lookahead caches
also distinguish actual candidate sets and the language implementation.

## Verification

- Byte-parser and independent symbolic validation accepted all **4,500** reference
  plans with the benchmark profile and 16-operation bound.
- Instruction filters activated on **1,286** examples. One reference disagrees
  with the resulting candidate set: `select_drink_common_sense/example44` names
  `jumpstart`, which is present in the scene, but its reference selects `cola`.
  The rule retains the explicit instruction interpretation; this disagreement
  is reported, not repaired using the reference answer.
- Focused native, metric, reward, and EAI auxiliary tests: **90 passed**.
- Isolated VLABench, EAI auxiliary, and joint compatibility suite:
  **261 passed, 12 previously known failures**. Failure identities match the
  historical suite exactly; no new failures appeared.
- Native-token tests cover alternate JSON formatting/member order, escape and
  delimiter boundary tokens, strict raw-JSON diagnostics, candidate enforcement,
  exclusion abstention, goal-label isolation, raw generation without masking,
  evaluator resume, and separate native/graph validity. HMM acceptance with
  whitespace cycles matches exhaustive enumeration; zero HMM weight retains
  DFA-only masking.
- Physical GPU5 completed **12 validation cases per configuration** for raw
  Qwen, grammar-only DFA, and instruction-filtered DFA, plus a final one-case
  integration check after the diagnostic changes. All generations completed
  without errors. Each constrained variant was native-valid on all 12 cases.

| Smoke configuration | Cases | Native validity | Pooled planning score |
|---|---:|---:|---:|
| Raw Qwen | 12 | 66.67% | 60.92 |
| Benchmark DFA, no object filter | 12 | 100% | 65.96 |
| Benchmark DFA, instruction filter | 12 | 100% | 65.96 |

The smoke manifest covers primitive, semantic, spatial, push, and long-plan task
families from validation. It is a protocol integration check, **not a full
accuracy estimate or evidence of an object-filter improvement**. Old HMM weights
must be reselected on validation before reporting new-grammar HMM results.

Remote isolated evidence:
`/workspace/.codex-experiments/vlabench_refined_20261014/results/` in
`vigorous_easley` on `gpu2.ihmc.us`. Logs include `raw_none.log`, `dfa_none.log`,
`dfa_instruction.log`, and `regression_tests.log`; structured evidence includes
`reference_audit.json`, `regression_comparison.json`, and per-example JSONL files.
The GPU UUID is `GPU-f1ffcf33-15e9-956d-c69e-40892f2aeb86` (visible as `cuda:0`).
Generated evidence remains ignored by Git.

## Controlled evaluation commands

Use a fresh output directory and the identical manifest/model for every variant:

```bash
export CUDA_VISIBLE_DEVICES=GPU-f1ffcf33-15e9-956d-c69e-40892f2aeb86
python -m test_regr.VLABenchAgentInterface.planning_experiment evaluate \
  --mode raw --grammar-profile benchmark --object-constraints none \
  --max-operations 16 --planning-dir /path/to/planning \
  --manifest /path/to/validation_manifest.json \
  --model /path/to/Qwen3-VL-8B-Instruct --output /path/to/new_run/raw
```

Repeat with `--mode dfa --object-constraints none --output /path/to/new_run/dfa`,
then `--mode dfa --object-constraints instruction --output /path/to/new_run/grounded`.
Keep the remaining arguments identical. Validation should establish the revised
configuration before a new complete test comparison is reported.

## Complete 450 example test comparison

The subsequent full run evaluated all three configurations on the identical
450-example manifest, with fresh raw-Qwen outputs under the revised prompt.
Independent checks against the pinned official scorer matched all **1,350**
records and aggregates. All 2,250 test data files matched their initial hashes
at the end of the run; code, model, and tokenizer fingerprints matched across
configurations.

| Configuration | Macro | Pooled score | Native validity |
|---|---:|---:|---:|
| Fresh raw Qwen | 50.2310 | 55.7787 | 78.89% |
| Revised DFA without object filter | 50.1428 | 56.1046 | 100% |
| Revised DFA with instruction filter | 50.3884 | 56.4046 | 100% |

Object filtering adds **0.2456 Macro points** and **0.3000 pooled points** over
DFA alone: three examples improve, none worsen, and 447 retain the same score.
The rules activate on 134 examples but change only three plans. Each of the
three changed scores rises from 20 to 65; remaining action-sequence errors
prevent perfect scores. Grammar-only DFA preserves all 355 structurally valid
raw plans; eight examples improve, ten worsen, and 432 retain their raw score.

Relative to fresh raw Qwen, the combined change is **+0.1574 Macro points** and
**+0.6259 pooled points**. The 95% paired-bootstrap Macro interval is
**[-0.5200, +0.8521]**, so this is a small observed gain rather than clear
evidence of a general planning improvement. The bootstrap uses 20,000 resamples
with seed 42, preserving dimension strata for Macro. Relative to the earlier
compact DFA Macro of 50.3063, the revised filtered result improves by only
**0.0821 points**; that historical comparison combines prompt and grammar
changes. The scene-table exclusion changed none of these 450 test tables.
The new filtered Macro is also 0.2284 points below the earlier raw-Qwen result
of 50.6168; the current prompt differs, so the fresh raw baseline is the
controlled comparison.

The measured comparison is retrospective on previously evaluated examples.
An untouched evaluation collection is needed before claiming improved
generalization. No new HMM result, simulator-success result, or comparison
against published CLAMP accuracy is inferred from these planning scores.

Full evidence is retained under
`/workspace/.codex-experiments/vlabench_refined_20261014/results_full/`, including
`full_comparison.json`, `independent_verification.json`, paired changes,
per-example generations, checksums, logs, and host telemetry. Physical GPU5 was
shared with an unrelated simulator-training process; runtimes reflect that
shared use. The reproducible sequential driver is `run_refined_full.py` and the
independent analysis script is `analyze_full.py` in the isolated code directory.
