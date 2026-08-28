# Constraint-Compilation Validation Report

**Run date:** 2026-08-27 through 2026-08-28  
**Revision inspected:** `d6c64aead1b963a4c6357b845beb11bdcc5af225` plus the
uncommitted constraint-compilation and benchmark changes documented here  
**Device:** NVIDIA T550 Laptop GPU (4,096 MiB), CUDA 12.8  
**Runtime:** Windows 11, Python 3.12.11, PyTorch 2.10.0+cu128

> **Scope warning:** this is not a GraphQA accuracy report. The constraints and
> connected proof neighborhoods come from real C2-C6 KB facts, but the scene
> features and executable labels are synthetic. The results validate
> constraint compilation, dynamic activation, loss wiring, and scalability.

## Executive Result

The staged workload and both full ablation runs completed. The required
weight-1 improvement gate passed:

| Metric | Before training | After training | Change | Result |
|---|---:|---:|---:|---|
| `miotaL` exact-set accuracy | 0.0000 | 0.0550 | +0.0550 | PASS |
| Hard active-KB-rule satisfaction | 0.728880 | 0.999937 | +0.271057 | PASS |
| Concept accuracy (supporting only) | 0.481167 | 0.869896 | +0.388729 | Informational |

The global-loss ablation was mixed. Weight 1 produced substantially higher KB
satisfaction, but weight 0 was slightly better on exact-set and concept
accuracy after two epochs:

| Final metric | `global_weight=0` | `global_weight=1` | Weight-1 delta |
|---|---:|---:|---:|
| `miotaL` exact-set accuracy | **0.0565** | 0.0550 | -0.0015 |
| KB-rule satisfaction | 0.966387 | **0.999937** | +0.033550 |
| Concept accuracy | **0.875667** | 0.869896 | -0.005771 |

Both ablations used the identical data fingerprint:
`4dddf15103f72b9c9fef1ea231d88679714a31af00d798172340d2436aab7655`.

## Method

- Concept accuracy is binary accuracy for the supervised name and attribute
  concepts over all scene objects.
- Exact-set accuracy thresholds the Product-t-norm conjunction of capability
  and attribute probabilities at 0.5, then requires the predicted object-index
  set to equal the synthetic gold set.
- KB satisfaction thresholds local concept probabilities at 0.5 and measures
  the fraction of active rule/object groundings satisfying `not source or
  target`. Inactive source/target rules are excluded.
- CPU memory is peak process RSS sampled every 50 ms. CUDA memory is PyTorch's
  peak allocated and reserved allocator memory.
- Full weight-0 and weight-1 runs executed concurrently after all smaller gates
  passed. Their individual timings therefore include resource contention and
  should not be treated as single-process peak-throughput measurements.

Every run used `batch_size=1`, Product t-norm, seed 17, CUDA, compiled logical
constraints, grouped executable labels, and the bundled 3,387-fact KB snapshot
covering 15 predicates.

## Staged Results

| Profile | Concepts / rules / scenes / epochs | Construction | Epoch time(s) | Examples/s | CPU RSS MiB | CUDA alloc/reserved MiB |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 | 64 / 250 / 20 / 1 | 5.26 s | 21.09 | 0.948 | 1,556.63 | 25.55 / 30.00 |
| Stage 2 | 256 / 2,000 / 200 / 1 | 6.51 s | 188.71 | 1.060 | 1,741.57 | 70.00 / 74.00 |
| Full-rule smoke | 1,024 / 10,000 / 20 / 1 | 11.79 s | 16.39 | 1.221 | 1,610.78 | 31.08 / 36.00 |
| Full weight 0 | 1,024 / 10,000 / 2,000 / 2 | 22.31 s | 2,350.86; 2,922.91 | 0.758 | 2,427.97 | 246.69 / 252.00 |
| Full weight 1 | 1,024 / 10,000 / 2,000 / 2 | 27.33 s | 2,393.58; 3,014.82 | 0.740 | 2,437.98 | 245.81 / 250.00 |

The full weight-0 training time was 5,273.77 seconds; weight 1 was 5,408.40
seconds. Each emitted and checkpointed one flushed `epoch_complete` record per
epoch.

### Loss and active-rule measurements

| Profile | Mean executable loss | Mean global loss | Zero global items | Active rules min/mean/max | Inactive mismatches |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 15.6463 | 52.5125 | 0 | 85 / 117.70 / 124 | 0 |
| Stage 2 | 14.0734 | 14.2333 | 5 | 43 / 86.30 / 138 | 0 |
| Full-rule smoke | 18.2669 | 74.8996 | 0 | 47 / 66.25 / 93 | 0 |
| Full weight 0 | 11.7045 | 21.6635 | 0 | 38 / 63.92 / 148 | 0 |
| Full weight 1 | 11.4316 | 1.3870 | 2,529 | 38 / 63.92 / 148 | 0 |

All 8,240 recorded executable and global losses were finite. Zero global losses
in stage 2 and weight 1 are satisfied rule sets, not skipped evaluation: every
zero-loss item retained a positive compiled active-rule count equal to its
expected count. Product implication loss is exactly zero when all active
implications are satisfied.

## Required Check Matrix

| Requirement | Evidence | Result |
|---|---|---|
| Finite, non-zero loss on a deliberately violated initial example | Full probe global loss 209.9975 | PASS |
| Global loss backpropagates into the shared MLP | Full probe gradient norm 302.2021 | PASS |
| Only active source/target rules execute | 0 expected-versus-compiled mismatches in all runs | PASS |
| Graph activation resets after iteration | `graph_reset=true` in all five results | PASS |
| Sequential `batch_size=1` | 2,000 optimizer items/epoch; no concurrent loader workers | PASS |
| One shared model computation per scene | 4,000 forwards for 2,000 scenes × 2 epochs | PASS |
| Summary grouping is correct | 2,000 items/epoch and 25 executable rows/item | PASS |
| Full executable formulas are interned | 2 formulas and 49,998 reused rows | PASS |
| Parameterized templates are active | `parameterized_executable_templates=true` | PASS |
| Formula compiler does not fall back | 0 batched-formula and candidate fallbacks | PASS |
| Construction and epoch times are separate | Reported in the staged-results table | PASS |
| Same data for weight ablation | Identical SHA-256 fingerprints | PASS |
| Exact-set accuracy improves for weight 1 | 0.0000 to 0.0550 | PASS |
| KB satisfaction improves for weight 1 | 0.728880 to 0.999937 | PASS |
| Zero loss does not result from skipped constraints | 0 zero-loss items without active rules; 0 zero-loss rule-count mismatches; violated probe loss 209.9975 with gradient norm 302.2021 | PASS |

The original literal zero-loss gate was replaced because a correctly satisfied
implication must have zero loss. The semantic gate fails on a non-finite loss,
a zero or gradient-free deliberately violated probe, a zero-loss item with no
active rules, or an expected-versus-compiled active-rule mismatch. Stage 2 and
the full weight-1 run had zero occurrences of either suspicious zero-loss
category. Their 5 and 2,529 zero-loss items respectively are retained as an
informational count of satisfied active rule sets.

## Compiler and Provenance Findings

- The full manifest compiled 50,000 executable source rows into two canonical
  parameterized formulas and reused them 49,998 times.
- The shared MLP executed exactly once per scene per epoch.
- The full profiles used 3,387 normalized facts across 15 KB predicates and
  connected real-fact neighborhoods.
- Active full-profile scenes evaluated 38–148 of the 10,000 global rules, with
  a mean of 63.9225. No inactive rules appeared in the compiled count.
- The full weight-1 run reached 0.999937 hard KB satisfaction, but this did not
  improve synthetic exact-set accuracy over weight 0. More constraint
  satisfaction therefore does not by itself demonstrate better answer quality.

## Commands

The environment was synchronized and all runs used:

```powershell
uv run --extra cu128 --extra dev python `
  -m test_regr.tiny_dynamic_graph.to_run_dynamic_graphqa_global_constraints `
  --config <profile.json> `
  --device cuda `
  --confirm-run `
  --output test_regr/tiny_dynamic_graph/results/<run>.pt `
  --results-json test_regr/tiny_dynamic_graph/results/<run>.json
```

The full ablation additionally passed `--global-weight 0` or
`--global-weight 1`. Checkpoints contain model, optimizer, constraint-model and
constraint-optimizer states, completed epoch, timing and item records, data
fingerprint, and Python/CPU/CUDA RNG states. A stopped run resumes with
`--resume <checkpoint>`; `--output` may point to the same file.

Raw JSON and checkpoint artifacts remain under the ignored
`test_regr/tiny_dynamic_graph/results/` directory. They are intentionally not
committed.

## Automated Validation

The focused CPU/compiler, dynamic-activation, GraphQA-adapter, metric,
resource-recording, CLI-override, dataset-hash, and deterministic-resume suite
passed under the CUDA-enabled `cu128` environment:

```text
70 passed in 33.83s
```

The command was:

```powershell
uv run --extra cu128 --extra dev pytest -q `
  test_regr/fixes/test_graph_active_concepts.py `
  test_regr/solver/test_compiled_coverage.py `
  test_regr/GraphQA/test_graphqa_adapter.py `
  test_regr/tiny_dynamic_graph/test_graphqa_stress_profiles.py
```

The compiled-coverage suite explicitly moves its complete program, including
auto-device reader and edge sensors, to CPU. This keeps the parity suite a true
CPU test even when CUDA is available; the benchmark profiles themselves ran on
CUDA as reported above.

## Limitations

- These measurements do not use real C2-C6 task pickles, visual features, or
  GraphQA answer labels.
- The two full ablations ran concurrently, so their throughput and peak process
  memory include contention.
- Hard satisfaction and exact-set metrics use a fixed 0.5 threshold; they do
  not measure calibrated probability quality.
- Two epochs are sufficient for this validation but do not establish
  convergence or generalization.
