# Dynamic Constraint Improvements and Run Results

This document records the performance and amortized-dual improvements made to
the GraphQA-scale dynamic-constraint workload in
`to_run_dynamic_graphqa_global_constraints.py`, together with the CUDA run
results used to validate them.

## Purpose

The workload holds one union graph containing many learned concepts and KB
rules. Each scene activates only a small proof neighborhood. It combines:

- dynamically active concepts and global constraints;
- supervised executable constraints, including a set-valued `miotaL` query;
- compiled fuzzy constraint evaluation through `compile_lc=True`;
- an optional per-grounding amortized `DualCritic` profile.

The committed full manifest contains 1,024 learned concepts, 10,000 global
rules, 2,000 scenes, 12 objects per scene, and five epochs.

## Environment Used

The measurements below were collected with:

- `uv 0.11.23`;
- PyTorch `2.10.0+cu128`;
- CUDA build 12.8;
- NVIDIA T550 Laptop GPU with 4 GB memory;
- Windows and PowerShell.

The environment was prepared with:

```powershell
uv sync --extra cu128 --extra dev
```

## Improvements Implemented

### 1. Cached dynamic activation metadata

Previously, every property activation check reconstructed the complete concept
index by traversing the graph. With a large union graph, this happened once per
property and dominated execution before tensor evaluation reached CUDA.

The graph now:

- caches the activation concept index and its identity-membership set;
- performs O(1) membership validation for `Concept` objects;
- invalidates the index when a concept or subgraph is added or removed;
- propagates invalidation from a modified subgraph to its parent graphs.

Implementation: `domiknows/graph/graph.py`.

### 2. One optimizer item per scene

Each scene produces two binary labels per object and one set-valued query. For
12 objects, this is 25 executable constraints. Previously, these were treated
as 25 separate optimizer items, repeating the same scene-level model forward
pass and global constraint calculation.

The workload now:

- groups all 25 executable labels into one scene payload;
- allows the executable-constraint switch to select a tuple of constraints;
- preserves the original single-constraint string behavior;
- scans only executable labels active in the current scene;
- updates only executable constraints whose active state changed;
- restricts compiled executable plan binding and evaluation to active names.

The full profile therefore performs 2,000 optimizer items per epoch rather than
50,000, so executable rows no longer cause 50,000 repeated model updates per
epoch.

Implementation areas:

- `domiknows/graph/executable.py`;
- `domiknows/program/model/pytorch.py`;
- `domiknows/program/model/lossModel.py`;
- `domiknows/graph/dataNode.py`;
- `domiknows/solver/compiled/formula.py`;
- `domiknows/solver/lossCalculator.py`;
- `test_regr/tiny_dynamic_graph/to_run_dynamic_graphqa_global_constraints.py`.

### 3. Parameterized executable-formula templates

Many scenes use the same logical structure with different concepts, logical
variable names, inputs, or labels. `Graph.compile_executable(...,
parameterize=True)` now replaces concept identifiers with typed template slots
and alpha-normalizes variable/path literals such as `"x"` and `path="x"`.
Repeated rows bind their actual concepts and labels at runtime while sharing
one immutable formula traversal plan.

Variable equality patterns are preserved: `p("x"), q(path="x")` does not share
a template with `p("x"), q(path="y")`. Numeric, Boolean, and semantic operator
settings also remain part of the template key, so different thresholds or
counting targets cannot be merged accidentally.

The optimization is opt-in and requires compiled executable loss evaluation.
Existing callers retain their previous per-row identities by default. Canonical
exact-expression interning remains available separately through
`deduplicate=True`. The GraphQA stress workload enables parameterization
explicitly.

For the full committed manifest:

| Measurement | Result |
| --- | ---: |
| Source executable rows | 50,000 |
| Parameterized compiled templates | 2 |
| Rows reusing a compiled template | 49,998 |
| Compiled-object reduction | 99.996% |
| Optimizer items after scene grouping | 2,000 |
| Full construction time | 13.482 seconds |

Implementation areas:

- `domiknows/graph/executable.py`;
- `domiknows/graph/graph.py`;
- `test_regr/fixes/test_compile_executable_deduplication.py`;
- `test_regr/tiny_dynamic_graph/to_run_dynamic_graphqa_global_constraints.py`.

### 4. Compiled amortized primal-dual profile

A new CLI profile directly exercises the compiled per-grounding
`groundingFeatures` through a `DualCritic`:

```python
PrimalDualProgram(
    ...,
    compile_lc=True,
    dual_granularity="amortized",
)
```

Select it with:

```text
--program-profile primal-dual-amortized
```

The existing default remains:

```text
--program-profile inference
```

The profiles have different purposes:

| Profile | Constraint objective |
| --- | --- |
| `inference` | Combined supervised executable loss and graph-global loss |
| `primal-dual-amortized` | Compiled global KB loss weighted per grounding by the `DualCritic` |

The amortized profile excludes executable formulas and their supporting
subformulas from the global dual system. They remain grouped scene inputs. The
profile also:

- disables primal-only critic warmup for one-epoch smoke runs;
- moves prediction and constraint-side models to the selected device;
- aligns compiled violation and feature tensors with the dual model's device;
- verifies that critic parameters actually change during training.

## Run Results Before the Improvements

### Full committed profile

The documented full CUDA command started, constructed the graph, and entered
training with 50,000 optimizer rows per epoch. It remained at `0/50000` after
3 minutes 43 seconds of epoch time and was stopped after approximately 4.9
minutes total.

The interrupt stack showed time being spent rebuilding activation metadata in
`Graph._activation_concepts` while iterating active properties. CUDA had not yet
become the meaningful bottleneck.

### Recommended stage-1 profile

The reduced profile used 64 learned concepts, 250 rules, 20 scenes, 12 objects,
and one epoch.

| Measurement | Before |
| --- | ---: |
| Graph construction | 4.466 seconds |
| Optimizer items | 500 |
| Training time | 713.374 seconds |
| Last executable loss | 0.374647 |
| Last global loss | 0.710584 |
| Graph state reset | Yes |

The loss values above came from the final single-row update in the ungrouped
workload.

## Run Results After the Improvements

The same 64-concept, 250-rule, 20-scene, one-epoch CUDA profile was rerun after
the improvements.

### Combined inference profile

| Measurement | After |
| --- | ---: |
| Source executable rows | 500 |
| Parameterized compiled templates | 2 |
| Rows reusing a compiled template | 498 |
| Optimizer items | 20 |
| Executable labels per item | 25 |
| Construction time (post-template validation) | 4.427 seconds |
| Training time (original grouped run) | 16.451 seconds |
| Training time (post-template validation) | 14.092 seconds |
| Last grouped executable loss | 14.619164 |
| Last global loss | 41.226566 |
| Graph state reset | Yes |

The training-time speedup was approximately:

```text
713.374 / 16.451 = 43.36x
```

Grouped and ungrouped loss magnitudes should not be compared directly. The new
executable value aggregates the scene's 25 supervised constraints, while the
old value represented one final executable row.

### Amortized primal-dual profile

| Measurement | Result |
| --- | ---: |
| Optimizer items | 20 |
| Training time | 20.761 seconds |
| Critic device | `cuda:0` |
| Global dual constraints | 250 |
| Critic weights changed | Yes |
| Graph state reset | Yes |

This run confirms that the compiled grounding features are consumed by a real
CUDA-resident amortized critic and influence its parameter update.

## Validation

The affected regression selection completed with:

```text
72 passed, 1 skipped
```

The focused interning and grouped-profile tests completed with:

```text
7 passed
```

Coverage includes:

- activation cache reuse and invalidation;
- parent-cache invalidation after subgraph mutation;
- one grouped optimizer item per scene;
- canonical formula reuse with row-specific runtime labels;
- preservation of the default one-object-per-row behavior;
- tuple-based executable constraint switching;
- compiled grounding-feature alignment;
- CPU amortized critic updates;
- CUDA feature, violation, and critic device alignment;
- graph activation reset after iteration.

The regression tests are in:

- `test_regr/fixes/test_graph_active_concepts.py`;
- `test_regr/tiny_dynamic_graph/test_graphqa_stress_profiles.py`.

## Commands

### Combined inference profile

```powershell
uv run --extra cu128 --extra dev python `
  -m test_regr.tiny_dynamic_graph.to_run_dynamic_graphqa_global_constraints `
  --config test_regr/tiny_dynamic_graph/mock_graphqa_stress_config.json `
  --device cuda `
  --program-profile inference `
  --confirm-run
```

### Amortized primal-dual profile

```powershell
uv run --extra cu128 --extra dev python `
  -m test_regr.tiny_dynamic_graph.to_run_dynamic_graphqa_global_constraints `
  --config test_regr/tiny_dynamic_graph/mock_graphqa_stress_config.json `
  --device cuda `
  --program-profile primal-dual-amortized `
  --confirm-run
```

Without `--confirm-run`, the command prints the resolved workload and exits
before graph construction. The summary now explicitly reports:

- `compiled_executable_rows`;
- `optimizer_items_per_epoch`;
- `executable_rows_per_item`;
- the selected program profile.

On successful completion, it also prints construction time, training time,
compiled-template count, reused-row count, parameterization status, compiler
status, dual granularity, critic parameter count, available final loss values,
and graph-reset status.

## Remaining Limitations

- The committed 1,024-concept/10,000-rule, 2,000-scene, five-epoch profile has
  not yet completed end to end after these improvements.
- Parameterized executable templates currently require `compile_lc=True` and
  `sample=False`; interpreter and sampled executable loss do not consume
  runtime concept slots.
- Active-rule and probability binding still perform some Python work per scene;
  the workload is not ahead-of-time compiled into a single static CUDA graph.
- These results measure a synthetic GraphQA-like wiring and scalability test,
  not GraphQA task accuracy.
- Timing is specific to the NVIDIA T550 test machine and should be remeasured on
  the target training GPU.
