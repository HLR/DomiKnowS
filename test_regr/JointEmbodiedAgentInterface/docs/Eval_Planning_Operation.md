# Eval_Planning_Operation

This document describes `test_regr.JointEmbodiedAgentInterface.eval_planning`:
how it reconstructs a joint agent, generates VLABench plans from static inputs,
scores them against reference plans, and writes an evaluation report.

It evaluates planning rather than physical execution. The command constructs
and loads the controller to satisfy the combined checkpoint contract, but does
not run controller actions or simulator episodes. Its planning scores are
separate from the simulator end-to-end success described in
[Joint_Operation.md](Joint_Operation.md).

## Operation at a glance

```text
CLI arguments + joint checkpoint + planning dataset + EAI vocabulary source
   |
   v
Apply knowledge tier and held-out task declarations
   |
   v
Load EAI rows to reconstruct vocabulary and joint graph compatibility
Load planning episodes, observation facts, and requested scoring labels
   |
   v
Partition held-out tasks, create seeded train/validation/test splits
Build both domain world/generation graphs and compile their policies
   |
   v
Instantiate shared Qwen/LoRA, EAI/VLABench label heads, and controller
Load checkpoint and validate runtime/model/knowledge compatibility
   |
   v
Select split -> dimension filter -> evaluation limit
   |
   +-- learned tier: predict task identity and rebuild task-derived facts
   +-- optional input probe: alter selected observations
   |
   v
Activate VLABench domain; evaluate without gradients
   |
   +-- greedy compact-label generation under DFA or free-decoding policy
   +-- convert labels to operations
   +-- score SR, PR, SPR, PM independently of validity gating
   +-- separately compute validity-gated reward and grounding diagnostics
   +-- count generation failures as empty predictions
   |
   v
Aggregate overall/dimension scores and print scorecard
Optionally save JSON report and token-decision JSONL trace
```

## Entry points and a typical command

Run from the repository root. For example, in PowerShell:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.eval_planning `
  --checkpoint test_regr/JointEmbodiedAgentInterface/checkpoints/joint_stage2_best.pt `
  --planning-dir test_regr/VLABenchAgentInterface/data/planning `
  --split test --seed 42 `
  --observation-mode visual --knowledge task `
  --output test_regr/JointEmbodiedAgentInterface/results/planning_test_visual.json
```

Replace the checkpoint path with an existing compatible joint checkpoint.
Match the EAI source, model, vocabulary capacities, controller configuration,
and operation budget to that checkpoint. Evaluation does not infer all these
settings from the file before constructing the runtime/model.

The Python API is:

```python
from test_regr.JointEmbodiedAgentInterface.eval_planning import (
    build_eval_parser,
    run_planning_evaluation,
)

args = build_eval_parser().parse_args([
    "--checkpoint", "path/to/joint_checkpoint.pt",
    "--split", "test",
    "--knowledge", "task",
])
report = run_planning_evaluation(args)
```

`run_planning_evaluation()` returns the report dictionary. The module's
`main()` parses arguments, runs evaluation, and returns zero on normal
completion. `main.py evaluate-planning` delegates to the same runner, but has
its own parser with a different option surface; this document's commands use
the dedicated module parser.

## 1. Establish the runtime and checkpoint contract

The runner first checks that `--checkpoint` exists, applies knowledge and
holdout settings, and seeds Python and PyTorch with `--seed` (default 42).
The effective knowledge tier must be selected before graph construction.

`_prepare_eval()` loads EAI rows to reconstruct its vocabulary, action/object
compatibility sets, and generation graph. No EAI predictions or EAI goal
metrics are evaluated here. The EAI source remains necessary because the
checkpoint stores both domains' heads and runtime compatibility information.

Default reconstruction settings include:

| Setting | Default | Purpose |
|---|---|---|
| `--eai-dataset`, `--eai-split` | `all`, `train` | Recreate EAI data/vocabulary source |
| `--eai-data-path`, `--eai-limit` | Unset | Optional local source and row cap |
| `--eai-max-steps` | 60 | EAI generation-policy budget |
| `--planner-model` | `Qwen/Qwen3-VL-8B-Instruct` | Reconstruct shared backbone |
| `--load-in-4bit` | Enabled | Load the base model in NF4 |
| `--planner-decoder-hidden-dim` | 512 | Retained configuration/compatibility value |
| `--max-entities` | 64 | Compact VLABench pointer vocabulary capacity |
| `--max-operations` | 12 | VLABench plan policy budget |
| Controller vision model | `google/siglip-base-patch16-224` | Reconstruct controller state |
| `--image-size` | `224 224` | Controller encoder dimensions |
| `--action-horizon`, `--hidden-dim`, `--max-views` | 16, 256, 4 | Controller compatibility |

The current joint planner has two linear domain heads over the shared causal
Qwen transformer, not a separate recurrent planner decoder. The decoder-width
option is retained in configuration; it does not create a GRU. Controller
settings are for deserialization and do not resize the planner's static images
through the controller.

The runner constructs the planner and controller, then calls
`load_joint_checkpoint()` without optimizer arguments. The loader restores
learned weights and saved random states, validates compatibility, and resets
graph activation. Thus the requested seed determines data splitting before
load, while checkpoint loading subsequently restores its saved RNG states.
Generation itself is greedy in evaluation mode.

Checks cover domain/runtime definitions, vocabularies, DFAs, model/controller
configuration, parameter ownership, and recorded logical-knowledge provenance.
Unsupported or standalone checkpoint formats fail. Stage 2 checkpoints marked
`retention_eligible: false` are rejected, as are older Stage 2 versions that
predate the supported robot-frame contract.

`--allow-planning-schema-migration` permits the explicitly supported audited
VLABench schema/budget migration while retaining other compatibility checks.
It is not a general ignore-mismatch switch. Knowledge activation changes and
definition changes are tracked separately; historical files without a
knowledge manifest report `unrecorded_legacy`.

`--device` defaults to `cuda:0` when CUDA is available, otherwise CPU. The
planner loader independently uses automatic device mapping when CUDA is
available; the argument is not a guarantee that every planner tensor resides
on that one device. For GPU run records, retain the visible physical GPU
assignment and actual model placement alongside the command.

## 2. Load static episodes and choose scoring labels

`--planning-dir` defaults to the standalone VLABench package's `data/planning`
directory. The loader traverses episode `output` directories in sorted order
and reads the associated instruction, reference operations, input images, and
optional environment configuration.

Each `PlanningExample` includes:

- episode ID, task name, and mapped dimension;
- instruction and ordinary/segmented image paths;
- scene entity names indexed by compact object pointers;
- reference operation sequence and dependency metadata;
- observation-grounded planning facts and their coverage/profile;
- available task-contract observations and targets.

Entity names come from scene `components`, `scene`, and `entities` fields.
Reference operations are scoring targets. Dependency metadata uses the output
payload first, then environment configuration, then the bundled task default.
The fallback is sequential unless the task is declared sequence-independent.

By default, references are the labels stored on disk. `--updated-data` (alias
`--updated-labels`) computes the corrected density material/button reference
in memory. It changes scoring references without rewriting the dataset or
adding corrected answers to model inputs. Reports record either `as-stored`
or `density-material-button-v1`. As-stored means original only when the supplied
dataset actually contains original labels.

## 3. Select examples in the implemented order

The evaluator loads the full planning collection before applying its evaluation
limit. Held-out tasks are partitioned first, then the remaining collection is
shuffled with the requested seed and split 80/10/10, with adjustments for small
collections.

| `--split` | Selection |
|---|---|
| `validation` (default) | Seeded validation subset of seen tasks |
| `test` | Seeded test subset of seen tasks |
| `train` | Seeded training subset of seen tasks |
| `heldout` | Episodes assigned to `--holdout-tasks` |
| `all` | Reload the supplied full collection without selecting a held-out subset |

`--dimension` then performs a case-insensitive substring match against each
example's dimension or task name. It is not restricted to the six dimension
labels. Quote `"M&T"` in PowerShell.

Next, positive `--limit N` takes the first N remaining examples. It limits
evaluation after splitting/filtering, rather than the collection used to form
the split, and does not avoid loading the model or compiling both domains.
Omit the limit for full evaluation. The current truthiness check treats zero
as no cap, while negative values perform Python tail slicing; use positive
values for bounded evaluations. Empty selections raise an error.

For `--knowledge learned`, task prediction happens after selection, and the
classifier rejects overlap with its recorded training episode IDs. Input
probes run last, so their donor pool is the selected evaluation collection.

### Held-out knowledge behavior

`--holdout-tasks` accepts comma-separated task names or `default` for the
bundled task-family selection. Without `--declare-heldout-knowledge`, those
task declarations are excluded before graph construction, and the `heldout`
examples lose task-derived planning facts, targets, and declared skill sequence.
Their task names remain available for scoring metadata.

The loader receives an explicit knowledge-ablation allowance for held-out
declaration removal. EAI compatibility remains checked. Keeping held-out
knowledge is an explicit experimental alternative.

`--split all` reloads raw evaluation examples rather than using the partitioned
`heldout` list, so it does not apply that list's per-example fact stripping.
Use `heldout` for the intended held-out-task comparison. No evaluation flag
changes which episodes a previously trained checkpoint actually saw.

## 4. Keep observation and knowledge settings explicit

### Observation modes

`--observation-mode visual` is the default. For physics QA, it leaves property
choice to the model from images, instruction, and scene option identities.
It does not use material-property metadata, reflection sidecars, or stored
answer targets to choose the button. Structural QA constraints still apply.
Unavailable symbolic property checks are reported as coverage limitations;
they do not remove those episodes from answer scoring.

`metadata-assisted` enables explicit property observations and qualitative
adapters. Pair it explicitly with `--knowledge metadata` and report it
separately. Visual mode is a QA property-choice contract within the broader
scene/entity interface, not a promise to remove every metadata field from
every task.

`--ground-complex` defaults to enabled and adds requirements grounded from
input language/scene information. `--no-ground-complex` selects the
representation-only comparison; it leaves scoring references in place.

### Knowledge tiers

| `--knowledge` | Constraint/fact profile |
|---|---|
| `grammar` | Plan-language grammar only |
| `generic` | Generic domain rules |
| `scene` | Generic and scene-grounded rules |
| `learned` | Task rules driven by predicted task identity |
| `task` | Task rules driven by the episode task name |
| `metadata` | Task profile with metadata/property observations available |

When omitted, the effective default comes from `VLABENCH_KNOWLEDGE` or the
package default `task`. Use an explicit value for reproducible CLI runs.
The `task` tier uses privileged benchmark task identity. The `learned` tier
requires `--task-classifier`; it rebuilds task-derived observations using the
prediction instead of simply retaining the true task's facts.

Changing a tier can change which rules and facts are enforced. It does not
change metric formulas. Checkpoint definition/provenance checks still apply;
a tier flag is not permission to ignore an incompatible model or vocabulary.

## 5. Generate one plan per selected example

Evaluation runs under `torch.no_grad()` and the VLABench domain scope. The
planner is in evaluation mode. The shared helper prefers segmented images
when available and otherwise uses ordinary images.

For each example:

1. Bind the VLABench policy to the example's supplied context.
2. Pass instruction, selected images, and entity table to the domain planner.
3. Rebuild the prompt plus generated compact-label prefix at each step.
4. Run Qwen with `use_cache=False` and project the boundary hidden state through
   `label_heads["vlabench"]`.
5. Mask disallowed labels and choose the highest allowed logit.
6. Advance the policy until EOS or the decoding budget, requiring an accepting
   final state for constrained decoding.
7. Convert labels into structured operations for scoring.

The runtime's compact-label budget is `max_operations * 5 + 1`, default 61.
This is not a native Qwen token cap: one graph label can span several native
tokens. The command does not expose a `--max-new-tokens` option. It uses neither
beam search nor ILP candidate selection in this path.

### Free decoding

`--no-dfa` replaces the planner's decoding policy with `FreeDecoding`, which
allows every compact label and accepts every state. The model still predicts
through the trained VLABench label head; this is not an unconstrained native
VLM JSON-generation baseline.

`--no-dfa-min-steps N` withholds EOS for the first N label decisions when free
decoding is enabled. Its default zero permits immediate EOS. The option has
no decoding effect without `--no-dfa`. The operation converter can ignore
unattached/misordered tokens, so metrics assess reconstructed operations, not
raw label-sequence grammatical correctness.

Free decoding does not disable runtime/checkpoint construction or the separate
post-generation world-validity diagnostics. Read `decoding`, component scores,
and validity together.

### Input probes and decision traces

`--input-probe` defaults to `original`. Other choices are diagnostics:

- `permuted-entities`: reorder entity pointers and remap references/targets to
  preserve scene meaning;
- `no-images`: remove both image collections;
- `shuffled-images`: assign images from a seeded shuffled donor collection;
- `shuffled-instructions`: assign donor instructions while keeping references.

Donor shuffling is not guaranteed to move every example. These probe results
are separate ablations, not ordinary benchmark scores.

`--decision-trace path.jsonl` writes and flushes one record per generated label:
episode ID, prefix, chosen label, allowed labels, probabilities normalized over
the allowed set, and unmasked logits. It observes decisions without changing
their selection. The trace file opens in write mode and is overwritten on a
new run.

## 6. Score planning independently of validity gating

The shared evaluator computes two tracks for every prediction.

### Planning metrics, on a 0-100 scale

| Metric | Meaning |
|---|---|
| SR | Multiset recall of skill names |
| PR | Multiset recall of entity/container values with their argument roles |
| SPR | Multiset recall of skill/entity/container tuples |
| PM | Fraction of matched benchmark subtask dependency-graph nodes |

Numeric entity pointers are resolved through the example's table before
comparison. PM groups operations using the bundled benchmark subtask patterns
and matches node signatures, graph layers, and predecessor signatures without
reusing predicted nodes. It is partial credit, not binary whole-plan equality.
Sequential, sequence-independent, and explicit dependency mappings can yield
different PM values for the same operation ordering.

```text
score        = 0.4 * SR + 0.4 * PR + 0.1 * SPR + 0.1 * PM
legacy_score = 0.4 * SR + 0.4 * PR
```

A perfect reference prediction scores 100 and 80, respectively. For a
reference `pick(apple), place(bowl)` and prediction `pick(banana), place(bowl)`,
the independent metric track gives SR 100, PR 50, SPR 50, PM 0, and score 65.
Recall-style components do not directly penalize extra predicted operations;
PM can change when extra operations alter dependency structure.

Readable plans are scored even if schema or world-validity checks reject them.
Malformed or empty predictions receive zero components and stay in the
denominator. Invalid references or invalid reference dependency graphs raise
errors instead of being silently assigned zero.

### Validity-gated diagnostics, on a 0-1 scale

`score_vlabench_plan()` separately validates operation schemas, pointer/task
restrictions, and applicable world constraints. It produces `reward`, `valid`,
`skill_match`, `entity_match`, `skill_with_entity_match`, and
`exact_graph_match` averages in `detailed_metrics`.

Invalid plans receive zero gated reward. Component matches can remain nonzero,
so a high planning score and a lower validity-gated reward are compatible.
This `reward` is a reference-plan diagnostic, not the simulator return used
by joint Stage 2 training.

## 7. Account for failures and dimension coverage

Exceptions raised inside `generate_plan()` are recorded with example index,
episode ID, error type, and message. The evaluator substitutes an empty
prediction and continues. `SemanticSearchLimitExceeded` contributes to both
`generation_failures` and `semantic_search_limit_failures`.

These failed examples remain in the denominator with zero planning components.
An infrastructure/model error caught during generation can therefore lower
the reported score; inspect failure counts before interpreting a model change.
Errors outside that generation block, such as loading images, reconstructing
the checkpoint, or validating references, can still abort the command.

`overall` is the mean of per-example scores. Dimension means cover M&T,
Spatial, ComSense, Semantic, PhysLaw, and Complex. `macro_score` equally
averages the six dimension scores, independent of their example counts. A
dimension with no evaluated examples has null scores; Macro is null until all
six dimensions are represented.

Unknown tasks count toward overall scores and `unknown_task_count`, but no
dimension, even when a dimension hint is present. `full_task_coverage` exposes
whether all evaluated tasks are mapped. A non-null Macro does not itself
prove full task coverage.

The report preserves the pinned scoring `source_revision` and
`dimension_mapping_revision` in its nested planning metrics. Its
`official_mapping` block additionally aggregates only upstream-mapped tasks
using upstream dimensions. That block still describes this internal static
input/split protocol; it is not the standalone standard VLM protocol with
upstream prompts and names-free inputs.

## 8. Read and save the report

The runner prints progress and a scorecard, then returns a dictionary. JSON
is written only when `--output` is supplied; parent directories are created,
and an existing output file is overwritten after evaluation completes.

| Report field | Interpretation |
|---|---|
| `checkpoint`, `stage`, `epoch` | Loaded checkpoint and saved selection metadata |
| `split`, `examples_evaluated` | Requested selection and actual count |
| `observation_mode`, `knowledge` | Input and constraint contracts |
| `reference_labels`, `updated_data` | Stored or in-memory-corrected labels |
| `decoding`, `free_decoding_min_steps` | DFA/free policy and end-label withholding |
| `holdout_tasks`, `declare_heldout_knowledge`, `task_classifier` | Task-knowledge experiment settings |
| `ground_complex`, `input_probe` | Grounding and diagnostic input variants |
| `knowledge_manifest`, `knowledge_provenance` | Current declarations and load-time provenance result |
| `overall`, `dimensions` | Per-example and per-dimension planning means |
| `macro_score`, `legacy_macro_score` | Equal-dimension totals, possibly null |
| Failure counters | Generation failures and semantic search-limit subset |
| `detailed_metrics` | Gated reward/validity, planning aggregation, errors, and coverage |

`detailed_metrics` includes `generation_errors`, `grounding_coverage`,
`observation_profiles`, `observation_reports`, and `complex_grounding_reports`.
Unavailable grounding describes verification coverage rather than automatically
establishing an incorrect answer.

The aggregate JSON is not a complete per-episode prediction archive or a resume
checkpoint. This module does not automatically produce `.run.json` sidecars,
prediction JSONL, dataset fingerprints, or a saved CLI manifest. Retain the
command, data/checkpoint identities, seed, dimension/limit settings, device
placement, and optional trace alongside a report when comparisons require them.
Rerunning evaluates the selected examples again; an existing report does not
skip completed episodes.

## Principal implementation locations

- [`eval_planning.py`](../eval_planning.py): dedicated parser, runtime setup,
  selection, checkpoint load, scorecard, and JSON report.
- [`models.py`](../models.py): causal compact-label generation, linear heads,
  greedy decoding, and `FreeDecoding`.
- [`checkpoint.py`](../checkpoint.py): joint compatibility checks and supported
  schema migrations.
- [`utils/generation_runtime.py`](../utils/generation_runtime.py): domain
  policies, contextual binding, and graph activation.
- [`utils/knowledge_manifest.py`](../utils/knowledge_manifest.py): formula,
  activation, and compiled-policy provenance.
- [`VLABench dataset`](../../VLABenchAgentInterface/dataset.py): episode loading,
  observation preparation, entity tables, and seeded splits.
- [`Knowledge tiers`](../../VLABenchAgentInterface/knowledge/__init__.py),
  [`holdout`](../../VLABenchAgentInterface/knowledge/holdout.py), and
  [`task classifier`](../../VLABenchAgentInterface/knowledge/task_classifier.py):
  rule activation and task-identity sources.
- [`Input probes`](../../VLABenchAgentInterface/planning_augmentation.py):
  entity remapping and diagnostic observation transformations.
- [`Shared evaluation helper`](../../VLABenchAgentInterface/training.py):
  image lifecycle, generation errors, independent/gated scoring, and coverage.
- [`Planning metrics`](../../VLABenchAgentInterface/planning_metrics.py) and
  [`planner reward`](../../VLABenchAgentInterface/reward.py): metric formulas,
  dependency matching, aggregation, and validity-gated diagnostics.
- [`Density reference repair`](../../VLABenchAgentInterface/repair_density_labels.py):
  optional in-memory reference corrections.
- [`Decision trace`](../../../domiknows/generation/decision_trace.py): per-label
  logits, masks, probabilities, and selections.
