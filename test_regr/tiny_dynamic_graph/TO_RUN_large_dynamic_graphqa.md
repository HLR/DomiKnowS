# To Run: GraphQA-Scale Dynamic Global Constraints

The full workload is intentionally **not part of the regression suite**. It is
a collaborator experiment for checking whether regular DomiKnowS training
remains differentiable and stable when one union graph holds many KB rules but
each GraphQA-like instance activates only a small proof neighborhood.

The recommended 64-concept/250-rule/20-scene stage-1 profile has been validated
on an NVIDIA T550 with CUDA 12.8. With scene grouping, the inference profile
completed one epoch in 14.09 seconds with parameterized templates (previously
713.37 seconds across 500 separate rows), and the amortized profile completed
in 20.76 seconds before template parameterization with its critic on `cuda:0`
and updated critic weights. The original committed full
1,024-concept/10,000-rule profile completed five epochs in 28,723.06 seconds;
that run identified repeated graph-wide scans as the dominant remaining cost.

The runtime-indexing update removes those per-item scans. After switching to
real KB proof neighborhoods, a reproducible 1,024-concept/10,000-rule/20-scene
CUDA smoke run completes its epoch in 23.31 seconds (1.165 seconds/item), versus
the original full run's wall-clock average of 2.872 seconds/item. The earlier
synthetic-rule smoke was 22.31 seconds; the real-neighborhood workload adds
connected-fact extraction and a more varied active-rule mix.

## What It Mimics

- Learned visual predicates: `name_*` and `attribute_*`.
- KB concepts: `semantic_*` and `capability_*` slots backed by normalized VQAR
  symbols.
- KB consistency rules: real `TypeOf` and open-attribute facts from the VQAR
  C2-C6 knowledge base, projected through name, attribute, semantic, and
  capability predicate layers with source-fact provenance.
- Set-valued object answers through `miotaL(andL(...))`.
- One shared MLP attached to all atomic predicates through `ModuleLearner`.
- One union graph reused across examples through `Graph.set_active_concepts`.
- Combined executable and graph-global loss through `InferenceProgram.train`.
- Optional compiled per-grounding duals through
  `PrimalDualProgram(..., dual_granularity="amortized")`.

The default manifest declares 1,024 learned concepts and 10,000 global rules,
then creates 2,000 mock scenes with 12 objects each. Each scene anchors on a
real KB fact and activates up to 24 additional learned concepts reached by a
deterministic breadth-first traversal of that fact's proof neighborhood.
Inactive sensors and rules should be skipped by the dynamic graph.

The bundled KB resolves to 3,387 unique normalized facts, including all 1,426
two-column `is_a.facts` edges. The full rule projection uses 3,298 distinct real
facts across 15 predicates. Every committed full-profile scene activates 28
learned concepts from a connected neighborhood containing 11–22 facts.

The 25 executable labels belonging to a scene are grouped into one optimizer
item. The default profile therefore performs 2,000 optimizer items per epoch,
not 50,000 repeated model forwards. Parameterized executable templates also
reduce the full manifest's 50,000 source rows to two compiled constraint
objects; 49,998 rows reuse one of those templates while keeping their own
runtime concept bindings and labels. Full-manifest construction measured 13.482
seconds on the NVIDIA T550.
The graph also caches its activation concept index and invalidates it
automatically when concepts or subgraphs are added or removed.

The compiled loss runtime additionally:

- uses an O(1) graph constraint-revision token;
- indexes `fixedL` constraints once per graph revision;
- retains global constraint snapshots and unary-rule adjacency across items;
- caches active rule subsets by the graph's active-concept frozenset;
- computes the shared GraphQA MLP logits once per scene and reuses concept
  slices until backward clears the autograd-safe cache.

## Safety Gate

Running the module without `--confirm-run` prints the resolved workload size and
exits before graph construction. The command shape and stage-1 profiles below
are validated; the full manifest remains a guarded long run.

### Conda

```bash
conda run --no-capture-output -n CLEVER \
  python -m test_regr.tiny_dynamic_graph.to_run_dynamic_graphqa_global_constraints \
  --config test_regr/tiny_dynamic_graph/mock_graphqa_stress_config.json \
  --device cuda \
  --confirm-run
```

### uv Alternative

Prepare the project environment with a CUDA extra first, for example
`uv sync --extra cu128`, then run:

```bash
uv run python -m test_regr.tiny_dynamic_graph.to_run_dynamic_graphqa_global_constraints \
  --config test_regr/tiny_dynamic_graph/mock_graphqa_stress_config.json \
  --device cuda \
  --confirm-run
```

The bundled C2-C6 KB files are used by default. To use the full external VQAR
copy, add `--kb-dir /path/to/VQAR_data/data/knowledge_base`.

To directly exercise compiled `groundingFeatures` with a per-grounding
`DualCritic`, add the amortized profile:

```bash
uv run --extra cu128 --extra dev python \
  -m test_regr.tiny_dynamic_graph.to_run_dynamic_graphqa_global_constraints \
  --config test_regr/tiny_dynamic_graph/mock_graphqa_stress_config.json \
  --device cuda \
  --program-profile primal-dual-amortized \
  --confirm-run
```

The default `--program-profile inference` retains the combined supervised
executable/global objective. The amortized profile scopes its critic to the
global KB rules; executable labels remain grouped scene inputs but are excluded
from the dual system. Both profiles print construction/training times and their
resolved compiler/dual diagnostics when they finish.
Every epoch also emits a flushed `{"event": "epoch_complete", ...}` JSON record
with its duration and current executable/global losses, even when tqdm is
disabled. The records are retained in `context.epoch_timings`, included in the
final diagnostics as `epoch_timings` (with `mean_epoch_seconds`), and saved in
the checkpoint metadata when `--output` is provided. `--output` is written
atomically after every epoch and can be resumed with `--resume`; use
`--results-json` for atomic machine-readable final diagnostics and
`--global-weight` for a data-preserving ablation override.

The committed stage profiles are:

```bash
# 64 concepts, 250 rules, 20 scenes, 1 epoch
--config test_regr/tiny_dynamic_graph/mock_graphqa_stage1_config.json

# 256 concepts, 2,000 rules, 200 scenes, 1 epoch
--config test_regr/tiny_dynamic_graph/mock_graphqa_stage2_config.json

# 1,024 concepts, 10,000 rules, 20 scenes, 1 epoch
--config test_regr/tiny_dynamic_graph/mock_graphqa_full_rules_smoke_config.json

# 1,024 concepts, 10,000 rules, 2,000 scenes, 2 epochs
--config test_regr/tiny_dynamic_graph/mock_graphqa_full_two_epoch_config.json
```

The completed staged and full ablation results are recorded in
`ConstraintCompilationValidationReport.md`.

## Recommended Run Order

1. Run `mock_graphqa_stage1_config.json` (64 concepts, 250 rules, 20 scenes,
   one epoch). Confirm finite, non-zero executable and global losses, global-loss
   gradients in the shared MLP, and a reset active graph.
2. Run `mock_graphqa_stage2_config.json` and record peak CPU RAM, GPU RAM,
   examples/second, active-rule count, and both loss components. Stop on a
   non-finite loss, a zero deliberate-violation loss, an active-rule mismatch,
   or a graph-reset failure. A zero training loss with a positive matching
   active-rule count means that the active implications are satisfied.
3. Run `mock_graphqa_full_rules_smoke_config.json` (1,024 concepts, 10,000
   rules, 20 scenes, one epoch). Confirm its real-KB provenance diagnostics and
   compare its construction and per-item training time with stage 2.
4. Compare `global_weight=0` against `global_weight=1` at the largest completed
   stage using identical seeds and data. The comparison needs separate output
   files; changing the weight must not regenerate the proof neighborhoods.
5. Only after steps 1–4 pass, run
   `mock_graphqa_full_two_epoch_config.json` (2,000 scenes, two epochs). Defer the
   five-epoch `mock_graphqa_stress_config.json` run until resumable checkpointing
   is implemented.
6. Report synthetic `miotaL` exact-set accuracy and KB-rule satisfaction in
   addition to concept accuracy, clearly labeled as stress-workload metrics.
   Repeat them as GraphQA metrics only after actual C2-C6 task labels replace
   the mock labels.

## Required Checks Before Reporting

- `program.cmodel.last_executable_loss` and `last_global_loss` are finite and
  non-zero on deliberately violated initial examples.
- Confirm `evaluation.global_loss_validation_gate.passed == true`. Training
  items may have zero global loss when all active implications are satisfied;
  the gate fails only for non-finite loss, a zero/missing-gradient violated
  probe, a zero-loss item with no active rules, or an expected-versus-compiled
  active-rule mismatch.
- Backpropagation from global loss reaches the shared MLP.
- Only constraints whose source and target concepts are active are evaluated.
- The graph resets to all concepts active after dataset iteration, including on
  exceptions.
- Training is sequential with `batch_size=1`; active concepts are mutable graph
  state and are not safe with concurrent data-loader workers.
- Confirm the summary reports `optimizer_items_per_epoch == examples` and
  `executable_rows_per_item == 2 * objects_per_example + 1`.
- Confirm completion diagnostics report `compiled_executable_formulas == 2`,
  `reused_executable_rows == 49998`, and
  `parameterized_executable_templates == true` for the unchanged full manifest.
- Confirm `shared_model_forward_calls == examples * epochs`; a larger count
  indicates that shared logits are being recomputed within an optimizer item.
- Confirm real-KB provenance: `kb_fact_count == 3387`, all expected predicates
  are listed in `kb_predicates`, `proof_neighborhood_count == examples`, and
  `mean_neighborhood_facts` is non-zero when using the bundled KB snapshot.
- Capture exactly one flushed `epoch_complete` record per epoch. Its
  `epoch_seconds`, executable loss, and global loss must agree with the final
  `epoch_timings` diagnostics.
- Report graph construction time separately from epoch training time (the CLI
  prints both automatically), plus examples/second and internally sampled peak
  process RSS and CUDA allocated/reserved memory.
- Do not interpret this workload as GraphQA accuracy. Its constraints and proof
  neighborhoods come from real C2-C6 KB facts, but its scene features and
  executable labels remain synthetic; it currently tests scalability, loss
  wiring, and KB-grounding provenance only.

## Follow-Up TODO

- **Completed:** replace deterministic mock proof chains with connected proof
  neighborhoods extracted from the real C2-C6 KB facts. The workload now keeps
  rule and neighborhood provenance, loads all 1,426 two-column `TypeOf` facts,
  and reports the KB predicate and neighborhood statistics used by each run.
- Add relation concepts over object pairs and inverse-relation constraints.
- **Completed:** run the optimized real-KB 2,000-scene/two-epoch profile for
  both `global_weight=0` and `global_weight=1`, with per-epoch JSON records and
  a deterministic shared-data fingerprint.
- **Completed:** add atomic resumable checkpoints containing model, optimizer,
  constraint state, completed epoch, measurements, and CPU/CUDA RNG state.
- Run the five-epoch profile and compare all five epoch records with the
  original 28,723.06-second synthetic baseline if longer convergence evidence
  is needed.
- Replace the remaining mock scene features and executable labels with actual
  C2-C6 task instances when those task files are available; real KB
  neighborhoods alone do not constitute a GraphQA accuracy evaluation.
- Add a separately marked slow test after the workload has passed manually.
