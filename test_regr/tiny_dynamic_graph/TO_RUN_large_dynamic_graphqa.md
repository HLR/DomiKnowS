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
and updated critic weights. The committed full
1,024-concept/10,000-rule profile has not yet been completed end to end.

## What It Mimics

- Learned visual predicates: `name_*` and `attribute_*`.
- KB concepts: `semantic_*` and `capability_*`.
- KB consistency rules: name-to-class, class hierarchy, class-to-capability,
  attribute-to-class, and deterministic distractor implications.
- Set-valued object answers through `miotaL(andL(...))`.
- One shared MLP attached to all atomic predicates through `ModuleLearner`.
- One union graph reused across examples through `Graph.set_active_concepts`.
- Combined executable and graph-global loss through `InferenceProgram.train`.
- Optional compiled per-grounding duals through
  `PrimalDualProgram(..., dual_granularity="amortized")`.

The default manifest declares 1,024 learned concepts and 10,000 global rules,
then creates 2,000 mock scenes with 12 objects each. Each scene activates its
target proof chain, candidate concepts, and 24 distractor concepts. Inactive
sensors and rules should be skipped by the dynamic graph.

The 25 executable labels belonging to a scene are grouped into one optimizer
item. The default profile therefore performs 2,000 optimizer items per epoch,
not 50,000 repeated model forwards. Parameterized executable templates also
reduce the full manifest's 50,000 source rows to two compiled constraint
objects; 49,998 rows reuse one of those templates while keeping their own
runtime concept bindings and labels. Full-manifest construction measured 13.482
seconds on the NVIDIA T550.
The graph also caches its activation concept index and invalidates it
automatically when concepts or subgraphs are added or removed.

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

## Recommended Run Order

1. Copy the manifest and reduce it to 64 concepts, 250 rules, 20 examples, and
   one epoch. Confirm non-zero executable and global gradients.
2. Increase to 256 concepts, 2,000 rules, and 200 examples. Record peak CPU RAM,
   GPU RAM, examples/second, active-rule count, and both loss components.
3. Run the committed 1,024-concept/10,000-rule profile only if stage 2 resets
   active graph state correctly and has no zero-loss samples.
4. Compare `global_weight=0` against `global_weight=1` using the same seed.
5. Verify that final `miotaL` exact-set accuracy and KB-rule satisfaction both
   improve; concept accuracy alone is not sufficient.

## Required Checks Before Reporting

- `program.cmodel.last_executable_loss` and `last_global_loss` are finite and
  non-zero on deliberately violated initial examples.
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
- Report graph construction time separately from epoch training time (the CLI
  prints both automatically).
- Do not interpret this synthetic workload as GraphQA accuracy. It only tests
  scalability and training wiring before using real GraphQA instances.

## Follow-Up TODO

- Add relation concepts over object pairs and inverse-relation constraints.
- Replace deterministic mock proof chains with proof neighborhoods extracted
  from real C2-C6 KB facts.
- Cache per-instance active-concept and active-rule closures.
- Add checkpointing and per-epoch JSONL diagnostics before any long run.
- Add a separately marked slow test after the workload has passed manually.
