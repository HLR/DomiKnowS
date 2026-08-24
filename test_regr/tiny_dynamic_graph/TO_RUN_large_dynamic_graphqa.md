# To Run: GraphQA-Scale Dynamic Global Constraints

This workload is intentionally **not part of the regression suite** and has not
been executed. It is a collaborator experiment for checking whether regular
DomiKnowS training remains differentiable and stable when one union graph holds
many KB rules but each GraphQA-like instance activates only a small proof
neighborhood.

## What It Mimics

- Learned visual predicates: `name_*` and `attribute_*`.
- KB concepts: `semantic_*` and `capability_*`.
- KB consistency rules: name-to-class, class hierarchy, class-to-capability,
  attribute-to-class, and deterministic distractor implications.
- Set-valued object answers through `miotaL(andL(...))`.
- One shared MLP attached to all atomic predicates through `ModuleLearner`.
- One union graph reused across examples through `Graph.set_active_concepts`.
- Combined executable and graph-global loss through `InferenceProgram.train`.

The default manifest declares 1,024 learned concepts and 10,000 global rules,
then creates 2,000 mock scenes with 12 objects each. Each scene activates its
target proof chain, candidate concepts, and 24 distractor concepts. Inactive
sensors and rules should be skipped by the dynamic graph.

## Safety Gate

Running the module without `--confirm-run` prints the resolved workload size and
exits before graph construction. The command below is therefore a plan, not a
command that has already been validated:

```bash
conda run --no-capture-output -n CLEVER \
  python -m test_regr.tiny_dynamic_graph.to_run_dynamic_graphqa_global_constraints \
  --config test_regr/tiny_dynamic_graph/mock_graphqa_stress_config.json \
  --device cuda \
  --confirm-run
```

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
- Report graph construction time separately from epoch training time.
- Do not interpret this synthetic workload as GraphQA accuracy. It only tests
  scalability and training wiring before using real GraphQA instances.

## Follow-Up TODO

- Add relation concepts over object pairs and inverse-relation constraints.
- Replace deterministic mock proof chains with proof neighborhoods extracted
  from real C2-C6 KB facts.
- Cache per-instance active-concept and active-rule closures.
- Add checkpointing and per-epoch JSONL diagnostics before any long run.
- Add a separately marked slow test after the workload has passed manually.
