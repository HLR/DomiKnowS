# Dynamic Graph With Global-Constraint Training

`example_dynamic_global_constraints.py` is a collaborator-facing smoke test for
regular differentiable training with a dynamic graph. It declares one union
schema and a shared six-output MLP through `ModuleLearner`. The mock samples in
`mock_global_constraint_samples.json` activate one concept pair at a time:

- `red -> colored`
- `dog -> animal`
- `tree -> plant`

Atomic labels are compiled into executable `existsL` questions; they are not
attached directly to concept sensors. The implications are normal graph-global
`ifL` constraints. `InferenceProgram` uses
`include_global_constraint_loss=True`, so:

```text
closs = executable_weight * executable_loss
      + global_weight * global_loss
loss  = mloss + beta * closs
```

The MLP starts with all three implications violated. The test confirms that
each sample activates only its own rule, global loss has a non-zero gradient
into the MLP, `program.train` reduces both loss components, the MLP overfits the
mock labels, and active graph state is reset.

```bash
conda run --no-capture-output -n CLEVER \
  python -m test_regr.tiny_dynamic_graph.example_dynamic_global_constraints

conda run --no-capture-output -n CLEVER \
  python -m unittest \
  test_regr.tiny_dynamic_graph.test_example_dynamic_global_constraints
```

The active-concept switch is mutable graph state. Consume
`ActiveConceptDataset` sequentially with `batch_size=1`; do not share this graph
instance across concurrent workers.
