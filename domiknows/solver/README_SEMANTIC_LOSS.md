# Exact semantic loss

DomiKnowS can compile a grounded logical constraint to an SDD or reduced
decision diagram and calculate its exact weighted model count (WMC) with
PyTorch. The training loss is:

```text
semantic_loss = -log(P(constraint is satisfied)) = -log(WMC)
```

The circuit is symbolic, but leaf weights are the model's softmax tensors, so
the loss remains differentiable. Repeated grounded variables use the stable key
`(concept_name, instance_id, class_index)` and therefore denote one logical
variable even when the DSL traversal encounters them multiple times.

## Use from a DataNode

```python
losses = datanode.calculateLcLoss(circuit=True)
for name, result in losses.items():
    print(name, result["probability"], result["loss"])
```

Each result reports `backend`, `nodeCount`, `groundingSignature`, and `cacheHit`
in addition to the WMC probability and loss. Labeled executable `queryL` and
`sumL` constraints are compiled too. An unlabeled `queryL` reports
`queryProbabilities` but has no scalar loss.

## Train with exact semantic loss

```python
from domiknows.program import SemanticLossProgram
from domiknows.program.model.pytorch import PoiModel

program = SemanticLossProgram(
    graph,
    PoiModel,
    poi=[scene, entity, answer],
    beta=1.0,
    circuit_backend="auto",
    circuit_max_nodes=100_000,
    circuit_size_limit_action="raise",
    device="cpu",
)
program.train(train_data, train_epoch_num=20, c_warmup_iters=0)
```

`SemanticLossProgram` uses `model_loss + beta * semantic_loss`. By default the
semantic model sums exact losses directly. Pass `lambda_weighted=True` to use
the existing learned per-constraint multipliers.

## Grounding aggregation

A head constraint has many groundings. `circuit_aggregation` chooses how they
become a loss:

| value | loss | when to use |
|---|---|---|
| `"joint"` (default) | one `-log P(all groundings hold)` | exact joint semantics — a variable shared by several groundings stays **one** logical variable, so cross-grounding dependence is preserved (the t-norm path loses this) |
| `"per_grounding"` | a `[G]` vector of `-log P(grounding)` | keeps the loss scale independent of the grounding count (joint `-log` grows roughly linearly with `G`, so `beta` otherwise needs retuning per task size), and is required by per-grounding dual mechanisms |

```python
losses = datanode.calculateLcLoss(circuit=True, circuitAggregation="per_grounding")
```

Each result also reports `aggregation` and `groundingCount`.

## Composing with the dual mechanisms

`SemanticLossProgram(training_style="primal_dual")` runs the exact loss through
the primal-dual epoch, so the dual machinery applies to it. Set
`lambda_weighted=True` so the multipliers are actually used:

```python
# exact semantic loss under augmented-Lagrangian duals
program = SemanticLossProgram(
    graph, Model, lambda_weighted=True,
    dual_algorithm="augmented", training_style="primal_dual",
)

# exact semantic loss under amortized per-grounding duals;
# per-grounding aggregation is selected automatically
program = SemanticLossProgram(
    graph, Model, lambda_weighted=True,
    dual_granularity="amortized", training_style="primal_dual",
)
```

`training_style="fixed"` (the default) keeps the classic
`model_loss + beta * semantic_loss` objective.

## Reporting how exact a run really was

Constraints whose circuit exceeds the node budget fall back to the Product
t-norm. `SemanticLossModel.exact_fraction` is the fraction of evaluated
constraints that used the exact circuit (`1.0` = fully exact); it resets each
epoch. Check it before trusting a run labelled "exact".

## Backends and installation

The built-in multi-valued BDD backend has no extra dependency. It treats an
`EnumConcept` instance as one categorical random variable, so its WMC uses the
softmax class probabilities rather than an invalid product of independent
Bernoulli class indicators.

When `pysdd` is importable, `backend="auto"` selects the SDD implementation.
Install it as an optional extra:

```powershell
uv sync --extra semantic-loss
```

The SDD WMC arithmetic is evaluated in PyTorch rather than through pysdd's
numeric WMC API, preserving gradients. To force the dependency-free backend in
tests or custom integrations:

```python
from domiknows.solver.circuitBooleanMethods import circuitBooleanMethods

processor = circuitBooleanMethods(backend="bdd")
```

## Supported constraints

The backend implements Boolean operators (`notL`, `andL`, `orL`, `nandL`,
`norL`, `xorL`, `ifL`, and equivalence/`iffL`), cardinality constraints,
count-to-count comparisons, labeled `sumL`, multiclass `sameL`/`differentL`,
exact unique-selection semantics for `iotaL` and `queryL`, and independent
multi-answer selection semantics for `miotaL`.

For `iotaL`, candidate expansions for the same primary entity are OR-combined
into one entity predicate before uniqueness is imposed. For `queryL`, class
`c` is compiled as `OR_i(selection_i AND entity_i_has_class_c)`.
For `miotaL`, each candidate condition remains an independent circuit output;
there is no existence or uniqueness cardinality constraint.

## Circuit size and fallback

`circuitBooleanMethods` accepts `max_nodes` (default `100000`) and
`size_limit_action` (`"raise"` or `"warn"`). The solver catches the default
raise at a constraint boundary, discards the partial circuit, emits a warning,
and returns a Product t-norm result marked with:

```python
result["exact"] = False
result["fallback"] = "circuit-size-limit"
```

This prevents dense relational groundings from stopping a training run while
making the loss-of-exactness explicit. Increase the budget only after checking
`nodeCount` and memory use. Circuit structure is hash-consed and cached for a
constraint plus its grounding and stable-leaf signatures; changed grounding
shapes compile a new structure.
