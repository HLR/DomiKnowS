# Compiled Logical-Constraint Execution

This package accelerates DomiKnowS logical-constraint evaluation by separating
the static constraint formula from the batch-specific `DataNode` topology and
prediction tensors. It preserves the native logical operators and solver
semantics while avoiding repeated Python-level plan construction and
per-grounding probability reads.

## Enable It

Use `compile_lc=True` when constructing a constraint model:

```python
from domiknows.program.model.lossModel import PrimalDualModel

model = PrimalDualModel(
    graph,
    tnorm="P",
    device="cpu",
    compile_lc=True,
)
```

The equivalent direct `DataNode` APIs accept `compiled=True`:

```python
losses = root_datanode.calculateLcLoss(tnorm="P", compiled=True)
verification = root_datanode.verifyResultsLC(compiled=True)
root_datanode.inferILPResults(*concepts, compiled=True)
answers = root_datanode.inferExecutableResults(
    *concepts,
    mode="tnorm",
    compiled=True,
)
```

Normal applications should use these flags rather than construct the classes
in this package directly.

## What Is Compiled

The compilation boundary is deliberately split into static plans and a
per-`DataNode` binding:

1. `CompiledPlanCache` stores immutable `CompiledFormulaPlan` instances on the
   solver. Plans are rebuilt when a logical constraint revision changes.
2. `ProbabilityStore` materializes each concept's prediction matrix once for
   the current data item and gathers literal probabilities in batches.
3. `TensorizedCandidateResolver` executes common identity, forward-relation,
   reversed-relation, expansion, and intersection paths with tensors.

Complex or irregular candidate paths retain the established candidate resolver,
so compilation does not replace their existing semantics. Formula plans invoke
each constraint's normal boolean processor implementation; the t-norm and
solver numerics are not reimplemented here.

## Supported Execution Modes

The same cached formula plans are shared by:

- differentiable t-norm loss;
- sampled logical-constraint loss;
- exact circuit / weighted-model-counting loss;
- logical-constraint verification;
- ILP constraint construction; and
- executable constraint inference.

`eqL` is a structural candidate filter rather than a truth-valued formula. It
can participate in candidate resolution but is not independently evaluated as
a compiled formula.

## Module Map

| Module | Responsibility |
| --- | --- |
| `formula.py` | Executes compiled formula plans and provides `CompiledLossCalculator` and `CompiledModeExecutor`. |
| `grounding.py` | Provides `ProbabilityStore`, including batched prediction gathering and `fixedL` handling. |
| `plan.py` | Defines formula and candidate plans, cache invalidation, and tensorized candidate-path execution. |

The package exports the principal extension types from `domiknows.solver.compiled`:
`CompiledConstraintEvaluator`, `CompiledLossCalculator`,
`CompiledModeExecutor`, `CompiledPlanCache`, `ProbabilityStore`, and
`TensorizedCandidateResolver`.

## Validation

The regression coverage compares compiled and interpreter results and gradients
for nested formulas and relation paths, and verifies plan sharing across
sampling, circuit, verification, ILP, and executable modes:

```powershell
pytest test_regr/solver/test_compiled_lc.py test_regr/solver/test_compiled_modes.py -q
```