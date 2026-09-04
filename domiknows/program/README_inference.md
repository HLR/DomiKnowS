# InferenceProgram

`InferenceProgram` trains model parameters from executable logical constraints:
logical expressions are evaluated as differentiable soft-logic programs, and the
result is compared with a per-example label.

Use `InferenceProgram(..., use_gumbel=True)` when constraint training should use
Gumbel-Softmax.

## Constraint Types

There are two relevant constraint registries on a graph:

- `graph.logicalConstrains`: graph-level constraints declared directly in the
  graph with functions such as `ifL(...)`, `andL(...)`, `existsL(...)`, or
  `exactL(...)`.
- `graph.executableLCs`: per-example executable constraints created by
  wrapping a logical expression with `execute(...)`. `Graph.compile_executable`
  creates these from strings in the dataset.

`graph.allLogicalConstrainsRecursive` exposes both:

- normal graph constraints as themselves
- executable constraints as their wrapped `innerLC`

That shared registry is used by the loss code for lookup, but the executable
training path still filters by the labels present on the current data item.

## Building Executable Constraints

Typical data has a logical expression and a label:

```python
item = {
    "logic_str": 'existsL(person("x"))',
    "logic_label": torch.tensor([1.0]),
    ...
}
```

Compile it before training:

```python
train_dataset = graph.compile_executable(
    raw_items,
    logic_keyword="logic_str",
    logic_label_keyword="logic_label",
)
```

For each item, `compile_executable` does the following:

1. Reads the string from `logic_keyword`.
2. Wraps it in `execute(...)` unless it is already wrapped.
3. Evaluates it in a namespace containing the graph variables.
4. Stores the wrapper in `graph.executableLCs` as `ELC0`, `ELC1`, ...
5. Adds a constraint `ReaderSensor` on `graph.constraint`.
6. Returns a `LogicDataset`.

`LogicDataset.__getitem__` injects item-local fields such as:

```python
{
    "_constraint_ELC0": logic_label,
    "_constraint_curr_lc_name": "ELC0",
    "_constraint_do_switch": None,
    ...
}
```

The constraint reader stores `_constraint_ELC0` on the constraint datanode as
`ELC0/label`. That label is what activates the executable constraint for the
current example.

## InferenceModel Loss

`InferenceProgram` uses `InferenceModel` as its constraint model.

At each constraint-loss forward pass, `InferenceModel.forward(builder, ...)`:

1. Builds a full `DataNode` from the model output builder.
2. Optionally applies Gumbel-Softmax to local predictions.
3. Reads executable labels from the constraint datanode with
   `datanode.getExecutableConstraintLabels()`.
4. Prepares a shared logical-constraint loss context with
   `datanode._prepareLcLossContext(...)`.
5. Iterates over `self.constr`, which was initialized from
   `graph.allLogicalConstrainsRecursive`.
6. Skips any constraint whose `"{lcName}/label"` is not present in the current
   data item.
7. Computes `datanode.calculateSingleLcLoss(lcName, ...)`.
8. Compares the soft-logic output, `conversionSigmoid`, with the item label
   using binary cross entropy.
9. Optionally computes graph-global constraint loss from
   `graph.logicalConstrains`.
10. Combines the enabled loss components.

The training target is therefore:

```text
executable_loss = sum(
    BCE(soft_logic_value(ELC_i), label_i)
    for ELC_i active on this item
)

global_loss = raw graph logical-constraint loss

closs =
    executable_constraint_loss_weight * executable_loss
  + global_constraint_loss_weight * global_loss

loss = mloss + beta * closs
```

`sumL` executable constraints are special-cased internally: their label is
forced to `1.0`, so they are trained as constraints that should be satisfied.

## Global vs Executable Constraints

Executable behavior is label-filtered:

- Executable constraints are trained when their `ELC*/label` is present on the
  current item.
- Normal graph-level constraints in `graph.logicalConstrains` are trained by the
  optional global loss component.
- If global loss is disabled, `InferenceModel.forward` still raises
  `ValueError` when no executable labels are found on the datanode.
- If global loss is enabled, global-only training items are allowed.

There is one switch for graph-global constraints:

- `include_global_constraint_loss=False`: train only executable constraints.
- `include_global_constraint_loss=True`: train executable constraints plus
  graph-global constraints.

The default is `False` for every training style. If you want graph-global
constraints, turn the switch on explicitly.

Example:

```python
program = InferenceProgram(
    graph,
    SolverModel,
    include_global_constraint_loss=True,
    executable_constraint_loss_weight=1.0,
    global_constraint_loss_weight=1.0,
)
```

Graph-global loss is computed through `datanode.calculateLcLoss(...)`, which
iterates `graph.logicalConstrains`. Executable constraints live in
`graph.executableLCs`, so they are not double-counted in the global component.

The combined constraint loss is:

```text
closs =
    executable_constraint_loss_weight * executable_loss
  + global_constraint_loss_weight * graph_global_loss
```

`sampleGlobalLoss` is different. It is an older sampled-loss aggregation option
used by sample-loss style training. It does not enable or disable the combined
InferenceProgram loss above, and it does not decide whether executable and
graph-global constraints are trained together.

## Training Styles

`InferenceProgram` supports two training styles.

### `training_style="default"`

This is the default behavior:

```python
program = InferenceProgram(
    graph,
    SolverModel,
    poi=[..., graph.constraint],
    tnorm="G",
    inferTypes=["local/argmax"],
)
```

For each item:

```text
mloss, metric, datanode, builder = model(data)
closs = InferenceModel(builder)
loss = mloss + beta * closs
backward(loss)
optimizer.step()
```


### `training_style="primal_dual"`

This style uses the primal-dual epoch loop:

- `use_gumbel=False`: delegates to `PrimalDualProgram.train_epoch`.
- `use_gumbel=True`: delegates to `GumbelPrimalDualProgram.train_epoch`.

Both paths use the same primal-dual scheduling machinery:

- warmup through `c_warmup_iters`
- constraint update frequency through `c_freq`
- optional constraint-only phases from the shared `LossProgram.train` path

The constraint model is still `InferenceModel`. Graph-global constraints are
included only when `include_global_constraint_loss=True`.

## Gumbel-Softmax

Gumbel can be enabled directly:

```python
program = InferenceProgram(
    graph,
    SolverModel,
    use_gumbel=True,
    initial_temp=1.0,
    final_temp=0.1,
    anneal_start_epoch=0,
    hard_gumbel=False,
)
```

When enabled, the constraint model does:

```python
datanode.inferLocal(keys=("softmax",))
datanode.inferGumbelLocal(temperature=current_temp, hard=hard_gumbel)
```

Then the executable soft-logic loss is computed from the Gumbel-modified local
softmax values.

Temperature comes from `GumbelTemperatureMixin`:

- before `anneal_start_epoch`, use `initial_temp`
- during the annealing window, linearly interpolate toward `final_temp`
- after the window, stay at `final_temp`

`InferenceProgram.train(...)` accepts the normal DomiKnowS `train_epoch_num`.
If `anneal_epochs` is omitted and Gumbel is enabled, that epoch count is used as
the annealing window.

## Evaluation

`evaluate_condition(...)` is for executable constraints. It:

1. Populates datanodes.
2. Reads active executable constraint names from the current item's labels.
3. Sets only those executable constraints active.
4. Runs local inference.
5. Verifies each active executable constraint with
   `datanode.verifySingleConstraint(...)`.

It does not evaluate every graph-global constraint unless that constraint is part
of the active executable-label path.

## Practical Checklist

- Include `graph.constraint` in `poi`; otherwise the constraint label reader will
  not populate the `ELC*/label` attributes.
- Call `graph.compile_executable(...)` before training an `InferenceProgram`.
- Use `logic_label` values compatible with BCE, usually `0.0` or `1.0`.
- Use `pos_weight > 1.0` when positive executable labels are rare and the model
  collapses toward negative predictions.
- Use `use_gumbel=True` when the executable constraint needs more discrete-like
  local decisions during training.
- Pass `include_global_constraint_loss=True` when graph-global constraints
  should be trained together with executable constraints.
