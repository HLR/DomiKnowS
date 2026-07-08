# CLEVR Inference vs Gumbel Inference

This task compares `InferenceProgram` and `InferenceProgram(..., use_gumbel=True)`
on the same 20 CLEVR question examples.

The original `20_examples_string_CLEVR.json` file contains only question text
and answers. `data/clevr_20_programs.json` is a compact task-local snapshot
created by exact-matching those questions against the regression CLEVR train
questions and scenes. Runtime does not depend on the large regression
`train/questions.json` or `train/scenes.json` files.

## What It Compares

- `InferenceProgram`: executable query constraints with standard soft
  predictions.
- `InferenceProgram(..., use_gumbel=True)`: same graph, sensors, data, seed,
  optimizer, and loss, but constraint training uses Gumbel-Softmax.

Both programs use lightweight trainable classifiers over deterministic
object/pair features. This keeps the example small while still exercising
constraint-driven learning.

Graph-global visual constraints are included in the constraint loss by default
with weight `0.1`. In the default `legacy` relation syntax this includes the
opposite-direction spatial constraints from `apply_opposite_constraints(...)`;
executable per-question `queryL(...)` constraints are trained at the same time.

## Run

```powershell
uv run --project Tasks\clevr_inference_vs_gumbel python Tasks\clevr_inference_vs_gumbel\main.py
```

Fast smoke run:

```powershell
uv run --project Tasks\clevr_inference_vs_gumbel python Tasks\clevr_inference_vs_gumbel\main.py --epochs 1 --train-items 4 --eval-items 2
```

The output reports before/after executable-query accuracy, average constraint
loss, executable/global loss components, gradient diagnostics, Gumbel
temperature settings, and a winner explanation. Query accuracy is the primary
metric and is computed as `argmax(queryDistribution) == logic_label`.

For `InferenceProgram(..., use_gumbel=True)`, the task reports both
deterministic query accuracy and a sampled Gumbel query accuracy. Deterministic
accuracy remains the winner criterion because it measures the model used at
normal inference time; sampled Gumbel accuracy is diagnostic and can vary with
the sampled Gumbel-Softmax path.

The average constraint loss includes both executable query loss and graph-global
visual constraints when global loss is enabled:

```text
constraint_loss = executable_loss + global_constraint_loss_weight * global_loss
```

A lower constraint loss does not necessarily mean better CLEVR answer accuracy;
it can reflect improved satisfaction of the graph-global constraints while the
query class argmax stays wrong. The default evaluation split has only four
examples, so query accuracy moves in coarse 25-point steps.

To compare executable constraints only, pass:

```powershell
--disable-global-constraint-loss
```

The global and executable loss weights can be adjusted with
`--global-constraint-loss-weight` and `--executable-constraint-loss-weight`.

## Constraint Translation

`clevr_constraints.py` is the cleaned task-local translator from CLEVR
functional programs into DomiKnowS executable constraints. Query questions are
translated into:

```python
queryL(attribute_parent, iotaL(...))
```

Answers are mapped to class indices for the queried attribute and trained
directly with `NBCrossEntropyLoss`.
