# CLEVR Inference vs Gumbel Inference

This task compares `InferenceProgram` and `InferenceProgram(..., use_gumbel=True)`
on the same compact CLEVR question examples.

`data/clevr_20_programs.json` contains 40 task-local examples: the original 20
exact-matched regression examples plus 20 generated examples from
`clevr-dataset-gen` covering all nine CLEVR 1.0 template files. The generated
append includes query, count, existence, comparison, same-relation, AND, OR,
and multi-hop families. Runtime does not depend on the large regression
`train/questions.json` or `train/scenes.json` files, or on the generated image
files.

`append_generated_examples.py` can rebuild the generated append from a
`clevr-dataset-gen/output/domiknows_balanced` question/scene pool. It removes
any prior generated append with the same marker before selecting a fresh
balanced 20-example set.

## Generated Examples

The appended examples were generated with the local CLEVR generator at:

```text
..\clevr-dataset-gen
```

This assumes [Blender](https://studio.blender.org/welcome/) is already
installed and available on `PATH`, and that
[facebookresearch/clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen)
has already been cloned from GitHub as a sibling directory of this repo at
`..\clevr-dataset-gen`. On Windows, Blender can be installed with Scoop:

```powershell
scoop install blender
```

Run the generation process from the `RelationalGraph` repo root:

```powershell
Push-Location ..\clevr-dataset-gen\image_generation

blender --background --python render_images.py -- `
  --num_images 12 `
  --filename_prefix CLEVR `
  --split domiknows_balanced `
  --output_image_dir ..\output\domiknows_balanced\images `
  --output_scene_dir ..\output\domiknows_balanced\scenes `
  --output_scene_file ..\output\domiknows_balanced\CLEVR_scenes.json `
  --width 160 --height 120 `
  --render_num_samples 8 `
  --render_min_bounces 2 `
  --render_max_bounces 2 `
  --min_pixels_per_object 0

Pop-Location
Push-Location ..\clevr-dataset-gen\question_generation

python generate_questions.py `
  --input_scene_file ..\output\domiknows_balanced\CLEVR_scenes.json `
  --output_questions_file ..\output\domiknows_balanced\CLEVR_questions.json `
  --template_dir CLEVR_1.0_templates `
  --templates_per_image 60 `
  --instances_per_template 1 `
  --reset_counts_every 1000

Pop-Location

python Tasks\clevr_inference_vs_gumbel\append_generated_examples.py
```

`CLEVR_1.0_templates` has nine template files, and the appended 20-example
selection keeps every one represented:

- `zero_hop.json`
- `one_hop.json`
- `two_hop.json`
- `three_hop.json`
- `same_relate.json`
- `single_and.json`
- `single_or.json`
- `comparison.json`
- `compare_integer.json`

The compact dataset currently covers these nine executable final constraint
forms:

- `query_color`
- `query_size`
- `query_shape`
- `query_material`
- `count`
- `exist`
- `equal_size`
- `equal_integer`
- `less_than`

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
Generated non-query examples train as executable boolean or numeric count
constraints using their CLEVR answers as labels.

## Run

```powershell
uv run --project Tasks\clevr_inference_vs_gumbel python Tasks\clevr_inference_vs_gumbel\main.py
```

Fast smoke run:

```powershell
uv run --project Tasks\clevr_inference_vs_gumbel python Tasks\clevr_inference_vs_gumbel\main.py --epochs 1 --train-items 4 --eval-items 2 --ilp-benchmark-warmup 0 --ilp-benchmark-repeats 1
```

Show English questions from the compact dataset with their translated
DomiKnowS executable constraints:

```powershell
uv run --project Tasks\clevr_inference_vs_gumbel python Tasks\clevr_inference_vs_gumbel\show_translated_constraints.py --limit 5
```

Show only generated examples:

```powershell
uv run --project Tasks\clevr_inference_vs_gumbel python Tasks\clevr_inference_vs_gumbel\show_translated_constraints.py --generated-only
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
query class argmax stays wrong. By default the task trains on 80% of the
compact dataset and evaluates on the remaining 20%; pass `--train-items` and
`--eval-items` to reproduce smaller fixed splits.

To compare executable constraints only, pass:

```powershell
--disable-global-constraint-loss
```

The global and executable loss weights can be adjusted with
`--global-constraint-loss-weight` and `--executable-constraint-loss-weight`.

## Full-Graph vs Dynamic-Graph ILP Benchmark

After training, the task first evaluates one temporary executable count query
with all three `inferExecutableResults` backends: `mode="tnorm"` for fuzzy
graph traversal, `mode="circuit"` for exact weighted model counting, and
`mode="ilp"` for a constrained MAP-world answer. This is an **ad hoc** query
because it is supplied through `queries=...`, returned directly, and not kept
as another registered executable constraint on the graph or DataNode.

The task then benchmarks ILP on the standard `InferenceProgram`. It uses that
same first relation-free count question from the compact dataset and executes
the learned query twice in each pair:

1. **Full graph:** all graph concepts, properties, and applicable constraints
   are active.
2. **Dynamic graph:** only concepts referenced by the current executable query
   are requested. `Graph.set_active_concepts(...)` automatically adds their
   ontology ancestors and the graph constraint concept.

Each configuration receives a fresh DataNode. The benchmark keeps only the
selected sample's registered executable activation label and invokes the
in-place ILP method directly:

```python
datanode.inferILPResults()
```

`DataNode.inferILPResults()` reads the active `ELC*/label`, builds and solves the
constrained ILP hypotheses, writes the selected `<concept>/ILP` assignment, and
persists the executable answer as `ELC*/answer`. These writes occur only on the
fresh benchmark DataNode, which is discarded after that measurement.

No concept arguments are passed to either timed call. `inferILPResults()`
therefore collects every predicate present in that fresh DataNode. The full and
dynamic calls are identical; their only experimental difference is whether the
DataNode was built after `Graph.set_active_concepts(None)` or after activating
the query-specific concept set.

Model forward execution and DataNode construction happen before the timer, so
the reported duration covers the complete `inferILPResults()` call, including
ILP model construction, hypothesis solving, assignment population, and answer
decoding.
The full graph is always measured before the dynamic graph, and the graph is
restored to its all-active state afterward.

The default benchmark discards one warm-up pair and reports the median of three
measured pairs. Configure those counts with:

```powershell
--ilp-benchmark-warmup 1 --ilp-benchmark-repeats 3
```

The report includes:

- full and dynamic median wall-clock time in milliseconds;
- requested query concepts and effective active-concept counts;
- predicates collected into each ILP problem;
- milliseconds saved, percentage time reduction, and speedup ratio;
- both native ILP answers and whether they agree.

Timing is diagnostic rather than a pass/fail threshold because wall-clock
results vary by CPU load and Gurobi environment. Answer agreement and the
reduction in active concepts and ILP predicates provide the semantic and
structural comparison.

## Constraint Translation

`clevr_constraints.py` is the cleaned task-local translator from CLEVR
functional programs into DomiKnowS executable constraints. Attribute-query
questions are translated into:

```python
queryL(attribute_parent, iotaL(...))
```

Generated non-query examples translate to executable boolean or count
constraints such as:

```python
sumL(...)
existsL(...)
equalCountsL(...)
lessL(...)
same_size(...)
```

Answers are mapped to class indices for queried attributes and trained directly
with `NBCrossEntropyLoss`. Count questions keep their integer answer as a
numeric label for `sumL(...)`; boolean CLEVR questions keep a 0/1 label.
