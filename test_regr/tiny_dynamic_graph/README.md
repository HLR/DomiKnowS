# Tiny Dynamic-Graph Examples

## Fresh graph per example

`example.py` creates two independent DomiKnowS graphs:

- `red_dog` declares `red`, `dog`, and `cat`.
- `cat_tree` declares `red`, `cat`, and `tree`.

The graph registry is cleared and rebuilt for each instance, but all predicate
views reference one `SharedConceptClassifier`. A single optimizer is also kept
across graph builds, so training state survives even though the symbolic schema
changes. Each executable is represented by candidate-membership `existsL`
queries, avoiding a fixed answer-label hierarchy.

Run it with:

```bash
conda run -n CLEVER python -m unittest test_regr.tiny_dynamic_graph.test_example
```

Alternatively, run it with `uv`:

```bash
uv run python -m unittest test_regr.tiny_dynamic_graph.test_example
```

## One reusable graph

`example_dynamic_graph.py` declares the union schema (`red`, `dog`, `cat`, and
`tree`) once and constructs one `InferenceProgram`. Before each sequential
training or inference step it calls `Graph.set_active_concepts(...)`. The graph
automatically retains required ancestors and its constraint concept, skips the
inactive predicate sensors, and ignores constraints that reference inactive
concepts. The same graph, program, shared classifier, and optimizer are reused
for both examples.

Run the reusable-graph test with Conda:

```bash
conda run -n CLEVER python -m unittest test_regr.tiny_dynamic_graph.test_example_dynamic_graph
```

Or run it with `uv`:

```bash
uv run python -m unittest test_regr.tiny_dynamic_graph.test_example_dynamic_graph
```
