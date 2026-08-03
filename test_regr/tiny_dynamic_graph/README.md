# Tiny Dynamic-Graph Example

This example creates two independent DomiKnowS graphs:

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
