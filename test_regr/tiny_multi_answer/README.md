# Tiny Multi-Answer Example

This example contains three objects and three learned unary concepts: `red`,
`dog`, and `cat`. The query asks for all objects satisfying:

```python
andL(red("o"), orL(dog(path="o"), cat(path="o")))
```

Because `iotaL` is a unique selector, a set-valued answer is compiled into one
`existsL` membership query per candidate. The expected answer is `{o1, o2}`.
Each unary concept is attached to a tiny linear `ModuleLearner`.

`example_multiAnswers.py` expresses the same question with one set-valued
constraint:

```python
miotaL(
    andL(red("o"), orL(dog(path="o"), cat(path="o"))),
    threshold=0.5,
)
```

Its label is the multi-hot vector `[1, 1, 0]`. `miotaL` keeps independent
membership probabilities (they are not normalized to sum to one), accepts an
empty answer set, and selects values inclusively at `probability >= threshold`.
Use `hard=True` for thresholded forward values with straight-through gradients.
It may feed relation, Boolean, or counting constraints; `queryL` rejects it
because `queryL` represents one multiclass answer.

Run it with:

```bash
conda run -n CLEVER python -m unittest test_regr.tiny_multi_answer.test_example
```

Alternatively, run it with `uv`:

```bash
uv run python -m unittest test_regr.tiny_multi_answer.test_example
```

Run the multi-answer regression with:

```bash
uv run python -m pytest test_regr/tiny_multi_answer/test_example_multi_answers.py
```
