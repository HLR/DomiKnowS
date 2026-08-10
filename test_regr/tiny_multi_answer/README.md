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
It may feed relation, Boolean, or counting constraints.

`example_multiAnswerQuery.py` separately demonstrates candidate-aligned
multi-answer querying:

```python
queryL(
    kind,
    miotaL(andL(red("o"), orL(dog(path="o"), cat(path="o")))),
)
```

Its label `[0, 1, -1]` gives one class ID per candidate, with `-1` marking an
unselected candidate. This keeps the original `miotaL` selection example
focused while documenting the distinct multi-answer `queryL` result shape.

`example_relationAnswers.py` demonstrates entity-aligned relation selection:

```python
(pair_src, pair_dst) = pair.has_a(pair_src=object, pair_dst=object)

left("r", path=("x", pair_src.reversed))
ball("y", path=("r", pair_dst))
```

The `.reversed` source role walks from candidate object `x` to its pair rows;
the destination role finds each related `y`. The first bound variable is the
answer axis. Complete `(x, r, y)` groundings are conjoined, then fuzzy-ORed per
`x`, so repeated matching relations still yield one object position. `iotaL`
selects the unique red object left of the ball; `miotaL` independently selects
both objects left of it. The module also runs `inferILPResults`: the active
executable `miotaL` is solved and persisted by `AnswerSolver`, while the
ordinary relation-aware `iotaL` remains a hard graph constraint. It prints the
ILP-derived unique object and multi-answer set alongside the soft results.

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

Run the relation example and its focused regression with:

```bash
uv run python -m test_regr.tiny_multi_answer.example_relationAnswers
uv run python -m pytest test_regr/tiny_multi_answer/test_example_relation_answers.py
```
