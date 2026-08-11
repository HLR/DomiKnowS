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

## Nested Relation Hops

Relation-aware selectors can continue through additional pair variables while
remaining aligned to the first bound entity. For example:

```python
miotaL(andL(
    object("x"),
    left("r1", path=("x", pair_src.reversed)),
    ball("y", path=("r1", pair_dst)),
    left("r2", path=("y", pair_src.reversed)),
    ball("z", path=("r2", pair_dst)),
), threshold=0.5)
```

This means: return every `x` for which there is a complete
`(x, r1, y, r2, z)` grounding satisfying all predicates. The result still has
one position per `x`; intermediate objects and relation rows never become
output positions. Earlier-hop values are projected onto the later expanded
rows, complete paths are conjoined, and paths sharing the same `x` are
fuzzy-ORed.

The focused regression generates equivalent constraints with 3, 4, and 5
relation hops. It uses three objects that are each simultaneously `red` and
`ball`, and requires both properties at every hop destination. For each depth
it verifies:

- a closed relation cycle returns `[1, 1, 1]`;
- a broken chain returns `[0, 0, 0]`;
- the distribution length remains three rather than growing with relation
  groundings;
- gradients remain available through the selector.

A separate property-mismatch regression keeps both relation hops present but
makes the intermediate object a yellow ball. A chain requiring `blue(y)`
returns `[0, 0, 0]`; changing only the predicate to `yellow(y)` returns
`[1, 0, 0]`. This distinguishes an existing object with a false property from
a missing hop represented by `None`.

`build_relation_answer_example` accepts custom `object_ids` and `features` for
these fixtures. The feature matrix must contain exactly one row per object ID
and four columns in `red`, `ball`, `blue`, `yellow` order.

Grounding cost still grows quickly: with `N` candidates and `H` unconstrained
binary hops, the complete grounding table can approach `N ** (H + 1)`. Product
fuzzy OR can also accumulate many weak paths. Use Gödel t-norm or ILP when
checking crisp graph reachability.

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
