# Tiny Multi-Answer Example

This example contains three objects and three learned unary concepts: `red`,
`dog`, and `cat`. The query asks for all objects satisfying:

```python
andL(red("o"), orL(dog(path="o"), cat(path="o")))
```

Because `iotaL` is a unique selector, a set-valued answer is compiled into one
`existsL` membership query per candidate. The expected answer is `{o1, o2}`.
Each unary concept is attached to a tiny linear `ModuleLearner`.

Run it with:

```bash
conda run -n CLEVER python -m unittest test_regr.tiny_multi_answer.test_example
```
