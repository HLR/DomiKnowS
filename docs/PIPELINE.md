# Pipeline

Programmer's steps to using our framework.

- [Pipeline](#pipeline)
  - [Knowledge Declaration](#knowledge-declaration)
  - [Model Declaration](#model-declaration)
  - [Training and Testing](#training-and-testing)
  - [Inference](#inference)

## Knowledge Declaration

In knowledge declaration, the user defines a collection of concepts and the way they are related to each other, representing the domain knowledge a the task.
We provide a graph language based on python for knowledge declaration with notation of `Graph`, `Concept`, `Property`, `Relation`, and `LogicalConstrain`.
`Graph` instances are basic container of the `Concept`s, `Relation`s, constaints and other instances in the framework.

The output of the Knowledge Declaration step is a `Graph`, within which there are `Concept`s, `Relation`s, and `LogicalConstrain`s.

Follows is an example showing how to declare a graph.

```python
with Graph('global') as graph:
  sentence = Concept(name='sentence')
  word = Concept(name='word')
  (rel_sentence_contains_word,) = sentence.contains(word)
  pair = Concept(name='pair')
  (rel_pair_word1, rel_pair_word2) = pair.has_a(arg1=word, arg2=word)

  people = word(name='people')
  organization = word(name='organization')
  disjoint(people, organization)

  work_for = pair(name='work_for')
  work_for.has_a(people, organization)
```

The above snippest shows the declaration of a `Graph` named `'global'` as variable `graph`.
With in the graph, there are `Concept`s named `'sentence'`, `'word'`, and `'pair'` as python variable `sentence`, `word`, and `pair`. `sentence` contains `word`s. `pair` has two arguments named `arg1` and `arg2`, which are both `word`.
`people` and `organization` are inheritance `Concept` extended from `word`. That means `people` and `organization` are `word`s.
`people` and `organization` are disjoint, which means if an instance of `word` is `people`, it must not be an `organization`, and vice versa.
`work_for` extends `pair` by limiting the first argument (which was a `word` for `pair`) to be `people` and the first argument (which was also a `word` for `pair`) to be `organization`.

There are inheritance (relation `IsA` or logical `ifL()`), disjoint (relation `NotA`, or logical `nandL()`), and composition (relation `HasA` or a compositional logical expression) constraints implied in the above `graph`.
One can add more complex logical constraints with our logical expression notations.
See [here](KNOWLEDGE.md) for more details about declaring graph and constraints.

Notice that this graph, the *"conceptual graph"*, declare a data structure. When real data come, they will be populate based on the conceptual graph to generate `Datanode`s and the `Datanode`s will be connected and form a *"data graph"*.

## Model Declaration

## Training and Testing

## Inference
