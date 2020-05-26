# Knowledge Declaration

In knowledge declaration, the user defines a collection of concepts and the way they are related to each other, representing the domain knowledge a the task.
We provide a graph language based on python for knowledge declaration with notation of "graph", "concept", "property", "relation", and "constraints".

## Class Overview

### Graph classes

- Package `regr.graph`: the set of class for above-mentioned notations as well as some base classes.
- Class `Graph`: a basic container for other components. It can contain sub-graphs for flexible modeling.
- Class `Concept`: a collection of `Property`s that can be related with each other by `Relation`s. It is a none leaf node in the `Graph`.
- Class `Property`: a key attached to a `Concept` that can be associated with certain value assigned by a sensor or a learner. It is a leaf node in the `Graph`.
- Class `Relation`: a relation between two `Concept`s. It is an edge in the `Graph`.

### Constraints classes

- Package `regr.graph.logicalConstrain`: a set of functions with logical symantics, that one can express logical constraints in first order logic.
- Function `*L()`: functions based on logical notations. Linear constraints can be generated based on the locigal constraints. Some of these functions are `ifL()`, `notL()`, `andL()`, `orL()`, `nandL()`, `existL()`, `equalL()`, `inSetL()`, etc.

## Graph

`Graph` instances are basic container of the `Concept`s, `Relation`s, constaints and other instances in the framework.
A `Graph` object is constructed either by manually coding or compiled from `OWL` (deprecated).
Each `Graph` object can contain other `Graph` objects as sub-graphs. No cyclic reference in graph hierarchy is allowed.

You can either write an owl file initializing your concepts and relations or to write your graph with our specific python classes.

Each `Graph` object can contain `Concept`s.

The graph is a partial program, and there is no sensor or learner, which are data processing units, connected. There is no behavior associated. It is only a data structure to express domain knowledge.

### Example

The following snippest shows an example of a `Graph`.

```python
with Graph() as graph:
    word = Concept('word')
    pair = Concept(word, word)
    with Graph('sub') as sub_graph:
        people = word('people')
        organization = word('organization')
        work_for = pair(people, organization)
```

#### Graph declaration and `with` statement

The first `with` statement creates a graph, assigns it to python variable `graph`, and declares that anything below are attached to the graph.

The second `with` statement declare another graph with an explicit name `'sub'`. It will also be attached to the enclosing graph, and become a subgraph. However, everything below this point will be attached to the subgraph, instead of the first graph.

#### Concept declaration

##### Direct declaration

`word = Concept('word')` creates a concept with name `'word'` (implicitly attached to the enclosing graph) and assign it to python variable `word`.

`pair = Concept(word, word)` is syntactic sugar to creates a concept with two `word`s being its arguments (with two `HasA` relations).
It does not has an explicit name. If a explicit name is designable, use keyword argument `name=`. For example, `pair = Concept(word, word, name='pair')`.
It will also be attached to the enclosing graph.
A `HasA` relation will be added between the new concept and each argument concept. That implies, two `HasA` concepts will be created from `pair` to `word`.
It is equivalent to the following statements:

```python
pair = Concept(name='pair')
pair.has_a(word)
pair.has_a(word)
```

##### Inherit declaration

`people = word('people')` and `organization = word('organization')` create another two concepts extending `word`, setting name `'people'` and `'organization'`, and assign them to python variable `people` and `organization`. They are attached to enclosing subgraph `sub_graph`.
Inherit declaration is syntactic sugar for creating concepts and `IsA` relations.
An `IsA` relation will be create for each of these statements. It is equivalent to the following statements:

```python
people = Concept('people')
people.is_a(word)
organization = Concept('organization')
organization.is_a(word)
```

### Access the nodes

 All the sub-`Graph`s and `Concept` instances can be retrieved from a graph (or sub-graph) with a (relative) pathname.
 For example, to retieve `people` from the above example, one can do `graph['sub/people']` or `sub_graph['people']`.

## Constraints
