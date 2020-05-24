# Knowledge Declaration

## Class Overview

* package `regr.graph`:
* `Graph`:
* `Concept`:
* `Property`:
* `Relation`:
* Constraints:

## Graph

`Graph` instances are basic container of the `Concept`s, `Relation`s, constaints and other instances in the framework.
A `Graph` object is constructed either by manually coding or compiled from `OWL` file(s).
Each `Graph` object can contain other `Graph` objects as sub-graphs. No cyclic reference in graph hierarchy is allowed.

You can either write an owl file initializing your concepts and relations or to write your graph with our specific python classes.

Each `Graph` object can contain certain `Concept` objects.

The graph is a partial program, and there is no sensor or learner, which are data processing units, connected. There is no behavior associated. It is only a data structure to express domain knowledge.

The following snippest shows an example of a `Graph`.

```python
with Graph() as graph:
    word = Concept('word')
    pair = Concept(word, word)
    with Graph() as sub_graph('sub'):
        people = word('people')
        organization = word('organization')
        work_for = pair(people, organization)
```

 All the sub-`Graph`s and `Concept` instances can be retrieved from a graph (or sub-graph) with a (relative) pathname.
 For example, to retieve `people` from the above example, one can do `graph['sub/people']` or `sub_graph['people']`.

## Constraints
