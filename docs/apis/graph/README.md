# `regr.graph`

## Conceptual Graph

### `regr.graph.Graph`

Inheriting from `BaseGraphTree` and has a local namespace.
`Graph` is a named container for `Graph`, `Concept` and `Relation`.

#### `Graph` Attributes

- `subgraphs`: An `OrderedDict` of its subgraphs, keyed by the names of the subgraphs.
- `concepts`: An `OrderedDict` of `Concept`s directly contained in this `Graph`, keyed by yhe names of the concepts.
- `logicalConstrains`: An `OrderedDict` of `LogicalConstrain`s which are declared under the scope of the graph, keyed by yhe names of the constraints.
- `relations`: An `OrderedDict` of `Relation`s which are declared under the scope of the graph (through concept method call), keyed by yhe names of the relation.
- `ontology`: An ontology indicator `Graph.Ontology`, which is a named tuple containing `iri` and `local`.

#### `Graph` Methods

##### `__init__(self, name=None, ontology=None, iri=None, local=None)`

Initiate the base class with proposal name, set upt `ontology` with arguments, and initiate `concepts`, `logicalConstrains`, and `relations`.

- Parameters:
  - `name`: A name proposal for the graph. If `None` is given, it will try to assign `'graph'` as the name. If the name is taken, an auto-incremental index will be appended to the name. Default: `None`.
  - `ontology`: An ontology indicator `Graph.Ontology`, which is a named tuple containing `iri` and `local`. If provided, the following arguments `iri` and `local` are ignored. Default: `None`.
  - `iri`: If `ontology` is not provided, use `iri` and `local` to construct the ontology indicator. Default: `None`.
  - `local`: If `ontology` is not provided, use `iri` and `local` to construct the ontology indicator. Default: `None`.

##### `__setitem__(self, name, obj)` / `graph[name] = obj`

Attach the `obj`, which is a subgraph, a concept, or a relation, to the graph. `obj` of other types is dropped.

- Parameters:
  - `name`: A key for the `obj`
  - `obj`: An instance of `Graph`, `Concept`, or `Relation`.

##### `__getitem__(self, name)` / `graph[name]`

Retrieve the attached subgraph, concept, or relation with the `name`.

- Parameters:
  - `name`: The key for the attached subgraph, concept, or relation.

- Return Value:
  - The subgraph, concept, or relation stored with specific key `name`.

##### `__iter__(self)` / `iter(graph)`

Retrieve all the attached subgraphs, concepts, and relations.
The behavior is anology to `dict.__iter__` which yields the keys.

- Return Value:
  - An iterator going through all the keys of attached subgraphs, concepts, and relations.

##### `get_sensors(self, *tests)`

Find all the `Sensor` instances nested in this graph (or its subgraph), which passes all the filter functions `tests`.

- Parameters:
  - `*tests`: A list of filter functions that the selected `Sensor` must tested to be `True`.

- Return Value:
  - A list of `Sensor`s, each of which passes all the filter functions `tests`.

##### `get_properties(self, *tests)`

Similar to `get_sensors(self, *tests)`, find all the `Property` instances nested in this graph (or its subgraph), which passes all the filter functions `tests`.

- Parameters:
  - `*tests`: A list of filter functions that the selected `Property` must tested to be `True`.

- Return Value:
  - A list of `Property`s, each of which passes all the filter functions `tests`.

##### `what(self, *tests)`

Return a `dict` structure that represent the current graph, including its subgraph and concepts.

- Return Value:
  - A `dict` structure that represent the current graph, including its subgraph and concepts

### `regr.graph.Concept`

### `regr.graph.Relation`

### `regr.graph.Property`

### `regr.graph.LogicalConstrain`

## Data Graph

### `regr.graph.DataNode`

### `regr.graph.DataNodeBuilder`
