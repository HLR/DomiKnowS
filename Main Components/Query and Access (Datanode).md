# Query and Access (Datanode)

The following is the methods of query and access to datanodes.

- [Query and Access](#query-and-access)
  - [Class Overview](#class-overview)
  - [DataNode and Data Graph](#dataNode-and-data-graph)
    - [Data Graph Query](#data-graph-query)
    - [Data Graph construction](#data-graph-construction)

## Class Overview

- `DataNode`
- `DataNodeBuilder`

## DataNode and Data Graph

An example in the learning process has its Data Graph built based on sensors included in the model.
Sensors usually partition the example into different types of elements corresponding to different concepts from the [knowledge graph](Knowledge%20Declaration%20%28Graph%29.md).

Each example element has its own DataNode build, which is linked to other Data Nodes in the Data Graph corresponding to other elements from the example through relation links. The Data Node stores the following information about the token:

- **ontology concepts**  - of the element from the associated [knowledge graph](Knowledge%20Declaration%20%28Graph%29.md),

- **id** - of the element, unique in the scope of all aspects of the given knowledge concept type,

- **relation links**  - for this element, it is a dictionary with names of relations and references to related DataNodes,

- **impact links** - for the element, it is a dictionary with references to DataNodes impacting this DataNode by having it as a subject of their relations,

- **attributes** - of the element, it is a dictionary with keys corresponding to the sensor that produced the given attribute and the output value of the given sensor.

DataNode methods facilitate access to its content:

- **children**: `getChildDataNodes(conceptName=None)`

The method returns a list of DataNode children (related to the DataNode, though it contains a relation). If *conceptName* parameter is provided then only DataNodes of the given type are returned. The example:

```python
getChildDataNodes(conceptName=char) # get all children DataNode with *char* type
```

- **relations**: `getRelationLinks(relationName=None)`

The method returns a list of related DataNode. If *relationName* is provided, only DataNodes related through the given relation are returned. The example:

```python
getRelationLinks(relationName=pair) # get list of related DataNodes through *pair* relation
```
 
- **attributes**: `getAttribute(*keys)`

The method returns the value of the attribute. The *keys* are concatenated into a single key string used to access the attribute in the DataNode. The example:

```python
getAttribute(work_for, 'ILP')* - get value of the attribute storing the result of the ILP solver solution for the concept *work_for*
```

### Data Graph Query

The Data Graph can be queried for specific DataNodes using the method called on any DataNode in the graph:

```python
findDatanodes(dns = None, select = None, indexes = None)
```

The method returns the list of DataNodes that satisfy the query provided in the *select* argument; additionally, the *indexes* argument can specify queries for related data nodes that the returned DataNodes must fulfil.

The data nodes are searched in the graph starting from the DataNode on which this method is called and below it.
The examples:

- **dataNode.findDatanodes(select = word)** - find all dataNodes of type *word*

- **dataNode.findDatanodes(select = (char, 'raw', 'J'))** - find dataNode of type *char* with with *raw* attribute equal *J*

- **dataNode.findDatanodes(select = word,  indexes = {"contains" : (char, 'raw', 'J')** - find dataNodes of type *word* and containing char with *raw* attribute equal *J*

- **dataNode.findDatanodes(select = word,  indexes = {"contains" : ((char, 'raw', 'o'), (char, 'raw', 'h'))** - find dataNodes of type *word* and containing dataNode of type *char* with *raw* attribute equal *o* and dataNode of type*char* with *raw* attribute equal *h*

- **dataNode.findDatanodes(select = pair, indexes = {"arg1" : 0, "arg2": 3})** - find dataNode of type *pair* linking dataNodes with id 0 and 3

- **dataNode.findDatanodes(select = pair, indexes = {"arg1" : (word, 'raw', 'John'), "arg2": (word, 'raw', "IBM")})** - find dataNode of type *pair* linking dataNode of type *word* with *raw* attribute equal *John* and dataNode of type *word* with *raw* attribute equal *IBM*

### Data Graph construction

Class **DataNodeBuilder** builds a Data Graph consisting of DataNodes during the learning process based on the sensor context update.
The data nodes are populated based on the concepts in the graph, and each DataNode's attributes are also updated from the value associated with the corresponding concept's property.

Each sensor has its context dictionary implemented with the object of the DataNodeBuilder class, which also implements the Dictionary interface but overloads its methods.
It creates a Data Graph based on the sensors' context update.

The overloaded method:

```python
__setitem__(self, key, value)
```

Updates the created DataNode with information submitted to the dictionary by the sensors.

The *key* is assumed to be a `str`, `Sensor`, or `Property`.

- `str` key

For the `str` key, it is used for general-purpose data storage, for example, the result of a data reader. The `DataNodeBuilder` instance is initialized by a `data_item` from a data reader, and all the keys (and corresponding values) in `data_item` are stored in it.
It can be used to store other resources by string keys.

- Sensor` key

When a `Sensor` instance is used as a key, it triggers a creation/update to data nodes.
If there is no DataNode of this sensor's concept, DataNodes will be created first.
The value of the sensor will be used to update the DataNodes' attributes.
The attributes will have the same name as the sensor's property.

If the sensor is an `EdgeSensor`, the relation link will be updated based on the source and destination of the `EdgeSensor`.

- `Property` key

When a `Property` instance is used as a key, it serves only as a retrieval shortcut based on the `property`.
Our design updates the value based on sensors and retrieves the value based on properties. Every time a sensor updates its value, it passes the value to the builder with itself as the key and also its property as the key. So other sensors can retrieve this value by just querying the property.
However, the dataNode builder will only store the value for properties and won't trigger updates to dataNodes based on property key, because updates of sensor key already include all creation and updating activities.

#### Values

The values provided to the builder are interpreted as an ordered collection of attribute values that are to be distributed to each data node of the associated concept.
Basically, it should be a Python `list` or a ' torch.Tensor`.
If a list is assigned, the length of the list must match the number of dataNodes of this concept, and each element in this list, whatever type it is, should be associated with the attribute of a dataNode in the same order as the dataNode index order.
If a tensor is provided, the first dimension of the sensor must match the number of dataNodes of this concept, and each "row" containing the rest of the dimensions should be associated with the attribute of a dataNode in the same order as the dataNode index order.

Examples:

```python
sentence['text'] = DummySensor00()
# sensor return:
['John works for IBM .']

sentence['ids'] = DummySensor01()
# sensor return:
tensor([[48, 97, 72, 9, 83]])  # shape = (1,5)
```

Where `DummySensor00` returns a list of only one `str`, indicating there is only one sentence dataNode. The sentence dataNode has a `'text'` attribute with `'John works for IBM .'`.
`DummySensor01` returns a tensor where the first dimension is 1, which also matches the only sentence. Then the vector `tensor([48, 97, 72, 9, 83])` should be associated with the sentence.

```python
word['text'] = DummySensor02()
# sensor return:
['John', 'works', 'for', 'IBM', '.']

word['emb'] = DummySensor03()
# sensor return:
tensor([[0.63, 1.12, ..., -0.83],
        [0.05, -0.94, ..., 2.72],
        [0.91, 0.24, ..., 0.12],
        [0.84, -0.22, ..., -0.72],
        [0.08, 1.10, ..., 0.01]])  # shape = (5,100)
```

Here `DummySensor02` returns a list of 5 `str`s. Each of them will be associated with a corresponding word dataNode in the `'text'` attribute.
`DummySensor03`, similarly, returns a tensor whose first dimension is 5. Each row of the 100-dim vector will be associated with a word dataNode in the `emb` attribute.
For instance, the second dataNode of word have `'text'` attribute as string `'work'` and `'emb'` attribute as tensor `tensor([0.05, -0.94, ..., 2.72])`.

#### DataNode(s) for concept

If the *<conceptOrRelationName>* part of the *key* is a *concept* from the  [knowledge graph](Knowledge%20Declaration%20%28Graph%29.md), then the **DataNodeBuilder** analyzes the *value* and determines how many elements of the type specified by the third part of the *key* will be created or updated by this *value*.
The *value* is assumed to provide a **single element** (contribute to single DataNode) if the value is:
- not Tensor or List, e.g., string,
- Tensor of dimension is 0,
- Tensor or List of length one and with dimension 1.
- Tensor of length two, but its key *attributeKey* part has the first word embedded in '<' - such a tensor represents a single set of two probabilities (negative and positive).

In this case, a single new DataNode is created or updated (if the DataNode already exists).
If the single DataNode is determined to be created for the *root* concept, then the *READER* key is used as the new DataNode ID; if the key is not set, then the ID is set to 0.

#### DataNode creation

When the builder receives an update to a sensor whose concept's DataNode is not yet initiated, the builder will create it first.

The *value* determines how many new DataNodes are created.
If the value is a list, then dataNodes are created based on the length of the list. If the value is a tensor, then dataNodes are created based on the first dimension of the tensor. Each DataNode will be initiated with an index that indicates its matching with elements in the list or "row" in the tensor.

The value cannot be None; otherwise, the set method logs an error and returns.

#### DataNode Attribute Update

The subsequent submission *values* length needs to match the number of DataNodes created for the given concept or relation.

For instance, a Tensor with shape [3, 7, 4] will update 3 data nodes, each with a Tensor of shape [7, 4].

To assign an attribute value to a dataNode created with a Tensor with shape[3,4], the tensor will need to have its first shape element equal 12.

#### Relation link Update

Datanodes-wise relation links are also built based on the value passed to the builder with `Sensor` keys.
The key is an `EdgeSensor`, the relation link will be created/updated.
A relation (between concepts in the graph) will be associated to the edge sensor, with another argument `mode='forward'|'backward'` which indicates the direction for the sensor.
The `dst` of the sensor is always the sensor's concept, and the `src` of the sensor indicates the other end of the relation. The `relation` of the sensor is used to indicate which relation link to be updated.
The relation is represented as a matrix tensor, whose first dimension represents the number of dataNodes of the `dst` concept and the second dimension matches the number of `src` concept's dataNodes.

Example:

```oython
word[phrace_contains_word.forward] = DummyEdgeSensor10()
# sensor return:
tensor([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])  # shape = (5, 4)
```

Here, `phrace_contains_word.forward` helps `DummyEdgeSensor10` to identify the relation and the forward/backward mode to be used. In this example, the sensor's `src` is the concept `phrase` and `dst` is the concept `word`, which matches the concept we are assign property to.

The return value of the sensor is a five-by-four matrix which indicates the existence of a relation link between dataNodes by ones, and the rest are zeros.
The first dimension, 5, indicates that there should be 5 words. And the second dimension 4 matches the number of phrase DataNodes.
In this example, the matrix indicates that word with index zero is connected to phrase 0, words 1 and 2 are connected to phrase 1, word three is connected to phrase 2, and word four is connected to phrase 3.
This format is used consistently for 1-to-1, 1-to-many, many-to-1, and many-to-many relations.

If this is the first property populated with data for the concept, it will also trigger the creation of DataNodes. Otherwise, the dimensions must match, as mentioned above.

Multiple relations can be used simultaneously.

```python
pair[pair_arg1.backward, pair_arg2.backward] = DummyEdgeSensor11()
# for example, the sensor will filter out a self-connected pair of words
# sensor return first element:
tensor([[1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        ...,
        [0, 0, 1, 0, 0],
        ...,
        [0, 0, 0, 1, 0],
        ...,
        [0, 0, 0, 0, 1],
        ...])  # shape = (20,5)
# sensor return second element:
tensor([[0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        ...])  # shape = (20,5)
```

Where the matrix dimension implies there are 20 dataNode instances of `pair`, and each is to be connected with word dataNodes separately.

In this example, the 7th pair's arguments it is as simple as getting the 7th row from `pair[pair_arg1.backward]`

```python
        [0, 1, 0, 0, 0]  #  indicating the second word
```

and 7th from `pair[pair_arg2.backward]`

```python
        [0, 0, 0, 1, 0]  #  indicating the fourth word
```

#### Returning Constructed DataNodes

Two methods are returning constructed dataNodes.

- The first DataNode in the list is returned by the method:

```python
getDataNode(self)
```

- All constructed DataNodes are returned by the method:

```python
getBatchDataNodes(self)
```
