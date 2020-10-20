# Query and Access

- [Query and Access](#query-and-access)
  - [Class Overview](#class-overview)
  - [DataNode and Data Graph](#datanode-and-data-graph)
    - [Data Graph Query](#data-graph-query)
    - [Data Graph construction](#data-graph-construction)

## Class Overview

- `DataNode`
- `DataNodeBuilder`

## DataNode and Data Graph

Example in the learning process has its Data Graph built based on sensors included in the model.
The example is usually partitioned by sensors into a different types of elements corresponding to different concepts from the [knowledge graph](KNOWLEDGE.md).

Each example element has its own DataNode build which is linked to other Data Nodes in the Data Graph corresponding to other elements from the example through relation links. The Data Node stores the following information about the token:

- **ontology concepts**  - of the element from the associated [knowledge graph](KNOWLEDGE.md),

- **id** - of the element, unique in the scope of all elements of the given knowledge concept type,

- **relation links**  - for this element, it is a dictionary with names of relations and references to related DataNodes,

- **impact links** - for the element, it is a dictionary with references of DataNodes impacting this DataNode by having it as a subject of their relations,

- **attributes** - of the element, it is a dictionary with key corresponding to the sensor which produced the given attribute and the output value of the given sensor.

DataNode methods facilitate access to its content:

- **children**: `getChildDataNodes(conceptName=None)`

The method returns a list of DataNode children DataNode (related to the DataNode though contains relation). If *conceptName* parameter is provided then only DataNodes of the given type are returned. The example:

```python
getChildDataNodes(conceptName=char) # get all children DataNode with *char* type
```

- **relations**: `getRelationLinks(relationName=None)`

The method returns a list of related DataNode. If *relationName* is provided only DataNodes related through the given relation are returned. The example:

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

The method returns the list of DataNodes satisfying the query provided in the *select* argument, additionally the *indexes* argument can specify queries for related data nodes which have to be satisfied by the returned Data Nodes.

The data nodes are searched in the graph starting from the DataNode on which this method is called and below it.
The examples:

- **datanode.findDatanodes(select = word)** - find all dataNodes of type *word*

- **datanode.findDatanodes(select = (char, 'raw', 'J'))** - find dataNode of type *char* with with *raw* attribute equal *J*

- **datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'raw', 'J')** - find dataNodes of type *word* and containing char with *raw* attribute equal *J*

- **datanode.findDatanodes(select = word,  indexes = {"contains" : ((char, 'raw', 'o'), (char, 'raw', 'h'))** - find dataNodes of type *word* and containing dataNode of type *char* with *raw* attribute equal *o* and dataNode of type*char* with *raw* attribute equal *h*

- **datanode.findDatanodes(select = pair, indexes = {"arg1" : 0, "arg2": 3})** - find dataNode of type *pair* linking dataNodes with id 0 and 3

- **datanode.findDatanodes(select = pair, indexes = {"arg1" : (word, 'raw', 'John'), "arg2": (word, 'raw', "IBM")})** - find dataNode of type *pair* linking dataNode of type *word* with *raw* attribute equal *John* and dataNode of type *word* with *raw* attribute equal *IBM*

### Data Graph construction

Class **DataNodeBuilder** builds Data Graph consisting of DataNodes during the learning process based on the sensor context update.
The datanodes a populated based on the concepts in the graph and each datanodes atttributes are also update from the value associated to the corresponding concepts' property.

Each sensor has its context dictionary implemented with the object of DataNodeBuilder class which also implements Dictionary interface but overloads its methods.
It creates Data Graph based on sensors' context update.

The overloaded method:

```python
__setitem__(self, key, value)
```

updates the created DataNode with information submitted to the dictionary by the sensors.

The *key* is assumed to be a `str`, `Sensor`, or `Property`.

#### `str` key

For `str` key, it is used for general purpose data storage, for example, the result of data reader. The `DataNodeBuilder` instance is initialized by a `data_item` from data reader, and all the keys (and corresponding values) in `data_item` are stored it.
It can be used to store other resources by string keys.

#### `Sensor` key

When a `Sensor` instance is used as a key, it will trigger an creation/update to datanodes.
If there is no datanode of this sensor's concept, datanodes will be created first.
The value of the sensor will be used to update the datanodes' attributes.
The attributes will has the same name as the sensor's property.

If the sensor is an `EdgeSensor`, relation link will be updated based on the source and destination of the `EdgeSensor`.

#### `Property` key

When a `Property` instance is used as a key, it is just to serve the retrieval shortcut based on `property`.
Our design is update value by sensors and retrieve value by properties. Every time a sensor update its value, it will pass the value to the builder with itself as key and also its property as key. So other sensor can retrieve this value by just query the propery.
However, the datanode builder will just store the value for properties and won't trigger updating to datanodes based on property key, because updates of sensor key already include all creation and updating activity.

#### Values

The values provided to the builder is interprete as a ordered collection of attributes values that are to be distributed to each datanode of the associated concept.
Basically, it should be a python `list` or a`torch.Tensor`.
If a list is assigned, the length of the list must match the number of datanodes of this concept and each element in this list, whatever type it is, should be associated to the attribute of a datanode in the same order as the datanode index order.
If a tensor is provided, the first dimension of the sensor must match the number of datanodes of this concept and each "row" containing the rest of dimension should be associated to the attribute of a datanode in the same order as the datanode index order.

Examples:

```python
sentence['text'] = DummySensor00()
# sensor return:
['John works for IBM .']

sentence['ids'] = DummySensor01()
# sensor return:
tensor([[48, 97, 72, 9, 83]])  # shape = (1,5)
```

Where `DummySensor00` returns a list of only one `str`, indicating there is only one sentence datanode. The sentence datanode has a `'text'` attribute with `'John works for IBM .'`.
`DummySensor01` returns a tensor where the first dimension is 1, which also match the only one sentence. Then the vector `tensor([48, 97, 72, 9, 83])` should be associated with the sentence.

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

Here `DummySensor02` returns a list of 5 `str`s. Each of them will be associated with a corresponding word datanode in `'text'` attribute.
`DummySensor03`, similarly, returns a tensor of whose first dimension if 5. Each raw of the 100-dim vector will be associated with a word datanode in `emb` attribute.
For instance, the second datanode of word have `'text'` attribute as string `'work'` and `'emb'` attribute as tensor `tensor([0.05, -0.94, ..., 2.72])`.

#### DataNode creation

When the builder recieve an update to a sensor whose concept's datanode are not yet initiated, the builder will create them first.

The *value* determine how many new DataNodes are created.
If the value is a list, then datanodes are created based on the length of the list. If the value is a tensor, then datanodes are created based on the first dimension of the tensor. Each datanode will be initiated with an index which indicated its matching with elements in the list or "row" in the tensor.

The *value* cannot be **None** otherwise the *set* method logs an error and returns.

#### DataNode(s) for concept

If the *<conceptOrRelationName>* part of the *key* is a *concept* from the  [knowledge graph](KNOWLEDGE.md) then the **DataNodeBuilder** analyses the *value* and determines how many elements of the type specified by the third part of the *key* will be created or updated by this *value*.
The *value* is assumed to provide **single element** (contribute to single DataNode) if the value is:
- not Tensor or List, e.g. string,
- Tensor but of the dimension equals 0,
- Tensor or List of length 1 and with dimension 1.
- Tensor of length 2 but its key *attributeKey* part has first word embedded in '<' - such a tensor represent a single set of two probabilities (negative and positive).

In this case a single new DataNode is created or updated (if the DataNode already exists).
If the single DataNode is determined to be for created for the *root* concept then the *READER* key is used as the new DataNode id, if the key is not set then the id is set to 0.

#### DataNode Attribute Update

The subsequent submission *values* length need to match the number of DataNodes created for the given concept or relation.

So for instance: Tensor with shape[3,7,4] will update 3 dataNodes each with Tensor of shape[7,4].

In order to assign attribute value to dataNode created with Tensor with shape[3,4] the tensor will need to have it first shape element equal 12.

#### DataNode for Relation (TO BE UPDATED)

If the *<conceptOrRelationName>* part of the *key* is a *concept* from the [knowledge graph](KNOWLEDGE.md) then the **DataNodeBuilder* collected from the Data Graph DataNotes related by the given relation based on the relation definition which specify types of DataNotes linked by the relation.
The *value* dimension has to be the larger or equal to the number of elements linked by the given relation.
The length of each dimension  has to be equal to the number of DataNoded found for the given concept linked by the relation.

If the dimension and lengths matches the [knowledge graph](KNOWLEDGE.md) definition and the number of existing DataNodes for the given in the concepts then the DataNodes representing the relation are created (or updated if already exists).

For instance:

- key *global/linguistic/pair/<work_for>/testsensor-19* indicates that this is infromation about *work_for* relation.
- value:
```python
    tensor([[[0.6000, 0.4000],
             [0.8000, 0.2000],
             [0.8000, 0.2000],
             [0.3700, 0.6300]],
             [[1.0000,    nan],
             [1.0000,    nan],
             [0.6000, 0.4000],
             [0.7000, 0.3000]],
             [[0.9800, 0.0200],
             [0.7000, 0.0300],
             [0.9500, 0.0500],
             [0.9000, 0.1000]],
             [[0.3500, 0.6500],
             [0.8000, 0.2000],
             [0.9000, 0.1000],
             [0.7000, 0.3000]]])
```

is a tensor of *torch.Size([4, 4, 2])* the relation *work_for* relates two elements thus the first dimensions of the tensor will be used to create DataNode for relation. The last dimension of the tensor will be as attribute value to the newly created DataNodes.

Additionally if the *<conceptOrRelationName>* part of the *key* contain has "_Candidate_" string attached then this indicate that the *value* has information (in the form of 0 or 1) if the given DataNode for relation are to be created.

#### Returning Constructed DataNodes

There are two methods returning constructed dataNodes.

- the fist DataNode in the list is returned by the method:

```python
getDataNode(self)
```

- all constructed DataNodes are returned by the method:

```python
getBatchDataNodes(self)
```

#### DataNode logging

*DataNode* and *DataNodeBuilde*r log their activities, warning and errors to a log file.
The *path* to the log file is printed to the program standard output.

The example of log from the DataNodeBuilder activity is provided below:

```log
	2020-09-29 17:32:51,441 - INFO - dataNodeBuilder:__init__ - 
	2020-09-29 17:32:51,441 - INFO - dataNodeBuilder:__init__ - Called
	2020-09-29 17:32:51,442 - INFO - dataNodeBuilder:__setitem__ - key - global/sentence/index/constantsensor,  value - <class 'str'>
	2020-09-29 17:32:51,442 - INFO - dataNodeBuilder:__createInitialdDataNode - Creating initial dataNode - provided value has length 1
	2020-09-29 17:32:51,443 - INFO - dataNodeBuilder:__createInitialdDataNode - Created single dataNode with id 0 of type sentence
	2020-09-29 17:32:51,443 - INFO - dataNodeBuilder:__updateRootDataNodeList - Updated elements in the root dataNodes list - [sentence 0]
	2020-09-29 17:32:51,443 - INFO - dataNodeBuilder:__setitem__ - key - global/sentence/index,  value - <class 'str'>
	2020-09-29 17:32:51,444 - INFO - dataNodeBuilder:__updateDataNodes - Adding attribute index in existing dataNodes - found 1 dataNodes of type sentence
	2020-09-29 17:32:51,456 - INFO - dataNodeBuilder:__setitem__ - key - global/word/spacy/spacygloverep,  value - <class 'torch.Tensor'>, shape torch.Size([17, 300])
	2020-09-29 17:32:51,457 - INFO - dataNodeBuilder:__createMultiplyDataNode - Adding 17 sets of dataNodes of type word
	2020-09-29 17:32:51,458 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 0, ...... word 299]
	2020-09-29 17:32:51,513 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 300, .... word 599]
	2020-09-29 17:32:51,515 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 600, .... word 899]
	2020-09-29 17:32:51,516 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 900, .... word 1199]
	2020-09-29 17:32:51,518 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 1200, ... word 1499]
	2020-09-29 17:32:51,520 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 1500, ... word 1799]
	2020-09-29 17:32:51,522 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 1800, ... word 2099]
	2020-09-29 17:32:51,523 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 2100, ... word 2399]
	2020-09-29 17:32:51,525 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 2400, ... word 2699]
	2020-09-29 17:32:51,527 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 2700, ... word 2999]
	2020-09-29 17:32:51,528 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 3000, ... word 3299]
	2020-09-29 17:32:51,530 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 3300, ... word 3599]
	2020-09-29 17:32:51,532 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 3600, ... word 3899]
	2020-09-29 17:32:51,533 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 3900, ... word 4199]
	2020-09-29 17:32:51,535 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 4200, ... word 4499]
	2020-09-29 17:32:51,536 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 4500, ... word 4799]
	2020-09-29 17:32:51,538 - INFO - dataNodeBuilder:__createMultiplyDataNode - Added 300 new dataNodes [word 4800, ... word 5099]
	2020-09-29 17:32:51,538 - ERROR - dataNodeBuilder:__createMultiplyDataNode - Number of dataNode sets 17 is different then the number of 1 dataNodes of type sentence - abandon the update
	2020-09-29 17:32:51,541 - INFO - dataNodeBuilder:__updateRootDataNodeList - Updated elements in the root dataNodes list - [sentence 0, word 0, ... word 5099]
	2020-09-29 17:32:51,542 - INFO - dataNodeBuilder:__setitem__ - key - global/word/spacy,  value - <class 'torch.Tensor'>, shape torch.Size([17, 300])
	2020-09-29 17:32:51,824 - INFO - dataNodeBuilder:__updateDataNodes - Adding attribute spacy in existing dataNodes - found 5100 dataNodes of type word
	2020-09-29 17:32:51,824 - WARNING - dataNodeBuilder:__updateDataNodes - Provided value has length 17 but found 5100 existing dataNode - abandon the update
```
