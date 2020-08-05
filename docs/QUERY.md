# Query and Access

- [Query and Access](#query-and-access)
  - [Class Overview](#class-overview)
  - [DataNode and Data Graph](#datanode-and-data-graph)
    - [Data Graph Query](#data-graph-query)
    - [Data Graph construction](#data-graph-construction)

## Class Overview

- `Datanode`:

## DataNode and Data Graph

Every example in the learning process has its Data Graph built based on sensors included in the model.
The example is partitioned by sensors into a different types of tokens corresponding to different linguistic concepts from the [knowledge graph](KNOWLEDGE.md).

Each example token has its own DataNode build which is linked to other Data Nodes in the Data Graph corresponding to other tokens from the example through relation links. The Data Node stores the following information about the token:

- **ontology concepts**  - from the associated [knowledge graph](KNOWLEDGE.md),

- **id** - unique in the scope of all tokens of the given knowledge concept type,

- **relation links**  - dictionary with names of relations and related DataNodes,

- **impact links** - dictionary with DataNodes impacting this DataNode by having it as a subject of its relation,

- **attributes** dictionary - with key corresponding to the sensor which produced the given attribute and its value for the given token.

DataNode methods facilitate access to its content:

- **children**: `getChildDataNodes(conceptName=None)`

The method returns a list of DataNode children DataNode. If *conceptName* is provided only DataNodes of the given knowledge type are returned. The example:

```python
getChildDataNodes(conceptName=char) # get all children DataNode with *char* type
```

- **relations**: `getRelationLinks(relationName=None)`

The method returns a list of related DataNode. If *relationName* is provided only DataNodes related through the given relation are returned. The example:

```python
getRelationLinks(relationName=pair) # get list of related DataNodes through *pair* relation
```

- **attributes**: `getAttribute(*keys)`

The method returns the value of the attribute. The *keys* are connected into a single key used to access the attribute in the DataNode. The example:

```python
getAttribute(work_for, 'ILP')* - get value of the attribute storing the result of the ILP solver solution for the concept *work_for*
```

### Data Graph Query

The Data Graph can be queried for specific DataNodes using the method called on any DataNode:

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

Class **DataNodeBuilder** builds Data Graph consisting of DataNodes during the learning process based on the sensor context update. Each sensor has its previous context dictionary replaced with the object of DataNodeBuilder class which also implements Dictionary interface but overloads its methods. It creates Data Graph based on sensors' context update.

The primary used overloaded method is:

```python
__setitem__(self, key, value)
```

The method assumes that key identifies the sensor and the value is a tensor of appropriate dimension.
