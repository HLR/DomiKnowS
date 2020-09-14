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

Class **DataNodeBuilder** builds Data Graph consisting of DataNodes during the learning process based on the sensor context update. 

Each sensor has its context dictionary implemented with the object of DataNodeBuilder class which also implements Dictionary interface but overloads its methods. 
It creates Data Graph based on sensors' context update.

The overloaded method:

```python
__setitem__(self, key, value)
```

updates the created DataNode with information submitted to the dictionary by the sensors.

The *key* is assumed to consists of three parts: *<graphPath> <conceptOrRelationName> <attributeKey>*.

The *grapthPath* and *attrubteKey* can be built of multiple names separated by /. The *conceptOrRelationName* is assumed to be a single word.

the examples of key:
- *global/application/neighbor/city2* - where *global/application* is the graph path, *neighbor* is the concept name and *city2* is an attribute key.


- *tweet/tweet/<PositiveLabel>/readersensor-1* - where first *tweet* is the graph path, the next *tweet* is the concept name and *<PositiveLabel>/readersensor-1* is an attribute key.

The key needs all these **three** elements otherwise the *set* method logs an error and returns.

The *conceptOrRelationName* part is used to **create** the new DataNode when first provided in the key to the *DataNodeBuilder*. The next time it will be used to create or update **attribute** in existing DataNodes of this *conceptOrRelationName* type.

# DataNode creation

The value determine how many new DataNodes are created. 

The value cannot be **None** otherwise the *set* method logs an error and returns.

The value is assumed to provide **single element** (contribute to single DataNode) if the value is:
- not Tensor or List,
- Tensor but of the dimension 0,
- Tensor or List of length 1 and with dimension 1.
- Tensor of length 2 but its key *attributeKey* part has first word embedded in '<'.

In this case a single new DataNode is created. If the single DataNode is determined to be for created for the root concept then the *READER* key is used as the new DataNode id, if the key is not set then the id is set to 0.

If the value is Tensor or List and not assumed to represent single value then its dimension is determine - for Tensor it is based on *dim()* method, for List based on nest level of the first list element.
The value with dimension equal 1 is assumed to represent single set of DataNodes. 

The value with dimension 2 represent **multiply** sets of DataNodes to be created. Each set of new DataNodes to be assigned to a subsequent **parent** DataNode, e.g.:

	 [['J', 'o', 'h', 'n'], ['w', 'o', 'r', 'k', 's'], ['f', 'o', 'r'], ['I', 'B', 'M']]

This represent 4 sets of characters each assigned to the word with the index equal to the index of the specific set in the list (or tensor).

The sets of newly created dataNode are matched with dataNodes created for the parent concept. So for instance Tensor with shape[1,5] will result with creation of 5 dataNodes and their assignment as children to a single parent dataNode.
Tensor with shape[3,4] will result with creation of 4 dataNodes sets of 4 dataNodes and their assignment as children to a 3 parent dataNodes.

Additionally the value can carry information about **child connection** between newly created DataNodes and exiting DataNodes, e.g.:

	[[(0, 0), (1, 2), (3, 3, 0)]]

This represent a single set of values (connected to a single parent DataNode as described above) which each element tuple specifies range if indexes of children DataNode to be connected with a given new DataNode.

**DataNode Attribute Update**

The subsequent submission values length need to match the number of DataNodes created for the given concept or relation. 

So for instance: Tensor with shape[3,7,4] will update 3 dataNodes each with Tensor of shape[7,4].

In order to assign attribute value to dataNode created with Tensor with shape[3,4] the tensor will need to have it first shape element equal 12.

**Returning Constructed DataNodes**

There are two methods returning constructed dataNodes.

- the fist DataNode in the list is returned by the method:

```python
getDataNode(self)
```

- all constructed DataNodes are returned by the method:

```python
getBatchDataNodes(self)
```
    