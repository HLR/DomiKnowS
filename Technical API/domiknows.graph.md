# domiknows.graph package

## Submodules

## domiknows.graph.base module

### *class* domiknows.graph.base.BaseGraphShallowTree(name=None)

Bases: [`BaseGraphTree`](#domiknows.graph.base.BaseGraphTree)

#### parse_query_apply(func, \*names, delim='/', trim=True)

#### *property* scope_key

### *class* domiknows.graph.base.BaseGraphTree(name=None)

Bases: [`AutoNamed`](domiknows.md#domiknows.base.AutoNamed), [`NamedTree`](domiknows.md#domiknows.base.NamedTree)

#### *classmethod* clear() → None.  Remove all items from od.

### *class* domiknows.graph.base.BaseGraphTreeNode(name=None)

Bases: [`AutoNamed`](domiknows.md#domiknows.base.AutoNamed), [`NamedTreeNode`](domiknows.md#domiknows.base.NamedTreeNode)

#### *classmethod* clear()

## domiknows.graph.candidates module

### *class* domiknows.graph.candidates.CandidateSelection(\*e, name=None)

Bases: [`LcElement`](#domiknows.graph.logicalConstrain.LcElement)

### *class* domiknows.graph.candidates.combinationC(\*e, name=None)

Bases: [`CandidateSelection`](#domiknows.graph.candidates.CandidateSelection)

#### strEs()

### domiknows.graph.candidates.findDatanodesForRootConcept(dn, rootConcept)

### domiknows.graph.candidates.getCandidates(dn, e, variable, lcVariablesDns, lc, logger, integrate=False)

### domiknows.graph.candidates.getEdgeDataNode(dn, path, currentIndexDN, lcVariablesDns)

### domiknows.graph.candidates.intersection_of_lists(lists)

## domiknows.graph.concept module

### *class* domiknows.graph.concept.Concept(name=None, batch=False)

Bases: [`BaseGraphTree`](#domiknows.graph.base.BaseGraphTree)

#### aggregate(vals, confs)

The aggregation used in this concept to reduce the inconsistent values.

#### assign_suggest_name(name=None)

#### bvals(prop)

Properties: get all binded values

* **Parameters:**
  **prop** (*str*) – property name
* **Returns:**
  Return vals and confs where vals is a list of values binded to the property
  and confs is a list of values representing the confidence of each binded value.
  An element of vals should have the shape:
  ( batch, vdim(s…) )
  Return None is if never binded to this property.
* **Return type:**
  [barray,…], [barray,…]

#### candidates(root_data, query=None, logger=None)

#### distances(p, q)

The “distance” used in this concept to measure the consistency.
Lower value indicates better consistency.
Feature(s) of one instance is always on only the last axis.
p, q - [(batch, vdim(s…)),…] \* nval

#### findRootGraph(superGraph)

#### getOntologyGraph()

#### get_batch()

#### get_canonical_concept() → [Concept](#domiknows.graph.concept.Concept)

#### get_equal_concepts(transitive: bool = False) → List[[Concept](#domiknows.graph.concept.Concept)]

#### get_equal_relations() → List[[Equal](#domiknows.graph.relation.Equal)]

#### get_equivalence_class() → List[[Concept](#domiknows.graph.concept.Concept)]

#### get_multiassign()

#### *classmethod* get_new_variable_index()

#### get_var_name()

#### is_equal_to(other_concept: [Concept](#domiknows.graph.concept.Concept)) → bool

#### is_equal_to_transitive(other_concept: [Concept](#domiknows.graph.concept.Concept)) → bool

#### merge_equal_concepts(property_merge_strategy: str = 'first') → Dict[str, Any]

#### newVariableIndex *= 0*

#### processLCArgs(\*args, conceptT=None, \*\*kwargs)

#### relate_to(concept, \*tests)

#### *classmethod* relation_type(name=None)

#### rvals(prop, hops=1)

Properties: get all values from relations

#### *property* scope_key

#### set_apply(name, sub)

#### vals(prop, hops=1)

#### what()

### *class* domiknows.graph.concept.EnumConcept(name=None, values=[])

Bases: [`Concept`](#domiknows.graph.concept.Concept)

#### *property* attributes

#### *property* enum

#### get_concept(value)

#### get_index(value)

#### get_value(index)

## domiknows.graph.dataNode module

### *class* domiknows.graph.dataNode.DataNode(myBuilder=None, instanceID=None, instanceValue=None, ontologyNode=None, graph=None, relationLinks={}, attributes={})

Bases: `object`

Represents a single data instance in a graph with relation links to other data nodes.

Attributes:
: - myBuilder (DatanodeBuilder): DatanodeBuilder used to construct this datanode.
  - instanceID (various): The data instance ID (e.g., paragraph number, sentence number).
  - instanceValue (various): Optional value of the instance (e.g., text, bitmap).
  - ontologyNode (Node): Reference to the node in the ontology graph.
  - graph (Graph): Graph to which the DataNode belongs.
  - relationLinks (dict): Dictionary mapping relation name to RelationLinks.
  - impactLinks (dict): Dictionary with dataNodes impacting this dataNode.
  - attributes (dict): Dictionary with node’s attributes.
  - current_device (str): The current device being used (‘cpu’ or ‘cuda’).
  - gurobiModel (NoneType): Placeholder for Gurobi model.
  - myLoggerTime (Logger): Logger for time measurement.

#### *exception* DataNodeError

Bases: `Exception`

Exception raised for DataNode-related errors.

#### addChildDataNode(dn)

Add a child DataNode to the current DataNode.

Args:
: dn (DataNode): The DataNode to be added as a child.

#### addEqualTo(equalDn, equalName='equalTo')

Add a DataNode that is considered equal to the current DataNode.

Args:
: equalDn (DataNode): The DataNode to be added.
  equalName (str, optional): The name of the relation for equality. Defaults to “equalTo”.

#### addRelationLink(relationName, dn)

Add a relation link between the current DataNode and another DataNode.

This method establishes a relation link from the current DataNode to another
DataNode (‘dn’) under a given relation name. It also updates the impactLinks
for the target DataNode.

Args:
: relationName (str): The name of the relation to add.
  dn (DataNode): The target DataNode to link to.

Returns:
: None

#### calculateLcLoss(tnorm='P', counting_tnorm=None, sample=False, sampleSize=0, sampleGlobalLoss=False)

Calculate the loss for logical constraints (LC) based on various t-norms.

Parameters:
- tnorm: str, optional

> Specifies the t-norm used for calculations. Supported t-norms are ‘L’ (Lukasiewicz),
> ‘G’ (Godel), and ‘P’ (Product). Default is ‘P’.
- sample: bool, optional
  : Specifies whether sampling is to be used. Default is False.
- sampleSize: int, optional
  : Specifies the sample size if sampling is enabled. A value of -1 indicates Semantic Sample.
    Default is 0.
- sampleGlobalLoss: bool, optional
  : Specifies whether to calculate the global loss in case of sampling. Default is False.

Returns:
- lcResult: object

> The calculated loss for logical constraints, typically a numeric value or data structure.

Raises:
- DataNodeError: When an unsupported tnorm is provided or other internal errors occur.

#### *classmethod* clear()

Clear DataNode class state.

This method resets the class-level ID counter and clears any cached
state to ensure clean state for testing and other scenarios where 
DataNode instances need to be reset.

#### collectConceptsAndRelations(conceptsAndRelations=None)

Collect all the concepts and relations from the data graph and transform them into tuple form.

Args:
: conceptsAndRelations (set, optional): A set to store the found concepts and relations. Defaults to None.

Returns:
: list: A list of tuples, each representing a concept or relation with additional information.

#### collectInferredResults(concept, inferKey)

Collect inferred results based on the given concept and inference key.

Args:
: concept (Concept or tuple): The concept for which to collect inferred results.
  inferKey (str): The type of inference, e.g., ‘ILP’, ‘softmax’, ‘argmax’.

Returns:
: torch.Tensor: Tensor containing collected attribute list.

#### collectedConceptsAndRelations *= None*

#### conceptsMap *= {}*

#### findConcept(conceptName, usedGraph=None)

Find concept based on the name in the ontology graph.

Args:
: conceptName (str or Concept): The name of the concept to find.
  usedGraph (object): The ontology graph to search within if not provided, defaults to the ontology graph associated with self master graph

Returns:
: tuple or None: A tuple containing details about the found concept or None if not found.

#### findConceptsAndRelations(dn, conceptsAndRelations=None, visitedDns=None)

Recursively search for concepts and relations in the data graph starting from a given dataNode (dn).

This method will traverse through linked dataNodes to find concepts and relations. If ‘variableSet’
is present in the attributes, it will return those concepts directly.

Args:
: dn (DataNode): The dataNode from which to start the search.
  conceptsAndRelations (set, optional): A set to store found concepts and relations. Defaults to None.
  visitedDns (set, optional): A set to keep track of visited dataNodes to prevent cycles. Defaults to None.

Returns:
: set: A set containing the names of all found concepts and relations.

#### findConceptsNamesInDatanodes(dns=None, conceptNames=None, relationNames=None)

Finds all unique concept and relation names in a list of DataNodes.

Args:
: dns (list, optional): A list of DataNodes to be searched. Defaults to None.
  conceptNames (set, optional): A set to store the names of concepts found. Defaults to None.
  relationNames (set, optional): A set to store the names of relations found. Defaults to None.

Returns:
: tuple: A tuple containing two sets: (conceptNames, relationNames).

#### findDatanodes(dns=None, select=None, indexes=None, visitedDns=None, depth=0)

Find and return DataNodes based on the given query conditions.

Args:
: dns (list): List of DataNodes to start with.
  select (object): Query condition for selecting DataNodes.
  indexes (dict): Optional query filtering.
  visitedDns (OrderedSet): Keeps track of already visited DataNodes.
  depth (int): Depth of the recursive call.

Returns:
: list: List of DataNodes that satisfy the query condition.

#### findRootConceptOrRelation(relationConcept, usedGraph=None)

Finds the root concept or relation of a given relation or concept.

Args:
: relationConcept (str or Object): The relation or concept to find the root for.
  usedGraph (Object, optional): The ontology graph where the relation or concept exists. Defaults to None.

Returns:
: Object or str: The root concept or relation.

#### getAttribute(\*keys)

Retrieve a specific attribute using a key or a sequence of keys.

The method accepts multiple keys in the form of positional arguments,
combining them to identify the attribute to retrieve.

Args:
: ```
  *
  ```
  <br/>
  keys (str or tuple or Concept): The key(s) to identify the attribute.

Returns:
: object: The value of the attribut   e if it exists, or None otherwise.

#### getAttributes()

Get all attributes of the DataNode.

Returns:
: dict: Dictionary containing all attributes of the DataNode.

#### getChildDataNodes(conceptName=None)

Retrieve child DataNodes based on the concept name.

Args:
: conceptName (str, optional): The name of the concept to filter the child DataNodes.
  : Defaults to None.

Returns:
: list: A list of child DataNodes that match the given concept name. Returns None if
  : there are no child DataNodes.

#### getDnsForRelation(rel)

Get DataNodes associated with a given relation.

The method first finds the root concept or relation for the given ‘rel’.
Depending on what it finds, it returns the corresponding DataNodes.

Args:
: rel (str/Object): The relation or concept for which DataNodes are needed.

Returns:
: list: A list of DataNodes corresponding to the relation, or [None] if not found.

#### getEqualTo(equalName='equalTo', conceptName=None)

Retrieve DataNodes that are equal to the current DataNode.

Args:
: equalName (str, optional): The name of the relation for equality. Defaults to “equalTo”.
  conceptName (str, optional): The name of the concept to filter the DataNodes.
  <br/>
  > Defaults to None.

Returns:
: list: A list of DataNodes that are considered equal to the current DataNode.

#### getILPSolver(conceptsRelations=None)

Get the ILP Solver instance based on the given concepts and relations.

Args:
: conceptsRelations (list, optional): A list of concepts and relations to be considered. Defaults to None.

Returns:
: tuple: An instance of ILP Solver and the list of processed concepts and relations.

Raises:
: DataNodeError: If the ILP Solver is not initialized.

#### getInferMetrics(\*conceptsRelations, inferType='ILP', weight=None, average='binary')

Calculate inference metrics for given concepts and relations.

Parameters:
- conceptsRelations: tuple or list

> Concepts and relations for which metrics are to be calculated. If empty, it collects all.
- inferType: str, optional (default is ‘ILP’)
  : The inference type to use. Can be ‘ILP’ or other types supported.
- weight: torch.Tensor or None, optional
  : Weight tensor to be used in the calculation.
- average: str, optional (default is ‘binary’)
  : Type of average to be used in metrics calculation. Can be ‘binary’, ‘micro’, etc.

Returns:
- result: dict

> Dictionary containing calculated metrics (TP, FP, TN, FN, P, R, F1) for each concept.

Logging:
- Various logs are printed for debugging and information.

#### getInstanceID()

Get the instance ID of the DataNode object.

Returns:
: various: Instance ID of the DataNode object.

#### getInstanceValue()

Get the instance value of the DataNode object.

Returns:
: various: Instance value of the DataNode object.

#### getLinks(relationName=None, conceptName=None)

Get links associated with the DataNode based on the relation and concept names.

This method retrieves the DataNodes linked to the current DataNode through
either relation links or impact links. You can filter these links based on
the name of the relation or the name of the concept (ontology node).

Args:
: relationName (str, optional): The name of the relation to filter by.
  : Defaults to None.
  <br/>
  conceptName (str, optional): The name of the ontology node (concept) to filter by.
  : Defaults to None.

Returns:
: dict or list: A dictionary containing the DataNodes linked through relation or
  : impact links. If relationName or conceptName is provided,
    returns a list of DataNodes that match the criteria.

#### getOntologyNode()

Get the ontology node related to the DataNode object.

Returns:
: Node: Ontology node related to the DataNode object.

#### getRelationAttrNames(conceptRelation, usedGraph=None)

Get attribute names for a given relation or concept that is a relation.

Args:
: conceptRelation (Concept): The concept or relation to check for attributes.
  usedGraph (object, optional): The ontology graph to use. Defaults to the ontology graph associated with self.

Returns:
: OrderedDict or None: An ordered dictionary of attribute names and their corresponding concepts, or None if no attributes found.

#### getRelationLinks(relationName=None, conceptName=None)

Retrieve relation links for a given relation and concept name.

This method retrieves relation links based on the relation name and/or
the concept name. It supports the flexibility to look up based on either
just a relation name, just a concept name, or both. If neither is given,
it returns all relation links.

Args:
: relationName (str or None): The name of the relation to filter by. If None, no filtering is done based on the relation name.
  conceptName (str or None): The name of the concept to filter by. If None, no filtering is done based on the concept name.

Returns:
: list: A list of DataNodes that match the given relation and concept names, or an empty list if no matches are found.

#### getRootDataNode()

Get the root DataNode.

Returns:
: object: The root DataNode.

#### hasAttribute(key)

Check if the DataNode has a specific attribute.

Args:
: key (str): The key of the attribute to check for.

Returns:
: bool: True if the attribute exists, False otherwise.

#### infer()

Calculate argMax and softMax for the ontology-based data structure.

#### inferGBIResults(\*\_conceptsRelations, model, kwargs)

Infer Grounded Belief Inference (GBI) results based on given concepts and relations.

Parameters:
- \_conceptsRelations: tuple or list

> Concepts and relations for which GBI is to be calculated. If empty, collects all from the graph.
- model: object
  : Solver model to be used in the GBI calculation.

Returns:
None. The function modifies the state of the self.graph object to store GBI results.

Logging:
- Logs whether the function was called with an empty or non-empty list of concepts and relations.
- Logs other debug and informational messages.

Side Effects:
- Modifies the state of the self.graph object to store GBI results.

#### inferGumbelLocal(temperature=1.0, hard=False)

Apply Gumbel-Softmax to local inference results for differentiable discrete sampling.

This method modifies the local/softmax attributes in-place to use Gumbel-Softmax
instead of standard softmax, enabling better gradient flow for discrete decisions.

Args:
: temperature (float): Controls sharpness of distribution (lower = more discrete)
  hard (bool): If True, use straight-through estimator (discrete forward, soft backward)

#### inferILPResults(\*\_conceptsRelations, key=('local', 'softmax'), fun=None, epsilon=1e-05, minimizeObjective=False, ignorePinLCs=False, Acc=None)

Calculate ILP (Integer Linear Programming) prediction for a data graph using this instance as the root.
Based on the provided list of concepts and relations, it initiates ILP solving procedures.

Parameters:
- 

```
*
```

\_conceptsRelations: tuple

> The concepts and relations used for inference.
- key: tuple, optional
  : The key to specify the inference method, default is (“local”, “softmax”).
- fun: function, optional
  : Additional function to be applied during ILP, default is None.
- epsilon: float, optional
  : The small value used for any needed approximations, default is 0.00001.
- minimizeObjective: bool, optional
  : Whether to minimize the objective function during ILP, default is False.
- ignorePinLCs: bool, optional
  : Whether to ignore pin constraints, default is False.
- Acc: object, optional
  : An accumulator for collecting results, default is None.

Raises:
- DataNodeError: When no concepts or relations are found for inference.

Returns:
- None: This function operates in-place and does not return a value.

#### inferLocal(keys=('softmax', 'argmax'), Acc=None)

Infer local probabilities and information for given concepts and relations.

Args:
: keys (tuple): Tuple containing the types of information to infer (‘softmax’, ‘argmax’, etc.).
  Acc (dict, optional): A dictionary containing some form of accumulated data for normalization.

Attributes affected:
: - This function manipulates the ‘attributes’ dictionary attribute of the class instance.

Notes:
: - The method uses PyTorch for tensor operations.
  - Logging is done to capture the time taken for inferring local probabilities.

#### isRelation(conceptRelation, usedGraph=None)

Check if a concept is a relation.

Args:
: conceptRelation (str or Concept): The concept or relation to check.
  usedGraph (object, optional): The ontology graph to use. Defaults to the one associated with self.

Returns:
: bool: True if the concept is a relation, otherwise False.

#### removeChildDataNode(dn)

Remove a child DataNode from the current DataNode.

Args:
: dn (DataNode): The DataNode to be removed.

#### removeEqualTo(equalDn, equalName='equalTo')

Remove a DataNode that is considered equal to the current DataNode.

Args:
: equalDn (DataNode): The DataNode to be removed.
  equalName (str, optional): The name of the relation for equality. Defaults to “equalTo”.

#### removeRelationLink(relationName, dn)

Remove a relation link between the current DataNode and another DataNode.

This method removes a relation link from the current DataNode to another
DataNode (‘dn’) under a given relation name. It also updates the impactLinks
for the target DataNode.

Args:
: relationName (str): The name of the relation to remove.
  dn (DataNode): The target DataNode to unlink from.

Returns:
: None

#### resetChildDataNode()

Reset all child DataNodes from the current DataNode.

#### setActiveLCs()

#### verifyResultsLC(key='/local/argmax')

Verify the results of ILP (Integer Linear Programming) by checking the percentage of
results satisfying each logical constraint (LC).

Parameters:
- key: str, optional

> Specifies the method used for verification. Supported keys are those containing “local” or “ILP”.
> Default is “/local/argmax”.

Raises:
- DataNodeError: When an unsupported key is provided.

Returns:
- verifyResult: object

> The result of the verification, typically a data structure containing percentages of
> results that satisfy each logical constraint.

#### visualize(filename: str, inference_mode='ILP', include_legend=False, open_image=False)

Visualize the current DataNode instance and its attributes.

This method creates a graph visualization using the Graphviz library. The
visualization includes attributes and relationships.

Args:
: filename (str): The name of the file where the Graphviz output will be stored.
  inference_mode (str, optional): The mode used for inference (“ILP” by default).
  include_legend (bool, optional): Whether or not to include a legend in the visualization.
  open_image (bool, optional): Whether or not to automatically open the generated image.

Raises:
: Exception: If the specified inference_mode is not found in the DataNode.

### *class* domiknows.graph.dataNode.DataNodeBuilder(\*args, \*\*kwargs)

Bases: `dict`

DataNodeBuilder class that extends Python’s built-in dictionary.

Attributes:
- context (str): The context in which the DataNodeBuilder is being used, defaults to “build”.
- myLoggerTime: Logger time instance for logging purposes.
- skeletonDataNode: Data structure for the basic DataNode skeleton.
- skeletonDataNodeFull: Data structure for the full DataNode skeleton.

Methods:
- \_\_init_\_: Initializes the DataNodeBuilder instance.
- \_\_getitem_\_: Overrides dict’s \_\_getitem_\_ to fetch item for a given key.
- \_\_changToTuple: Converts list elements to tuple form for use as dictionary keys.

#### *classmethod* clear()

Clear DataNodeBuilder class state.

This method resets any class-level state that might persist
between test runs or other scenarios where clean state is needed.

#### collectTime(start)

Collects the time taken for the \_\_setitem_\_ operation and stores it in internal lists.

This method calculates the time elapsed for a \_\_setitem_\_ operation and appends that,
along with the start and end timestamps, to respective lists stored in the object.

Args:
: start (int): The start time of the \_\_setitem_\_ operation in nanoseconds.

Notes:
: - The time taken for each \_\_setitem_\_ operation is stored in a list named ‘DataNodeTime’.
  - The start time for each \_\_setitem_\_ operation is stored in a list named ‘DataNodeTime_start’.
  - The end time for each \_\_setitem_\_ operation is stored in a list named ‘DataNodeTime_end’.

#### context *= 'build'*

#### createBatchRootDN()

Creates a batch root DataNode when certain conditions are met.

Conditions for creating a new batch root DataNode:
- If the DataNodeBuilder object already has a single root DataNode, no new root DataNode will be created.
- If the DataNodeBuilder object has DataNodes of different types, a batch root DataNode cannot be created.

### Parameters:

None

### Returns:

None

### Side Effects:

- Modifies the ‘dataNode’ attribute of the DataNodeBuilder object.
- Logs messages based on the production mode status and whether a new root DataNode is created or not.

### Raises:

- ValueError: When the DataNodeBuilder object has no DataNodes, or existing DataNodes have no connected graph.

### Notes:

- This method makes use of internal logging for debugging and timing.

#### createFullDataNode(rootDataNode)

Method to create a full data node based on the current skeleton of the DataNodeBuilder object.

### Parameters:

rootDataNode
: The root data node to which attributes will be added.

### Returns:

None

### Side Effects:

- Modifies internal state to reflect that a full data node has been created.
- Logs time taken to create the full data node.

### Notes:

- This method operates under the assumption that the DataNodeBuilder is initially in skeleton mode.

#### findDataNodesInBuilder(select=None, indexes=None)

Method to find data nodes that meet certain criteria within the DataNodeBuilder object.

### Parameters:

select
: A function to apply to each DataNode to determine if it should be selected. Defaults to None.

indexes
: A list of indexes to specifically look for. Defaults to None.

### Returns:

list
: A list of DataNodes that meet the given criteria.

#### getBatchDataNodes()

Retrieves and returns all DataNodes stored in the DataNodeBuilder object.

### Returns:

list or None
: Returns a list of all existing DataNodes if they exist; otherwise returns None.

### Side Effects:

- Logs various messages about the internal state and time usage of the DataNodeBuilder object.

### Raises:

None

### Notes:

- This method makes use of internal logging for debugging and timing.

#### getDataNode(context='interference', device='auto')

Retrieves and returns the first DataNode from the DataNodeBuilder object based on the given context and device.

### Parameters:

context
: The context under which to get the DataNode, defaults to “interference”.

device
: The torch device to set for the DataNode, defaults to ‘auto’.

### Returns:

DataNode or None
: Returns the first DataNode if it exists, otherwise returns None.

### Side Effects:

- Updates the torch device for the returned DataNode based on the ‘device’ parameter.
- Logs various messages based on the context and production mode.

### Raises:

None

### Notes:

- This method makes use of internal logging for debugging and timing.

## domiknows.graph.dataNodeConfig module

## domiknows.graph.dataNodeDummy module

### domiknows.graph.dataNodeDummy.addDatanodes(concept, conceptInfos, datanodes, allDns, level=1)

### domiknows.graph.dataNodeDummy.construct_ls_path_string(value)

### domiknows.graph.dataNodeDummy.createDummyDataNode(graph)

### domiknows.graph.dataNodeDummy.findConcept(conceptName, usedGraph)

### domiknows.graph.dataNodeDummy.findConceptInfo(usedGraph, concept)

### domiknows.graph.dataNodeDummy.ifConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, currentLc, ifResult, lcTestIndex, lcSatisfactionMsg, headLc)

### domiknows.graph.dataNodeDummy.lcConstrainSatisfactionMsg(lcSatisfactionTest, lcIterator, currentLc, lcResult, lcTestIndex, lcSatisfactionMsg, headLc)

### domiknows.graph.dataNodeDummy.satisfactionReportOfConstraints(dn)

## domiknows.graph.equality_mixin module

Equality mixin and an opt-in applier for Concept-like classes.

Usage (no auto-apply to avoid circular imports):
: from domiknows.graph.equality_mixin import apply_equality_mixin
  apply_equality_mixin(Concept)
  apply_equality_mixin(EnumConcept)  # optional

### *class* domiknows.graph.equality_mixin.EqualityMixin

Bases: `object`

#### get_canonical_concept() → [Concept](#domiknows.graph.concept.Concept)

#### get_equal_concepts(transitive: bool = False) → List[[Concept](#domiknows.graph.concept.Concept)]

#### get_equal_relations() → List[[Equal](#domiknows.graph.relation.Equal)]

#### get_equivalence_class() → List[[Concept](#domiknows.graph.concept.Concept)]

#### is_equal_to(other_concept: [Concept](#domiknows.graph.concept.Concept)) → bool

#### is_equal_to_transitive(other_concept: [Concept](#domiknows.graph.concept.Concept)) → bool

#### merge_equal_concepts(property_merge_strategy: str = 'first') → Dict[str, Any]

### domiknows.graph.equality_mixin.apply_equality_mixin(cls: type) → None

Apply EqualityMixin methods to the given class (e.g., Concept).

## domiknows.graph.executable module

### *class* domiknows.graph.executable.LogicDataset(data: Sequence[data_type], lc_name_list: list[str], logic_keyword: str = 'constraint', logic_label_keyword: str = 'label')

Bases: `Sequence`[`data_type`]

Wrapper around dataset containing executable logical expressions.

#### KEYWORD_FMT *: str* *= '_constraint_{lc_name}'*

#### *property* curr_lc_key *: str*

This key in each data item specifies which LC is currently active.
The value is the LC name (e.g., LC2).

#### *property* do_switch_key *: str*

This key (when present in the data item) indicates that we’re switching between LCs.

Only the presence of the key in the data item is used. The value has no meaning.

This is used in SolverModel.inference: when present will speed up searching through properties
by ignoring properties that are logical constraints but aren’t the current active LC
(set by self.curr_lc_key).

### domiknows.graph.executable.add_keyword(expr_str: str, kwarg_name: str, kwarg_value: Any) → str

Takes string containing logical expression without name parameter and
adds a name keyword argument to top-most expression.

e.g., andL(x, y) -> andL(x, y, name=”xyz”)

### domiknows.graph.executable.get_full_funcs(expr_str: str) → str

Converts logical expression to version with full important name.
Done recursively (not just to top-most expression); see: \_recurse_call(…)

e.g., andL(x, y) -> domiknows.graph.logicalConstrain.andL(x, y)

## domiknows.graph.graph module

### *class* domiknows.graph.graph.Graph(name=None, ontology=None, iri=None, local=None, auto_constraint=None, reuse_model=False)

Bases: [`BaseGraphTree`](#domiknows.graph.base.BaseGraphTree)

Represents a graph structure, extending from BaseGraphTree.

Class Attributes:
: varNameReversedMap (dict): Class-level variable to store a reversed mapping for variable names.

Instance Attributes:
: \_concepts (OrderedDict): Stores the concepts in an ordered dictionary.
  \_logicalConstrains (OrderedDict): Stores the logical constraints in an ordered dictionary.
  \_relations (OrderedDict): Stores the relations in an ordered dictionary.
  \_batch (NoneType): Currently not set, reserved for batch operations.
  cacheRootConcepts (dict): Cache for root concepts, initialized as an empty dictionary.
  auto_constraint (Any): Specifies whether to automatically create constraints.
  reuse_model (bool): Flag to indicate whether to reuse an existing model.
  ontology (tuple or Graph.Ontology): The ontology associated with the graph.

#### *class* Ontology(iri, local)

Bases: `tuple`

Namedtuple for Ontology data structure.

Attributes:
: iri (str): The IRI of the ontology. Default is None.
  local (str): The local identification of the ontology. Default is None.

#### iri

Alias for field number 0

#### local

Alias for field number 1

#### are_keys_new(given_dict, dict_list)

Check if all keys in ‘given_dict’ are not present in any dictionary within ‘dict_list’.

This method iterates over each key in ‘given_dict’ and checks if it exists in any of the dictionaries
contained within ‘dict_list’. If a key from ‘given_dict’ is found in any dictionary in ‘dict_list’,
the method returns False, indicating that not all keys are new. Otherwise, it returns True,
indicating all keys in ‘given_dict’ are new (i.e., not present in any dictionary in ‘dict_list’).

Parameters:
given_dict (dict): A dictionary whose keys are to be checked.
dict_list (list of dict): A list of dictionaries against which the keys of ‘given_dict’ are to be checked.

Returns:
bool: True if all keys in ‘given_dict’ are new, False otherwise.

#### *property* auto_constraint

Determines whether to automatically enforce constraints.

If the auto_constraint is not set, it will defer to the sup attribute.

Returns:
bool: True if constraints should be automatically enforced, False otherwise.

#### *property* batch

Gets the current batch.

Returns:
Any: The current batch.

#### check_if_all_used_variables_are_defined(lc, found_variables, used_variables=None, headLc=None)

Checks if all variables used in a logical constraint are properly defined.

This method traverses through the elements of a logical constraint to identify all the variables 
that are used but not defined. It also handles variable names in different types of paths.

Args:
lc (LogicalConstrain): The logical constraint to be processed.
found_variables (dict): Dictionary containing all variables that have been defined.

> The key is the variable name and the value is a tuple containing information
> about the variable.

used_variables (dict, optional): Dictionary to store variables that are used. The key is the variable name,
: and the value is a list of tuples, each containing the logical constraint,
  variable name, the type of the element that uses it, and the path to the variable.
  Defaults to None.

headLc (str, optional): The name of the parent logical constraint. Defaults to None.

Returns:
dict: A dictionary containing all used variables.

Raises:
Exception: If there are variables that are used but not defined, or if there are errors in the path definitions.

#### check_path(path, resultConcept, variableConceptParent, lc_name, foundVariables, variableName)

Checks the validity of a path in terms of relations and concepts.

This function checks the validity of a given path, including ensuring that each relation
or concept in the path has the correct type. It raises exceptions with informative error messages 
if the path is not valid.

Args:
path (list): The path to check, starting from the source concept.
resultConcept (tuple): The expected end concept of the path.
variableConceptParent (Concept): The parent concept for the source of the path.
lc_name (str): The name of the logical constraint where the path is defined.
foundVariables (dict): Dictionary of found variables in the scope.
variableName (str): The name of the variable being checked.

Raises:
Exception: Various types of exceptions are raised for different kinds of path invalidity.

#### collectVarMaps(lc, varMapsList)

Collects variable mappings (VarMaps) from a logical constraint (lc) and updates the list of collected VarMaps.

This method recursively traverses the elements of the logical constraint ‘lc’ to identify and process VarMaps.
It differentiates between the definition of new variables and the usage of existing ones. For new variables, 
it clones the current VarMap, adds the name of the logical constraint, and appends it to ‘varMapsList’. For 
existing variables, it updates the path variable in the current VarMap to match the one used in their 
definition. The method modifies ‘lc’ by removing VarMaps that define new variables.

Parameters:
lc (LogicalConstrain): The logical constraint from which VarMaps are to be collected.
varMapsList (list): A list that accumulates VarMaps. This list collects only the definitions of variables.

Returns:
list: The updated list of variable mappings (VarMaps) after processing ‘lc’.

Note:
- The method assumes the existence of specific types and structures within ‘lc’, such as ‘VarMaps’ tuples.
- The method is recursive and alters the structure of ‘lc’ by removing defining VarMaps.

#### compile_logic(data, logic_keyword='constraint', logic_label_keyword='label', extra_namespace_values={}, verbose=False)

Takes a dataset containing keys logic_keyword and logic_label_keyword and
converts it to a LogicDataset and adds the expressions to the graph.
Using the LogicDataset during e.g., training lets you switch between these constraints.
data: and iterable of dicts containing the keys specified by logic_keyword and logic_label_keyword
extra_namespace_values: dict[str, Any], any values added to this dictionary get added to the namespace used when executing the logical expressions (the variable names are the keys).

#### *property* concepts

Getter for the concepts.

Returns:
: dict: Dictionary of concepts.

#### findConcept(conceptName)

Finds the root concept or relation for a given concept or relation.

This method performs a recursive search to identify the root concept or relation.
If a result has been previously computed, it retrieves the result from cache to avoid redundant computation.

Args:
relationConcept (Any): The concept or relation for which the root is to be found. The type depends on the implementation.

Returns:
Any: The root concept or relation.

Raises:
AttributeError, TypeError: If the attribute ‘is_a’ is not available or if the type is incorrect.

#### findConceptInfo(concept)

Finds and returns various information related to a given concept.

This method compiles a dictionary containing different attributes and relations of the concept. 
It looks for the ‘has_a’, ‘contains’, and ‘is_a’ relationships and also identifies if the concept is a root.

Args:
concept (Any): The concept for which information is to be found. The type depends on the implementation.

Returns:
OrderedDict: A dictionary containing the following keys:

> - ‘concept’: The original concept.
> - ‘relation’: Boolean indicating if the concept has a ‘has_a’ relationship.
> - ‘has_a’: List of ‘has_a’ relations.
> - ‘relationAttrs’: An ordered dictionary of relation attributes.
> - ‘contains’: List of concepts that the original concept contains.
> - ‘containedIn’: List of concepts that contain the original concept.
> - ‘is_a’: List of concepts that the original concept is a type of.
> - ‘root’: Boolean indicating if the concept is a root concept.

#### findRootConceptOrRelation(relationConcept)

#### find_lc_variable(lc, found_variables=None, headLc=None)

Finds all variables defined in a logical constraint and reports errors for duplicates.

This method traverses through the elements of a logical constraint to find all the variables 
that have been defined. It checks for incorrect cardinality definitions, multiple definitions of the 
same variable, and variables that are not associated with any concept among other things.

Args:
lc (LogicalConstrain): The logical constraint to be processed.
found_variables (dict, optional): Dictionary to store found variables. The key is the variable name,

> and the value is a tuple containing the logical constraint, variable name, 
> and the concept associated with the variable.
> Defaults to None.

headLc (str, optional): The name of the parent logical constraint. Defaults to None.

Returns:
dict: A dictionary containing all found variables.

Raises:
Exception: If there are issues with the variable definitions or cardinality.

#### getPathStr(path)

Converts a path of concepts and relations to a string representation.

This method iterates over a given path, which can include instances of the Relation and Concept classes,
and constructs a string representation of the path.

Args:
path (list): A list of path elements which can be instances of Relation or Concept classes.

> The first element in the list is not processed, and the list should be non-empty.

Returns:
str: A string representation of the path, excluding the first element.

#### get_apply(name)

Finds and returns the concept or relation with the given name.

Args:
name (str): The name of the concept or relation to find.

Returns:
Object: The concept or relation with the specified name, or the result of BaseGraphTree.get_apply if not found.

#### get_constraint_concept()

#### get_properties(\*tests)

Finds and returns properties that meet the given conditions.

Args:
tests (callable): Variable-length argument list of test conditions each property must pass.

Returns:
list: A list of properties that meet all the given test conditions.

#### get_sensors(\*tests)

Finds and returns sensors that meet the given conditions.

Args:
tests (callable): Variable-length argument list of test conditions each sensor must pass.

Returns:
list: A list of sensors that meet all the given test conditions.

#### handleVarsPath(lc, varMaps)

Processes and updates the variable paths in a logical constraint (lc) based on the mappings provided in varMaps.

This method iterates through the elements of ‘lc’ and performs various transformations based on the type
of each element and the presence of variable mappings in ‘varMaps’. The method handles nested logical 
constraints recursively, updates variables already in V form, and modifies variable paths using mappings 
from ‘varMaps’. Additionally, it removes all ‘VarMaps’ elements from ‘lc’.

Parameters:
lc (LogicalConstrain): The logical constraint to be processed.
varMaps (dict): A dictionary containing mappings of variable names to their respective V instances or paths.

Note:
- The method assumes a specific structure of ‘lc’ and ‘varMaps’, with ‘lc’ containing elements like

> LogicalConstrain, V, Concept, and tuples with ‘VarMaps’.
- It employs a flag ‘needsVariableUpdate’ to track if the next element requires variable path updates.
- The method is recursive for nested logical constraints and alters the structure of ‘lc’.

#### *property* logicalConstrains

Getter for the logical constraints.

Returns:
: dict: Dictionary of logical constraints.

#### *property* logicalConstrainsRecursive

Generator function that yields logical constraints recursively.

This method goes through all nodes and yields the logical constraints 
if the node is an instance of Graph.

Yields:
: tuple: A tuple containing key-value pairs of logical constraints.

#### *property* ontology

Gets the ontology associated with the object.

Returns:
Graph.Ontology: The ontology associated with the object.

#### print_predicates()

Generate and return a list of predicates with their variables.

This method goes through the subgraphs and concepts to generate a list of predicates,
determining the variables they use based on their relations and superclass structure.

Returns:
: list: A list of strings, each representing a predicate and its variables.

#### *property* relations

Getter for the relations.

Returns:
: dict: Dictionary of relations.

#### set_apply(name, sub)

Sets the Concept or Relation for a given name in the object.

If the sub is a Graph, it delegates to the parent class BaseGraphTree.set_apply.
If it is a Concept or Relation, it adds it to the respective dictionary.

Args:
name (str): The name to set for the Concept or Relation.
sub (Graph|Concept|Relation): The object to set for the name. This can be an instance of Graph,

> Concept, or Relation.

TODO:
1. Handle cases where a concept has the same name as a subgraph.
2. Handle other types for the ‘sub’ parameter.

#### *property* subgraphs

Ordered dictionary containing the subgraphs.

Returns:
: OrderedDict: An ordered dictionary of the subgraphs.

#### varNameReversedMap *= {}*

#### visualize(filename, open_image=False)

Visualizes the graph using Graphviz.

Creates a directed graph and populates it with nodes for concepts and
edges for relations. It also recursively adds subgraphs. Finally, it
either saves the graph to a file or returns the graph object depending on
the filename parameter.

Args:
filename (str|None): The name of the file where the graph will be saved.

> If None, the graph object is returned instead.

open_image (bool): Whether to open the image after rendering.
: Defaults to False.

Returns:
graphviz.Digraph|None: The graph object if filename is None, otherwise None.

#### what()

Method to get the summary of the graph tree.

This method provides a dictionary containing the base graph tree and its concepts.

Returns:
: dict: Dictionary containing ‘concepts’ and the base graph tree.

## domiknows.graph.logicalConstrain module

### *class* domiknows.graph.logicalConstrain.LcElement(\*e, name=None)

Bases: `object`

#### *exception* LcElementError

Bases: `Exception`

### *class* domiknows.graph.logicalConstrain.LogicalConstrain(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LcElement`](#domiknows.graph.logicalConstrain.LcElement)

#### createILPAccumulatedCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName='COUNT')

#### createILPCompareCounts(model, myIlpBooleanProcessor, v, headConstrain, compareOp, diff, integrate, , logicMethodName='COUNT_CMP')

Build ILP constraints (and optionally return indicator vars) enforcing
compareOp between the **counts** of two variable sets.

compareOp : one of ‘>’, ‘>=’, ‘<’, ‘<=’, ‘==’, ‘!=’
diff      : constant offset  (we enforce  count(A) - count(B) ∘ diff)

#### createILPConstrains(lcName, lcFun, model, v, headConstrain)

#### createILPCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName='COUNT')

#### createSingleVarILPConstrains(lcName, lcFun, model, v, headConstrain)

#### createSummation(model, myIlpBooleanProcessor, v, headConstrain, integrate, logicMethodName='SUMMATION')

#### getLcConcepts()

#### strEs()

### *class* domiknows.graph.logicalConstrain.V(name, v)

Bases: `tuple`

#### name

Alias for field number 0

#### v

Alias for field number 1

### *class* domiknows.graph.logicalConstrain.andL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.atLeastAL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_AccumulatedCountBaseL`

#### limitOp *: str* *= '>='*

### *class* domiknows.graph.logicalConstrain.atLeastL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CountBaseL`

#### limitOp *: str* *= '>='*

### *class* domiknows.graph.logicalConstrain.atMostAL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_AccumulatedCountBaseL`

#### limitOp *: str* *= '<='*

### *class* domiknows.graph.logicalConstrain.atMostL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CountBaseL`

#### limitOp *: str* *= '<='*

### *class* domiknows.graph.logicalConstrain.eqL(\*e, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.equalCountsL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CompareCountsBaseL`

#### compareOp *= '=='*

### *class* domiknows.graph.logicalConstrain.equivalenceL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.exactAL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_AccumulatedCountBaseL`

#### limitOp *: str* *= '=='*

### *class* domiknows.graph.logicalConstrain.exactL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CountBaseL`

#### limitOp *: str* *= '=='*

### *class* domiknows.graph.logicalConstrain.existsAL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_AccumulatedCountBaseL`

#### fixedLimit *: int | None* *= 1*

#### limitOp *: str* *= '>='*

### *class* domiknows.graph.logicalConstrain.existsL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CountBaseL`

#### fixedLimit *: int | None* *= 1*

#### limitOp *: str* *= '>='*

### *class* domiknows.graph.logicalConstrain.fixedL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.forAllL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.greaterEqL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CompareCountsBaseL`

#### compareOp *= '>='*

### *class* domiknows.graph.logicalConstrain.greaterL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CompareCountsBaseL`

#### compareOp *= '>'*

### *class* domiknows.graph.logicalConstrain.ifL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.lessEqL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CompareCountsBaseL`

#### compareOp *= '<='*

### *class* domiknows.graph.logicalConstrain.lessL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CompareCountsBaseL`

#### compareOp *= '<'*

### *class* domiknows.graph.logicalConstrain.nandL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.norL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.notEqualCountsL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: `_CompareCountsBaseL`

#### compareOp *= '!='*

### *class* domiknows.graph.logicalConstrain.notL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.orL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### *class* domiknows.graph.logicalConstrain.sumL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

### domiknows.graph.logicalConstrain.use_grad(grad)

### *class* domiknows.graph.logicalConstrain.xorL(\*e, p=100, active=True, sampleEntries=False, name=None)

Bases: [`LogicalConstrain`](#domiknows.graph.logicalConstrain.LogicalConstrain)

## domiknows.graph.property module

### *class* domiknows.graph.property.Property(prop_name)

Bases: [`BaseGraphShallowTree`](#domiknows.graph.base.BaseGraphShallowTree)

#### attach(sub)

#### attach_to_context(name=None)

#### find(\*sensor_tests)

#### get_fullname(delim='/')

## domiknows.graph.relation module

### *class* domiknows.graph.relation.Contains(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`OTMRelation`](#domiknows.graph.relation.OTMRelation)

#### *classmethod* name()

#### relation_cls_name *= 'contains'*

### *class* domiknows.graph.relation.Equal(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`OTORelation`](#domiknows.graph.relation.OTORelation)

#### *classmethod* name()

#### relation_cls_name *= 'equal'*

### *class* domiknows.graph.relation.HasA(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`MTORelation`](#domiknows.graph.relation.MTORelation)

#### *classmethod* name()

#### relation_cls_name *= 'has_a'*

### *class* domiknows.graph.relation.HasMany(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`OTMRelation`](#domiknows.graph.relation.OTMRelation)

#### *classmethod* name()

#### relation_cls_name *= 'has_many'*

### *class* domiknows.graph.relation.IsA(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`OTORelation`](#domiknows.graph.relation.OTORelation)

#### *classmethod* name()

#### relation_cls_name *= 'is_a'*

### *class* domiknows.graph.relation.MTMRelation(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`Relation`](#domiknows.graph.relation.Relation)

### *class* domiknows.graph.relation.MTORelation(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`Relation`](#domiknows.graph.relation.Relation)

### *class* domiknows.graph.relation.NotA(src, dst, \*args, \*\*kwargs)

Bases: [`OTORelation`](#domiknows.graph.relation.OTORelation)

#### *classmethod* name()

#### relation_cls_name *= 'not_a'*

### *class* domiknows.graph.relation.OTMRelation(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`Relation`](#domiknows.graph.relation.Relation)

### *class* domiknows.graph.relation.OTORelation(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`Relation`](#domiknows.graph.relation.Relation)

### *class* domiknows.graph.relation.Relation(src, dst, argument_name, reverse_of=None, auto_constraint=None)

Bases: [`BaseGraphTree`](#domiknows.graph.base.BaseGraphTree)

#### *property* auto_constraint

#### *property* dst

#### *property* graph

#### *property* mode

#### *classmethod* name()

#### set_apply(name, sub)

#### *property* src

#### what()

### *class* domiknows.graph.relation.Transformed(relation, property, fn=None)

Bases: `object`

### domiknows.graph.relation.disjoint(\*concepts)

## domiknows.graph.trial module

### *class* domiknows.graph.trial.Trial(data=None, obsoleted=None, name=None)

Bases: `object`

#### *classmethod* clear()

#### *classmethod* default()

#### items()

#### keys()

#### *property* sup

#### values()

### *class* domiknows.graph.trial.TrialTree(trial, name=None)

Bases: [`BaseGraphTree`](#domiknows.graph.base.BaseGraphTree)

#### *property* trial

#### what()

## Module contents
