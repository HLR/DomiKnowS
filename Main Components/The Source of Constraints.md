# The Source of Constraints 

The following explains how to define constraints for our solver.

- [Inroduction](#inroduction)
- [Constrains Definition](#constrains-definition)
  - [Graph with logical Constraints](#graph-with-logical-constraints)
- [Ontology File as Constraints Definition](#ontology-file-as-constraints-definition)


## Inroduction

The solver constructs the ILP (Integer Linear Programming) model based on the constraints defined in the learning model and the prediction data for assigning graph concepts and relations to example tokens.
The actual used ILP is Zero-one linear programming, in which the variables are restricted to be either 0 or 1.
It solves the ILP model and provides the most optimized assignment.

The solver can be called on the DataNode (usually the root DataNode of the Data Graph) with the DataNode method:
 
```
inferILPConstrains(*_conceptsRelations, fun=None)
```
The method retrieves the constraints from the ontology graph associated with the Data Graph and the probabilities from the attributes of the Data Graph nodes.
It has two arguments:
* *_conceptsRelations* is a collection of concepts and relations for which the ILP model should be solved. 
They can be provided as Concepts (nodes in the model graph) or strings representing concept or relation names. 
If this collection is empty, then the methods will use all concepts and relations in the Data Graph.


* *fun* is an optional function modifying the original probability in the Data Graph before they are used in the ILP model.

The results of the ILP solution are added to nodes in the Data Graph with the key ILP.

## Constraints Definition

ILP constraints are the same as the constraints defined in the [knowledge graph constraints](Knowledge%20Declaration%20(Graph).md#graph-constraints). The ILP constraints could be specified in the **ontology graph itself with defined logical constraints** or in the **ontology (in OWL file)** provided as a URL in the ontology graph. Each method is implemented in the backend to mimic a constraint using the Gurobi Python package [gurobipy](https://pypi.org/project/gurobipy/).

