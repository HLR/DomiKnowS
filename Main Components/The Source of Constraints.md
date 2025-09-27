# The Source of Constrains 

The following explains how to define constraints for our solver.

- [Inroduction](#inroduction)
- [Constrains Definition](#constrains-definition)
  - [Graph with logical Constrains](#graph-with-logical-constrains)
- [Ontology File as Constrains Definition](#ontology-file-as-constrains-definition)


## Inroduction

The solver builds the ILP (Integer Linear Programming) model based on the constrains defined in the learning model and the prediction data for graph concepts and relations assignment to example tokens.
The actual used ILP is Zero-one linear programming in which the variables are restricted to be either 0 or 1.
It solves the ILP model and provides the most optimized assignment.

The solver can be called on the DataNode (usually the root DataNode of the Data Graph) with DataNode method:
 
```
inferILPConstrains(*_conceptsRelations, fun=None)
```
The method retrieves the constrains from the ontology graph associated with the Data Graph and the probabilities from Data Graph nodes attributes.
It has two arguments:
* *_conceptsRelations* is a collection of concepts and relations for which the ILP model should be solved. 
They can be provide as Concepts (nodes in the model graph) or strings representing concepts or relations names. 
If this collection is empty then the methods will use all concepts and relations in the Data Graph.


* *fun* is a optional function modifying the original probability in the Data Graph before they are used in the ILP model.

The results of the ILP solution are added to nodes in the Data Graph with key ILP.

## Constraints Definition

ILP constraints are the same as the constraints defined the [knowledge graph constraints](Knowledge%20Declaration%20(Graph).md#graph-constraints). The ILP constrains could be specified in the **ontology graph itself with defined logical constrains** or in the **ontology (in OWL file)** provided as url in the ontology graph. Each method is implemented in the backend to mimic a constraint using Gurobi Python package [gurobipy](https://pypi.org/project/gurobipy/).

