# Inference

- [Inference](#inference)
  - [Class Overview](#class-overview)
    - [ILP Solver](#ilp-solver)

## Class Overview

- package `regr.solver`:
- `ilpOntSolver`:

### ILP Solver

The solver builds the ILP (Integer Linear Programming) model based on the constrains defined in the learning model and the prediction data for graph concepts and relations assignment to example tokens.
The actual used ILP is Zero-one linear programming in which the variables are restricted to be either 0 or 1.
It solves the ILP model and provides the most optimized assignment.

The solver can be called on the DataNode (usually the root DataNode of the Data Graph) with DataNode method:

```python
inferILPConstrains(*_conceptsRelations, fun=None)
```

The method retrieves the constrains from the ontology graph associated with the Data Graph and the probabilities from Data Graph nodes attributes.
It has two arguments:

* `_conceptsRelations` is a collection of concepts and relations for which the ILP model should be solved.
They can be provide as Concepts (nodes in the model graph) or strings representing concepts or relations names.
If this collection is empty then the methods will use all concepts and relations in the Data Graph.

* `fun` is a optional function modifying the original probability in the Data Graph before they are used in the ILP model.

The results of the ILP solution are added to nodes in the Data Graph with key ILP.
