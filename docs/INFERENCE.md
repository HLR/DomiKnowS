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

- `_conceptsRelations` is a collection of concepts and relations for which the ILP model should be solved.
They can be provide as Concepts (nodes in the model graph) or strings representing concepts or relations names.
If this collection is empty then the methods will use all concepts and relations in the Data Graph.

- `fun` is a optional function modifying the original probability in the Data Graph before they are used in the ILP model.

The solver [implementation using Gurobi](https://github.com/kordjamshidi/RelationalGraph/blob/master/regr/solver/gurobiILPOntSolver.py) is called with probabilities for token classification obtained from learned model. 

The solver encodes mapping from constrains to the appropriate equivalent logical expression for the given graph and the provided probabilities.
The `regr.solver.ilpBooleanMethods.ilpBooleanProcessor` encodes basic logical expressions into the ILP equations. Supported logical operations are:

- "NOT": `notVar()`
- "AND": `and2Var`, `andVar()`
- "OR": `or2Var()`, `orVar()`
- "IF": `ifVar()`
- "NAND": `nand2Var()`, `nandVar()`
- "XOR": `xorVar()`
- "EPQ": `epqVar()`
- "NOR": `nor2Var()`, `norVar()`

The solver ILP model is solved by Gurobi and the found solutions for optimal classification of tokens and relations is returned.

The results of the ILP solution are added to nodes in the Data Graph with key ILP. The value can be access using the `getAttribute` method of the DataNode, e.g.:

```python
dn.getAttribute(work_for, 'ILP').item()
```
