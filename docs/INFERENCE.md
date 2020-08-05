# Inference

- [Inference](#inference)
  - [Class Overview](#class-overview)
    - [ILP Solver](#ilp-solver)

## Class Overview

- package `regr.solver`:
- `ilpOntSolver`:

### ILP Solver

The solver builds the ILP (Integer Linear Programming) model based on the constrains defined in the learning model and the prediction data for graph concepts and relations assignment to example tokens.
The actual used ILP is Zero-One linear programming in which the variables are restricted to be either 0 or 1.
It solves the ILP model and provides the most optimized assignment. The model objective is by default maximized. However it can be optionally minimized if appropriate parameter is provided to the method.

The solver can be called on the [DataNode](QUERY.md) (usually the root DataNode of the Data Graph) with the method:

```python
inferILPConstrains(*_conceptsRelations, fun=None, epsilon = 0.00001,  minimizeObjective = False)
```

It has arguments:

- `_conceptsRelations` is a collection of concepts and relations for which the ILP model should be solved.
They can be provide as Concepts (nodes in the knowledge graph) or strings representing concepts or relations names.
If this collection is empty then the methods will use all concepts and relations in the Data Graph.


- `fun` is a optional function modifying the original probability in the Data Graph before they are used in the ILP model.


- `epsilon` is a value by which  the original probability in the Data Graph are modify before they are used in the ILP model.


- `minimizeObjective` if set to True then the ILP model will minimize the objective.

The solver [implementation using Gurobi](/regr/solver/gurobiILPOntSolver.py) is called with probabilities for token classification obtained from learned model. 

The method retrieves the probabilities from Data Graph nodes attributes. The key use to retrieve it has a pattern:

```python
'<' + conceptRelation + '>'
```
where `conceptRelation` is the name of the concept or relation for which the probability is being retrieved.

Additionally, the method determines if some of the concepts or relations are **hard constrained**. 
The concept or relation is hard constrained if there is a dedicated DataNode with ontology type equal to this concept or relation.
Alternatively, if the DataNode for the parent concept or relation to the given concept or relation does has a attribute with the key:

```python
conceptRelation
```
where `conceptRelation` is the name of the concept or relation for which the hard constrain is being retrieved.
The value of this attribute can be 0 or 1.

The method retrieves the constrains from the knowledge graph associated with the Data Graph.
The solver encodes mapping from constrains to the appropriate equivalent logical expression for the given graph and the provided probabilities.
The `regr.solver.ilpBooleanMethods.ilpBooleanProcessor` encodes basic logical expressions into the ILP equations. Supported logical operations are:

- "NOT": `notVar()`,
- "AND": `and2Var`, `andVar()`,
- "OR": `or2Var()`, `orVar()`,
- "IF": `ifVar()`,
- "NAND": `nand2Var()`, `nandVar()`,
- "XOR": `xorVar()`,
- "EPQ": `epqVar()`,
- "NOR": `nor2Var()`, `norVar()`,
- "COUNT": `countVar()`.

The solver ILP model is solved by Gurobi and the found solutions for optimal classification of tokens and relations is returned.

The results of the ILP solution are added to nodes in the Data Graph with key ILP. The value can be access using the `getAttribute` method of the DataNode, e.g.:

```python
dn.getAttribute(work_for, 'ILP').item()
```

The solver can also verify if the constrains and the ground truth for the given DataNode are consistent, with the method:

```python
verifySelection(*_conceptsRelations)
```
It has arguments:

- `_conceptsRelations` is a collection of concepts and relations for which the ILP model should be solved.
They can be provide as Concepts (nodes in the knowledge graph) or strings representing concepts or relations names.
If this collection is empty then the methods will use all concepts and relations in the Data Graph.

The method retrieves the ground truth from Data Graph nodes attributes. The key use to retrieve it has a pattern:

```python
'<' + conceptRelation + '>/label'
```
where `conceptRelation` is the name of the concept or relation for which the ground truth is being retrieved.

The result of the verification is either True or False.


