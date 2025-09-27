# Inference

There are two main inference tools in DomiKnows framework:

- [ILP](#ILP)
- [GBI](#gbi)

## ILP
The following is the tutorial on the ILP solver for infernece.

- [Class Overview](#class-overview)
- [ILP Solver](#ilp-solver)
- [ILP Inference](#ilp-inference)
- [ILP Loss for Logical Constrains](#ilp-loss-for-logical-constrains)
- [Softmax and Argmax Inference](#softmax-and-argmax-inference)
- [Inference Result Access](#inference-result-access)
- [Inference Metrics](#inference-metrics)

### Class Overview

- package `domiknows.solver`:
- `ilpOntSolver`,
- `ilpOntSolverFactory`,
- `ilpConfig`,
- `gurobiILPOntsolver`,
- `ilpBooleanMethods`,
- `lcLossBooleanMethods`,
- `gurobiILPOntsolver`,
- `gurobiILPBooleanMethods`,


- package `domiknows.graph`:
- `dataNode`:

### ILP Solver

The solver builds the ILP (Integer Linear Programming) model based on the constrains defined in the learning model and the prediction data for graph concepts and relations assignment to example tokens.
The actual used ILP is Zero-One linear programming in which the variables are restricted to be either 0 or 1.
It solves the ILP model and provides the most optimized assignment. The model objective is by default maximized. However it can be optionally minimized if appropriate parameter is provided to the method.

The solver can be called on the [DataNode](Query%20and%20Access%20%28Datanode%29.md) (usually the root DataNode of the Data Graph) with the method:

#### ILP Inference
```python
inferILPResults(*_conceptsRelations, fun=None, epsilon = 0.00001, minimizeObjective = False):
```

It has arguments:

- `_conceptsRelations` is a collection of concepts and relations for which the ILP model should be solved.
They can be provide as Concepts (nodes in the knowledge graph) or strings representing concepts or relations names.
If this collection is empty then the methods will use all concepts and relations in the Data Graph.


- `fun` is a optional function modifying the original probability in the Data Graph before they are used in the ILP model.


- `epsilon` is a value by which  the original probability in the Data Graph are modify before they are used in the ILP model.


- `minimizeObjective` by default objective is maximized however if this variable is set to True then the ILP model will minimize the objective.

The solver implementation, using Gurobi, is called with classification probabilities obtained from learned model. 

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
where `conceptRelation` is the name of the concept or relation for which the hard constraint is being retrieved.
The value of this attribute can be 0 or 1.

The method retrieves the constrains from the knowledge graph associated with the Data Graph.
The solver translates constrains defined in the graph to the equivalent logical expressions.
The `domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor` encodes basic logical expressions into the ILP equations. Supported logical operations are:

- "NOT": `notVar()`,
- "AND": `and2Var`, `andVar()`,
- "OR": `or2Var()`, `orVar()`,
- "IF": `ifVar()`,
- "NAND": `nand2Var()`, `nandVar()`,
- "XOR": `xorVar()`,
- "EPQ": `epqVar()`,
- "NOR": `nor2Var()`, `norVar()`,
- "COUNT": `countVar()`.

The solver ILP model is solved by Gurobi and the found solutions for optimal classification of variables and relations is returned.

#### ILP Loss for Logical Constrains 

Solver can also calculate the loss for each logical constraint:

```python
dn.calculateLcLoss()
```
The method returns a dictionary with entry for each logical constraint and tensor with calculated losses.

Example of loss calculation:

```
 # nandL(people,organization)
 #                               John    works   for     IBM
 'lc0LossTensor' : torch.tensor([0.2000, 0.0000, 0.0000, 0.5100], device=device),
        
 # ifL(work_for('x'), andL(people(path=('x', rel_pair_word1.name)), organization(path=('x', rel_pair_word2.name))))
 #                                 John           works          for      IBM
 'lc2LossTensor' : torch.tensor([0.2000,         0.2000,        0.2000,  0.0200,  # John
                                 float("nan"),  float("nan"),  0.4000,  0.2900,  # works
                                 0.0200,         0.0300,        0.0500,  0.1000,  # for
                                 0.5500,         0.2000,        0.1000,  0.0000], # IBM
                                 device=device)
```

#### Softmax and Argmax Inference

In addition to ILP inference the standard softmax and agmax can be also calculated on learned predictions.

The method calculating this inference across all datanotes for each concepts is:

```python
dn.infer()
```

It adds attributes: <concept>/softmax and <concept>/argmax to each datanote.

The method calculating this inference locally for prediction in the given  datanotes for each concepts is:

```python
dn.inferLocal()
```

It adds attributes: <concept>/local/softmax and <concept>/local/argmax to each datanote.

#### Inference Result Access

The results of the inference solution are added to nodes in the Data Graph with key appropriate for the inference (e.g.: ILP). 

The value can be access using the `getAttribute` method of the DataNode:

```python
dn.getAttribute(concept, inferKey).item()
```

The inference results for collection of dataNodte can be collected using method:

```python
dn.collectInferedResults(concept, inferKey),
```

- `concept`  - is a name of the concept for which results are requested, e.g. `work_for`

- `inferKey` - is a name of the key  for the inference type e.g. `ILP`, `argmax`, `softmax`, `local/argmax`, `local/softmax`


#### Inference Metrics

The inference results can be compared to ground truth (stored in the dataNote with key `label`) and metrics can be calculated using  dataNode method:

```python
dn.getInferMetric(*conceptsRelations, inferType='ILP', weight = None):
```

```python
'<' + conceptRelation + '>'
```
where `conceptRelation` is the name of the concept or relation for which the probability is being retrieved.

- `inferType` - is a name of the key  for the inference type e.g. `ILP`, `argmax`, `softmax`,

- `weight` - modifies the inference result used to calculate metrics.

The method returns dictionary with entry for each concept and additional entry 'Total'. Each record in the dictionary has entry for:
- `TP` -  true positive
- `FP` - false positive
- `TN` - true negative
- `FN` - false negative
- `P` - precision
- `R` - recall
- `F1` - F1 score
 
Example of the ILP metrics:

```
ILP metrics Total {'TP': tensor(1.), 'FP': tensor(15.), 'TN': tensor(13.), 'FN': tensor(1.), 'P': tensor(0.0625), 'R': tensor(0.5000), 'F1': tensor(0.1111)}
ILP metrics work_for {'TP': tensor(0.), 'FP': tensor(0.), 'TN': tensor(4.), 'FN': tensor(0.)}
ILP metrics people {'TP': tensor(1.), 'FP': tensor(1.), 'TN': tensor(0.), 'FN': tensor(0.), 'P': tensor(0.5000), 'R': tensor(1.), 'F1': tensor(0.6667)}
```

## GBI
### Overview
Gradient-based inference (GBI) [1] performs gradient steps to optimize over some black-box function. In DomiKnowS, the function being optimized is the number of constraint violations.

For each example, GBI will optimize the following loss:
```python
log_probs * (num_constraint_violations / total_constraints) + reg_strength * regularization_loss
```

Here, `log_probs` refers to the mean log probabilities over all the outputs of the model, and `regularization_loss` refers to the L2 distance between the original (frozen) parameters and the new parameters. `reg_strength` is a hyperparameter that controls how much GBI should stray from the original parameters.

GBIModel is defined in `domiknows.model.gbi`. Several hyperparameters can be specified at initialization:

* **gbi_iters**: The maximum number of gradient update steps to perform. GBI will exit early if all constraints are specified.
* **lr**: The step size of each update step.
* **reg_weight**: The regularization strength, as described previously.
* **reset_params**: If set to `True`, the parameters of the model will be reset to the original (non-optimized) parameters after GBI is complete. If set to `False`, the parameters will *only* be reset if the loss becomes `NaN` or the constraints aren't satisfied after `gbi_iters` updates. **During inference, this should be set to `True`.** The default value for this parameter is `True`.

### Usage
In DomiKnowS, GBI can be used in a similar way as ILP. For example, instead of calling `inferILPResults` the user can invoke `inferGBIResults`. The user can also add `GBI` to the `inferTypes` parameter. For example:

```python
program = SolverPOIProgram(
    graph,
    poi=(image_batch, image, image_pair),
    inferTypes=['local/argmax', 'local/softmax', 'GBI']
)
```

When populating the program, make sure to set `grad=True` as GBI requires gradients to perform updates. For example:
```python
node = program.populate_one(dataitem, grad=True)
```

Parameters for `GBIModel` can be specified when initializing the program. For example:
```python
program = SolverPOIProgram(
    graph,
    poi=(image_batch, image, image_pair),
    inferTypes=['local/argmax', 'local/softmax', 'GBI'],
    gbi_iters=100,
    lr=1e-1,
    reg_weight=1.5,
    reset_params=True
)
```

Here, by setting `reset_params=True` (which is also the default), GBI will be used for inference only, as the parameters are reset after each iteration.

[1] "Gradient-Based Inference for Networks with Output Constraints" Jay Yoon Lee et. al

