# Workflow

The following describes the workflow that enables users to train their model using the chosen method.

- [Workflow](#workflow)
  - [Class Overview](#class-overview)
  - [Workflows](#workflows)
    - [Initiate](#initiate)
    - [Train: `train()`](#train-train)
    - [Test: `test()`](#test-test)
    - [Eval: `eval()`](#eval-eval)
  - [Programs](#Programs)
    - [Inferece Mask Loss (IML)](#inferece-mask-loss-iml-1)
    - [Priaml Dual (PD)](#priaml-dual-pd-2)
    - [GBI](#gbi)
    
## Class Overview

- package `domiknows.program`:
- `SolverPOIProgram`:

## Workflows

There are built-in workflows that are commonly used in machine learning tasks. Also, user can extend the `LearningBasedProgram` class to build their own workflows.

### Initiate

A basic program can be initiated with a graph. For example:

```python
program = LearningBasedProgram(graph)
```

Our main and most basic `LearningBasedProgram`, which serves as the parent of other programs that we will introduce further, is `SolverPOIProgram`. Let's look at an example:

```python
program = SolverPOIProgram(graph,poi=[relation[org]],inferTypes=['local/softmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()))
```

The first parameter is always our graph, which we defined earlier. `poi`, which stands for Point of Interest, defines the label in our graph on which we want to train our parameters. For example, imagine that your relation concept has two labels, `org` and `family`. In this scenario, you may want to create and define a separate `SolverPOIProgram` to train them. You can also train them together by including them in the same list `poi=[relation[org], relation[family]]` or `poi=[relation]`.

Keep in mind that POI should include the final label(s) you want to train, as well as your concept path to reach them in the knowledge graph; otherwise, DomiKnows may not know how to navigate to the desired label in your graph pathing. Imagine this graph:

```python
with Graph(name='global') as graph:
    x = Concept(name='x')
    y = Concept(name='y')
    y_label = y(name='y_label')

    x_contain_y, =x.contains(y)
```

In this example, if we want to train a model to estimate y_label, we need to include x_contain_y in our poi: `poi=[y,x[x_contain_y]]`.


### Train: `train()`

`train()` takes a required argument `training_set` as input. Additionally, the user can provide `valid_set` and/or `test_set`. There are three arguments: reader objects, which are essentially `Iterable`s. 
There are other optional arguments in `train()`.

`train()` will invoke `train_epoch()` for `train_epoch_num` times and invoke `test()` on `valid_set` if provided, for each time. And after training is done, `test()` will be invoked with `test_set` if provided.

`Optim` specifies an optimizer class to initiate an optimizer and use for training the model.

Loss and metric will be stored in `program.model.loss` and `program.model.metric` for access.

Example:

```python
program.train(train_reader, test_set=test_reader, train_epoch_num=10, Optim=torch.optim.Adam)
print(program.model.loss)  # last test loss will be print
```


### Test: `test()`

Takes a reader object as input, enumerates through all the samples or batches, and updates the statistics of loss and metric.
Can be invoked via `train()` or used independently.
Loss, metric, and a root `DataNode` will be returned.

```python
loss, metric, output = program.test(test_reader)
print(program.model.metric)
```

### Eval: `eval()`

Takes a reader object as input. Labels will be ignored. Loss and metric are also ignored. Just return the updated root `DataNode`.

## Programs

Different types of programs can be used to train the data. `SolverPOIProgram` is the simplest program that uses only the input and label of given data to learn its parameters.

However, Domiknows can leverage the defined constraints in the graph to teach the model not to violate those constraints and, in the process, improve the overall performance of the model.

### Inferece Mask Loss (IML) [[1]](#1)

To use IML, the program must be initialized by `IMLProgram`. IMLProgram extends `LearningBasedProgram` and shares most of the same input arguments.

```python
program = IMLProgram(graph, poi=(image, ), inferTypes=['ILP', 'local/argmax'], loss=..., metric=...})
```

The only new parameter here is `inferTypes`, which is set to `['ILP', 'local/argmax']`. These parameters are optional and can be used to calculate the metric.

### Priaml Dual (PD) [[2]](#2)

Primal-Dual model can also be easily defined by:

```python
prgram = PrimalDualProgram( graph, Model=SolverModel, poi=(sentence, phrase, pair), inferTypes=['local/argmax'],loss=..., metric=...)
            
```

Here, instead of `POImodel`, we use `SolverModel`.

### GBI [[3]](#3)

#### Overview
Gradient-based inference (GBI) [[3]](#3) performs gradient steps to optimize over some black-box function. In DomiKnowS, the function being optimized is the number of constraint violations.

For each example, GBI will optimize the following loss:
```python
log_probs * (num_constraint_violations / total_constraints) + reg_strength * regularization_loss
```

Here, `log_probs` refers to the mean log probabilities over all the outputs of the model, and `regularization_loss` refers to the L2 distance between the original (frozen) parameters and the new parameters. `reg_strength` is a hyperparameter that controls how much GBI should stray from the original parameters.

GBIModel is defined in `domiknows.model.gbi`. Several hyperparameters can be specified at initialization:

* **gbi_iters**: The maximum number of gradient update steps to perform. GBI will exit early if all constraints are specified.
* **lr**: The step size of each update step.
* **reg_weight**: The regularization strength, as described previously.
* **reset_params**: If set to `True`, the parameters of the model will be reset to the original (non-optimized) parameters after GBI is complete. If set to `False`, the parameters will *only* be reset if the loss becomes `NaN` or the constraints aren't satisfied after `gbi_iters` updates. **During training, this should be set to `False`.**

#### GBI for training
Normally, GBI is used only during inference. Here, GBI is adapted for training by simply omitting the final parameter reset at the end of certain inference steps. *Because of this, the workflow for using GBI during training will be similar to the workflow for using GBI for inference-only.*

#### Usage
In DomiKnowS, GBI can be used in a similar way to ILP. For example, instead of calling `inferILPResults`, the user can invoke `inferGBIResults`. The user can also add `GBI` to the `inferTypes` parameter. **Importantly, to use GBI for training, the user should specify `reset_params=False`.** For example:

```python
program = SolverPOIProgram(
    graph,
    poi=(image_batch, image, image_pair),
    inferTypes=['local/argmax', 'local/softmax', 'GBI'],
    reset_params=False
)
```

When populating the program, ensure that `grad=True` is set, as GBI requires gradients to perform updates. For example:
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
    reset_params=False
)
```

The training loop can then be specified as:
```python
for dataitem in dataloader:
    node = program.populate_one(dataitem, grad=True)
    node.inferLocal()
```

## References


<a id="1">[1] https://hlr.github.io/publications/Inference-Masked-Loss </a> 
Inference-Masked Loss for Deep Structured Output Learning, Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI 2020).


<a id="2">[2] https://papers.nips.cc/paper/2019/file/cf708fc1decf0337aded484f8f4519ae-Paper.pdf </a> 
A Primal-Dual Formulation for Deep Learning with Constraints, 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

<a id="3">[3] https://ojs.aaai.org/index.php/AAAI/article/view/4316 </a> 
"Gradient-Based Inference for Networks with Output Constraints" Jay Yoon Lee et. al
