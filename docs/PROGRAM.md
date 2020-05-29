# Program

- [Program](#program)
  - [Class Overview](#class-overview)
  - [Workflows](#workflows)
    - [Initiate](#initiate)
    - [Train: `train()`](#train-train)
    - [Test: `test()`](#test-test)
    - [Eval: `eval()`](#eval-eval)

## Class Overview

- package `regr.program`:
- `LearningBasedProgram`:

## Workflows

There are built-in workflows that are commonly used in machine learning tasks. Also, user can extend the `LearningBasedProgram` class to build their own workflows.

### Initiate

A basic program can be initiate with a graph. For example:

```python
program = LearningBasedProgram(graph)
```

Additional arguments may be required for other programs.

### Train: `train()`

`train()` takes a required argument `training_set` as input. Additionally, user can provide `valid_set` and/or `test_set`. There three arguments are reader objects, which are essentially `Iterable`s. 
There are other optional arguments in `train()`.

`train()` will invoke `train_epoch()` for `train_epoch_num` times and invoke `test()` on `valid_set` if provided, for each time. And after taining is done, `test()` will be invoke with `test_set` if provided.

`Optim` specifies a optimizor class to initiate an optimizor with and be used to train the model.

Loss and metric will be store in `program.model.loss` and `program.model.mettic` for access.

Example:

```python
program.train(train_reader, test_set=test_reader, train_epoch_num=10, Optim=torch.optim.Adam)
print(program.model.loss)  # last test loss will be print
```


### Test: `test()`

Takes a reader object as input, enumerate through all the samples or batches, and update the statics of loss and metric.
Can be invoke via `train()` or used independently.
Loss, metric, and a root `DataNode` will be return.

```python
loss, metric, output = program.test(test_reader)
print(program.model.metric)
```

### Eval: `eval()`

Takes a reader object as input. Labels will be ignored. Loss and metric are also ignored. Just return the updated root `DataNode`.
