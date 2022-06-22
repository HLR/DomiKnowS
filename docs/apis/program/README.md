# `regr.program`

<!-- TOC depthto:3 withlinks:true -->

- [1. Base Program](#1-base-program)
    - [1.1. `regr.program.LearningBasedProgram`](#11-regrprogramlearningbasedprogram)
- [2. `Model`](#2-model)
    - [2.1. `regr.program.model.pytorch.TorchModel`](#21-regrprogrammodelpytorchtorchmodel)
    - [2.2. `regr.program.model.pytorch.PoiModel`](#22-regrprogrammodelpytorchpoimodel)
    - [2.3. `regr.program.model.pytorch.SolverModel`](#23-regrprogrammodelpytorchsolvermodel)
    - [2.4. `regr.program.model.pytorch.IMLModel`](#24-regrprogrammodelpytorchimlmodel)
    - [2.5. `regr.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss` (*experimental*)](#25-regrprogrammodelpytorchpoimodeltoworkwithlearnerwithloss-experimental)
- [3. Implemented Programs](#3-implemented-programs)
    - [3.1. `regr.program.POIProgram`](#31-regrprogrampoiprogram)
    - [3.2. `regr.program.IMLProgram`](#32-regrprogramimlprogram)
    - [3.3. `regr.program.POILossProgram`](#33-regrprogrampoilossprogram)

<!-- /TOC -->

## 1. Base Program

### 1.1. `regr.program.LearningBasedProgram`

`LearningBasedProgram` is a model transfered form a graph (with sensors and learners) and machine learning pipelines.

#### 1.1.1. `Program` Attributes

- `graph`: A `Graph` the program initiated from.
- `model`: A `Model` that carries out caluclation.
- `logger`: A logger.
- `opt`: An optimior that is used for current session.

#### 1.1.2. `Program` Methods

##### 1.1.2.1. `__init__(self, graph, Model, **kwargs)`

##### 1.1.2.2. `update_nominals(self, dataset)`

##### 1.1.2.3. `to(self, device='auto')`

##### 1.1.2.4. `train(self, training_set, valid_set=None, test_set=None, device=None, train_epoch_num=1, Optim=None)`

##### 1.1.2.5. `train_epoch(self, dataset)`

##### 1.1.2.6. `test(self, dataset, device=None)`

##### 1.1.2.7. `test_epoch(self, dataset, device=None)`

##### 1.1.2.8. `populate(self, dataset, device=None)`

##### 1.1.2.9. `populate_one(self, data_item, device=None)`

##### 1.1.2.10. `populate_epoch(self, dataset, device=None)`

## 2. `Model`

`Model`s are the core part of the program that carry out the calculation based on the knowledge declaration and model declaration. They are `torch.nn.Module` and they are directly connected to `torch`.

### 2.1. `regr.program.model.pytorch.TorchModel`

Inheriting from `torch.nn.Module`. This is an abstract class.

#### 2.1.1. `TorchModel` Attributes

Inheriting attributes from `torch.nn.Module`.

- `graph`: The graph where this model is derived.

#### 2.1.2. `TorchModel` Methods

##### 2.1.2.1. `__init__(self, graph)`

##### 2.1.2.2. `mode(self, mode=None)`

##### 2.1.2.3. `reset(self)`

##### 2.1.2.4. `move(self, value, device=None)`

##### 2.1.2.5. `forward(self, data_item)`

##### 2.1.2.6. `populate(self)`

### 2.2. `regr.program.model.pytorch.PoiModel`

Inheriting from `regr.program.model.pytorch.TorchModel`. 

#### 2.2.1. `PoiModel` Attributes

#### 2.2.2. `PoiModel` Methods

### 2.3. `regr.program.model.pytorch.SolverModel`

#### 2.3.1. `SolverModel` Attributes

#### 2.3.2. `SolverModel` Methods

### 2.4. `regr.program.model.pytorch.IMLModel`

#### 2.4.1. `IMLModel` Attributes

#### 2.4.2. `IMLModel` Methods


### 2.5. `regr.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss` (*experimental*)

#### 2.5.1. `PoiModelToWorkWithLearnerWithLoss` Attributes

#### 2.5.2. `PoiModelToWorkWithLearnerWithLoss` Methods

## 3. Implemented Programs

### 3.1. `regr.program.POIProgram`

### 3.2. `regr.program.IMLProgram`

### 3.3. `regr.program.POILossProgram`
