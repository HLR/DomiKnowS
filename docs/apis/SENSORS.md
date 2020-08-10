# `regr.sensor`

## Base classes

### `regr.sensor.Sensor`

The base class of all sensors and learners. All sensors are designed to be callable objects.
When a sensor is called (with an data item retrieved from the data reader), dependency will be checked and prerequired sensors (or learners) will be called recursively.
The functionality of the sensor is implemented in its `forward()` function. When all dependency are evaluated, `forward()` function of the current sensor will be invoke.

#### `Sensor` Attributes

Inheriting attributes from `regr.graph.base.BaseGraphTreeNode`.

- `name`: a short name of the sensor. If not specified when creation, it will be the lowercase classname followed by a hyphen and an auto-increment index.
- `fullname`: a full path from the root of the graph to a property of a concepy and the `name` of this sensor.
- `sup`: the parent node of this sensor, which is a concepy's property to which this sensor is associated.

#### `Sensor` Methods

##### `forward(data_item: Dict[str, Any]): Any`

The abstract function for implementing the functionality of the current sensor.

- Parameters:
  - `data_item: Dict[str, Any]`: A data item retrieved from the data reader. It is a python `dict` that contain string keys and any (mostly `torch.Tensor`s) value.

- Return value:
  - any value that is generated based on the functionality of the sensor. If `None` is returned, the framework consider there is no return from the sensor, which is differently treated internally.

### `regr.sensor.Learner`

The base class of all the learners.
The only difference is that `Learner`s has learnable parameters that will be updated during training.

#### `Learner` Attributes

Inheriting attributes from `regr.sensor.Sensor` except an additional `paramters`.

Parameters:

- `paramters`: attribute or `@property` function that indicate the learnable paramters in this learner.

#### `Learner` Methods

Inheriting from `regr.sensor.Sensor`

## `regr.sensor.pytorch`

This package contains the sensors implemented specific to work with pytorch.

### `regr.sensor.pytorch.TorchSensor`

#### `TorchSensor` Attributes

#### `TorchSensor` Methods

### `regr.sensor.pytorch.TorchLearner`

#### `TorchLearner` Attributes

#### `TorchLearner` Methods

### `regr.sensor.pytorch.FunctionalSensor`

#### `FunctionalSensor` Attributes

#### `FunctionalSensor` Methods

### `regr.sensor.pytorch.ModuleSensor`

#### `ModuleSensor` Attributes

#### `ModuleSensor` Methods

### `regr.sensor.pytorch.ModuleLearner`

#### `ModuleLearner` Attributes

#### `ModuleLearner` Methods

### `regr.sensor.pytorch.TorchEdgeSensor`

#### `TorchEdgeSensor` Attributes

#### `TorchEdgeSensor` Methods

### `regr.sensor.pytorch.QuerySensor`

#### `QuerySensor` Attributes

#### `QuerySensor` Methods

### `regr.sensor.pytorch.DataNodeSensor`

#### `DataNodeSensor` Attributes

#### `DataNodeSensor` Methods

### `regr.sensor.pytorch.CandidateSensor`

#### `CandidateSensor` Attributes

#### `CandidateSensor` Methods

## `regr.sensor.pytorch.sensors`

### `regr.sensor.pytorch.ConstantSensor`

#### `ConstantSensor` Attributes

#### `ConstantSensor` Methods

### `regr.sensor.pytorch.ReaderSensor`

#### `ReaderSensor` Attributes

#### `ReaderSensor` Methods

### `regr.sensor.pytorch.CandidateReaderSensor`

#### `CandidateReaderSensor` Attributes

#### `CandidateReaderSensor` Methods

### `regr.sensor.pytorch.TorchEdgeReaderSensor`

#### `TorchEdgeReaderSensor` Attributes

#### `TorchEdgeReaderSensor` Methods

## `regr.sensor.pytorch.learners`

## Deprecated

Notice that, sensors and learners in `regr.sensor.torch` are **deprecated**. Use `regr.sensor.pytorch` instead.
