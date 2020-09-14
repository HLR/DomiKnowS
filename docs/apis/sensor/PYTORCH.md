# [↑ `regr.sensor`](../SENSORS.md)

## `regr.sensor.pytorch`

This package contains the sensors implemented specific to work with pytorch.

### `regr.sensor.pytorch.TorchSensor`

#### `TorchSensor` Attributes

- `pres`
- `context_helper`
- `inputs`
- `edges`
- `label`
- `device`

#### `TorchSensor` Methods

##### `__init__(self, *pres, edges=None, label=False, device='auto')`

##### `__call__(self, data_item: Dict[str, Any]) -> Dict[str, Any]`

##### `update_context(self, data_item: Dict[str, Any], force=False) -> Dict[str, Any]`

##### `update_pre_context(self, data_item: Dict[str, Any]) -> Any`

#####

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
