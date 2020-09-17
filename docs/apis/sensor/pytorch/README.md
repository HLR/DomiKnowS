# [↑ `regr.sensor`](..)

<!-- TOC depthto:3 withlinks:true -->

- [1. `regr.sensor.pytorch`](#1-regrsensorpytorch)
    - [1.1. `regr.sensor.pytorch.TorchSensor`](#11-regrsensorpytorchtorchsensor)
    - [1.2. `regr.sensor.pytorch.TorchLearner`](#12-regrsensorpytorchtorchlearner)
    - [1.3. `regr.sensor.pytorch.FunctionalSensor`](#13-regrsensorpytorchfunctionalsensor)
    - [1.4. `regr.sensor.pytorch.ModuleSensor`](#14-regrsensorpytorchmodulesensor)
    - [1.5. `regr.sensor.pytorch.ModuleLearner`](#15-regrsensorpytorchmodulelearner)
    - [1.6. `regr.sensor.pytorch.TorchEdgeSensor`](#16-regrsensorpytorchtorchedgesensor)
    - [1.7. `regr.sensor.pytorch.QuerySensor`](#17-regrsensorpytorchquerysensor)
    - [1.8. `regr.sensor.pytorch.DataNodeSensor`](#18-regrsensorpytorchdatanodesensor)
    - [1.9. `regr.sensor.pytorch.CandidateSensor`](#19-regrsensorpytorchcandidatesensor)
- [2. `regr.sensor.pytorch.sensors`](#2-regrsensorpytorchsensors)
    - [2.1. `regr.sensor.pytorch.ConstantSensor`](#21-regrsensorpytorchconstantsensor)
    - [2.2. `regr.sensor.pytorch.ReaderSensor`](#22-regrsensorpytorchreadersensor)
    - [2.3. `regr.sensor.pytorch.CandidateReaderSensor`](#23-regrsensorpytorchcandidatereadersensor)
    - [2.4. `regr.sensor.pytorch.TorchEdgeReaderSensor`](#24-regrsensorpytorchtorchedgereadersensor)
- [3. `regr.sensor.pytorch.learners`](#3-regrsensorpytorchlearners)

<!-- /TOC -->

## 1. `regr.sensor.pytorch`

This package contains the sensors implemented specific to work with pytorch.

### 1.1. `regr.sensor.pytorch.TorchSensor`

Inheriting from `regr.sensor.Sensor`. The base class of all sensors and learners designed to support interacting with PyTorch.

#### 1.1.1. `TorchSensor` Attributes

Inheriting attributes from `regr.sensor.Sensor`. There are the following additional attributes for `TorchSensor`.

- `pres`: List of pre-required property names (`str`) of the associated `Concept`, which will be calculated automatically before invoking current sensor's `forward()` and used to fill the `inputs`.
- `context_helper`: The sensor-wise reference to the current processing `data_item` when it is called. *Do not count on this in case of multiprocessing scenario*.
- `inputs`: The inputs that is needed for current `forword()` call, calculated and collected from `pres`.
- `edges`: List of the pre-required edges' forward or backward property, which will be calculated automatically before invoking current sensor's `forward()`.
- `label`: Whether this sensor is a label in the data, which should not be used in forward calculation of the model.
- `device`: An indicator of device to be used for this sensor. It can be an indicators to instantiate a `torch.device` (`str`, `int`) or a `torch.device` instance.
- `prop`: The `Property` that the current sensor is associated to. Raise `ValueError` if it is not associated to any `Property`.
- `concept`: The `Concept` that the current sensor is associated to. Raise `ValueError` if it is not associated to any `Property` or `Concept`.

#### 1.1.2. `TorchSensor` Methods

##### 1.1.2.1. `__init__(self, *pres, edges=None, label=False, device='auto')`

Instantiate the sensor with attribute specified. Determine device if needed.

- Parameters:
  - `*pres`: All the positional arguments are treated as the `pres` attribute of the sensor.
  - `edges`: The `edges` attribute of the sensor. Default: `None`.
  - `label`: The `label` attribute of the sensor. Default: `False`.
  - `device`: The `device` attribute of the sensor. If `'auto'` is given, try to use the first available CUDA device or fall back to CPU automatically. Default: `'auto'`.

##### 1.1.2.2. `__call__(self, data_item: Dict[str, Any]) -> Any`

Override [`Sensor.__call__()`](../README.md#1121-callself-dataitem-dictstr-any-forcefalse---any) to initiate `context_helper` with current `data_item` before invoking `update_context()`.

##### 1.1.2.3. `update_context(self, data_item: Dict[str, Any], force=False)`

Override [`Sensor.update_context()`](../README.md#1123-updatecontextself-dataitem-dictstr-any-forcefalse) to update `pres` before current sensor via [`update_pre_context()`](#1124-updateprecontextself-dataitem-dictstr-any) and limit the propagation to only `Property` level.

##### 1.1.2.4. `update_pre_context(self, data_item: Dict[str, Any])`

This method collect `pres` and `edges` that are required for the current sensor, execute them with the current `data_item`, and avoid sensors with `label=True`.

- Parameters:
  - `data_item: Dict[str, Any]`: A data item retrieved from the data reader, passed from the call to the sensor.

##### 1.1.2.5. `fetch_value(self, pre, selector=None)`

This method retrieves the value of a pre-required property that is indicated by `pre` from the `context_helper`. There could be multiple sensors assigning to one property. User can pass a `selector` function to filter the sensor. The value of the first sensor will be return.

- Parameters:
  - `selector`: A function that returns `True` for any expected sensor and `False` otherwise. Default: `None`.

##### 1.1.2.6. `define_inputs(self)`

This method instantiate the `inputs` attribute of the sensor based on current `context_helper`, which refers to current `data_item`.

### 1.2. `regr.sensor.pytorch.TorchLearner`

#### 1.2.1. `TorchLearner` Attributes

#### 1.2.2. `TorchLearner` Methods

### 1.3. `regr.sensor.pytorch.FunctionalSensor`

#### 1.3.1. `FunctionalSensor` Attributes

#### 1.3.2. `FunctionalSensor` Methods

### 1.4. `regr.sensor.pytorch.ModuleSensor`

#### 1.4.1. `ModuleSensor` Attributes

#### 1.4.2. `ModuleSensor` Methods

### 1.5. `regr.sensor.pytorch.ModuleLearner`

#### 1.5.1. `ModuleLearner` Attributes

#### 1.5.2. `ModuleLearner` Methods

### 1.6. `regr.sensor.pytorch.TorchEdgeSensor`
Inheriting from `regr.sensor.pytorch.FunctionalSensor`.
This class is the base sensor for all edge sensors.
#### 1.6.1. `TorchEdgeSensor` Attributes
Inheriting attributes from `regr.sensor.pytorch.FunctionalSensor`. There are the following additional attributes for `TorchEdgeSensor`.

- `mode`: this attribute can take the values of `forward` or `backward` and helps detecting the source and the destination concept to specify where the inputs should be looked up from and where the output should be stored. 
- `to`: this specifies the target property that this edge is supposed to store the results in.
- `src`: this stores the source concept assigned to this edge based on the `mode` parameter.
- `dst`: this stores the target concept assigned to this edge based on the `mode` parameter.
#### 1.6.2. `TorchEdgeSensor` Methods
##### 1.6.2.1. `__attached__(self, sup)`

This function assigns proper values to `src` and `dst` variables and adds a dummy sensor to the `dst[to]` property to enable triggering in the model execution.

##### 1.6.2.2. `update_context(self, data_item: Dict[str, Any], force=False)`

Override [`FnctionalSensor.update_context()`]() to change the storing property to match `dst[to]`.

##### 1.6.2.3. `update_pre_context(self, data_item: Dict[str, Any], force=False)`

Override [`FnctionalSensor.update_pre_context()`]() to change the check on required properties on the source concept with `src[*pres]`.

##### 1.6.2.3. `fetch_value(self, pre, selector=None)`

Override [`FnctionalSensor.fetch_value()`]() to change the fetching to occur from the properties stored on the source concept `src[pre]`.

### 1.7. `regr.sensor.pytorch.QuerySensor`

#### 1.7.1. `QuerySensor` Attributes

#### 1.7.2. `QuerySensor` Methods

### 1.8. `regr.sensor.pytorch.DataNodeSensor`

#### 1.8.1. `DataNodeSensor` Attributes

#### 1.8.2. `DataNodeSensor` Methods

### 1.9. `regr.sensor.pytorch.CandidateSensor`

#### 1.9.1. `CandidateSensor` Attributes

#### 1.9.2. `CandidateSensor` Methods

## 2. `regr.sensor.pytorch.sensors`

### 2.1. `regr.sensor.pytorch.ConstantSensor`

#### 2.1.1. `ConstantSensor` Attributes

#### 2.1.2. `ConstantSensor` Methods

### 2.2. `regr.sensor.pytorch.ReaderSensor`
Inheriting from `regr.sensor.pytorch.ConstantSensor`.
This class is the base sensor used for Reading from the dictionary structured input.
#### 2.2.1. `ReaderSensor` Attributes
Inheriting attributes from `regr.sensor.pytorch.ConstantSensor`. There are the following additional attributes for `ReaderSensor`.

- `keyword`: This variable defines the `keyword` that the reader is supposed to read from the dictionary. This should be accessible in from the root node and can take form of a single string or tuples of string.
- `data`: this variable is filled with one of the internal functions called during execution which holds the current instance data with respect to the keywords of the input.

#### 2.2.2. `ReaderSensor` Methods
##### 2.2.2.1. `fill_data(self, data_item)`

This function is used to fill the `data` variable of the class by fetching relevant `keyword` from the `data_item`
- Parameters:
  - `data_item`: The data instance passed from the Reader Class to the model execution

##### 2.2.2.2. `forward(self, _*)`
Override [`ConstantSensor.forward()`]() to return `data` variable using the parent `forward` function.

### 2.3. `regr.sensor.pytorch.CandidateReaderSensor`

#### 2.3.1. `CandidateReaderSensor` Attributes

#### 2.3.2. `CandidateReaderSensor` Methods

### 2.4. `regr.sensor.pytorch.TorchEdgeReaderSensor`

#### 2.4.1. `TorchEdgeReaderSensor` Attributes

#### 2.4.2. `TorchEdgeReaderSensor` Methods

## 3. `regr.sensor.pytorch.learners`
