# `regr.sensor`

<!-- TOC depthto:2 withlinks:true -->

- [1. Base Classes `regr.sensor`](#1-base-classes-regrsensor)
- [2. PyTorch Sensors `regr.sensor.pytorch`](#2-pytorch-sensors-regrsensorpytorch)
- [3. Torch Sensors `regr.sensor.torch` (*Deprecated*)](#3-torch-sensors-regrsensortorch-deprecated)
- [4. AllenNLP Sensors `regr.sensor.allennlp` (*Deprecated*)](#4-allennlp-sensors-regrsensorallennlp-deprecated)

<!-- /TOC -->

## 1. Base Classes `regr.sensor`

### 1.1. `regr.sensor.Sensor`

The base class of all sensors and learners. All sensors are designed to be callable objects.
When a sensor is called (with an data item retrieved from the data reader), dependency will be checked and prerequired sensors (or learners) will be called recursively.
The functionality of the sensor is implemented in its `forward()` function. When all dependency are evaluated, `forward()` function of the current sensor will be invoke.

#### 1.1.1. `Sensor` Attributes

Inheriting attributes from `regr.graph.base.BaseGraphTreeNode`.

- `name`: a short name of the sensor. If not specified when creation, it will be the lowercase classname followed by a hyphen and an auto-increment index.
- `fullname`: a full path from the root of the graph to a property of a concepy and the `name` of this sensor.
- `sup`: the parent node of this sensor, which is a concepy's property to which this sensor is associated.

#### 1.1.2. `Sensor` Methods

##### 1.1.2.1. `__call__(self, data_item: Dict[str, Any], force=False) -> Any`

A sensor is a callable object. This is the basic interface the sensor used. All the sensors (and learners) are expected to be invoke with this function anywhere in this framework. Interally, it will trigger [`update_context()`](#1123-updatecontextself-dataitem-dictstr-any-forcefalse---dictstr-any) to update the `data_item` based on the return value of [`forward()`](#1122-forwarddataitem-dictstr-any-any) which implement the functionality of this spesific sensor.

- Parameters:
  - `data_item: Dict[str, Any]`: A data item retrieved from the data reader. It is a python `dict` that contain string keys and any (mostly `torch.Tensor`s) value.

- Return value:
  - any value that is generated based on the functionality of the sensor. If `None` is returned, the framework consider there is no return from the sensor, which is differently treated internally.

##### 1.1.2.2. `forward(data_item: Dict[str, Any]): Any`

The abstract function for implementing the functionality of the current sensor.

- Parameters:
  - `data_item: Dict[str, Any]`: A data item retrieved from the data reader, passed from the call to the sensor.

- Return value:
  - any value that is generated based on the functionality of the sensor. The return value will be cached in the `data_item` with the sensor's and its associations' idendities. If `None` is returned, the framework consider there is no return from the sensor and does not update the data item.

##### 1.1.2.3. `update_context(self, data_item: Dict[str, Any], force=False)`

Update the `data_item` based on the return value of [`forward()`](#1122-forwarddataitem-dictstr-any-any) if it has not been calculated before, with the sensor's idendity. Also update the sensor's associated `Property`, `Concept`, etc. via [`propagate_context()`](#1124-propagatecontextself-dataitem-node-forcefalse) if feasible.

- Parameters:
  - `data_item: Dict[str, Any]`: A data item retrieved from the data reader, passed from the call to the sensor.
  - `force`: Whether force recaluclating the value if the sensor's idendity is already in the `data_item`. Default: `False`.

##### 1.1.2.4. `propagate_context(self, data_item, node, force=False)`

Update the `node`'s parent node `node.sup` in the `data_item` with value cached for the `node`. The update propagate to `node.sup` and its parents automatically.

- Parameters:
  - `data_item: Dict[str, Any]`: A data item retrieved from the data reader, passed from the call to the sensor.
  - `node`: The node, which is a `Sensor`, `Property`, `Concept`, etc., based on which we propagate the value.
  - `force`: Whether force update if the parent's value is already assigned. Default: `False`.

### 1.2. `regr.sensor.Learner`

The base class of all the learners.
The only difference is that `Learner`s has learnable parameters that will be updated during training.

#### 1.2.1. `Learner` Attributes

Inheriting attributes from `regr.sensor.Sensor` except an additional `paramters`.

- `paramters`: The attribute that indicates the learnable paramters in this learner.

#### 1.2.2. `Learner` Methods

Inheriting from `regr.sensor.Sensor`

## 2. PyTorch Sensors `regr.sensor.pytorch`

This package contains the sensors implemented specific to work with PyTorch.

[Here](./sensor/PYTORCH.md) is a list of avalible sensors.

[Here](../developer/MODEL.md#sensor) for more information about customizing your own sensor.

## 3. Torch Sensors `regr.sensor.torch` (*Deprecated*)

Notice that, sensors and learners in `regr.sensor.torch` are *deprecated*.
Use [`regr.sensor.pytorch`](#2-pytorch-sensors-regrsensorpytorch) instead.

## 4. AllenNLP Sensors `regr.sensor.allennlp` (*Deprecated*)

Notice that, sensors and learners in `regr.sensor.allennlp` are dedicated to AllenNLP examples and thus *deprecated* for common use.
Use [`regr.sensor.pytorch`](#2-pytorch-sensors-regrsensorpytorch) instead.
