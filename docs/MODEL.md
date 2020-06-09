# Model Declaration

- [Model Declaration](#model-declaration)
  - [Class Overview](#class-overview)
  - [Reader](#reader)
  - [Sensor](#sensor)
    - [Initiate a Sensor](#initiate-a-sensor)
    - [Invoke a Sensor](#invoke-a-sensor)
    - [`forward()`](#forward)
      - [Example: `ReaderSensor`](#example-readersensor)
      - [Example: `ConcatSensor`](#example-concatsensor)
      - [Overriding `forward()`](#overriding-forward)
    - [Caching output](#caching-output)
    - [Managing invocation path](#managing-invocation-path)
    - [Converting input (WIP)](#converting-input-wip)
      - [Example: `TorchSensor.forward()`](#example-torchsensorforward)
      - [Example: `QuerySensor.forward(datanode)`](#example-querysensorforwarddatanode)
  - [Sensor Assignment to Property](#sensor-assignment-to-property)
  - [Learner](#learner)
  - [Multiple Assigment Convention](#multiple-assigment-convention)
  - [Detach](#detach)
    - [`detach` use cases](#detach-use-cases)

## Class Overview

- package `regr.sensor`:
- "Reader":
- `Sensor`:
- `Learner`:

## Reader

User will have to write their own reader to read source data.
To be flexible, the framework does not require user to implement a specific reader class.
Instead, the user will need to provide an `Iterable` object for the input of the program and yield an sample, or a batch of samples, in a `dict`.
The reader will be provided to the program for specific workflow like `train()`, `test()` or `eval()`.
Sensors will also be invoked with a `dict` retrieved from the reader each time (detailed later).

For example, a `list` of `dict` is a simplest input reader.
Also, `torch.utils.data.DataLoader` instance is a good choice when working with PyTorch.
The framework also has a simple reader for JSON format input file.

There is also a default Reader class implemented in the framework.
```
class RegrReader:
    def __init__(self, file, type="json"):
        self.file = file
        if type == "json":
            with open(file, 'r') as myfile:
                data = myfile.read()
            # parse file
            self.objects = json.loads(data)
        else:
            self.objects = self.parse_file()

    # you should return the objects list here
    def parse_file(self):
        pass

    def make_object(self, item):
        result = {}
        pattern = re.compile("^get.+val$")
        _list = [method_name for method_name in dir(self)
                 if callable(getattr(self, method_name)) and pattern.match(method_name)]
        for func in _list:
            name = func.replace("get", "", 1)
            k = name.rfind("val")
            name = name[:k]
            result[name] = getattr(self, func)(item)
        return result

    def run(self):
        for item in self.objects:
            yield self.make_object(item)
```
if you are loading a json file you do not have to write the `parse_file` function for your customized reader. Otherwise, you have to load a file and write a parser to parse it to a list of objects. 
To generate the outputs of each example you have to write functions in the format of `get$name$val`. each time the object is provided to your function and you should return the value of `$name` inside this function. At the end the output for each example will contain all the keys from the output of function `get$name$val` as `$name`. For getting one example of the reader, you have to call `run()` and it will yield one example at a time.
## Sensor

`Sensor`s are procedures to access external resources and procedures. For example, reading from raw data, feature engineering staffs, and preprocessing procedures.
"Sensors" are looked at as blackboxes in a program.
Users use sensors as [callable objects](#callable).
Underlying `forward()` function will be used to calculate an output. One can override `forward()` function to customize how the sensor get the output.
Base classes of `Sensor` will handle details like caching calcuated results, managing invocation path, and converting input.

### Initiate a Sensor

Sensors should always be initiated by a implmented class, rather the abstract `Sensor` or any base sensor.

```python
sensor = ImplementedSensor(...)
```

The framework contains some default sensors that users can use directly. Users can also customize their own by extending these sensors. The main body of the sensor should be written in a method called forward() inside Sensor class. 

### Invoke a Sensor

A `Sensor` is a callable object that can be invoked by passing a `dict` context variable or a `DataNodeBuilder`.
When working in a program, the item yield by the reader will be used as the context.
One can invoke it by

```python
context = {...}  # input data organized in a python dictionary
output = sensor(context)
```

Sensor will calcuate `output` based on its implementation of `forward()` function.
The `context` will be updated with a `sensor.fullname` key and `output` value pair as a caching mechanism.

```python
assert context[sensor.fullname] == output
```
The full name is constructed TODO.
The output is interpreted TODO.

### `forward()`

`forward()` defines the output calculation of the sensor.
Defaultly, it takes the same `context` as the sensor being invoke with.

For example, following are some basic sensors.

#### Example: `ReaderSensor`

`ReaderSensor` retieves the value with a key, defined when being initiated, from the `context`.

```python
sensor = ReaderSensor(key='raw_input')
context = {'raw_input': 'hello world'}

output = sensor(context)

assert output == 'hello world'
```

#### Example: `ConcatSensor`

`ConcatSensor` concatenates the input tensors (at the last dimension).

```python
sensor = ConcatSensor(feat_sensor1, feat_sensor2, key='raw_input')
context = {
  feat_sensor1.fullname: torch.rand(5, 4, 3),
  feat_sensor2.fullname: torch.rand(5, 4, 7)}

output = sensor(context)

assert output.shape == (5, 4, 10)
```

#### Overriding `forward()`

One can override `forward()` function to customize sensor. For example, one can create a random number generator:

```python
import torch

class RNGSensor(Sensor):
  def __init__(self, shape):
    self.shape=shape

  def forward(self, *args):
    return torch.rand(shape)

sensor = RNGSensor(shape=(5, 5))

output1 = sensor({})
output2 = sensor({})

assert output1.shape == (5, 5)
assert output1 != output2
```

However, `context` will cache output (as will be detailed later). If the identical `context` is passed, the output will be the same, unless force re-calculate is indicated by passing `force=True`.

```python
context = {}

output1 = sensor(context)
output2 = sensor(context)

assert output1 == output2

output3 = sensor(context, force=True)

assert output1 != output3
```

### Caching output

`context` is used as a source of input and a cache of outputs at the same time.
All the `output` generated by `sensor` will be updated to the `context` using `sensor.fullname` as a key and `output` as the value automatically.
If the key `sensor.fullname` exist, the associated value will be return without re-calculating, unless `force=True` is spesified when incoking.

```python
output = sensor(context)

assert context[sensor.fullname] == output

output = sensor(context, force=True)
```
### Influence of the sensor call on the 
The cache effect will be propagate to upper level object if there isn't a value yet.
For example, if the sensor is assigned to a property of a concept in a graph, the output of the sensor will be propagate to the property `sensor.sup` with key `sensor.sup.fullname`, further to the concept `sensor.sup.sup` with key `sensor.sup.sup`, etc.

```python
concept = Concept()
concept['prop'] = sensor
context = {}
output = sensor(context)

assert context[sensor.fullname] == output
assert context[concept['prop'].fullname] == output
assert context[concept.fullname] == output
```

### Managing invocation path

In model declaration, the user will need to assign sensors to properties.
However, that should not nessesearily reflect the order of calculation that is needed.
Sensors just look at `context` and try to fetch whatever they need as input.
\TODO{this next sentecne is not readable:} Managing one should be call before another is a headache.

Some base sensor extentions (e.g. `TorchSensor` and its subclasses) are able to trace what it need beforehand and invoke automatically. Such automation forms an invocation path.

`TorchSensor` initiates with extra variables `*pres` to specify the properties that are required to be calculated before it and be used as input.
Just pass the names of the properties that are required when constructing the new sensor.
For example, the following sensor

```python
class NoiseSensor(TorchSensor):
  def forward(self):
    input = self.inputs[0]
    return input + torch.rand(input.shape)

concept['clean'] = sensor
concept['noisy'] = NoiseSensor('clean')
```

### Converting input (WIP)

Some base sensor extensions are overrided to accept different input or accept inputs different ways.

#### Example: `TorchSensor.forward()`

`TorchSensor.forward()` does not accept input directly. Instead, it uses a member variable `inputs` to access its inputs. `self.inputs` is assigned with outputs of `self.pres`, as mentioned in [invocation path](#managing-invocation-path).
One can extend `TorchSensor` with any PyTorch model. For example:

```python
class LeakyReLUSensor(TorchSensor):
  def __init__(self, *pres, negative_slope=0.01, output=None, edges=None, label=False):
    super().__init__(*pres, output, edges, label):
    self.module = torch.nn.LeakyReLU(negative_slope=negative_slope)

  def forward(self,) -> Any:
    return self.module(self.inputs)
```

#### Example: `QuerySensor.forward(datanode)`

As `DataNode` provides a flexible interface to retrieve data, it is desirable to program with `DataNode`.


## Sensor Assignment to Property

Sensors can be assigned to properties

```python
reader = Reader()
sentence['raw'] = ReaderSensor(reader, 'sentence')
sentence.contains()[word] = SimpleTokenizorSensor('raw')
people['label'] = LabelReaderSensor(reader, 'people')
organization['label'] = LabelReaderSensor(reader, 'organization')
work_for['label'] = LabelReaderSensor(reader, 'work_for')
```

## Learner

`Learner`s are essentially `Sensor`s, except they have the member function `parameters()` that returns a list of parameters used by this learner.
In our `Program`, parameters of the whole model is collected by enumerating all the `Learner`s attached to the graph.
There are a few leaners implemented by using corresponding torch module. User can also override `TorchLearner` to use any torch module. For example:

```python
class MultiheadAttentionLearner(TorchLearner):
  def __init__(self, *pres, output=None, edges=None, **kwargs):
    super().__init__(*pres, output=None, edges=None)
    self.model = torch.nn.MultiheadAttention(**kwargs)

  def parameters(self):
    return self.model.parameters()

  def forward(self):
    query, key, value = self.inputs
    output = self.model(query, key, value)
    return output
```

`Learner`s can be assigned to properties the same way as sensors.

```python
sentence['embed'] = BertLearner('raw')
people['label'] = LogisticRegression('embed')
organization['label'] = LogisticRegression('embed')
work_for['label'] = LogisticRegression('embed')
```

## Multiple Assigment Convention

A non-trivial semantic we introduced for our language is we allow assigning multiple sensors and learners to one property, while rather than a mean of overriding the variable name, we introduce the multiple assignments as a *consistency constraint*.

For example, the following code assigns a `Learner` and then a `Sensor` to the `"label"` property the `people`, which is a `Concept`.

```python
people['label'] = Learner()
people['label'] = Sensor()
```

The implication is **NOT** that the property has been updated or overridden with the sensor. It means we can get the actual value of `"label"` property with either the `Learner` or the `Sensor`, which should be consistent. However, initially, the output of the `Learner` may not match the output of the `Sensor`. *Learning* of the model should take place here to resolve the inconsistency, i.e. a loss function should be constructed based on the error whenever there is multiple assignments to a property.

To be more specific, a program will look for all properties that has multiple assigmnets and generate all pairs of sensors (including learners). When the pair of sensors is constituted by a sensor with option `label=True` and another with option `label=False` (which is default). They will be used to trigger calculation and apply any loss and metric on.

It should be noticed that even if the two (or more) assignments are not `Learner`, the loss can is still considered. A sensor may rely on the output of another learner. The inconsistency error here can help to update those learners indirectly. If there is no underlying learner is still would not hurt because it is just a constant regarding any parameter in the model.

## Detach

In the graph - sub-graph - concept - property - sensor hierarchy, each component can trigger `detach` to remove its direct children components. For example,

```python
with Graph() as graph:
  concept = Concept()

graph.detach(concept) # remove concept from graph
```

It should be notice that `concept.sup` will also be unset.

If no spesific child passed, all leaf nodes are removed. For example, remove all sensors in a graph:

```python
graph.detach()
```

If no spesific child passed and `all=True`, all direct children will be remove. For example, remove all properties of a concept:

```python
concept.detach(all=True)
```

### `detach` use cases

Since multiple assignment would not override the existing sensor, if you want to remove a sensor, `.detach(sensor)` is what you need.

Graph can be reuse by different part of program. In case it will be used multiple time in one python program and you have a different set of sensors and learners to assign to the properties. You need to reset the graph by removing all the sensors and learners by `detach()`.

```python
graph.detach()
```
