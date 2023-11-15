# domiknows.program.model package

## Submodules

## domiknows.program.model.base module

### *class* domiknows.program.model.base.Mode(value)

Bases: `Enum`

An enumeration.

#### POPULATE *= 3*

#### TEST *= 2*

#### TRAIN *= 1*

## domiknows.program.model.gbi module

### *class* domiknows.program.model.gbi.GBIModel(graph, solver_model=None, gbi_iters=100, device='auto')

Bases: `Module`

#### calculateGBISelection(datanode, conceptsRelations)

#### find_last_layers_in_submodels(model, name='')

#### forward(datanode, build=None, print_grads=False)

Defines the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

#### get_constraints_satisfaction(node)

Get constraint satisfaction from datanode
Returns number of satisfied constraints and total number of constraints

#### reg_loss(model_updated, model, exclude_names={})

#### reset()

#### reset_last_layers_in_submodels(model, last_layers)

#### training*: bool*

## domiknows.program.model.ilpu module

### *class* domiknows.program.model.ilpu.ILPUModel(graph, poi=None, loss=None, metric=None, inferTypes=None, inference_with=None, probKey=('local', 'softmax'), device='auto', probAcc=None, ignore_modules=False)

Bases: [`SolverModel`](#domiknows.program.model.pytorch.SolverModel)

#### poi_loss(data_item, prop, sensors)

The function calculates the loss for a given data item using a set of sensors.

* **Parameters:**
  **data_item** – The data_item parameter represents a single data item that is being

processed. It could be any type of data, depending on the context of your code
:param \_: The underscore “_” is a convention in Python to indicate that a variable is not going
to be used in the code. It is often used as a placeholder for a variable that needs to be
present for the function signature but is not actually used within the function. In this case,
it seems that the variable
:param sensors: The sensors parameter is a list of sensor functions. These sensor functions
take a data_item as input and return some output
:return: the calculated local_loss.

#### training*: bool*

## domiknows.program.model.iml module

### *class* domiknows.program.model.iml.IMLModel(graph, poi=None, loss=None, metric=None, inferTypes=None, inference_with=None, probKey=('local', 'softmax'), device='auto', probAcc=None, ignore_modules=False)

Bases: [`SolverModel`](#domiknows.program.model.pytorch.SolverModel)

#### poi_loss(data_item, prop, sensors)

The function calculates the loss for a given data item using a set of sensors.

* **Parameters:**
  **data_item** – The data_item parameter represents a single data item that is being

processed. It could be any type of data, depending on the context of your code
:param \_: The underscore “_” is a convention in Python to indicate that a variable is not going
to be used in the code. It is often used as a placeholder for a variable that needs to be
present for the function signature but is not actually used within the function. In this case,
it seems that the variable
:param sensors: The sensors parameter is a list of sensor functions. These sensor functions
take a data_item as input and return some output
:return: the calculated local_loss.

#### training*: bool*

## domiknows.program.model.lossModel module

### *class* domiknows.program.model.lossModel.LossModel(graph, tnorm='P', sample=False, sampleSize=0, sampleGlobalLoss=False, device='auto')

Bases: `Module`

#### forward(builder, build=None)

Defines the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

#### get_lmbd(key)

The function get_lmbd returns a clamped value from a dictionary based on a given key.

* **Parameters:**
  **key** – The key parameter is used to access a specific value in the lmbd dictionary
* **Returns:**
  the value of self.lmbd[self.lmbd_index[key]] after clamping it to a maximum value of

self.lmbd_p[self.lmbd_index[key]].

#### logger *= <Logger domiknows.program.model.lossModel (WARNING)>*

#### reset()

#### reset_parameters()

#### to(device)

Moves and/or casts the parameters and buffers.

This can be called as

#### to(device=None, dtype=None, non_blocking=False)

#### to(dtype, non_blocking=False)

#### to(tensor, non_blocking=False)

#### to(memory_format=torch.channels_last)

Its signature is similar to `torch.Tensor.to()`, but only accepts
floating point or complex `dtype`s. In addition, this method will
only cast the floating point or complex parameters and buffers to `dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

#### NOTE
This method modifies the module in-place.

Args:
: device (`torch.device`): the desired device of the parameters
  : and buffers in this module
  <br/>
  dtype (`torch.dtype`): the desired floating point or complex dtype of
  : the parameters and buffers in this module
  <br/>
  tensor (torch.Tensor): Tensor whose dtype and device are the desired
  : dtype and device for all parameters and buffers in this module
  <br/>
  memory_format (`torch.memory_format`): the desired memory
  : format for 4D parameters and buffers in this module (keyword
    only argument)

Returns:
: Module: self

Examples:

```default
>>> # xdoctest: +IGNORE_WANT("non-deterministic")
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
        [-0.5113, -0.2325]], dtype=torch.float64)
>>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
        [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j,  0.2382+0.j],
        [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j],
        [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```

#### training*: bool*

### *class* domiknows.program.model.lossModel.PrimalDualModel(graph, tnorm='P', device='auto')

Bases: [`LossModel`](#domiknows.program.model.lossModel.LossModel)

#### logger *= <Logger domiknows.program.model.lossModel (WARNING)>*

#### training*: bool*

### *class* domiknows.program.model.lossModel.SampleLossModel(graph, tnorm='P', sample=False, sampleSize=0, sampleGlobalLoss=False, device='auto')

Bases: `Module`

#### forward(builder, build=None)

The forward function calculates the loss for a PrimalDualModel using a DataNodeBuilder and
returns the loss value, the DataNode, and the builder.

* **Parameters:**
  **builder** – The builder parameter is an instance of the DataNodeBuilder class. It is

used to create a batch root data node and retrieve a data node
:param build: The build parameter is an optional argument that specifies whether the
DataNodeBuilder should be invoked or not. If build is None, then the value of self.build
is used. If build is True, then the createBatchRootDN() method
:return: three values: lmbd_loss, datanode, and builder.

#### get_lmbd(key)

The function get_lmbd returns the value of self.lmbd at the index specified by
self.lmbd_index[key], ensuring that the value is non-negative.

* **Parameters:**
  **key** – The key parameter is used to access a specific element in the self.lmbd list. It

is used as an index to retrieve the corresponding value from the list
:return: the value of self.lmbd[self.lmbd_index[key]].

#### logger *= <Logger domiknows.program.model.lossModel (WARNING)>*

#### reset()

#### reset_parameters()

#### training*: bool*

## domiknows.program.model.pytorch module

### *class* domiknows.program.model.pytorch.PoiModel(graph, poi=None, loss=None, metric=None, device='auto', ignore_modules=False)

Bases: [`TorchModel`](#domiknows.program.model.pytorch.TorchModel)

#### default_poi()

The function default_poi returns a list of properties from a graph that have more than one
instance of the TorchSensor class.
:return: a list of properties that have more than one instance of the TorchSensor class in the
graph.

#### find_sensors(prop)

The function find_sensors finds pairs of sensors in a given property that have one sensor
labeled as the target and the other sensor as the output.

* **Parameters:**
  **prop** – The parameter “prop” is expected to be an object that has a method called “find”

which takes a class name as an argument and returns a list of objects of that class. In this
case, it is being used to find objects of the class “TorchSensor”

#### poi_loss(data_item, \_, sensors)

The function calculates the loss for a given data item using a set of sensors.

* **Parameters:**
  **data_item** – The data_item parameter represents a single data item that is being

processed. It could be any type of data, depending on the context of your code
:param \_: The underscore “_” is a convention in Python to indicate that a variable is not going
to be used in the code. It is often used as a placeholder for a variable that needs to be
present for the function signature but is not actually used within the function. In this case,
it seems that the variable
:param sensors: The sensors parameter is a list of sensor functions. These sensor functions
take a data_item as input and return some output
:return: the calculated local_loss.

#### poi_metric(data_item, prop, sensors, datanode=None)

The poi_metric function calculates a local metric based on the given data item, property,
sensors, and optional datanode.

* **Parameters:**
  **data_item** – The data_item parameter is a single data item that is being processed. It

could be any type of data, depending on the context of your code
:param prop: The “prop” parameter is a property or attribute of the data item that is being
evaluated
:param sensors: The sensors parameter is a list of sensor functions. These sensor functions
take a data_item as input and return a value. The sensors list contains multiple sensor
functions that will be called to collect data for the metric calculation
:param datanode: The datanode parameter is an optional argument that represents the data node
for which the metric is being calculated. It is used as an input to the metric function to
provide additional context or information for the calculation
:return: the local_metric value.

#### populate(builder, datanode=None, run=True)

The populate function evaluates sensors, calculates loss and metrics, and returns the total
loss and metric values.

* **Parameters:**
  **builder** – The builder parameter is an object that is used to construct and populate data

nodes in a data structure. It is likely an instance of a class that provides methods for
creating and manipulating data nodes
:param datanode: The datanode parameter is an optional argument that represents a data node in
the builder. It is used to store the metric values calculated during the population process. If
datanode is not provided, a new batch root data node is created and assigned to datanode
:param run: The run parameter is a boolean flag that determines whether the sensors should be
evaluated or not. If run is True, the sensors will be evaluated by calling their \_\_call_\_
method. If run is False, the sensors will not be evaluated, defaults to True (optional)
:return: two values: loss and metric.

#### reset()

#### training*: bool*

### *class* domiknows.program.model.pytorch.PoiModelDictLoss(graph, poi=None, loss=None, metric=None, dictloss=None, device='auto')

Bases: [`PoiModel`](#domiknows.program.model.pytorch.PoiModel)

#### poi_loss(data_item, prop, sensors)

The function calculates the loss for a given data item using a set of sensors.

* **Parameters:**
  **data_item** – The data_item parameter represents a single data item that is being

processed. It could be any type of data, depending on the context of your code
:param \_: The underscore “_” is a convention in Python to indicate that a variable is not going
to be used in the code. It is often used as a placeholder for a variable that needs to be
present for the function signature but is not actually used within the function. In this case,
it seems that the variable
:param sensors: The sensors parameter is a list of sensor functions. These sensor functions
take a data_item as input and return some output
:return: the calculated local_loss.

#### populate(builder, run=True)

The populate function evaluates sensors, calculates loss and metrics, and returns the total
loss and metric values.

* **Parameters:**
  **builder** – The builder parameter is an object that is used to construct and populate data

nodes in a data structure. It is likely an instance of a class that provides methods for
creating and manipulating data nodes
:param datanode: The datanode parameter is an optional argument that represents a data node in
the builder. It is used to store the metric values calculated during the population process. If
datanode is not provided, a new batch root data node is created and assigned to datanode
:param run: The run parameter is a boolean flag that determines whether the sensors should be
evaluated or not. If run is True, the sensors will be evaluated by calling their \_\_call_\_
method. If run is False, the sensors will not be evaluated, defaults to True (optional)
:return: two values: loss and metric.

#### reset()

#### training*: bool*

### *class* domiknows.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss(graph, poi=None, device='auto')

Bases: [`TorchModel`](#domiknows.program.model.pytorch.TorchModel)

#### default_poi()

#### *property* loss

#### *property* metric

#### populate(builder)

#### reset()

#### training*: bool*

### *class* domiknows.program.model.pytorch.SolverModel(graph, poi=None, loss=None, metric=None, inferTypes=None, inference_with=None, probKey=('local', 'softmax'), device='auto', probAcc=None, ignore_modules=False)

Bases: [`PoiModel`](#domiknows.program.model.pytorch.PoiModel)

#### inference(builder)

The inference function takes a builder object, iterates over a list of properties, and performs
inference using different types of models based on the inferTypes list.

* **Parameters:**
  **builder** – The builder parameter is an object that is used to construct a computation

graph. It is typically used to define the inputs, operations, and outputs of a neural network
model
:return: the datanode object.

#### populate(builder, run=True)

The populate function takes a builder object, performs inference on it, and then calls the
populate method of the superclass with the resulting datanode, returning the datanode,
lose, and metric values.

* **Parameters:**
  **builder** – The “builder” parameter is an object that is used to build or construct the data

node. It is likely an instance of a class that has methods for creating and manipulating data
nodes
:param run: The “run” parameter is a boolean flag that determines whether to run the population
process immediately after populating the data node. If set to True, the population process will
be executed; if set to False, the population process will be skipped, defaults to True
(optional)
:return: three values: datanode, lose, and metric.

#### training*: bool*

### *class* domiknows.program.model.pytorch.SolverModelDictLoss(graph, poi=None, loss=None, metric=None, dictloss=None, inferTypes=['ILP'], device='auto')

Bases: [`PoiModelDictLoss`](#domiknows.program.model.pytorch.PoiModelDictLoss)

#### inference(builder)

#### populate(builder, run=True)

The populate function evaluates sensors, calculates loss and metrics, and returns the total
loss and metric values.

* **Parameters:**
  **builder** – The builder parameter is an object that is used to construct and populate data

nodes in a data structure. It is likely an instance of a class that provides methods for
creating and manipulating data nodes
:param datanode: The datanode parameter is an optional argument that represents a data node in
the builder. It is used to store the metric values calculated during the population process. If
datanode is not provided, a new batch root data node is created and assigned to datanode
:param run: The run parameter is a boolean flag that determines whether the sensors should be
evaluated or not. If run is True, the sensors will be evaluated by calling their \_\_call_\_
method. If run is False, the sensors will not be evaluated, defaults to True (optional)
:return: two values: loss and metric.

#### training*: bool*

### *class* domiknows.program.model.pytorch.TorchModel(graph, device='auto', ignore_modules=False)

Bases: `Module`

#### data_hash(data_item)

#### forward(data_item, build=None)

The forward function takes a data item, performs various operations on it based on the graph
structure, and returns the output.

* **Parameters:**
  **data_item** – The data_item parameter is the input data that will be passed through the

forward pass of the neural network. It can be either a dictionary or an iterable object. If it
is an iterable, it will be converted into a dictionary where the keys are the indices of the
elements in the iterable
:param build: The build parameter is a boolean flag that indicates whether the method should
build a data node or not. If build is True, the method will build a data node and return the
loss, metric, data node, and builder. If build is False, the method
:return: If the build parameter is True, the function returns a tuple containing (loss,
metric, datanode, builder). 
If the build parameter is False, the function returns a tuple containing the values returned
by the populate method.

#### mode(mode=None)

The mode function sets the mode of the object and performs certain actions based on the mode.

* **Parameters:**
  **mode** – The mode parameter is used to specify the mode of operation for the code. It can

take one of the following values:
:return: The method is returning the value of the \_mode attribute.

#### move(value, device=None)

#### populate()

#### reset()

#### training*: bool*

### domiknows.program.model.pytorch.model_helper(Model, \*args, \*\*kwargs)

## domiknows.program.model.torch module

## Module contents
