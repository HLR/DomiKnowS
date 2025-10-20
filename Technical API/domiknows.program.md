# domiknows.program package

## Subpackages

* [domiknows.program.model package](domiknows.program.model.md)
  * [Submodules](domiknows.program.model.md#submodules)
  * [domiknows.program.model.base module](domiknows.program.model.md#module-domiknows.program.model.base)
    * [`Mode`](domiknows.program.model.md#domiknows.program.model.base.Mode)
      * [`Mode.POPULATE`](domiknows.program.model.md#domiknows.program.model.base.Mode.POPULATE)
      * [`Mode.TEST`](domiknows.program.model.md#domiknows.program.model.base.Mode.TEST)
      * [`Mode.TRAIN`](domiknows.program.model.md#domiknows.program.model.base.Mode.TRAIN)
  * [domiknows.program.model.gbi module](domiknows.program.model.md#module-domiknows.program.model.gbi)
    * [`GBIModel`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel)
      * [`GBIModel.calculateGBISelection()`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel.calculateGBISelection)
      * [`GBIModel.forward()`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel.forward)
      * [`GBIModel.get_argmax_from_node()`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel.get_argmax_from_node)
      * [`GBIModel.get_constraints_satisfaction()`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel.get_constraints_satisfaction)
      * [`GBIModel.reg_loss()`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel.reg_loss)
      * [`GBIModel.reset()`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel.reset)
      * [`GBIModel.set_pretrained()`](domiknows.program.model.md#domiknows.program.model.gbi.GBIModel.set_pretrained)
  * [domiknows.program.model.ilpu module](domiknows.program.model.md#module-domiknows.program.model.ilpu)
    * [`ILPUModel`](domiknows.program.model.md#domiknows.program.model.ilpu.ILPUModel)
      * [`ILPUModel.poi_loss()`](domiknows.program.model.md#domiknows.program.model.ilpu.ILPUModel.poi_loss)
  * [domiknows.program.model.iml module](domiknows.program.model.md#module-domiknows.program.model.iml)
    * [`IMLModel`](domiknows.program.model.md#domiknows.program.model.iml.IMLModel)
      * [`IMLModel.poi_loss()`](domiknows.program.model.md#domiknows.program.model.iml.IMLModel.poi_loss)
  * [domiknows.program.model.lossModel module](domiknows.program.model.md#module-domiknows.program.model.lossModel)
    * [`InferenceModel`](domiknows.program.model.md#domiknows.program.model.lossModel.InferenceModel)
      * [`InferenceModel.forward()`](domiknows.program.model.md#domiknows.program.model.lossModel.InferenceModel.forward)
      * [`InferenceModel.logger`](domiknows.program.model.md#domiknows.program.model.lossModel.InferenceModel.logger)
    * [`LossModel`](domiknows.program.model.md#domiknows.program.model.lossModel.LossModel)
      * [`LossModel.forward()`](domiknows.program.model.md#domiknows.program.model.lossModel.LossModel.forward)
      * [`LossModel.get_lmbd()`](domiknows.program.model.md#domiknows.program.model.lossModel.LossModel.get_lmbd)
      * [`LossModel.logger`](domiknows.program.model.md#domiknows.program.model.lossModel.LossModel.logger)
      * [`LossModel.reset()`](domiknows.program.model.md#domiknows.program.model.lossModel.LossModel.reset)
      * [`LossModel.reset_parameters()`](domiknows.program.model.md#domiknows.program.model.lossModel.LossModel.reset_parameters)
      * [`LossModel.to()`](domiknows.program.model.md#domiknows.program.model.lossModel.LossModel.to)
    * [`PrimalDualModel`](domiknows.program.model.md#domiknows.program.model.lossModel.PrimalDualModel)
      * [`PrimalDualModel.logger`](domiknows.program.model.md#domiknows.program.model.lossModel.PrimalDualModel.logger)
    * [`SampleLossModel`](domiknows.program.model.md#domiknows.program.model.lossModel.SampleLossModel)
      * [`SampleLossModel.forward()`](domiknows.program.model.md#domiknows.program.model.lossModel.SampleLossModel.forward)
      * [`SampleLossModel.get_lmbd()`](domiknows.program.model.md#domiknows.program.model.lossModel.SampleLossModel.get_lmbd)
      * [`SampleLossModel.logger`](domiknows.program.model.md#domiknows.program.model.lossModel.SampleLossModel.logger)
      * [`SampleLossModel.reset()`](domiknows.program.model.md#domiknows.program.model.lossModel.SampleLossModel.reset)
      * [`SampleLossModel.reset_parameters()`](domiknows.program.model.md#domiknows.program.model.lossModel.SampleLossModel.reset_parameters)
  * [domiknows.program.model.pytorch module](domiknows.program.model.md#module-domiknows.program.model.pytorch)
    * [`PoiModel`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModel)
      * [`PoiModel.default_poi()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModel.default_poi)
      * [`PoiModel.find_sensors()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModel.find_sensors)
      * [`PoiModel.poi_loss()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModel.poi_loss)
      * [`PoiModel.poi_metric()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModel.poi_metric)
      * [`PoiModel.populate()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModel.populate)
      * [`PoiModel.reset()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModel.reset)
    * [`PoiModelDictLoss`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelDictLoss)
      * [`PoiModelDictLoss.poi_loss()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelDictLoss.poi_loss)
      * [`PoiModelDictLoss.populate()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelDictLoss.populate)
      * [`PoiModelDictLoss.reset()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelDictLoss.reset)
    * [`PoiModelToWorkWithLearnerWithLoss`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss)
      * [`PoiModelToWorkWithLearnerWithLoss.default_poi()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss.default_poi)
      * [`PoiModelToWorkWithLearnerWithLoss.loss`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss.loss)
      * [`PoiModelToWorkWithLearnerWithLoss.metric`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss.metric)
      * [`PoiModelToWorkWithLearnerWithLoss.populate()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss.populate)
      * [`PoiModelToWorkWithLearnerWithLoss.reset()`](domiknows.program.model.md#domiknows.program.model.pytorch.PoiModelToWorkWithLearnerWithLoss.reset)
    * [`SolverModel`](domiknows.program.model.md#domiknows.program.model.pytorch.SolverModel)
      * [`SolverModel.inference()`](domiknows.program.model.md#domiknows.program.model.pytorch.SolverModel.inference)
      * [`SolverModel.populate()`](domiknows.program.model.md#domiknows.program.model.pytorch.SolverModel.populate)
    * [`SolverModelDictLoss`](domiknows.program.model.md#domiknows.program.model.pytorch.SolverModelDictLoss)
      * [`SolverModelDictLoss.inference()`](domiknows.program.model.md#domiknows.program.model.pytorch.SolverModelDictLoss.inference)
      * [`SolverModelDictLoss.populate()`](domiknows.program.model.md#domiknows.program.model.pytorch.SolverModelDictLoss.populate)
    * [`TorchModel`](domiknows.program.model.md#domiknows.program.model.pytorch.TorchModel)
      * [`TorchModel.data_hash()`](domiknows.program.model.md#domiknows.program.model.pytorch.TorchModel.data_hash)
      * [`TorchModel.forward()`](domiknows.program.model.md#domiknows.program.model.pytorch.TorchModel.forward)
      * [`TorchModel.mode()`](domiknows.program.model.md#domiknows.program.model.pytorch.TorchModel.mode)
      * [`TorchModel.move()`](domiknows.program.model.md#domiknows.program.model.pytorch.TorchModel.move)
      * [`TorchModel.populate()`](domiknows.program.model.md#domiknows.program.model.pytorch.TorchModel.populate)
      * [`TorchModel.reset()`](domiknows.program.model.md#domiknows.program.model.pytorch.TorchModel.reset)
    * [`model_helper()`](domiknows.program.model.md#domiknows.program.model.pytorch.model_helper)
  * [domiknows.program.model.torch module](domiknows.program.model.md#domiknows-program-model-torch-module)
  * [Module contents](domiknows.program.model.md#module-domiknows.program.model)

## Submodules

## domiknows.program.batchprogram module

### *class* domiknows.program.batchprogram.BatchProgram(\*args, batch_size=1, \*\*kwargs)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

#### train_epoch(dataset)

The function train_epoch trains a model on a dataset for one epoch, updating the model’s
parameters based on the calculated loss and performing gradient descent if an optimizer is
provided.

* **Parameters:**
  **dataset** – The dataset parameter is the training dataset that contains the input data and

corresponding labels. It is used to iterate over the data items during training

## domiknows.program.callbackprogram module

### *class* domiknows.program.callbackprogram.CallbackProgram(\*args, \*\*kwargs)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

#### default_after_train_step(output=None)

#### default_before_train_step()

#### test(\*args, \*\*kwargs)

The function test is used to test a model on a given dataset, with an optional device argument
for specifying the device to run the test on.

* **Parameters:**
  **dataset** – The dataset parameter is the dataset object that contains the testing data. It

is used to evaluate the performance of the model on the testing data
:param device: The “device” parameter is used to specify the device on which the model should be
tested. It can be set to “None” if you want to test the model on the CPU, or it can be set to a
specific device such as “cuda” if you want to test the model on

#### test_epoch(dataset)

The function test_epoch is used to evaluate a model on a dataset during the testing phase,
yielding the loss, metric, and output for each data item.

* **Parameters:**
  **dataset** – The dataset parameter is the input dataset that you want to test your model

on. It could be a list, generator, or any other iterable object that provides the data items to
be tested. Each data item should be in a format that can be processed by your model

#### train(\*args, \*\*kwargs)

The train function is used to train a model on a given training set, with optional validation
and testing sets, for a specified number of epochs.

* **Parameters:**
  **training_set** – The training set is the dataset used to train the model. It typically

consists of input data and corresponding target labels
:param valid_set: The valid_set parameter is used to specify the validation dataset. It is
typically a separate portion of the training dataset that is used to evaluate the model’s
performance during training and tune hyperparameters
:param test_set: The test_set parameter is used to specify the dataset that will be used for
testing the model’s performance after each epoch of training. It is typically a separate dataset
from the training and validation sets, and is used to evaluate the model’s generalization
ability on unseen data
:param device: The device on which the model will be trained (e.g., ‘cpu’ or ‘cuda’)
:param train_epoch_num: The number of epochs to train the model for. An epoch is a complete pass
through the entire training dataset, defaults to 1 (optional)
:param test_every_epoch: The parameter “test_every_epoch” is a boolean flag that determines
whether to perform testing after every epoch during training. If set to True, testing will be
performed after each epoch. If set to False, testing will only be performed once at the end of
training, defaults to False (optional)
:param Optim: The Optim parameter is used to specify the optimizer to be used for training the
model. It should be a class that implements the optimization algorithm, such as
torch.optim.SGD or torch.optim.Adam. The optimizer is responsible for updating the model’s
parameters based on the computed gradients

#### train_epoch(dataset)

The function train_epoch trains a model on a dataset for one epoch, updating the model’s
parameters based on the calculated loss and performing gradient descent if an optimizer is
provided.

* **Parameters:**
  **dataset** – The dataset parameter is the training dataset that contains the input data and

corresponding labels. It is used to iterate over the data items during training

#### train_pure_epoch(dataset)

### *class* domiknows.program.callbackprogram.ProgramStorageCallback(program, fn)

Bases: `object`

### domiknows.program.callbackprogram.hook(callbacks, \*args, \*\*kwargs)

## domiknows.program.loss module

### *class* domiknows.program.loss.BCEFocalLoss(weight=None, pos_weight=None, reduction='mean', alpha=1, gamma=2, with_logits=True)

Bases: [`BCEWithLogitsLoss`](#domiknows.program.loss.BCEWithLogitsLoss)

#### forward(input, target, weight=None)

Runs the forward pass.

### *class* domiknows.program.loss.BCEWithLogitsFocalLoss(weight=None, reduction='mean', alpha=0.5, gamma=2)

Bases: `Module`

#### forward(input, target, weight=None)

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### *class* domiknows.program.loss.BCEWithLogitsIMLoss(lmbd, reduction='mean')

Bases: `Module`

#### forward(input, inference, target, weight=None)

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### *class* domiknows.program.loss.BCEWithLogitsLoss(weight: Tensor | None = None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight: Tensor | None = None)

Bases: `BCEWithLogitsLoss`

#### forward(input, target, weight=None)

Runs the forward pass.

### *class* domiknows.program.loss.NBCrossEntropyDictLoss(weight: Tensor | None = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0)

Bases: `CrossEntropyLoss`

#### forward(builder, prop, input, target, \*args, \*\*kwargs)

Runs the forward pass.

### *class* domiknows.program.loss.NBCrossEntropyIMLoss(lmbd, reduction='mean')

Bases: [`BCEWithLogitsIMLoss`](#domiknows.program.loss.BCEWithLogitsIMLoss)

#### forward(input, inference, target, weight=None)

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### *class* domiknows.program.loss.NBCrossEntropyLoss(weight: Tensor | None = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0)

Bases: `CrossEntropyLoss`

#### forward(input, target, \*args, \*\*kwargs)

Runs the forward pass.

### *class* domiknows.program.loss.NBMSELoss(size_average=None, reduce=None, reduction: str = 'mean')

Bases: `MSELoss`

#### forward(input, target, \*args, \*\*kwargs)

Runs the forward pass.

## domiknows.program.lossprogram module

### *class* domiknows.program.lossprogram.GBIProgram(graph, Model, poi, beta=1, \*\*kwargs)

Bases: [`LossProgram`](#domiknows.program.lossprogram.LossProgram)

#### logger *= <Logger domiknows.program.lossprogram (WARNING)>*

### *class* domiknows.program.lossprogram.InferenceProgram(graph, Model, beta=1, \*\*kwargs)

Bases: [`LossProgram`](#domiknows.program.lossprogram.LossProgram)

#### evaluate_condition(evaluate_data, device='cpu')

#### logger *= <Logger domiknows.program.lossprogram (WARNING)>*

### *class* domiknows.program.lossprogram.LossProgram(graph, Model, CModel=None, beta=1, \*\*kwargs)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

#### DEFAULTCMODEL

alias of [`PrimalDualModel`](domiknows.program.model.md#domiknows.program.model.lossModel.PrimalDualModel)

#### call_epoch(name, dataset, epoch_fn, \*\*kwargs)

The function call_epoch logs information about the loss and metrics of a model during an epoch
and updates a database if specified.

* **Parameters:**
  * **name** – The name of the epoch or task being performed. It is used for logging purposes
  * **dataset** – The dataset parameter is the input dataset that will be used for training or

evaluation. It is typically a collection of data samples that the model will process
:param epoch_fn: The epoch_fn parameter is a function that represents a single epoch of
training or evaluation. It takes the dataset as input and performs the necessary operations
for that epoch, such as forward and backward passes, updating model parameters, and calculating
metrics

#### logger *= <Logger domiknows.program.lossprogram (WARNING)>*

#### populate_epoch(dataset, grad=False)

The populate_epoch function is used to iterate over a dataset and yield the output of a model
for each data item, either with or without gradient calculations.

* **Parameters:**
  **dataset** – The dataset parameter is the input data that you want to use to populate the

model. It could be a list, array, or any other iterable object that contains the data items
:param grad: The grad parameter is a boolean flag that determines whether or not to compute
gradients during the epoch. If grad is set to False, the epoch will be executed in
evaluation mode without computing gradients. If grad is set to True, the epoch will be
executed in training, defaults to False (optional)

#### test_epoch(dataset, \*\*kwargs)

The function test_epoch is used to evaluate a model on a dataset during the testing phase,
yielding the loss, metric, and output for each data item.

* **Parameters:**
  **dataset** – The dataset parameter is the input dataset that you want to test your model

on. It could be a list, generator, or any other iterable object that provides the data items to
be tested. Each data item should be in a format that can be processed by your model

#### to(device)

#### train(training_set, valid_set=None, test_set=None, c_lr=0.05, c_momentum=0.9, c_warmup_iters=10, c_freq=10, c_freq_increase=5, c_freq_increase_freq=1, c_lr_decay=4, c_lr_decay_param=1, batch_size=1, dataset_size=None, print_loss=True, \*\*kwargs)

The train function is used to train a model on a given training set, with optional validation
and testing sets, for a specified number of epochs.

* **Parameters:**
  **training_set** – The training set is the dataset used to train the model. It typically

consists of input data and corresponding target labels
:param valid_set: The valid_set parameter is used to specify the validation dataset. It is
typically a separate portion of the training dataset that is used to evaluate the model’s
performance during training and tune hyperparameters
:param test_set: The test_set parameter is used to specify the dataset that will be used for
testing the model’s performance after each epoch of training. It is typically a separate dataset
from the training and validation sets, and is used to evaluate the model’s generalization
ability on unseen data
:param device: The device on which the model will be trained (e.g., ‘cpu’ or ‘cuda’)
:param train_epoch_num: The number of epochs to train the model for. An epoch is a complete pass
through the entire training dataset, defaults to 1 (optional)
:param test_every_epoch: The parameter “test_every_epoch” is a boolean flag that determines
whether to perform testing after every epoch during training. If set to True, testing will be
performed after each epoch. If set to False, testing will only be performed once at the end of
training, defaults to False (optional)
:param Optim: The Optim parameter is used to specify the optimizer to be used for training the
model. It should be a class that implements the optimization algorithm, such as
torch.optim.SGD or torch.optim.Adam. The optimizer is responsible for updating the model’s
parameters based on the computed gradients

#### train_epoch(dataset, c_lr=1, c_warmup_iters=10, c_freq_increase=1, c_freq_increase_freq=1, c_lr_decay=0, c_lr_decay_param=1, c_session={}, batch_size=1, dataset_size=None, print_loss=True, \*\*kwargs)

The function train_epoch trains a model on a dataset for one epoch, updating the model’s
parameters based on the calculated loss and performing gradient descent if an optimizer is
provided.

* **Parameters:**
  **dataset** – The dataset parameter is the training dataset that contains the input data and

corresponding labels. It is used to iterate over the data items during training

### *class* domiknows.program.lossprogram.PrimalDualProgram(graph, Model, beta=1, \*\*kwargs)

Bases: [`LossProgram`](#domiknows.program.lossprogram.LossProgram)

#### logger *= <Logger domiknows.program.lossprogram (WARNING)>*

### *class* domiknows.program.lossprogram.SampleLossProgram(graph, Model, beta=1, \*\*kwargs)

Bases: [`LossProgram`](#domiknows.program.lossprogram.LossProgram)

#### logger *= <Logger domiknows.program.lossprogram (WARNING)>*

#### train(training_set, valid_set=None, test_set=None, c_lr=0.05, c_momentum=0.9, c_warmup_iters=10, c_freq=10, c_freq_increase=5, c_freq_increase_freq=1, c_lr_decay=4, c_lr_decay_param=1, \*\*kwargs)

The train function is used to train a model on a given training set, with optional validation
and testing sets, for a specified number of epochs.

* **Parameters:**
  **training_set** – The training set is the dataset used to train the model. It typically

consists of input data and corresponding target labels
:param valid_set: The valid_set parameter is used to specify the validation dataset. It is
typically a separate portion of the training dataset that is used to evaluate the model’s
performance during training and tune hyperparameters
:param test_set: The test_set parameter is used to specify the dataset that will be used for
testing the model’s performance after each epoch of training. It is typically a separate dataset
from the training and validation sets, and is used to evaluate the model’s generalization
ability on unseen data
:param device: The device on which the model will be trained (e.g., ‘cpu’ or ‘cuda’)
:param train_epoch_num: The number of epochs to train the model for. An epoch is a complete pass
through the entire training dataset, defaults to 1 (optional)
:param test_every_epoch: The parameter “test_every_epoch” is a boolean flag that determines
whether to perform testing after every epoch during training. If set to True, testing will be
performed after each epoch. If set to False, testing will only be performed once at the end of
training, defaults to False (optional)
:param Optim: The Optim parameter is used to specify the optimizer to be used for training the
model. It should be a class that implements the optimization algorithm, such as
torch.optim.SGD or torch.optim.Adam. The optimizer is responsible for updating the model’s
parameters based on the computed gradients

#### train_epoch(dataset, c_warmup_iters=0, c_session={}, \*\*kwargs)

The function train_epoch trains a model on a dataset for one epoch, updating the model’s
parameters based on the calculated loss and performing gradient descent if an optimizer is
provided.

* **Parameters:**
  **dataset** – The dataset parameter is the training dataset that contains the input data and

corresponding labels. It is used to iterate over the data items during training

### domiknows.program.lossprogram.backward(loss, parameters)

### domiknows.program.lossprogram.reverse_sign_grad(parameters, factor=-1.0)

### domiknows.program.lossprogram.unset_backward(parameters)

## domiknows.program.metric module

### *class* domiknows.program.metric.CMWithLogitsMetric(\*args, \*\*kwargs)

Bases: `Module`

A utility class for computing confusion matrix metrics from logits.

Inherits from:
: torch.nn.Module

#### forward(input, target, \_, prop, weight=None)

Computes True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) values
from given logits and target.

Args:
: input (torch.Tensor): The logits tensor.
  target (torch.Tensor): The ground truth labels tensor.
  \_ : Placeholder, not used.
  prop: Placeholder, not used.
  weight (torch.Tensor, optional): Weights to apply to the input. Defaults to tensor of value 1.

Returns:
: dict: A dictionary containing TP, FP, TN, and FN values.

### *class* domiknows.program.metric.DatanodeCMMetric(inferType='ILP')

Bases: `Module`

A utility class for computing confusion matrix metrics using datanode inference results.

Inherits from:
: torch.nn.Module

Attributes:
: inferType (str): The type of inference used to derive metrics.

#### forward(input, target, data_item, prop, weight=None)

Computes the confusion matrix metrics using data from the provided datanode.

Args:
: input (torch.Tensor): The input tensor.
  target (torch.Tensor): The ground truth labels.
  data_item: The datanode containing the inference metrics.
  prop: The property associated with the inference.
  weight (torch.Tensor, optional): An optional weight tensor. Defaults to None.

Returns:
: dict/None: A dictionary containing the TP, FP, TN, FN values, or 
  : information on class names, labels, and predictions; 
    returns None if the property name is not found in the results.

### *class* domiknows.program.metric.MacroAverageTracker(metric)

Bases: [`MetricTracker`](#domiknows.program.metric.MetricTracker)

A utility class that extends the MetricTracker to compute macro-average of metrics for datanodes.

Inherits from:
: MetricTracker

#### forward(values)

Computes the macro-average for the given values.

Args:
: values (Any): The input values (can be single value, list, tensor, or dictionary of values).

Returns:
: Any: The macro-averaged value. The structure (tensor, list, or dictionary) of the output
  : mirrors the structure of the input.

### *class* domiknows.program.metric.MetricTracker(metric)

Bases: `Module`

A utility class for tracking metrics for all datanodes.

Attributes:
: metric (callable): The metric function to track.
  list (list): A list of metric values for all the datanodes.
  dict (defaultdict): A dictionary of metric values grouped by keys.

#### kprint(k)

Custom key printing function based on the type and properties of the key.

Args:
: k: The key to be printed.

Returns:
: str: A string representation of the key.

#### reset()

Resets the internal storage (both list and dict) to their initial empty state.

#### value(reset=False)

Retrieves the value(s) of the computed metric(s).

Args:
: reset (bool, optional): If True, resets the internal storage after retrieving the value. Defaults to False.

Returns:
: Any: The metric value(s), either as a single value, list, or dictionary.

### *class* domiknows.program.metric.PRF1Tracker(metric=CMWithLogitsMetric(), confusion_matrix=False)

Bases: [`MetricTracker`](#domiknows.program.metric.MetricTracker)

A tracker to calculate and monitor precision, recall, F1 score, and accuracy metrics.

Inherits from the MetricTracker class.

Methods:
- forward: Processes input values to compute various metrics like precision, recall, F1 score, and accuracy.

#### confusion_matrix

Initialize the PRF1Tracker instance.

Parameters:
- metric (Metric, optional): An instance of the metric to be tracked. Defaults to CMWithLogitsMetric().
- confusion_matrix (bool, optional): Whether to create confusion matrix values or not. Defaults to False.

#### forward(values)

Processes the input values and computes precision, recall, F1 score, and accuracy metrics.

Parameters:
- values: Input data containing raw class names and predictions.

Returns:
- dict: A dictionary containing calculated metrics.

If the input contains class names it means it is for a multiclass concept:
: Returns a classification report with metrics for each class and overall metrics 
  like ‘weighted avg’, ‘macro avg’, and ‘accuracy’ after negative classes are removed.

Else:

> Returns metrics: ‘P’ (Precision), ‘R’ (Recall), ‘F1’ (F1 Score), and ‘accuracy’ for the bincaryclass concept.

## domiknows.program.model_program module

### *class* domiknows.program.model_program.IMLProgram(graph, \*\*kwargs)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

### *class* domiknows.program.model_program.POILossProgram(graph, poi=None)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

### *class* domiknows.program.model_program.POIProgram(graph, \*\*kwargs)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

### *class* domiknows.program.model_program.SolverPOIDictLossProgram(graph, \*\*kwargs)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

### *class* domiknows.program.model_program.SolverPOIProgram(graph, \*\*kwargs)

Bases: [`LearningBasedProgram`](#domiknows.program.program.LearningBasedProgram)

## domiknows.program.program module

### *class* domiknows.program.program.LearningBasedProgram(graph, Model, logger=None, \*\*kwargs)

Bases: `object`

#### calculateMetricDelta(metric1, metric2)

The function calculates the difference between two metrics and returns the delta.

* **Parameters:**
  **metric1** – The first metric, represented as a dictionary. Each key in the dictionary

represents a category, and the corresponding value is another dictionary where the keys
represent subcategories and the values represent the metric values
:param metric2: The metric2 parameter is a dictionary representing a metric. It has a nested
structure where the keys represent categories and the values represent subcategories and their
corresponding values
:return: a dictionary called metricDelta.

#### call_epoch(name, dataset, epoch_fn, \*\*kwargs)

The function call_epoch logs information about the loss and metrics of a model during an epoch
and updates a database if specified.

* **Parameters:**
  * **name** – The name of the epoch or task being performed. It is used for logging purposes
  * **dataset** – The dataset parameter is the input dataset that will be used for training or

evaluation. It is typically a collection of data samples that the model will process
:param epoch_fn: The epoch_fn parameter is a function that represents a single epoch of
training or evaluation. It takes the dataset as input and performs the necessary operations
for that epoch, such as forward and backward passes, updating model parameters, and calculating
metrics

#### load(path, \*\*kwargs)

The function loads a saved model state dictionary from a specified path.

* **Parameters:**
  **path** – The path parameter is the file path to the saved model state dictionary

#### populate(dataset, device=None)

#### populate_epoch(dataset, grad=False)

The populate_epoch function is used to iterate over a dataset and yield the output of a model
for each data item, either with or without gradient calculations.

* **Parameters:**
  **dataset** – The dataset parameter is the input data that you want to use to populate the

model. It could be a list, array, or any other iterable object that contains the data items
:param grad: The grad parameter is a boolean flag that determines whether or not to compute
gradients during the epoch. If grad is set to False, the epoch will be executed in
evaluation mode without computing gradients. If grad is set to True, the epoch will be
executed in training, defaults to False (optional)

#### populate_one(data_item, grad=False, device=None)

#### save(path, \*\*kwargs)

The function saves the state dictionary of a model to a specified path using the torch.save()
function.

* **Parameters:**
  **path** – The path where the model’s state dictionary will be saved

#### test(dataset, device=None, \*\*kwargs)

The function test is used to test a model on a given dataset, with an optional device argument
for specifying the device to run the test on.

* **Parameters:**
  **dataset** – The dataset parameter is the dataset object that contains the testing data. It

is used to evaluate the performance of the model on the testing data
:param device: The “device” parameter is used to specify the device on which the model should be
tested. It can be set to “None” if you want to test the model on the CPU, or it can be set to a
specific device such as “cuda” if you want to test the model on

#### test_epoch(dataset, \*\*kwargs)

The function test_epoch is used to evaluate a model on a dataset during the testing phase,
yielding the loss, metric, and output for each data item.

* **Parameters:**
  **dataset** – The dataset parameter is the input dataset that you want to test your model

on. It could be a list, generator, or any other iterable object that provides the data items to
be tested. Each data item should be in a format that can be processed by your model

#### to(device='auto')

#### train(training_set, valid_set=None, test_set=None, device=None, train_epoch_num=1, test_every_epoch=False, Optim=None, \*\*kwargs)

The train function is used to train a model on a given training set, with optional validation
and testing sets, for a specified number of epochs.

* **Parameters:**
  **training_set** – The training set is the dataset used to train the model. It typically

consists of input data and corresponding target labels
:param valid_set: The valid_set parameter is used to specify the validation dataset. It is
typically a separate portion of the training dataset that is used to evaluate the model’s
performance during training and tune hyperparameters
:param test_set: The test_set parameter is used to specify the dataset that will be used for
testing the model’s performance after each epoch of training. It is typically a separate dataset
from the training and validation sets, and is used to evaluate the model’s generalization
ability on unseen data
:param device: The device on which the model will be trained (e.g., ‘cpu’ or ‘cuda’)
:param train_epoch_num: The number of epochs to train the model for. An epoch is a complete pass
through the entire training dataset, defaults to 1 (optional)
:param test_every_epoch: The parameter “test_every_epoch” is a boolean flag that determines
whether to perform testing after every epoch during training. If set to True, testing will be
performed after each epoch. If set to False, testing will only be performed once at the end of
training, defaults to False (optional)
:param Optim: The Optim parameter is used to specify the optimizer to be used for training the
model. It should be a class that implements the optimization algorithm, such as
torch.optim.SGD or torch.optim.Adam. The optimizer is responsible for updating the model’s
parameters based on the computed gradients

#### train_epoch(dataset, \*\*kwargs)

The function train_epoch trains a model on a dataset for one epoch, updating the model’s
parameters based on the calculated loss and performing gradient descent if an optimizer is
provided.

* **Parameters:**
  **dataset** – The dataset parameter is the training dataset that contains the input data and

corresponding labels. It is used to iterate over the data items during training

#### verifyResultsLC(data, constraint_names=None, device=None)

The function verifyResultsLC calculates and prints the accuracy of constraint verification
results for a given dataset.

* **Parameters:**
  **data** – The data parameter is the input data that will be used to populate the datanode.

It is passed to the populate method of the current object (self) along with an optional
device parameter
:param constraint_names: The constraint_names parameter is a list of constraint names that you
want to verify the results for. If this parameter is not provided or is set to None, then the
function will verify the results for all constraints available in the verifyResult dictionary
:param device: The device parameter is used to specify the device on which the calculations
should be performed. It is an optional parameter and if not provided, the default device will be
used
:return: None.

### *class* domiknows.program.program.dbUpdate(graph)

Bases: `object`

#### getTimeStamp()

### domiknows.program.program.get_len(dataset, default=None)

## domiknows.program.tracker module

### *class* domiknows.program.tracker.ConfusionMatrixTracker

Bases: [`Tracker`](#domiknows.program.tracker.Tracker)

#### reduce(values)

### *class* domiknows.program.tracker.MacroAverageTracker

Bases: [`Tracker`](#domiknows.program.tracker.Tracker)

#### reduce(values)

### *class* domiknows.program.tracker.Tracker

Bases: `dict`

#### append(dict_)

#### printer *= <domiknows.program.tracker.TrackerPrinter object>*

#### reduce()

#### reset()

#### value(reset=False)

### *class* domiknows.program.tracker.TrackerPrinter(indent=1, width=80, depth=None, stream=None, , compact=False, sort_dicts=True, underscore_numbers=False)

Bases: `PrettyPrinter`

## Module contents
