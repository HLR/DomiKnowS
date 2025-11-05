# domiknows.sensor.pytorch package

## Subpackages

* [domiknows.sensor.pytorch.tokenizers package](domiknows.sensor.pytorch.tokenizers.md)
  * [Submodules](domiknows.sensor.pytorch.tokenizers.md#submodules)
  * [domiknows.sensor.pytorch.tokenizers.spacy module](domiknows.sensor.pytorch.tokenizers.md#module-domiknows.sensor.pytorch.tokenizers.spacy)
    * [`SpacyTokenizer`](domiknows.sensor.pytorch.tokenizers.md#domiknows.sensor.pytorch.tokenizers.spacy.SpacyTokenizer)
      * [`SpacyTokenizer.forward()`](domiknows.sensor.pytorch.tokenizers.md#domiknows.sensor.pytorch.tokenizers.spacy.SpacyTokenizer.forward)
  * [domiknows.sensor.pytorch.tokenizers.transformers module](domiknows.sensor.pytorch.tokenizers.md#module-domiknows.sensor.pytorch.tokenizers.transformers)
    * [`TokenizerEdgeSensor`](domiknows.sensor.pytorch.tokenizers.md#domiknows.sensor.pytorch.tokenizers.transformers.TokenizerEdgeSensor)
      * [`TokenizerEdgeSensor.forward()`](domiknows.sensor.pytorch.tokenizers.md#domiknows.sensor.pytorch.tokenizers.transformers.TokenizerEdgeSensor.forward)
  * [Module contents](domiknows.sensor.pytorch.tokenizers.md#module-domiknows.sensor.pytorch.tokenizers)

## Submodules

## domiknows.sensor.pytorch.aggregation_sensor module

## domiknows.sensor.pytorch.learnerModels module

### *class* domiknows.sensor.pytorch.learnerModels.LSTMModel(input_dim, hidden_dim, bidirectional=False, num_layers=1, batch_size=1)

Bases: `Module`

Torch module for an LSTM.

#### forward(input)

Runs the LSTM over a sequence.

Args:
- input: Input of shape (seq_len, batch_size, input_dim).

Returns:
- Output features from the last layer of the LSTM for each timestep.

> Has shape (seq_len, batch_size, hidden_dim \* num_directions).

#### init_hidden()

Creates the zero-initialized hidden state and cell state.

Returns:
- Tuple of tensors corresponding to the initial hidden state and cell state.

> Each tensor has shape (num_layers \* 2, batch_size, hidden_dim) filled
> with zeros.

### *class* domiknows.sensor.pytorch.learnerModels.PyTorchFC(input_dim, output_dim)

Bases: `Module`

Torch module for a fully-connected layer over sequence inputs with softmax activations.

#### forward(x)

Compute class probabilities from a linear layer for the last timestep in
the sequence input.

Args:
- x: Sequence batch of shape (batch_size, seq_len, input_dim)

> The last timestep: x[:, -1, :] is used as input.

Returns:
- Softmax probabilities of shape (batch_size, output_dim).

### *class* domiknows.sensor.pytorch.learnerModels.PyTorchFCRelu(input_dim, output_dim)

Bases: `Module`

Torch module for a fully-connected layer with LeakyReLU activations.

#### forward(x)

Apply a linear layer followed by LeakyReLU.

Args:
- x: Input tensor whose last dimension equals input_dim

> Leading dimensions are preserved.

Returns:
- Activated features with the same leading dimensions as

> the input tensor and last dimension output_dim.

## domiknows.sensor.pytorch.learners module

### *class* domiknows.sensor.pytorch.learners.FullyConnectedLearner(\*pres, input_dim, output_dim, device='auto')

Bases: [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

A learner for a sequence input backed by a single fully-connected layer and a softmax.
Calculates the probabilities on the last time-step only.

Inherits from:
- TorchLearner: Provides learner and sensor functionality.

#### forward() → Any

Runs the linear layer and softmax on the first item of the stored inputs. Calculates the
probabilities for the last timestep of the sequence inputs only.

Expects self.inputs[0] to have shape (batch_size, seq_len, input_dim).

Returns:
- The linear layer and softmax output for the input.

> Has shape (batch_size, output_dim).

### *class* domiknows.sensor.pytorch.learners.FullyConnectedLearnerRelu(\*pres, input_dim, output_dim, device='auto')

Bases: [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

A learner backed by a single fully-connected layer with a leaky ReLU non-linearity.

Inherits from:
- TorchLearner: Provides learner and sensor functionality.

#### forward() → Any

Runs the linear layer and non-linearity on the first item of the stored inputs.

Returns:
- The linear layer and leaky ReLU output for the input.

### *class* domiknows.sensor.pytorch.learners.LSTMLearner(\*pres, input_dim, hidden_dim, num_layers=1, bidirectional=False, device='auto')

Bases: [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

A learner backed by an LSTM model.

Inherits from:
- TorchLearner: Provides learner and sensor functionality.

#### forward() → Any

Runs the LSTM on the first item of the stored inputs.

Returns:
- The LSTM output for the sequence input.

### *class* domiknows.sensor.pytorch.learners.ModuleLearner(\*pres, module, edges=None, loss=None, metric=None, label=False, \*\*kwargs)

Bases: [`ModuleSensor`](#domiknows.sensor.pytorch.sensors.ModuleSensor), [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

A learner that wraps around a torch module.

Inherits from:
- ModuleSensor: Injects a pre-built torch.nn.Module as self.model.
- TorchLearner: Learner behaviors (parameters, save/load, loss/metric).

#### update_parameters()

Because we provide the full module directly, we override update_parameters with a no-op.

### *class* domiknows.sensor.pytorch.learners.TorchLearner(\*pre, edges=None, loss=None, metric=None, label=False, device='auto')

Bases: [`Learner`](domiknows.sensor.md#domiknows.sensor.learner.Learner), [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

A PyTorch-based learner that behaves like a sensor (it updates/propagates
context and can be placed in the graph) but also owns trainable parameters
and optional loss/metric functions.

#### *property* device

The current device used to run torch operations on.

#### load(filepath)

Loads the underlying torch module given the folder name where the
module is saved.

If the file doesn’t exist, continues without loading.

Args:
- filepath: folder where module is loaded from

#### loss(data_item, target)

Computes the loss on a given data item instance according to the
configured function (if provided).

Args:
- data_item: The data item to perform inference on
- target: Sensor that provides the label

Returns:
- The loss value if a loss function was provided at initialization;

> otherwise returns None.

#### metric(data_item, target)

Computes the metric on a given data item instance according to the
configured function (if provided).

Args:
- data_item: The data item to perform inference on
- target: Sensor that provides the label

Returns:
- The metric value if a metric function was provided at initialization;

> otherwise returns None.

#### *property* model

Underlying torch module.

#### *abstract property* parameters *: Any*

Parameters of this learner (the underlying torch module).

Returns:
- An iterator (or collection) of parameters from self.model, if set.

> Returns None if self.model is None.

#### *property* sanitized_name

Sanitized identifier suitable for file-names.

Returns:
- Sanitized name

#### save(filepath)

Saves the underlying torch module state_dict to a folder using torch.save.

Args:
- filepath: folder where module is saved

#### update_parameters()

Attaches predecessor learners’ modules as submodules of this learner’s model. Will only
perform this action once, even if called multiple times.

## domiknows.sensor.pytorch.query_sensor module

### *class* domiknows.sensor.pytorch.query_sensor.DataNodeReaderSensor(\*pres, \*\*kwargs)

Bases: [`DataNodeSensor`](#domiknows.sensor.pytorch.query_sensor.DataNodeSensor), [`FunctionalReaderSensor`](#domiknows.sensor.pytorch.sensors.FunctionalReaderSensor)

### *class* domiknows.sensor.pytorch.query_sensor.DataNodeSensor(\*pres, \*\*kwargs)

Bases: [`QuerySensor`](#domiknows.sensor.pytorch.query_sensor.QuerySensor)

#### forward_wrap()

Wraps the forward method ensuring the results are on the appropriate device.

Returns:
- The result of the forward method, moved to the appropriate device if necessary.

### *class* domiknows.sensor.pytorch.query_sensor.QuerySensor(\*pres, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### *property* builder

#### *property* concept

Returns the concept associated with this sensor.

Raises:
- ValueError: If the sensor doesn’t have a concept associated with it.

#### define_inputs()

Defines the inputs for this sensor based on its predecessors.

#### forward_wrap()

Wraps the forward method ensuring the results are on the appropriate device.

Returns:
- The result of the forward method, moved to the appropriate device if necessary.

## domiknows.sensor.pytorch.relation_sensors module

### *class* domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor(\*pres, \*\*kwargs)

Bases: [`QuerySensor`](#domiknows.sensor.pytorch.query_sensor.QuerySensor)

Base class for Candidate Sensor used to query the input for creating the list of candidate
Inherits from:
- QuerySensor:

#### *property* args

#### define_inputs()

Defines the input/candidate concept of the query

### *class* domiknows.sensor.pytorch.relation_sensors.CandidateEqualSensor(\*pres, edges=None, forward=None, label=False, device='auto', relations=None)

Bases: [`CandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.CandidateSensor)

Candidate Sensor with Equal Relation. The behavior is similar to Candidate Sensor.
However, it adds identification of equality and the name of the equal concept type
Inherits from:
- CandidateSensor: Base class for Edge Candidate Sensor used to query the possible concept to link the relation

#### *property* args

#### forward_wrap()

Forward wrapper for returning candidate with equal relations

Return:
- Candidate of equal relations

### *class* domiknows.sensor.pytorch.relation_sensors.CandidateRelationSensor(\*pres, relations, edges=None, forward=None, label=False, device='auto')

Bases: [`CandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.CandidateSensor)

Candidate Sensor with Relation. The behavior is similar to Candidate Sensor.
However, it provides the relation’s candidate instead of property
Inherits from:
- CandidateSensor: Base class for Edge Candidate Sensor used to query the possible concept to link the relation

#### *property* args

Return:
- Order Dict of name of relation and concept/property linked to

### *class* domiknows.sensor.pytorch.relation_sensors.CandidateSensor(\*pres, relation, \*\*kwargs)

Bases: [`EdgeSensor`](#domiknows.sensor.pytorch.relation_sensors.EdgeSensor), [`BaseCandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor)

Base class for Edge Candidate Sensor used to query the possible concept to link the relation
Inherits from:
- EdgeSensor: Edge sensor used to create the link/edge between two concept to create relation
- BaseCandidateSensor: Base class for Candidate Sensor used to query the input for creating the list of candidate

#### *property* args

#### forward_wrap()

Forwards whether pair of concepts will be used to create the link/edge for relation

Return:
- List of output whether the pair of concepts will be used to create the link/edge for relation

### *class* domiknows.sensor.pytorch.relation_sensors.CompositionCandidateReaderSensor(\*args, relations, \*\*kwargs)

Bases: [`CompositionCandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.CompositionCandidateSensor), [`FunctionalReaderSensor`](#domiknows.sensor.pytorch.sensors.FunctionalReaderSensor)

Inherits from:
- CompositionCandidateSensor
- FunctionalReaderSensor

### *class* domiknows.sensor.pytorch.relation_sensors.CompositionCandidateSensor(\*args, relations, \*\*kwargs)

Bases: [`JointSensor`](#domiknows.sensor.pytorch.sensors.JointSensor), [`BaseCandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor)

Composition candidate Sensor used to join two ot more concepts for constructing relation
Inherits from:
- JointSensor: Represents a joint sensor that generates multiple properties.
- BaseCandidateSensor: Base class for Candidate Sensor used to query the input for creating the list of candidate

#### *property* args

#### forward_wrap()

Forward whether the mapping of candidate concepts ad their corresponding relations

Return:
- List of mapping indicating the link between two/more concepts with specific relation

### *class* domiknows.sensor.pytorch.relation_sensors.EdgeReaderSensor(\*args, keyword, is_constraint=False, \*\*kwargs)

Bases: [`ReaderSensor`](#domiknows.sensor.pytorch.sensors.ReaderSensor), [`EdgeSensor`](#domiknows.sensor.pytorch.relation_sensors.EdgeSensor)

Inherits from:
- ReaderSensor
- EdgeSensor

### *class* domiknows.sensor.pytorch.relation_sensors.EdgeSensor(\*pres, relation, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

Edge sensor used to create the link/edge between two concept to create relation

Inherits from:
: - FunctionalSensor:  A functional sensor with functionality for forward pass operations making it directly usable.

#### fetch_value(pre, selector=None, concept=None)

Fetch the value of the head of relation using the optional selector

Args:
- pre: data item to be updated the head of relation with
- selector: optional selector used to fetch data
- concept: optional concept to fetch data if not, concept to fetch data is source of relation

Return:
- the value of the head of relation

#### *property* relation

Returns the relation linked by this Edge Sensor

#### update_pre_context(data_item: Dict[str, Any], concept=None) → Any

Updates the concept of the relation by provided data items

Args:
: - data_item: data item to be updated the head of relation with

### *class* domiknows.sensor.pytorch.relation_sensors.JointEdgeReaderSensor(\*args, bundle_call=False, \*\*kwargs)

Bases: [`JointReaderSensor`](#domiknows.sensor.pytorch.sensors.JointReaderSensor), [`EdgeSensor`](#domiknows.sensor.pytorch.relation_sensors.EdgeSensor)

Inherits from:
- JointReaderSensor
- EdgeSensor

## domiknows.sensor.pytorch.sensors module

### *class* domiknows.sensor.pytorch.sensors.Cache

Bases: `object`

A base Cache interface that supports setting & getting.

### *class* domiknows.sensor.pytorch.sensors.CacheSensor(\*args, cache={}, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

FunctionalSensor with cached forward calls.
Can be backed by any dict-like object that supports \_\_getitem_\_ and \_\_setitem_\_.

Inherits from:
- FunctionalSensor: Parent class that performs forward passes using the provided

> function.

#### fill_hash(hash)

Sets the cache key to use for the current instance. Should be
unique for the instance.

Args:
- hash: unique identifier for the current instance

#### forward_wrap()

Wraps the parent forward_wrap by checking in the cache first.

If the key is not found, then it performs the regular forward_wrap
call and stores the resulting value.

The hash for the current data item must be set already by calling
self.fill_hash, otherwise None will be used as the cache key.

Returns:
- Cached forward_wrap call

### *class* domiknows.sensor.pytorch.sensors.ConcatSensor(\*pres, edges=None, label=False, device='auto')

Bases: [`TorchSensor`](#domiknows.sensor.pytorch.sensors.TorchSensor)

A sensor that concatenates the inputs on the last dimension for each
forward pass.

#### forward() → Any

Concatenate all tensors in self.inputs along the last dimension. Expects self.inputs
to already be tensors.

Returns:
- The concatenated tensor with shape identical to the inputs except for the

> last dimension, which is the sum of all inputs’ last-dimension sizes.

### *class* domiknows.sensor.pytorch.sensors.ConstantSensor(\*args, data, as_tensor=True, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

A sensor that provides a constant value.

Inherits from:
- FunctionalSensor: A sensor with defined forward functionality; this class

> replaces that with a constant value.

#### forward(\*\_, \*\*\_\_) → Any

Performs the forward function call by returning the set constant value.

Returns:
: - The set constant value, optionally creating a new tensor instance.
    : If the conversion fails, returns the set constant object as-is.

### *class* domiknows.sensor.pytorch.sensors.FunctionalReaderSensor(\*args, keyword, is_constraint=False, \*\*kwargs)

Bases: [`ReaderSensor`](#domiknows.sensor.pytorch.sensors.ReaderSensor)

Combines FunctionalSensor and ReaderSensor. Retrieves values
from input data dictionary for each instance, then applies the specified
forward function.

The given forward function must have a keyword argument data, which is
how the read values will be passed.

Similar to ReaderSensor, supports tuple keywords; the specified function
will be applied for each retrieved value individually.

Inherits from:
- ReaderSensor: A parent sensor class that retrieves values from the input data dictionary.

#### forward(\*args, \*\*kwargs) → Any

Computes the forward pass by applying the specified forward function to
the read values from the keyword.
Uses values that have already been read (from self.data): expects
self.fill_data to be called first for each input sample.

The forward function will always be called, but the passed data may be
None in certain circumstances, including if self.fill_data has not yet been
called (see: ReaderSensor.fill_data).

Returns:
- Read and processed values corresponding to the keyword; either a single

> value or a tuple of values if the keyword is a tuple.

### *class* domiknows.sensor.pytorch.sensors.FunctionalSensor(\*pres, forward=None, build=True, \*\*kwargs)

Bases: [`TorchSensor`](#domiknows.sensor.pytorch.sensors.TorchSensor)

A functional sensor extending the TorchSensor with functionality for forward pass operations making it directly usable.

Inherits from:
- TorchSensor: A base class for torch-based sensors in the graph.

#### fetch_value(pre, selector=None, concept=None)

Fetches the value for a predecessor using an optional selector. Extends the behavior to handle more types.

Args:
- pre: The predecessor to fetch the value for.
- selector (optional): An optional selector to find a specific value.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

Returns:
- The fetched value for the given predecessor.

#### forward(\*inputs, \*\*kwinputs)

Computes the forward pass for this functional sensor, making use of a provided forward function if available.

Args:
- 

```
*
```

inputs: Variable-length argument list of inputs for the forward function.
- 

```
**
```

kwinputs: Additional keyword inputs for the forward function.

Returns:
- The result of the forward computation.
- Calls the superclass forward method if no forward function was provided during initialization.

#### forward_wrap()

Wraps the forward method ensuring the results are on the appropriate device.

Returns:
- The result of the forward method, moved to the appropriate device if necessary.

#### update_context(data_item: Dict[str, Any], force=False, override=True)

Updates the context of the given data item for this functional sensor.

Args:
- data_item (Dict[str, Any]): The data dictionary to update.
- force (bool, optional): Flag to force recalculation even if result is cached. Default is False.
- override (bool, optional): Flag to decide if overriding the parent node is allowed. Default is True.

#### update_pre_context(data_item: Dict[str, Any], concept=None)

Updates the context for the predecessors of this sensor. Extends the behavior to handle more types.

Args:
- data_item (Dict[str, Any]): The data dictionary to update context for.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

### *class* domiknows.sensor.pytorch.sensors.JointReaderSensor(\*args, bundle_call=False, \*\*kwargs)

Bases: [`JointSensor`](#domiknows.sensor.pytorch.sensors.JointSensor), [`ReaderSensor`](#domiknows.sensor.pytorch.sensors.ReaderSensor)

Combines JointSensor and ReaderSensor. Retrieves values from the
input data dictionary into multiple properties.

Inherits from:
- JointSensor: A parent sensor class that calculates multiple properties.
- ReaderSensor: A parent sensor class that retrieves values from the input data dictionary.

### *class* domiknows.sensor.pytorch.sensors.JointSensor(\*args, bundle_call=False, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

Represents a joint sensor that generates multiple properties.

Inherits from:
- FunctionalSensor: A sensor with defined forward functionality.

#### attached(sup)

Configures the joint sensor when attached to a parent node.

Args:
- sup: The parent node to which this sensor is attached.

#### *property* components

Returns the list of component sensors associated with this joint sensor.

Returns:
- List of component sensors.

#### update_context(data_item: Dict[str, Any], force=False, override=True)

Updates the context of the given data item for this joint sensor.

Args:
- data_item (Dict[str, Any]): The data dictionary to update.
- force (bool, optional): Flag to force recalculation even if result is cached. Default is False.
- override (bool, optional): Flag to decide if overriding of the parent data is allowed. Default is True.

### *class* domiknows.sensor.pytorch.sensors.LabelReaderSensor(\*args, \*\*kwargs)

Bases: [`ReaderSensor`](#domiknows.sensor.pytorch.sensors.ReaderSensor)

A ReaderSensor that’s also a label. Equivalent to creating a ReaderSensor with the
label keyword argument set to True.

Inherits from:
- ReaderSensor: A parent sensor class that retrieves values from the input data dictionary.

### *class* domiknows.sensor.pytorch.sensors.ListConcator(\*pres, edges=None, label=False, device='auto')

Bases: [`TorchSensor`](#domiknows.sensor.pytorch.sensors.TorchSensor)

A sensor that stacks lists of tensors and concatenates them
on the last dimension for each forward pass.

#### forward() → Any

Stacks lists of tensors into a single tensor (in-place) and concatenates
all those inputs on the last dimension.

Returns:
- The concatenated tensor. The last dimension equals the sum of the

> last-dimension sizes of all (converted) inputs.

### *class* domiknows.sensor.pytorch.sensors.ModuleSensor(\*args, module, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### *property* device

#### forward(\*inputs)

Computes the forward pass for this functional sensor, making use of a provided forward function if available.

Args:
- 

```
*
```

inputs: Variable-length argument list of inputs for the forward function.
- 

```
**
```

kwinputs: Additional keyword inputs for the forward function.

Returns:
- The result of the forward computation.
- Calls the superclass forward method if no forward function was provided during initialization.

#### *property* model

### *class* domiknows.sensor.pytorch.sensors.NominalSensor(\*pres, vocab=None, edges=None, device='auto')

Bases: [`TorchSensor`](#domiknows.sensor.pytorch.sensors.TorchSensor)

A base sensor class that calculates the one-hot encoded form of the forward function.
This class must be inherited and reimplemeted and is not usable.

Inherits from:
- TorchSensor: A parent sensor class for torch-based sensors in the graph.

#### complete_vocab()

Adds values to the vocabulary based on the forward pass output.

#### one_hot_encoder(value)

Helper function for calculating the one-hot encoding of a given value or set of values.

One-hot encodings are calculated by indexing against the self.vocab attribute.

Args:
- value: A value or list of values to encode as a one-hot tensor.

Returns:
- A tensor of one-hot encoded values.

> If a single value is provided, then outputs a tensor of size (1, V).
> If a list of values is provided with non-zero size, then outputs a tensor of size (N, 1, V).
> If an empty list of values is provided, then outputs a tensor of size (1, 1, V) with all zeros.

#### update_context(data_item: Dict[str, Any], force=False)

Updates the context of the given data item for this sensor and calculates the one-hot encoding of the function output.

Args:
- data_item: The data dictionary to update.
- force (optional): Flag to force recalculation even if result is cached. Default is False.

### *class* domiknows.sensor.pytorch.sensors.PrefilledSensor(\*pres, forward=None, build=True, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

A sensor that returns the existing filled value of the property associated with this sensor.

Inherits from:
- FunctionalSensor: A sensor with defined forward functionality; this class

> replaces that with the pre-filled property value.

#### forward(\*args, \*\*kwargs) → Any

Performs the forward function call by returning the pre-filled property value.

Returns:
- The existing filled value corresponding to the property associated with this sensor.

### *class* domiknows.sensor.pytorch.sensors.ReaderSensor(\*args, keyword, is_constraint=False, \*\*kwargs)

Bases: [`ConstantSensor`](#domiknows.sensor.pytorch.sensors.ConstantSensor)

A sensor that retrieves values from input data dictionary for each instance.

Inherits from:
- ConstantSensor: A parent sensor class that just returns a constant value.

#### fill_data(data_item)

Read the target value (based on the set keyword attribute) from the given
data_item into self.data.
By default, expects the keyword to be present in the data_item. However,
if self.is_constraint is set, then it allows the keyword to be missing (and
instead just sets self.data to None).

If the keyword is a tuple of values, then will read each item individually.

Args:
- data_item: The data dictionary to read values from

Raises:
- KeyError if self.is_constraint is False and the desired keyword

> is missing from the input data_item.

#### forward(\*\_, \*\*\_\_) → Any

Computes the forward pass by returning the values read from the keyword.
Converts the data to torch tensors by default. Returns values that have
already been read (from self.data): expects self.fill_data to be called first
for each input sample.

May return None in certain conditions, including if self.fill_data has not
yet been called (see: self.fill_data).

Returns:
- Read values corresponding to the keyword; either a single value or a

> tuple of values if the keyword is a tuple.

### *class* domiknows.sensor.pytorch.sensors.TorchCache(path)

Bases: [`Cache`](#domiknows.sensor.pytorch.sensors.Cache)

Disk-based cache that serializes values with torch.save & uses file-names as keys.

Inherits from:
- Cache: Parent Cache interface supporting getting/setting.

#### file_path(name)

Gets the save/load path of the values given a cache key.

Args:
- name: Cache key

Returns:
- File-path where the cached values are located.

#### *property* path

Path of folder where cached values will be saved.

Returns:
- Save folder path

#### sanitize(name)

Helper function for creating the cache file name by removing/replacing
certain symbols.

Args:
- name: Cache key

Returns:
- Sanitized cache key

### *class* domiknows.sensor.pytorch.sensors.TorchEdgeSensor(\*pres, to, mode='forward', edges=None, forward=None, label=False, device='auto')

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### attached(sup)

#### *property* concept

Returns the concept associated with this sensor.

Raises:
- ValueError: If the sensor doesn’t have a concept associated with it.

#### fetch_value(pre, selector=None, concept=None)

Fetches the value for a predecessor using an optional selector. Extends the behavior to handle more types.

Args:
- pre: The predecessor to fetch the value for.
- selector (optional): An optional selector to find a specific value.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

Returns:
- The fetched value for the given predecessor.

#### modes *= ('forward', 'backward', 'selection')*

#### update_context(data_item: Dict[str, Any], force=False, override=True)

Updates the context of the given data item for this functional sensor.

Args:
- data_item (Dict[str, Any]): The data dictionary to update.
- force (bool, optional): Flag to force recalculation even if result is cached. Default is False.
- override (bool, optional): Flag to decide if overriding the parent node is allowed. Default is True.

#### update_pre_context(data_item: Dict[str, Any], concept=None) → Any

Updates the context for the predecessors of this sensor. Extends the behavior to handle more types.

Args:
- data_item (Dict[str, Any]): The data dictionary to update context for.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

### *class* domiknows.sensor.pytorch.sensors.TorchSensor(\*pres, edges=None, label=False, device='auto')

Bases: [`Sensor`](domiknows.sensor.md#domiknows.sensor.sensor.Sensor)

A second main sensor base class that builds on the bare sensor class and updates and propagates context based on the given data item.
This class must be inherited and reimplemeted and is not usable.

Inherits from:
- Sensor: The base class for sensors.

#### *property* concept

Returns the concept associated with this sensor.

Raises:
- ValueError: If the sensor doesn’t have a concept associated with it.

#### define_inputs()

Defines the inputs for this sensor based on its predecessors.

#### fetch_value(pre, selector=None, concept=None)

Fetches the value for a predecessor using an optional selector.

Args:
- pre: The predecessor to fetch the value for.
- selector (optional): An optional selector to find a specific value.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

Returns:
- The fetched value for the given predecessor.

Raises:
- Raises KeyError if the provided selector key doesn’t exist.

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

#### *static* non_label_sensor(sensor)

Checks if the provided sensor is not a label sensor.

Args:
- sensor: The sensor to check.

Returns:
- bool: True if the sensor is not a label sensor, otherwise False.

#### *property* prop

Returns the superior of this sensor. This property is used to get the property associated with the sensor.

Raises:
- ValueError: If the sensor doesn’t have a superior.

#### update_context(data_item: Dict[str, Any], force=False)

Updates the context of the given data item for this torch sensor. The fucntion that is callaed when \_\_call_\_ is used.

Args:
- data_item (Dict[str, Any]): The data dictionary to update.
- force (bool, optional): Flag to force recalculation even if result is cached. Default is False.

#### update_pre_context(data_item: Dict[str, Any], concept=None) → Any

Updates the context for the predecessors of this sensor.

Args:
- data_item (Dict[str, Any]): The data dictionary to update context for.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

### *class* domiknows.sensor.pytorch.sensors.TriggerPrefilledSensor(\*args, callback_sensor=None, \*\*kwargs)

Bases: [`PrefilledSensor`](#domiknows.sensor.pytorch.sensors.PrefilledSensor)

A sensor that returns the existing filled value of the property associated with this sensor
and triggers a callback to another Sensor.

Inherits from:
- PrefilledSensor: A sensor that returns the existing value of the property.

#### forward(\*args, \*\*kwargs) → Any

Performs the forward function call by returning the pre-filled property value, but first
triggering a callback to the set self.callback_sensor.

Returns:
- The existing filled value corresponding to the property associated with this sensor.

### domiknows.sensor.pytorch.sensors.cache(SensorClass, CacheSensorClass=<class 'domiknows.sensor.pytorch.sensors.CacheSensor'>)

### domiknows.sensor.pytorch.sensors.joint(SensorClass, JointSensorClass=<class 'domiknows.sensor.pytorch.sensors.JointSensor'>)

## domiknows.sensor.pytorch.tokenizer_sensors module

## domiknows.sensor.pytorch.utils module

### *class* domiknows.sensor.pytorch.utils.UnBatchWrap(module)

Bases: `Module`

#### batch(value)

#### forward(\*args, \*\*kwargs)

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

#### unbatch(value)

## Module contents
