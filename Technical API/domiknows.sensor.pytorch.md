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

## domiknows.sensor.pytorch.learnerModels module

### *class* domiknows.sensor.pytorch.learnerModels.LSTMModel(input_dim, hidden_dim, bidirectional=False, num_layers=1, batch_size=1)

Bases: `Module`

#### forward(input)

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

#### init_hidden()

### *class* domiknows.sensor.pytorch.learnerModels.PyTorchFC(input_dim, output_dim)

Bases: `Module`

#### forward(x)

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### *class* domiknows.sensor.pytorch.learnerModels.PyTorchFCRelu(input_dim, output_dim)

Bases: `Module`

#### forward(x)

Define the computation performed at every call.

Should be overridden by all subclasses.

#### NOTE
Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

## domiknows.sensor.pytorch.learners module

### *class* domiknows.sensor.pytorch.learners.FullyConnectedLearner(\*pres, input_dim, output_dim, device='auto')

Bases: [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

#### forward() → Any

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

### *class* domiknows.sensor.pytorch.learners.FullyConnectedLearnerRelu(\*pres, input_dim, output_dim, device='auto')

Bases: [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

#### forward() → Any

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

### *class* domiknows.sensor.pytorch.learners.LSTMLearner(\*pres, input_dim, hidden_dim, num_layers=1, bidirectional=False, device='auto')

Bases: [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

#### forward() → Any

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

### *class* domiknows.sensor.pytorch.learners.ModuleLearner(\*pres, module, edges=None, loss=None, metric=None, label=False, \*\*kwargs)

Bases: [`ModuleSensor`](#domiknows.sensor.pytorch.sensors.ModuleSensor), [`TorchLearner`](#domiknows.sensor.pytorch.learners.TorchLearner)

#### update_parameters()

### *class* domiknows.sensor.pytorch.learners.TorchLearner(\*pre, edges=None, loss=None, metric=None, label=False, device='auto')

Bases: [`Learner`](domiknows.sensor.md#domiknows.sensor.learner.Learner), [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### *property* device

#### load(filepath)

#### loss(data_item, target)

#### metric(data_item, target)

#### *property* model

#### *abstract property* parameters *: Any*

#### *property* sanitized_name

#### save(filepath)

#### update_parameters()

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

### *class* domiknows.sensor.pytorch.relation_sensors.EdgeSensor(\*pres, \*\*kwargs, relations)

**Objective**: Use to create the relation (edge) between two concepts, need to specify which relation to create

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.query_sensor.QuerySensor)

#### *property* args


#### relation()

Returns:
- The relation linked from this sensor.

#### update_pre_context(data_item, concept=None)

Update either concept or source of the edge (head of relation) if concept is not provided into provided data item

Args:
- data_item: The data item used to update concept
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.
- 


#### fetch_value(pre, selector=None, concept=None)

Args:
- pre: The predecessor to fetch the value for.
- selector (optional): An optional selector to find a specific value.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

Returns:
- The fetched value for the given predecessor.

Fetch the value of concept or source of the edge (head of relation) if concept is not provided using optional selector.

### *class* domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor(\*pres, \*\*kwargs)

**Objective**: Based class for Sensor to determine the connection between Concept


Bases: [`QuerySensor`](#domiknows.sensor.pytorch.query_sensor.QuerySensor)

#### *property* args

#### define_inputs()

Defines the inputs for this sensor based on its predecessors.

### *class* domiknows.sensor.pytorch.relation_sensors.CandidateEqualSensor(\*pres, edges=None, forward=None, label=False, device='auto', relations=None)

Bases: [`CandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.CandidateSensor)

#### *property* args

#### forward_wrap()

Wraps the forward method ensuring the results are on the appropriate device.

Returns:
- The result of the forward method, moved to the appropriate device if necessary.

### *class* domiknows.sensor.pytorch.relation_sensors.CandidateRelationSensor(\*pres, relations, edges=None, forward=None, label=False, device='auto')

Bases: [`CandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.CandidateSensor)

#### *property* args

### *class* domiknows.sensor.pytorch.relation_sensors.CandidateSensor(\*pres, relation, \*\*kwargs)

Bases: [`EdgeSensor`](#domiknows.sensor.pytorch.relation_sensors.EdgeSensor), [`BaseCandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor)

#### *property* args

#### forward_wrap()

Wraps the forward method ensuring the results are on the appropriate device.

Returns:
- The result of the forward method, moved to the appropriate device if necessary.

### *class* domiknows.sensor.pytorch.relation_sensors.CompositionCandidateReaderSensor(\*args, relations, \*\*kwargs)

**Objective**: Reading whether there is connection between concept considered within relations.


Bases: [`CompositionCandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.CompositionCandidateSensor), [`FunctionalReaderSensor`](#domiknows.sensor.pytorch.sensors.FunctionalReaderSensor)

### *class* domiknows.sensor.pytorch.relation_sensors.CompositionCandidateSensor(\*args, relations, \*\*kwargs)

**Objective**: Creating the mapping between multiple concepts and relations. Required the function to return whether there is connection based on defined relation.

Bases: [`JointSensor`](#domiknows.sensor.pytorch.sensors.JointSensor), [`BaseCandidateSensor`](#domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor)

#### *property* args

#### forward_wrap()

Wraps the forward method ensuring the results are on the appropriate device.

Returns:
- The result of the forward method, moved to the appropriate device if necessary.

### *class* domiknows.sensor.pytorch.relation_sensors.EdgeReaderSensor(\*args, keyword, is_constraint=False, \*\*kwargs)

**Objective**: Reading the edge sensor in the form of list from data item

Bases: [`ReaderSensor`](#domiknows.sensor.pytorch.sensors.ReaderSensor), [`EdgeSensor`](#domiknows.sensor.pytorch.relation_sensors.EdgeSensor)

### *class* domiknows.sensor.pytorch.relation_sensors.EdgeSensor(\*pres, relation, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### fetch_value(pre, selector=None, concept=None)

Fetches the value for a predecessor using an optional selector. Extends the behavior to handle more types.

Args:
- pre: The predecessor to fetch the value for.
- selector (optional): An optional selector to find a specific value.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

Returns:
- The fetched value for the given predecessor.

#### *property* relation

#### update_pre_context(data_item: Dict[str, Any], concept=None) → Any

Updates the context for the predecessors of this sensor. Extends the behavior to handle more types.

Args:
- data_item (Dict[str, Any]): The data dictionary to update context for.
- concept (optional): The concept associated with this sensor. Defaults to the sensor’s own concept.

### *class* domiknows.sensor.pytorch.relation_sensors.JointEdgeReaderSensor(\*args, bundle_call=False, \*\*kwargs)

Bases: [`JointReaderSensor`](#domiknows.sensor.pytorch.sensors.JointReaderSensor), [`EdgeSensor`](#domiknows.sensor.pytorch.relation_sensors.EdgeSensor)

## domiknows.sensor.pytorch.sensors module

### *class* domiknows.sensor.pytorch.sensors.AggregationSensor(\*pres, edges, map_key, deafault_dim=480, device='auto')

Bases: [`TorchSensor`](#domiknows.sensor.pytorch.sensors.TorchSensor)

#### get_data()

#### get_map_value()

#### update_context(data_item: Dict[str, Any], force=False)

Updates the context of the given data item for this torch sensor. The fucntion that is callaed when \_\_call_\_ is used.

Args:
- data_item (Dict[str, Any]): The data dictionary to update.
- force (bool, optional): Flag to force recalculation even if result is cached. Default is False.

### *class* domiknows.sensor.pytorch.sensors.Cache

Bases: `object`

### *class* domiknows.sensor.pytorch.sensors.CacheSensor(\*args, cache={}, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### fill_hash(hash)

#### forward_wrap()

Wraps the forward method ensuring the results are on the appropriate device.

Returns:
- The result of the forward method, moved to the appropriate device if necessary.

### *class* domiknows.sensor.pytorch.sensors.ConcatAggregationSensor(\*pres, edges, map_key, deafault_dim=480, device='auto')

Bases: [`AggregationSensor`](#domiknows.sensor.pytorch.sensors.AggregationSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

### *class* domiknows.sensor.pytorch.sensors.ConcatSensor(\*pres, edges=None, label=False, device='auto')

Bases: [`TorchSensor`](#domiknows.sensor.pytorch.sensors.TorchSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

### *class* domiknows.sensor.pytorch.sensors.ConstantSensor(\*args, data, as_tensor=True, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### forward(\*\_, \*\*\_\_) → Any

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

### *class* domiknows.sensor.pytorch.sensors.FirstAggregationSensor(\*pres, edges, map_key, deafault_dim=480, device='auto')

Bases: [`AggregationSensor`](#domiknows.sensor.pytorch.sensors.AggregationSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

### *class* domiknows.sensor.pytorch.sensors.FunctionalReaderSensor(\*args, keyword, is_constraint=False, \*\*kwargs)

Bases: [`ReaderSensor`](#domiknows.sensor.pytorch.sensors.ReaderSensor)

#### forward(\*args, \*\*kwargs) → Any

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

### *class* domiknows.sensor.pytorch.sensors.LastAggregationSensor(\*pres, edges, map_key, deafault_dim=480, device='auto')

Bases: [`AggregationSensor`](#domiknows.sensor.pytorch.sensors.AggregationSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

### *class* domiknows.sensor.pytorch.sensors.ListConcator(\*pres, edges=None, label=False, device='auto')

Bases: [`TorchSensor`](#domiknows.sensor.pytorch.sensors.TorchSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

### *class* domiknows.sensor.pytorch.sensors.MaxAggregationSensor(\*pres, edges, map_key, deafault_dim=480, device='auto')

Bases: [`AggregationSensor`](#domiknows.sensor.pytorch.sensors.AggregationSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

### *class* domiknows.sensor.pytorch.sensors.MeanAggregationSensor(\*pres, edges, map_key, deafault_dim=480, device='auto')

Bases: [`AggregationSensor`](#domiknows.sensor.pytorch.sensors.AggregationSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

### *class* domiknows.sensor.pytorch.sensors.MinAggregationSensor(\*pres, edges, map_key, deafault_dim=480, device='auto')

Bases: [`AggregationSensor`](#domiknows.sensor.pytorch.sensors.AggregationSensor)

#### forward() → Any

Computes the forward pass for this torch sensor.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

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

#### complete_vocab()

#### one_hot_encoder(value)

#### update_context(data_item: Dict[str, Any], force=False)

Updates the context of the given data item for this torch sensor. The fucntion that is callaed when \_\_call_\_ is used.

Args:
- data_item (Dict[str, Any]): The data dictionary to update.
- force (bool, optional): Flag to force recalculation even if result is cached. Default is False.

### *class* domiknows.sensor.pytorch.sensors.PrefilledSensor(\*pres, forward=None, build=True, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### forward(\*args, \*\*kwargs) → Any

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

### *class* domiknows.sensor.pytorch.sensors.ReaderSensor(\*args, keyword, is_constraint=False, \*\*kwargs)

Bases: [`ConstantSensor`](#domiknows.sensor.pytorch.sensors.ConstantSensor)

#### fill_data(data_item)

#### forward(\*\_, \*\*\_\_) → Any

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

### *class* domiknows.sensor.pytorch.sensors.SpacyTokenizorSensor(\*pres, forward=None, build=True, \*\*kwargs)

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### *class* English(vocab: Vocab | bool = True, , max_length: int = 1000000, meta: Dict[str, Any] = {}, create_tokenizer: Callable[[Language], Callable[[str], Doc]] | None = None, create_vectors: Callable[[Vocab], BaseVectors] | None = None, batch_size: int = 1000, \*\*kwargs)

Bases: `Language`

#### Defaults

alias of `EnglishDefaults`

#### default_config *= {'components': {}, 'corpora': {'dev': {'@readers': 'spacy.Corpus.v1', 'augmenter': None, 'gold_preproc': False, 'limit': 0, 'max_length': 0, 'path': '${paths.dev}'}, 'train': {'@readers': 'spacy.Corpus.v1', 'augmenter': None, 'gold_preproc': False, 'limit': 0, 'max_length': 0, 'path': '${paths.train}'}}, 'initialize': {'after_init': None, 'before_init': None, 'components': {}, 'init_tok2vec': '${paths.init_tok2vec}', 'lookups': None, 'tokenizer': {}, 'vectors': '${paths.vectors}', 'vocab_data': None}, 'nlp': {'after_creation': None, 'after_pipeline_creation': None, 'batch_size': 1000, 'before_creation': None, 'disabled': [], 'lang': 'en', 'pipeline': [], 'tokenizer': {'@tokenizers': 'spacy.Tokenizer.v1'}, 'vectors': {'@vectors': 'spacy.Vectors.v1'}}, 'paths': {'dev': None, 'init_tok2vec': None, 'train': None, 'vectors': None}, 'system': {'gpu_allocator': None, 'seed': 0}, 'training': {'accumulate_gradient': 1, 'annotating_components': [], 'batcher': {'@batchers': 'spacy.batch_by_words.v1', 'discard_oversize': False, 'size': {'@schedules': 'compounding.v1', 'compound': 1.001, 'start': 100, 'stop': 1000}, 'tolerance': 0.2}, 'before_to_disk': None, 'before_update': None, 'dev_corpus': 'corpora.dev', 'dropout': 0.1, 'eval_frequency': 200, 'frozen_components': [], 'gpu_allocator': '${system.gpu_allocator}', 'logger': {'@loggers': 'spacy.ConsoleLogger.v1'}, 'max_epochs': 0, 'max_steps': 20000, 'optimizer': {'@optimizers': 'Adam.v1', 'L2': 0.01, 'L2_is_weight_decay': True, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-08, 'grad_clip': 1.0, 'learn_rate': 0.001, 'use_averages': False}, 'patience': 1600, 'score_weights': {}, 'seed': '${system.seed}', 'train_corpus': 'corpora.train'}}*

#### factories *= {'en.lemmatizer': <function make_lemmatizer>, 'merge_entities': <function Language.component.<locals>.add_component.<locals>.factory_func>, 'merge_noun_chunks': <function Language.component.<locals>.add_component.<locals>.factory_func>, 'merge_subtokens': <function Language.component.<locals>.add_component.<locals>.factory_func>}*

#### lang *: str | None* *= 'en'*

#### forward(sentences)

Computes the forward pass for converting sentence into token using Spacy

Args:
- sentences: sentences to be converted into list(s) of tokens

Returns:
- The list(s) of token converted from provided sentence

#### nlp *= <spacy.lang.en.English object>*

### *class* domiknows.sensor.pytorch.sensors.BertTokenizorSensor(\*pres, forward=None, build=True, \*\*kwargs)

**Objective**: Sensor to convert sentence into token using BertTokenizer

Bases: [`FunctionalSensor`](#domiknows.sensor.pytorch.sensors.FunctionalSensor)

#### TRANSFORMER_MODEL *= 'bert-base-uncased'*

#### forward(sentences)

Computes the forward pass for converting sentence into token using BertTokenizer

Args:
- sentences: sentences to be converted into list(s) of tokens

Returns:
- The list(s) of token converted from provided sentence

#### *property* tokenizer

### *class* domiknows.sensor.pytorch.sensors.TorchCache(path)

Bases: [`Cache`](#domiknows.sensor.pytorch.sensors.Cache)

#### file_path(name)

#### *property* path

#### sanitize(name)

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

#### forward(\*args, \*\*kwargs) → Any

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

### domiknows.sensor.pytorch.sensors.cache(SensorClass, CacheSensorClass=<class 'domiknows.sensor.pytorch.sensors.CacheSensor'>)

### domiknows.sensor.pytorch.sensors.joint(SensorClass, JointSensorClass=<class 'domiknows.sensor.pytorch.sensors.JointSensor'>)

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
