# domiknows.sensor package

## Subpackages

* [domiknows.sensor.pytorch package](domiknows.sensor.pytorch.md)
  * [Subpackages](domiknows.sensor.pytorch.md#subpackages)
    * [domiknows.sensor.pytorch.tokenizers package](domiknows.sensor.pytorch.tokenizers.md)
      * [Submodules](domiknows.sensor.pytorch.tokenizers.md#submodules)
      * [domiknows.sensor.pytorch.tokenizers.spacy module](domiknows.sensor.pytorch.tokenizers.md#module-domiknows.sensor.pytorch.tokenizers.spacy)
      * [domiknows.sensor.pytorch.tokenizers.transformers module](domiknows.sensor.pytorch.tokenizers.md#module-domiknows.sensor.pytorch.tokenizers.transformers)
      * [Module contents](domiknows.sensor.pytorch.tokenizers.md#module-domiknows.sensor.pytorch.tokenizers)
  * [Submodules](domiknows.sensor.pytorch.md#submodules)
  * [domiknows.sensor.pytorch.learnerModels module](domiknows.sensor.pytorch.md#module-domiknows.sensor.pytorch.learnerModels)
    * [`LSTMModel`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.LSTMModel)
      * [`LSTMModel.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.LSTMModel.forward)
      * [`LSTMModel.init_hidden()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.LSTMModel.init_hidden)
      * [`LSTMModel.training`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.LSTMModel.training)
    * [`PyTorchFC`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.PyTorchFC)
      * [`PyTorchFC.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.PyTorchFC.forward)
      * [`PyTorchFC.training`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.PyTorchFC.training)
    * [`PyTorchFCRelu`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.PyTorchFCRelu)
      * [`PyTorchFCRelu.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.PyTorchFCRelu.forward)
      * [`PyTorchFCRelu.training`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learnerModels.PyTorchFCRelu.training)
  * [domiknows.sensor.pytorch.learners module](domiknows.sensor.pytorch.md#module-domiknows.sensor.pytorch.learners)
    * [`FullyConnectedLearner`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.FullyConnectedLearner)
      * [`FullyConnectedLearner.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.FullyConnectedLearner.forward)
    * [`FullyConnectedLearnerRelu`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.FullyConnectedLearnerRelu)
      * [`FullyConnectedLearnerRelu.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.FullyConnectedLearnerRelu.forward)
    * [`LSTMLearner`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.LSTMLearner)
      * [`LSTMLearner.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.LSTMLearner.forward)
    * [`ModuleLearner`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.ModuleLearner)
      * [`ModuleLearner.update_parameters()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.ModuleLearner.update_parameters)
    * [`TorchLearner`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner)
      * [`TorchLearner.device`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.device)
      * [`TorchLearner.load()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.load)
      * [`TorchLearner.loss()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.loss)
      * [`TorchLearner.metric()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.metric)
      * [`TorchLearner.model`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.model)
      * [`TorchLearner.parameters`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.parameters)
      * [`TorchLearner.sanitized_name`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.sanitized_name)
      * [`TorchLearner.save()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.save)
      * [`TorchLearner.update_parameters()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.learners.TorchLearner.update_parameters)
  * [domiknows.sensor.pytorch.query_sensor module](domiknows.sensor.pytorch.md#module-domiknows.sensor.pytorch.query_sensor)
    * [`DataNodeReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.DataNodeReaderSensor)
    * [`DataNodeSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.DataNodeSensor)
      * [`DataNodeSensor.forward_wrap()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.DataNodeSensor.forward_wrap)
    * [`QuerySensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.QuerySensor)
      * [`QuerySensor.builder`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.QuerySensor.builder)
      * [`QuerySensor.concept`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.QuerySensor.concept)
      * [`QuerySensor.define_inputs()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.QuerySensor.define_inputs)
      * [`QuerySensor.forward_wrap()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.query_sensor.QuerySensor.forward_wrap)
  * [domiknows.sensor.pytorch.relation_sensors module](domiknows.sensor.pytorch.md#module-domiknows.sensor.pytorch.relation_sensors)
    * [`BaseCandidateSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor)
      * [`BaseCandidateSensor.args`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor.args)
      * [`BaseCandidateSensor.define_inputs()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.BaseCandidateSensor.define_inputs)
    * [`CandidateEqualSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateEqualSensor)
      * [`CandidateEqualSensor.args`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateEqualSensor.args)
      * [`CandidateEqualSensor.forward_wrap()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateEqualSensor.forward_wrap)
    * [`CandidateRelationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateRelationSensor)
      * [`CandidateRelationSensor.args`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateRelationSensor.args)
    * [`CandidateSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateSensor)
      * [`CandidateSensor.args`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateSensor.args)
      * [`CandidateSensor.forward_wrap()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CandidateSensor.forward_wrap)
    * [`CompositionCandidateReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CompositionCandidateReaderSensor)
    * [`CompositionCandidateSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CompositionCandidateSensor)
      * [`CompositionCandidateSensor.args`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CompositionCandidateSensor.args)
      * [`CompositionCandidateSensor.forward_wrap()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.CompositionCandidateSensor.forward_wrap)
    * [`EdgeReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeReaderSensor)
    * [`EdgeSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor)
      * [`EdgeSensor.fetch_value()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor.fetch_value)
      * [`EdgeSensor.relation`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor.relation)
      * [`EdgeSensor.update_pre_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor.update_pre_context)
    * [`JointEdgeReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.JointEdgeReaderSensor)
  * [domiknows.sensor.pytorch.sensors module](domiknows.sensor.pytorch.md#module-domiknows.sensor.pytorch.sensors)
    * [`AggregationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.AggregationSensor)
      * [`AggregationSensor.get_data()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.AggregationSensor.get_data)
      * [`AggregationSensor.get_map_value()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.AggregationSensor.get_map_value)
      * [`AggregationSensor.update_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.AggregationSensor.update_context)
    * [`BertTokenizorSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.BertTokenizorSensor)
      * [`BertTokenizorSensor.TRANSFORMER_MODEL`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.BertTokenizorSensor.TRANSFORMER_MODEL)
      * [`BertTokenizorSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.BertTokenizorSensor.forward)
      * [`BertTokenizorSensor.tokenizer`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.BertTokenizorSensor.tokenizer)
    * [`Cache`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.Cache)
    * [`CacheSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.CacheSensor)
      * [`CacheSensor.fill_hash()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.CacheSensor.fill_hash)
      * [`CacheSensor.forward_wrap()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.CacheSensor.forward_wrap)
    * [`ConcatAggregationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ConcatAggregationSensor)
      * [`ConcatAggregationSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ConcatAggregationSensor.forward)
    * [`ConcatSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ConcatSensor)
      * [`ConcatSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ConcatSensor.forward)
    * [`ConstantSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ConstantSensor)
      * [`ConstantSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ConstantSensor.forward)
    * [`FirstAggregationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FirstAggregationSensor)
      * [`FirstAggregationSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FirstAggregationSensor.forward)
    * [`FunctionalReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalReaderSensor)
      * [`FunctionalReaderSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalReaderSensor.forward)
    * [`FunctionalSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalSensor)
      * [`FunctionalSensor.fetch_value()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalSensor.fetch_value)
      * [`FunctionalSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalSensor.forward)
      * [`FunctionalSensor.forward_wrap()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalSensor.forward_wrap)
      * [`FunctionalSensor.update_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalSensor.update_context)
      * [`FunctionalSensor.update_pre_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.FunctionalSensor.update_pre_context)
    * [`JointReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.JointReaderSensor)
    * [`JointSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.JointSensor)
      * [`JointSensor.attached()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.JointSensor.attached)
      * [`JointSensor.components`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.JointSensor.components)
      * [`JointSensor.update_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.JointSensor.update_context)
    * [`LabelReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.LabelReaderSensor)
    * [`LastAggregationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.LastAggregationSensor)
      * [`LastAggregationSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.LastAggregationSensor.forward)
    * [`ListConcator`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ListConcator)
      * [`ListConcator.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ListConcator.forward)
    * [`MaxAggregationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.MaxAggregationSensor)
      * [`MaxAggregationSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.MaxAggregationSensor.forward)
    * [`MeanAggregationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.MeanAggregationSensor)
      * [`MeanAggregationSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.MeanAggregationSensor.forward)
    * [`MinAggregationSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.MinAggregationSensor)
      * [`MinAggregationSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.MinAggregationSensor.forward)
    * [`ModuleSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ModuleSensor)
      * [`ModuleSensor.device`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ModuleSensor.device)
      * [`ModuleSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ModuleSensor.forward)
      * [`ModuleSensor.model`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ModuleSensor.model)
    * [`NominalSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.NominalSensor)
      * [`NominalSensor.complete_vocab()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.NominalSensor.complete_vocab)
      * [`NominalSensor.one_hot_encoder()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.NominalSensor.one_hot_encoder)
      * [`NominalSensor.update_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.NominalSensor.update_context)
    * [`PrefilledSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.PrefilledSensor)
      * [`PrefilledSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.PrefilledSensor.forward)
    * [`ReaderSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ReaderSensor)
      * [`ReaderSensor.fill_data()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ReaderSensor.fill_data)
      * [`ReaderSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.ReaderSensor.forward)
    * [`SpacyTokenizorSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.SpacyTokenizorSensor)
      * [`SpacyTokenizorSensor.English`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.SpacyTokenizorSensor.English)
      * [`SpacyTokenizorSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.SpacyTokenizorSensor.forward)
      * [`SpacyTokenizorSensor.nlp`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.SpacyTokenizorSensor.nlp)
    * [`TorchCache`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchCache)
      * [`TorchCache.file_path()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchCache.file_path)
      * [`TorchCache.path`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchCache.path)
      * [`TorchCache.sanitize()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchCache.sanitize)
    * [`TorchEdgeSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchEdgeSensor)
      * [`TorchEdgeSensor.attached()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchEdgeSensor.attached)
      * [`TorchEdgeSensor.concept`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchEdgeSensor.concept)
      * [`TorchEdgeSensor.fetch_value()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchEdgeSensor.fetch_value)
      * [`TorchEdgeSensor.modes`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchEdgeSensor.modes)
      * [`TorchEdgeSensor.update_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchEdgeSensor.update_context)
      * [`TorchEdgeSensor.update_pre_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchEdgeSensor.update_pre_context)
    * [`TorchSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor)
      * [`TorchSensor.concept`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.concept)
      * [`TorchSensor.define_inputs()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.define_inputs)
      * [`TorchSensor.fetch_value()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.fetch_value)
      * [`TorchSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.forward)
      * [`TorchSensor.non_label_sensor()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.non_label_sensor)
      * [`TorchSensor.prop`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.prop)
      * [`TorchSensor.update_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.update_context)
      * [`TorchSensor.update_pre_context()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TorchSensor.update_pre_context)
    * [`TriggerPrefilledSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TriggerPrefilledSensor)
      * [`TriggerPrefilledSensor.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.TriggerPrefilledSensor.forward)
    * [`cache()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.cache)
    * [`joint()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.joint)
  * [domiknows.sensor.pytorch.utils module](domiknows.sensor.pytorch.md#module-domiknows.sensor.pytorch.utils)
    * [`UnBatchWrap`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.utils.UnBatchWrap)
      * [`UnBatchWrap.batch()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.utils.UnBatchWrap.batch)
      * [`UnBatchWrap.forward()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.utils.UnBatchWrap.forward)
      * [`UnBatchWrap.training`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.utils.UnBatchWrap.training)
      * [`UnBatchWrap.unbatch()`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.utils.UnBatchWrap.unbatch)
  * [Module contents](domiknows.sensor.pytorch.md#module-domiknows.sensor.pytorch)

## Submodules

## domiknows.sensor.learner module

### *class* domiknows.sensor.learner.Learner(name=None)

Bases: [`Sensor`](#domiknows.sensor.sensor.Sensor)

#### *abstract property* parameters*: Any*

## domiknows.sensor.sensor module

### *class* domiknows.sensor.sensor.Sensor(name=None)

Bases: [`BaseGraphTreeNode`](domiknows.graph.md#domiknows.graph.base.BaseGraphTreeNode)

Represents the bare parent sensor that can update and propagate context of a datanode based on the given data and create new properties.

Inherits from:
- BaseGraphTreeNode: A parent node class that provides basic graph functionalities.

#### forward(data_item: Dict[str, Any])

Computes the forward pass for the given data item. This method should be implemented by subclasses. This function defines how to calcualte the new properties based on the current data.

Args:
- data_item (Dict[str, Any]): The data dictionary to compute the forward pass for.

Raises:
- NotImplementedError: Indicates that subclasses should provide their implementation.

#### propagate_context(data_item, node, force=False)

Propagates the context from this sensor to the given node’s superior, if needed. It ensures the data is consistent 
and updated throughout the graph.

Args:
- data_item (Dict[str, Any]): The data dictionary to propagate context through.
- node (BaseGraphTreeNode): The node to propagate context to.
- force (bool, optional): Flag to force propagation even if the result is cached. Default is False.

#### update_context(data_item: Dict[str, Any], force=False)

Updates the context of the given data item based on this sensor. If forced, or if the result isn’t cached,
it computes the forward pass and caches the result.

Args:
- data_item (Dict[str, Any]): The data dictionary to update.
- force (bool, optional): Flag to force recalculation even if result is cached. Default is False.

## Module contents
