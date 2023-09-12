# domiknows.sensor package

## Subpackages

* [domiknows.sensor.allennlp package](domiknows.sensor.allennlp.md)
  * [Submodules](domiknows.sensor.allennlp.md#submodules)
  * [domiknows.sensor.allennlp.base module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-base-module)
  * [domiknows.sensor.allennlp.learner module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-learner-module)
  * [domiknows.sensor.allennlp.module module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-module-module)
  * [domiknows.sensor.allennlp.sensor module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-sensor-module)
  * [Module contents](domiknows.sensor.allennlp.md#module-contents)
* [domiknows.sensor.pytorch package](domiknows.sensor.pytorch.md)
  * [Subpackages](domiknows.sensor.pytorch.md#subpackages)
    * [domiknows.sensor.pytorch.tokenizers package](domiknows.sensor.pytorch.tokenizers.md)
      * [Submodules](domiknows.sensor.pytorch.tokenizers.md#submodules)
      * [domiknows.sensor.pytorch.tokenizers.spacy module](domiknows.sensor.pytorch.tokenizers.md#domiknows-sensor-pytorch-tokenizers-spacy-module)
      * [domiknows.sensor.pytorch.tokenizers.transformers module](domiknows.sensor.pytorch.tokenizers.md#domiknows-sensor-pytorch-tokenizers-transformers-module)
      * [Module contents](domiknows.sensor.pytorch.tokenizers.md#module-contents)
  * [Submodules](domiknows.sensor.pytorch.md#submodules)
  * [domiknows.sensor.pytorch.learnerModels module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-learnermodels-module)
  * [domiknows.sensor.pytorch.learners module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-learners-module)
  * [domiknows.sensor.pytorch.query_sensor module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-query-sensor-module)
  * [domiknows.sensor.pytorch.relation_sensors module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-relation-sensors-module)
  * [domiknows.sensor.pytorch.sensors module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-sensors-module)
  * [domiknows.sensor.pytorch.utils module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-utils-module)
  * [Module contents](domiknows.sensor.pytorch.md#module-contents)
* [domiknows.sensor.torch package](domiknows.sensor.torch.md)
  * [Submodules](domiknows.sensor.torch.md#submodules)
  * [domiknows.sensor.torch.learner module](domiknows.sensor.torch.md#domiknows-sensor-torch-learner-module)
  * [domiknows.sensor.torch.sensor module](domiknows.sensor.torch.md#domiknows-sensor-torch-sensor-module)
  * [Module contents](domiknows.sensor.torch.md#module-domiknows.sensor.torch)

## Submodules

## domiknows.sensor.learner module

### *class* domiknows.sensor.learner.Learner(name=None)

Bases: [`Sensor`](#domiknows.sensor.sensor.Sensor)

#### *abstract property* parameters*: Any*

## domiknows.sensor.sensor module

### *class* domiknows.sensor.sensor.Sensor(name=None)

Bases: `BaseGraphTreeNode`

#### forward(data_item: Dict[str, Any])

#### propagate_context(data_item, node, force=False)

#### update_context(data_item: Dict[str, Any], force=False)

## Module contents
