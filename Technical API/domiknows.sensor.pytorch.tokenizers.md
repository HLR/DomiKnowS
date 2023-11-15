# domiknows.sensor.pytorch.tokenizers package

## Submodules

## domiknows.sensor.pytorch.tokenizers.spacy module

### *class* domiknows.sensor.pytorch.tokenizers.spacy.SpacyTokenizer(\*pres, relation, edges=None, label=False, device='auto', spacy=None)

Bases: [`EdgeSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor)

#### forward()

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

## domiknows.sensor.pytorch.tokenizers.transformers module

### *class* domiknows.sensor.pytorch.tokenizers.transformers.TokenizerEdgeSensor(\*pres, relation, edges=None, label=False, device='auto', tokenizer=None)

Bases: [`EdgeSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor), [`JointSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.JointSensor)

#### forward(text)

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

## Module contents
