# domiknows.sensor.pytorch.tokenizers package

## Submodules

## domiknows.sensor.pytorch.tokenizers.spacy module

### *class* domiknows.sensor.pytorch.tokenizers.spacy.SpacyTokenizer(\*pres, relation, edges=None, label=False, device='auto', spacy=None)

Bases: [`EdgeSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor)

Tokenize text using spaCy’s English tokenizer.
Inherits from:
- EdgeSensor:  Edge sensor used to create the link/edge between two concept to create relation

#### forward() → Any

Return:
- A list of token after calling spaCy’s English tokenizer.

## domiknows.sensor.pytorch.tokenizers.transformers module

### *class* domiknows.sensor.pytorch.tokenizers.transformers.TokenizerEdgeSensor(\*pres, relation, edges=None, label=False, device='auto', tokenizer=None)

Bases: [`EdgeSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.relation_sensors.EdgeSensor), [`JointSensor`](domiknows.sensor.pytorch.md#domiknows.sensor.pytorch.sensors.JointSensor)

Tokenize text using BERT tokenizer.
Inherits from:
- EdgeSensor:  Edge sensor used to create the link/edge between two concept to create relation

#### forward(text) → Any

Return:
- A list of token after calling BERT tokenizer and list of offset

## Module contents
