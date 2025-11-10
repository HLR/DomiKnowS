# DomiKnows Sensor Components

This directory contains the sensor implementations for the DomiKnows framework, which update and propagate context through the knowledge graph during data processing and model training.

---

## Sensor Hierarchy Overview

| Sensor Class | File | Parent Class | Primary Use Case |
|-------------|------|--------------|------------------|
| **Base Sensors** | | | |
| `Sensor` | `sensor.py` | `BaseGraphTreeNode` | Base sensor for context updates |
| `Learner` | `learner.py` | `Sensor` | Sensor with trainable parameters |
| **PyTorch Base Sensors** | | | |
| `TorchSensor` | `sensors.py` | `Sensor` | PyTorch-based sensor with device management |
| `FunctionalSensor` | `sensors.py` | `TorchSensor` | Sensor with custom forward function |
| `ModuleSensor` | `sensors.py` | `FunctionalSensor` | Wraps a torch.nn.Module |
| **PyTorch Learners** | | | |
| `TorchLearner` | `learners.py` | `Learner`, `FunctionalSensor` | PyTorch learner with parameters |
| `ModuleLearner` | `learners.py` | `ModuleSensor`, `TorchLearner` | Wraps a torch.nn.Module as learner |
| `LSTMLearner` | `learners.py` | `TorchLearner` | LSTM-based learner |
| `FullyConnectedLearner` | `learners.py` | `TorchLearner` | FC layer with softmax |
| `FullyConnectedLearnerRelu` | `learners.py` | `TorchLearner` | FC layer with LeakyReLU |
| **Specialized Sensors** | | | |
| `ConstantSensor` | `sensors.py` | `FunctionalSensor` | Returns constant value |
| `PrefilledSensor` | `sensors.py` | `FunctionalSensor` | Returns pre-filled property value |
| `TriggerPrefilledSensor` | `sensors.py` | `PrefilledSensor` | Triggers callback before returning value |
| `ReaderSensor` | `sensors.py` | `ConstantSensor` | Reads from input data dictionary |
| `FunctionalReaderSensor` | `sensors.py` | `ReaderSensor` | Reads and applies forward function |
| `LabelReaderSensor` | `sensors.py` | `ReaderSensor` | ReaderSensor marked as label |
| `NominalSensor` | `sensors.py` | `TorchSensor` | One-hot encoding sensor |
| `CacheSensor` | `sensors.py` | `FunctionalSensor` | Caches forward results |
| **Joint Sensors** | | | |
| `JointSensor` | `sensors.py` | `FunctionalSensor` | Generates multiple properties |
| `JointReaderSensor` | `sensors.py` | `JointSensor`, `ReaderSensor` | Reads into multiple properties |
| **Utility Sensors** | | | |
| `ConcatSensor` | `sensors.py` | `TorchSensor` | Concatenates tensors |
| `ListConcator` | `sensors.py` | `TorchSensor` | Stacks and concatenates lists |
| **Query Sensors** | | | |
| `QuerySensor` | `query_sensor.py` | `FunctionalSensor` | Queries DataNodes from graph |
| `DataNodeSensor` | `query_sensor.py` | `QuerySensor` | Applies forward to each DataNode |
| `DataNodeReaderSensor` | `query_sensor.py` | `DataNodeSensor`, `FunctionalReaderSensor` | Reads and processes per DataNode |
| **Relation Sensors** | | | |
| `EdgeSensor` | `relation_sensors.py` | `FunctionalSensor` | Creates edges between concepts |
| `BaseCandidateSensor` | `relation_sensors.py` | `QuerySensor` | Base for candidate generation |
| `CandidateSensor` | `relation_sensors.py` | `EdgeSensor`, `BaseCandidateSensor` | Generates relation candidates |
| `CandidateRelationSensor` | `relation_sensors.py` | `CandidateSensor` | Candidate sensor for relations |
| `CandidateEqualSensor` | `relation_sensors.py` | `CandidateSensor` | Candidate sensor for equality |
| `CompositionCandidateSensor` | `relation_sensors.py` | `JointSensor`, `BaseCandidateSensor` | Multi-relation composition |
| `TorchEdgeSensor` | `sensors.py` | `FunctionalSensor` | Edge sensor with directionality |
| **Aggregation Sensors** | | | |
| `AggregationSensor` | `aggregation_sensor.py` | `TorchSensor` | Base edge-based aggregation |
| `MaxAggregationSensor` | `aggregation_sensor.py` | `AggregationSensor` | Max pooling aggregation |
| `MinAggregationSensor` | `aggregation_sensor.py` | `AggregationSensor` | Min pooling aggregation |
| `MeanAggregationSensor` | `aggregation_sensor.py` | `AggregationSensor` | Mean pooling aggregation |
| `ConcatAggregationSensor` | `aggregation_sensor.py` | `AggregationSensor` | Concatenation aggregation |
| `FirstAggregationSensor` | `aggregation_sensor.py` | `AggregationSensor` | First element aggregation |
| `LastAggregationSensor` | `aggregation_sensor.py` | `AggregationSensor` | Last element aggregation |
| **Tokenizer Sensors** | | | |
| `SpacyTokenizer` | `spacy.py` | `EdgeSensor` | spaCy-based tokenization |
| `TokenizerEdgeSensor` | `transformers.py` | `EdgeSensor`, `JointSensor` | BERT tokenization |

### Quick Selection Guide

**Choose based on your needs:**

- **Simple forward computation**: `FunctionalSensor`
- **Trainable parameters**: `TorchLearner` or `ModuleLearner`
- **Read from input data**: `ReaderSensor` or `FunctionalReaderSensor`
- **Multiple outputs**: `JointSensor`
- **Query graph structure**: `QuerySensor` or `DataNodeSensor`
- **Create relations**: `EdgeSensor` or `CandidateSensor`
- **Aggregate over spans**: `MaxAggregationSensor`, `MeanAggregationSensor`, etc.
- **Constant values**: `ConstantSensor`
- **Cache results**: `CacheSensor`

---

## Core Components

### Base Sensor Classes

#### `Sensor` (`sensor.py`)
Foundation class for all sensors that update and propagate context.

**Key Features:**
- Updates DataNode context based on input data
- Caches results to avoid redundant computation
- Propagates context through graph hierarchy
- Force recalculation with `force=True`

**Key Methods:**
```python
sensor = Sensor()

# Call sensor to update context
result = sensor(data_item, force=False)

# Manual context update
sensor.update_context(data_item, force=False)

# Implement in subclass
def forward(self, data_item):
    # Compute new property value
    return computed_value
```

#### `Learner` (`learner.py`)
Extension of `Sensor` with trainable parameters.

**Key Feature:**
```python
class MyLearner(Learner):
    @property
    def parameters(self):
        # Return trainable parameters
        return self.model.parameters()
```

---

## PyTorch Sensor Classes (`sensors.py`)

### `TorchSensor`
Base PyTorch sensor with device management and predecessor handling.

**Key Features:**
- Automatic device placement (CPU/GPU)
- Predecessor sensors updated before forward pass
- Context helper for accessing DataNode state
- Label vs. non-label sensor distinction

**Parameters:**
- `*pres`: Predecessor properties/sensors
- `edges`: Edge sensors to update before forward
- `label`: Mark as label sensor (default: False)
- `device`: 'auto', 'cuda', or 'cpu'

**Usage:**
```python
class CustomSensor(TorchSensor):
    def forward(self):
        # Access predecessor values
        input1 = self.inputs[0]
        input2 = self.inputs[1]
        # Compute result
        return input1 + input2

# Attach to graph
concept['output'] = CustomSensor(concept['input1'], concept['input2'])
```

### `FunctionalSensor`
Most commonly used sensor with custom forward function.

**Usage:**
```python
# Simple function
def add_tensors(x, y):
    return x + y

concept['sum'] = FunctionalSensor(
    concept['a'], 
    concept['b'],
    forward=add_tensors
)

# Lambda function
concept['product'] = FunctionalSensor(
    concept['a'], 
    concept['b'],
    forward=lambda x, y: x * y
)

# With edges
concept['aggregated'] = FunctionalSensor(
    concept['local_feature'],
    edges=[has_a_relation],
    forward=lambda feat: torch.mean(feat, dim=0)
)
```

### `ModuleSensor`
Wraps a `torch.nn.Module` as a sensor.

**Usage:**
```python
import torch.nn as nn

linear = nn.Linear(768, 128)
concept['projection'] = ModuleSensor(
    concept['embeddings'],
    module=linear
)
```

---

## Learner Classes (`learners.py`)

### `TorchLearner`
Base learner with trainable parameters, loss, and metrics.

**Key Features:**
- Manages underlying torch module
- Save/load model checkpoints
- Loss and metric computation
- Parameter updates and device management

**Usage:**
```python
class CustomLearner(TorchLearner):
    def __init__(self, *pres, hidden_dim=128, **kwargs):
        super().__init__(*pres, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self):
        input_tensor = self.inputs[0]
        return self.model(input_tensor)

# Attach with loss and metric
concept['label'] = CustomLearner(
    concept['features'],
    loss=nn.CrossEntropyLoss(),
    metric=accuracy_metric
)

# Save/load
learner.save('checkpoints/')
learner.load('checkpoints/')
```

### `ModuleLearner`
Wraps a complete `torch.nn.Module` as a learner.

**Usage:**
```python
model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

concept['predictions'] = ModuleLearner(
    concept['embeddings'],
    module=model,
    loss=nn.CrossEntropyLoss(),
    metric=f1_metric
)
```

### Pre-built Learners

#### `LSTMLearner`
LSTM-based sequence learner.

**Usage:**
```python
concept['sequence_encoding'] = LSTMLearner(
    concept['token_embeddings'],
    input_dim=768,
    hidden_dim=256,
    num_layers=2,
    bidirectional=True
)
```

#### `FullyConnectedLearner`
FC layer with softmax for classification.

**Usage:**
```python
concept['class_probs'] = FullyConnectedLearner(
    concept['features'],
    input_dim=256,
    output_dim=10
)
```

#### `FullyConnectedLearnerRelu`
FC layer with LeakyReLU activation.

**Usage:**
```python
concept['hidden'] = FullyConnectedLearnerRelu(
    concept['input'],
    input_dim=768,
    output_dim=256
)
```

---

## Data Reading Sensors

### `ReaderSensor`
Reads values from input data dictionary.

**Key Features:**
- Reads by keyword from `data_item`
- Supports tuple keywords for multiple values
- Optional constraint mode (allows missing keys)
- Automatic tensor conversion

**Usage:**
```python
# Single keyword
concept['text'] = ReaderSensor(keyword='input_text')

# Multiple keywords
concept['inputs'] = ReaderSensor(keyword=('text', 'labels'))

# Constraint mode (allow missing)
concept['optional_label'] = ReaderSensor(
    keyword='label',
    is_constraint=True
)

# Usage in data pipeline
data_item = {'input_text': "Hello world", 'labels': [0, 1]}
sensor.fill_data(data_item)  # Reads from dictionary
result = sensor(data_item)    # Returns tensor
```

### `FunctionalReaderSensor`
Combines reading with custom forward function.

**Usage:**
```python
def preprocess_text(text, data):
    # data contains the read value
    return text.lower() if data else text

concept['processed'] = FunctionalReaderSensor(
    concept['text'],
    keyword='needs_lowercase',
    forward=preprocess_text,
    as_tensor=False
)
```

### `LabelReaderSensor`
Shorthand for `ReaderSensor(label=True)`.

**Usage:**
```python
# Mark as label for loss computation
concept['gold_label'] = LabelReaderSensor(keyword='label')
```

---

## Query Sensors (`query_sensor.py`)

### `QuerySensor`
Queries DataNodes from the graph structure.

**Key Features:**
- Retrieves DataNodes matching sensor's concept
- Requires DataNodeBuilder context
- Passes DataNodes to forward as keyword argument
- Must be assigned to a property

**Usage:**
```python
def count_entities(datanodes):
    return len(datanodes)

concept['entity_count'] = QuerySensor(
    forward=count_entities,
    build=True
)

# DataNodes passed automatically to forward function
```

### `DataNodeSensor`
Applies forward function to each DataNode individually.

**Key Features:**
- Iterates over retrieved DataNodes
- Aligns property inputs with DataNodes
- Repeats non-property inputs
- Returns tensor or list

**Usage:**
```python
def classify_word(embedding, threshold, datanode):
    # embedding: aligned with DataNode
    # threshold: repeated for each DataNode
    # datanode: current DataNode
    return (embedding > threshold).float()

word['prediction'] = DataNodeSensor(
    word['embedding'],
    global_threshold,
    forward=classify_word
)
```

### `DataNodeReaderSensor`
Combines DataNode iteration with data reading.

**Usage:**
```python
def process_token(features, datanode, data):
    # features: from DataNode
    # data: read from input dictionary
    return model(features, data)

word['processed'] = DataNodeReaderSensor(
    word['features'],
    keyword='raw_text',
    forward=process_token
)
```

---

## Relation Sensors (`relation_sensors.py`)

### `EdgeSensor`
Creates edges between concepts for relations.

**Key Features:**
- Links source and destination concepts
- Updates relation context
- Fetches values from source concept

**Usage:**
```python
def compute_relation_score(src_emb, dst_emb):
    return torch.dot(src_emb, dst_emb)

work_for['score'] = EdgeSensor(
    person['embedding'],
    organization['embedding'],
    relation=work_for,
    forward=compute_relation_score
)
```

### `CandidateSensor`
Generates candidate relation instances.

**Key Features:**
- Queries all possible (src, dst) pairs
- Applies forward to each candidate
- Returns tensor of shape (|src|, |dst|)

**Usage:**
```python
def should_link(src_emb, dst_emb, src, dst):
    # src, dst are DataNodes
    score = torch.sigmoid(torch.dot(src_emb, dst_emb))
    return (score > 0.5).long()

work_for = CandidateSensor(
    person['embedding'],
    organization['embedding'],
    relation=work_for,
    forward=should_link
)
```

### `CandidateRelationSensor`
Candidate sensor for multiple relations.

**Usage:**
```python
# Generate candidates for multiple relations simultaneously
concept['relations'] = CandidateRelationSensor(
    concept['features'],
    relations=[work_for, located_in],
    forward=classify_relation_type
)
```

### `CandidateEqualSensor`
Generates equality relation candidates.

**Usage:**
```python
def is_equal(emb1, emb2, dn1, dn2):
    return torch.cosine_similarity(emb1, emb2) > 0.9

entity['coreference'] = CandidateEqualSensor(
    entity['embedding'],
    relations=[entity.equal()],
    forward=is_equal
)
```

---

## Aggregation Sensors (`aggregation_sensor.py`)

### `AggregationSensor`
Base class for span-based tensor aggregation.

**Key Features:**
- Aggregates over has_a relations
- Uses span indices (start, end) pairs
- Multiple aggregation strategies
- Fallback to zero tensor if empty

**Common Parameters:**
- `edges`: has_a relation in backward mode
- `map_key`: Field name for source tensor map
- `default_dim`: Fallback dimension for zero tensors
- `device`: Device placement

### Aggregation Types

#### `MaxAggregationSensor`
Element-wise maximum over span.

**Usage:**
```python
# word has_a char relation
# Aggregate character features to word level
word['max_char_features'] = MaxAggregationSensor(
    char['span_indices'],
    edges=[word.has_a(char).reversed],
    map_key='char_embeddings',
    default_dim=768
)
```

#### `MinAggregationSensor`
Element-wise minimum over span.

#### `MeanAggregationSensor`
Element-wise mean over span.

**Usage:**
```python
sentence['avg_word_embedding'] = MeanAggregationSensor(
    word['span_indices'],
    edges=[sentence.has_a(word).reversed],
    map_key='word_embeddings',
    default_dim=768
)
```

#### `ConcatAggregationSensor`
Concatenates tensors along last dimension.

#### `FirstAggregationSensor`
Selects first tensor in span.

**Usage:**
```python
word['first_char'] = FirstAggregationSensor(
    char['span_indices'],
    edges=[word.has_a(char).reversed],
    map_key='char_embeddings',
    default_dim=128
)
```

#### `LastAggregationSensor`
Selects last tensor in span.

---

## Specialized Sensors

### `ConstantSensor`
Returns constant value.

**Usage:**
```python
# Constant threshold
concept['threshold'] = ConstantSensor(data=0.5)

# Constant tensor
concept['zero'] = ConstantSensor(
    data=torch.zeros(10),
    as_tensor=True
)

# Constant list (no conversion)
concept['labels'] = ConstantSensor(
    data=['positive', 'negative'],
    as_tensor=False
)
```

### `PrefilledSensor`
Returns pre-filled property value.

**Usage:**
```python
# Data already filled externally
concept['predictions'] = PrefilledSensor()

# Later, fill the value
data_item[concept['predictions']] = model_output
```

### `TriggerPrefilledSensor`
Triggers callback before returning pre-filled value.

**Usage:**
```python
def log_prediction(data_item):
    print(f"Prediction accessed: {data_item}")

concept['logged_pred'] = TriggerPrefilledSensor(
    callback_sensor=logging_sensor
)
```

### `NominalSensor`
One-hot encodes categorical values.

**Usage:**
```python
class CategorySensor(NominalSensor):
    def __init__(self, *pres, vocab=None, **kwargs):
        super().__init__(*pres, vocab=vocab, **kwargs)
    
    def forward(self):
        category = self.inputs[0]
        return category  # Returns string/int

concept['category'] = CategorySensor(
    concept['raw_category'],
    vocab=['cat_A', 'cat_B', 'cat_C']
)
# Automatically one-hot encoded: [1,0,0], [0,1,0], [0,0,1]
```

### `CacheSensor`
Caches forward computation results.

**Usage:**
```python
# In-memory caching
cache_dict = {}
concept['expensive_computation'] = CacheSensor(
    concept['input'],
    forward=expensive_function,
    cache=cache_dict
)

# Disk-based caching
from domiknows.sensor.pytorch.sensors import TorchCache
disk_cache = TorchCache('cache_dir/')
concept['computed'] = CacheSensor(
    concept['input'],
    forward=expensive_function,
    cache=disk_cache
)

# Set hash for each instance
sensor.fill_hash('unique_instance_id')
result = sensor(data_item)  # Cached by hash
```

---

## Joint Sensors

### `JointSensor`
Generates multiple properties from single computation.

**Key Features:**
- Single forward returns tuple/list
- Automatically creates component sensors
- Optional bundled calls
- Can iterate to create components

**Usage:**
```python
def compute_features(input_data):
    # Returns multiple values
    feature1 = process1(input_data)
    feature2 = process2(input_data)
    return feature1, feature2

# Method 1: Named properties
concept[('feature1', 'feature2')] = JointSensor(
    concept['input'],
    forward=compute_features
)

# Method 2: Iteration
joint = JointSensor(concept['input'], forward=compute_features)
f1, f2 = joint
concept['feature1'] = f1
concept['feature2'] = f2

# Access components
for component in joint:
    print(component)
```

### `JointReaderSensor`
Reads into multiple properties.

**Usage:**
```python
# Read tuple of values
concept[('text', 'label')] = JointReaderSensor(
    keyword=('input_text', 'gold_label')
)
```

---

## Utility Sensors

### `ConcatSensor`
Concatenates predecessor tensors on last dimension.

**Usage:**
```python
concept['combined'] = ConcatSensor(
    concept['feature1'],
    concept['feature2'],
    concept['feature3']
)
# Output shape: [..., dim1 + dim2 + dim3]
```

### `ListConcator`
Stacks lists then concatenates.

**Usage:**
```python
concept['sequence_concat'] = ListConcator(
    concept['token_list1'],
    concept['token_list2']
)
```

---

## Tokenizer Sensors

### `SpacyTokenizer` (`spacy.py`)
spaCy-based tokenization edge sensor.

**Usage:**
```python
from spacy.lang.en import English
nlp = English()

sentence.has_a(token)
sentence[token] = SpacyTokenizer(
    sentence['text'],
    relation=sentence.has_a(token),
    spacy=nlp
)
```

### `TokenizerEdgeSensor` (`transformers.py`)
BERT tokenizer edge sensor.

**Usage:**
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence.has_a(token)
sentence[(token, 'offsets')] = TokenizerEdgeSensor(
    sentence['text'],
    relation=sentence.has_a(token),
    tokenizer=tokenizer
)
# Returns: (token_ids, offset_mapping)
```

---

## Common Patterns

### Basic Sensor Chain
```python
# 1. Read input
concept['raw_text'] = ReaderSensor(keyword='text')

# 2. Preprocess
concept['lowercase'] = FunctionalSensor(
    concept['raw_text'],
    forward=lambda x: x.lower(),
    as_tensor=False
)

# 3. Embed
concept['embedding'] = ModuleLearner(
    concept['lowercase'],
    module=embedding_model
)

# 4. Classify
concept['prediction'] = FullyConnectedLearner(
    concept['embedding'],
    input_dim=768,
    output_dim=10,
    loss=nn.CrossEntropyLoss()
)
```

### Relation Detection
```python
# Detect work_for relations
person.has_a(organization, name='work_for')

work_for['exists'] = CandidateSensor(
    person['embedding'],
    organization['embedding'],
    relation=work_for,
    forward=lambda p_emb, o_emb, p, o: 
        torch.sigmoid(torch.dot(p_emb, o_emb)) > 0.5
)
```

### Hierarchical Aggregation
```python
# Sentence > Word > Char hierarchy
sentence.has_a(word)
word.has_a(char)

# Char-level features
char['embedding'] = CharEmbedding(char['raw'])

# Aggregate to word level
word['char_features'] = MaxAggregationSensor(
    char['span_indices'],
    edges=[word.has_a(char).reversed],
    map_key='char_embeddings'
)

# Word-level features
word['combined'] = ConcatSensor(
    word['char_features'],
    word['word_embedding']
)

# Aggregate to sentence level
sentence['features'] = MeanAggregationSensor(
    word['span_indices'],
    edges=[sentence.has_a(word).reversed],
    map_key='word_embeddings'
)
```

### Query-Based Features
```python
def count_coref_mentions(datanodes):
    return len(datanodes)

entity['mention_count'] = QuerySensor(
    forward=count_coref_mentions,
    build=True
)

def avg_mention_confidence(confidences, datanodes):
    return torch.mean(torch.stack([
        conf for conf, dn in zip(confidences, datanodes)
    ]))

entity['avg_confidence'] = DataNodeSensor(
    entity['confidence'],
    forward=avg_mention_confidence
)
```

### Custom Learner with Multiple Outputs
```python
class MultiTaskLearner(TorchLearner):
    def __init__(self, *pres, **kwargs):
        super().__init__(*pres, **kwargs)
        self.shared = nn.Linear(768, 256)
        self.task1_head = nn.Linear(256, 10)
        self.task2_head = nn.Linear(256, 5)
    
    def forward(self):
        x = self.inputs[0]
        shared_rep = self.shared(x)
        task1_out = self.task1_head(shared_rep)
        task2_out = self.task2_head(shared_rep)
        return task1_out, task2_out

concept[('task1', 'task2')] = MultiTaskLearner(
    concept['features']
)
```

---

## Device Management

All PyTorch sensors support device placement:

```python
# Auto-detect
sensor = TorchSensor(device='auto')

# Specific device
sensor = TorchSensor(device='cuda:0')
sensor = TorchSensor(device='cpu')

# Check device
print(sensor.device)  # cuda:0
```

---

## Context Flow

**Sensor Lifecycle:**

1. **Call**: `sensor(data_item, force=False)`
2. **Update predecessors**: `update_pre_context(data_item)`
3. **Define inputs**: `define_inputs()` → fills `self.inputs`
4. **Forward**: `forward()` → computes result
5. **Cache**: Store in `data_item[sensor]`
6. **Propagate**: Update property and hierarchy

**Example Trace:**
```python
# Graph structure
concept['a'] = ReaderSensor(keyword='input_a')
concept['b'] = FunctionalSensor(concept['a'], forward=lambda x: x * 2)
concept['c'] = FunctionalSensor(concept['b'], forward=lambda x: x + 1)

# Call chain
data_item = {'input_a': 5}
result = concept['c'](data_item)

# Execution:
# 1. concept['c'] called
# 2. Updates concept['b'] (predecessor)
#    - concept['b'] updates concept['a'] (predecessor)
#      - concept['a'].forward() reads 'input_a' → 5
#      - Caches: data_item[concept['a']] = 5
#    - concept['b'].forward(5) → 10
#    - Caches: data_item[concept['b']] = 10
# 3. concept['c'].forward(10) → 11
# 4. Caches: data_item[concept['c']] = 11
# 5. Returns: 11
```

---

## Sensor Factories

### `joint()` Factory
Create joint sensor variants:

```python
from domiknows.sensor.pytorch.sensors import joint, FunctionalSensor

JointFunctionalSensor = joint(FunctionalSensor)

# Use like JointSensor but with FunctionalSensor features
concept[('out1', 'out2')] = JointFunctionalSensor(
    concept['input'],
    forward=lambda x: (x, x*2)
)
```

### `cache()` Factory
Add caching to any sensor:

```python
from domiknows.sensor.pytorch.sensors import cache, ModuleLearner

CachedModuleLearner = cache(ModuleLearner)

concept['cached_output'] = CachedModuleLearner(
    concept['input'],
    module=expensive_model,
    cache=TorchCache('cache/')
)
```

---

## Best Practices

1. **Use FunctionalSensor for simple logic**: Avoid subclassing for one-off operations
2. **Leverage predecessors**: Let the framework handle update order
3. **Mark labels explicitly**: Use `label=True` for loss computation sensors
4. **Cache expensive operations**: Use `CacheSensor` for repeated computations
5. **Read data once**: Use `ReaderSensor` at graph entry points
6. **Aggregate hierarchically**: Use aggregation sensors for span-based features
7. **Query when needed**: Use `QuerySensor` for graph-structure-dependent features
8. **Save/load learners**: Use `save()`/`load()` for model persistence

---

## Debugging

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Sensor operations logged to console
sensor(data_item)
```

### Inspect Context
```python
# After sensor call
print(data_item.keys())  # All cached sensors
print(data_item[sensor])  # Specific sensor output
print(sensor.inputs)      # Predecessor values
```

### Force Recalculation
```python
# Bypass cache
sensor(data_item, force=True)
```

### Check Device Placement
```python
result = sensor(data_item)
if torch.is_tensor(result):
    print(f"Result device: {result.device}")
```

---

## Common Issues

### "Sensor must be assigned to property"
```python
# Wrong
sensor = FunctionalSensor(forward=lambda x: x)

# Correct
concept['property'] = FunctionalSensor(forward=lambda x: x)
```

### "Context helper not set"
```python
# Wrong - calling forward directly
sensor.forward()

# Correct - call sensor with data_item
sensor(data_item)
```

### "Circular dependencies"
```python
# Wrong - circular reference
concept['a'] = FunctionalSensor(concept['b'], forward=lambda x: x)
concept['b'] = FunctionalSensor(concept['a'], forward=lambda x: x)

# Solution - use intermediate sensor or restructure
```

### "Device mismatch"
```python
# Ensure all tensors on same device
sensor = TorchSensor(device='cuda:0')
# Sensors automatically move inputs to device in forward_wrap()

# For manual tensor operations
value = value.to(device=self.device)
```

### "ReaderSensor keyword not found"
```python
# Wrong - missing keyword in data_item
sensor = ReaderSensor(keyword='missing_key')
sensor(data_item)  # KeyError

# Solution 1 - Use constraint mode
sensor = ReaderSensor(keyword='missing_key', is_constraint=True)

# Solution 2 - Ensure data_item contains key
data_item = {'missing_key': value}
sensor.fill_data(data_item)
sensor(data_item)
```

### "QuerySensor requires DataNodeBuilder"
```python
# Wrong - using QuerySensor without build mode
sensor = QuerySensor(forward=lambda dns: len(dns))

# Correct - enable build mode in program
program.train(data, build=True)
# Or when calling populate
program.populate(data, build=True)
```

---

## Advanced Features

### Custom Context Propagation

Override `propagate_context` for custom behavior:

```python
class CustomPropagationSensor(TorchSensor):
    def propagate_context(self, data_item, node, force=False):
        # Custom propagation logic
        if node.sup is not None:
            # Apply transformation before propagating
            transformed = self.transform(data_item[self])
            data_item[node.sup] = transformed
            super().propagate_context(data_item, node.sup, force)
```

### Dynamic Predecessor Selection

```python
class DynamicSensor(FunctionalSensor):
    def define_inputs(self):
        # Select predecessors dynamically
        if some_condition:
            self.inputs = [self.fetch_value(self.pres[0])]
        else:
            self.inputs = [self.fetch_value(self.pres[1])]
```

### Stateful Sensors

```python
class StatefulSensor(FunctionalSensor):
    def __init__(self, *pres, **kwargs):
        super().__init__(*pres, **kwargs)
        self.state = []
    
    def forward(self, x):
        self.state.append(x)
        # Use accumulated state
        return torch.mean(torch.stack(self.state))
    
    def reset_state(self):
        self.state = []
```

### Sensor Composition

```python
def compose_sensors(*sensors):
    """Chain multiple sensors sequentially"""
    def composed_forward(*inputs):
        result = inputs[0]
        for sensor in sensors:
            result = sensor.forward(result)
        return result
    
    return FunctionalSensor(
        *sensors[0].pres,
        forward=composed_forward
    )

# Usage
preprocessing = compose_sensors(
    lowercase_sensor,
    tokenize_sensor,
    encode_sensor
)
concept['processed'] = preprocessing
```

### Conditional Execution

```python
def conditional_forward(x, condition):
    if condition:
        return expensive_computation(x)
    else:
        return cheap_computation(x)

concept['adaptive'] = FunctionalSensor(
    concept['input'],
    concept['should_use_expensive'],
    forward=conditional_forward
)
```

---

## Integration with Models

### Using Sensors in Models

```python
from domiknows.model import PoiModel

# Define sensors
concept['features'] = ReaderSensor(keyword='input_features')
concept['prediction'] = FullyConnectedLearner(
    concept['features'],
    input_dim=768,
    output_dim=10,
    loss=nn.CrossEntropyLoss(),
    metric=accuracy_metric
)

# Create model
model = PoiModel(
    graph,
    poi=[concept['prediction']]  # Points of interest
)

# Training
for batch in dataloader:
    loss, metrics, datanode, builder = model(batch)
    loss.backward()
    optimizer.step()
```

### Sensors with Constraints

```python
# Define logical constraint
from domiknows.graph import V, ifL, andL

ifL(
    work_for(V.x, V.y),
    andL(person(V.x), organization(V.y))
)

# Sensors provide values for constraint evaluation
person['prediction'] = FullyConnectedLearner(...)
organization['prediction'] = FullyConnectedLearner(...)
work_for['prediction'] = CandidateSensor(...)

# Model automatically evaluates constraints
model = SolverModel(graph, inferTypes=['ILP'])
```

### Sensors in Inference

```python
# During inference, sensors populate DataNode
for data_item in test_data:
    # Sensors update context
    datanode = program.populate_single(data_item, build=True)
    
    # Access results
    predictions = datanode[concept['prediction']]
    features = datanode[concept['features']]
```

---

## Performance Optimization

### Minimize Redundant Computation

```python
# Use caching for expensive operations
cache_dict = {}
concept['expensive'] = CacheSensor(
    concept['input'],
    forward=expensive_function,
    cache=cache_dict
)

# Set unique hash per instance
for idx, data_item in enumerate(dataset):
    concept['expensive'].fill_hash(f'instance_{idx}')
    result = concept['expensive'](data_item)
```

### Batch Operations

```python
# Process multiple items efficiently
def batch_forward(inputs_list):
    # Stack inputs
    batched = torch.stack(inputs_list)
    # Batch computation
    results = model(batched)
    # Unstack results
    return list(results)

concept['batched'] = FunctionalSensor(
    concept['inputs'],
    forward=batch_forward
)
```

### Lazy Evaluation

```python
# Only compute when accessed
class LazySensor(FunctionalSensor):
    def update_context(self, data_item, force=False):
        # Don't compute unless forced or property accessed
        if force or self.prop not in data_item:
            super().update_context(data_item, force)
```

### Memory-Efficient Aggregation

```python
# For large sequences, use chunked aggregation
class ChunkedAggregationSensor(AggregationSensor):
    def forward(self):
        results = []
        chunk_size = 1000
        for i in range(0, len(self.data), chunk_size):
            chunk = self.data[i:i+chunk_size]
            results.append(torch.mean(torch.cat(chunk, dim=0), dim=0))
        return torch.stack(results)
```

---

## Testing Sensors

### Unit Testing

```python
import unittest
import torch

class TestCustomSensor(unittest.TestCase):
    def setUp(self):
        self.graph = Graph('test')
        self.concept = Concept('test_concept')
        self.graph.add_concept(self.concept)
    
    def test_forward(self):
        # Define sensor
        self.concept['output'] = FunctionalSensor(
            self.concept['input'],
            forward=lambda x: x * 2
        )
        
        # Create data_item
        data_item = {self.concept['input']: torch.tensor([1, 2, 3])}
        
        # Call sensor
        result = self.concept['output'](data_item)
        
        # Assert
        expected = torch.tensor([2, 4, 6])
        self.assertTrue(torch.equal(result, expected))
    
    def test_caching(self):
        # Test that results are cached
        call_count = [0]
        
        def counting_forward(x):
            call_count[0] += 1
            return x * 2
        
        self.concept['cached'] = FunctionalSensor(
            self.concept['input'],
            forward=counting_forward
        )
        
        data_item = {self.concept['input']: torch.tensor([1])}
        
        # First call
        result1 = self.concept['cached'](data_item)
        self.assertEqual(call_count[0], 1)
        
        # Second call (should use cache)
        result2 = self.concept['cached'](data_item)
        self.assertEqual(call_count[0], 1)
        self.assertTrue(torch.equal(result1, result2))
        
        # Force recalculation
        result3 = self.concept['cached'](data_item, force=True)
        self.assertEqual(call_count[0], 2)
```

### Integration Testing

```python
def test_sensor_chain():
    # Setup graph
    graph = Graph('test')
    concept = Concept('entity')
    graph.add_concept(concept)
    
    # Define sensor chain
    concept['raw'] = ReaderSensor(keyword='text')
    concept['lower'] = FunctionalSensor(
        concept['raw'],
        forward=lambda x: x.lower(),
        as_tensor=False
    )
    concept['length'] = FunctionalSensor(
        concept['lower'],
        forward=lambda x: len(x)
    )
    
    # Test
    data_item = {'text': 'HELLO'}
    concept['raw'].fill_data(data_item)
    
    result = concept['length'](data_item)
    assert result == 5
    assert data_item[concept['lower']] == 'hello'
```

---

## Migration Guide

### From Legacy torch.py Sensors

```python
# Old (torch.py)
from domiknows.sensor.torch import TorchSensor

# New (pytorch.py)
from domiknows.sensor.pytorch import TorchSensor
```

### From Custom Sensor to FunctionalSensor

```python
# Old - custom class
class AddOneSensor(TorchSensor):
    def forward(self):
        return self.inputs[0] + 1

concept['output'] = AddOneSensor(concept['input'])

# New - FunctionalSensor
concept['output'] = FunctionalSensor(
    concept['input'],
    forward=lambda x: x + 1
)
```

### From Manual Context Update to Sensor

```python
# Old - manual update
def update_features(data_item):
    input_val = data_item['input']
    data_item['features'] = compute_features(input_val)

# New - sensor-based
concept['features'] = FunctionalSensor(
    concept['input'],
    forward=compute_features
)
```

---

## Examples

### Example 1: Named Entity Recognition

```python
# Graph structure
graph = Graph('ner')
sentence = Concept('sentence')
word = Concept('word')
sentence.has_a(word)

# Input sensors
sentence['text'] = ReaderSensor(keyword='text')
word['text'] = ReaderSensor(keyword='words')

# Embeddings
word['embedding'] = ModuleLearner(
    word['text'],
    module=embedding_model
)

# Sentence context (BiLSTM)
sentence['context'] = LSTMLearner(
    word['embedding'],
    input_dim=300,
    hidden_dim=256,
    bidirectional=True
)

# Word-level features
word['features'] = ConcatSensor(
    word['embedding'],
    word['contextual']  # From sentence context
)

# NER predictions
word['ner_tag'] = FullyConnectedLearner(
    word['features'],
    input_dim=300 + 512,
    output_dim=9,  # B-PER, I-PER, B-ORG, ...
    loss=nn.CrossEntropyLoss()
)

# Labels
word['gold_tag'] = LabelReaderSensor(keyword='ner_labels')
```

### Example 2: Relation Extraction

```python
# Graph structure
graph = Graph('relation_extraction')
sentence = Concept('sentence')
entity = Concept('entity')
relation = Relation('work_for')

sentence.has_a(entity)
entity.has_a(entity, name='work_for')

# Input
sentence['text'] = ReaderSensor(keyword='text')
entity['mention'] = ReaderSensor(keyword='mentions')

# Entity representations
entity['span_start'] = ReaderSensor(keyword='span_start')
entity['span_end'] = ReaderSensor(keyword='span_end')

# BERT embeddings
sentence['bert_output'] = ModuleLearner(
    sentence['text'],
    module=bert_model
)

# Extract entity spans from BERT output
def extract_span(bert_out, start, end):
    return bert_out[start:end+1].mean(dim=0)

entity['embedding'] = FunctionalSensor(
    sentence['bert_output'],
    entity['span_start'],
    entity['span_end'],
    forward=extract_span
)

# Relation candidate generation
def compute_relation_prob(e1_emb, e2_emb, e1, e2):
    combined = torch.cat([e1_emb, e2_emb, e1_emb * e2_emb], dim=-1)
    score = relation_classifier(combined)
    return torch.sigmoid(score) > 0.5

relation['prediction'] = CandidateSensor(
    entity['embedding'],
    entity['embedding'],
    relation=relation,
    forward=compute_relation_prob
)

# Labels
relation['gold'] = LabelReaderSensor(keyword='relations')
```

### Example 3: Hierarchical Text Classification

```python
# Graph structure
graph = Graph('hierarchical_classification')
document = Concept('document')
paragraph = Concept('paragraph')
sentence = Concept('sentence')
word = Concept('word')

document.has_a(paragraph)
paragraph.has_a(sentence)
sentence.has_a(word)

# Word level
word['text'] = ReaderSensor(keyword='words')
word['embedding'] = ModuleLearner(
    word['text'],
    module=word_embedding
)

# Sentence level (aggregate words)
sentence['word_features'] = MeanAggregationSensor(
    word['span_indices'],
    edges=[sentence.has_a(word).reversed],
    map_key='word_embeddings',
    default_dim=300
)

sentence['encoding'] = LSTMLearner(
    sentence['word_features'],
    input_dim=300,
    hidden_dim=256,
    bidirectional=True
)

# Paragraph level (aggregate sentences)
paragraph['sentence_features'] = MeanAggregationSensor(
    sentence['span_indices'],
    edges=[paragraph.has_a(sentence).reversed],
    map_key='sentence_encodings',
    default_dim=512
)

paragraph['encoding'] = LSTMLearner(
    paragraph['sentence_features'],
    input_dim=512,
    hidden_dim=256,
    bidirectional=True
)

# Document level (aggregate paragraphs)
document['paragraph_features'] = MeanAggregationSensor(
    paragraph['span_indices'],
    edges=[document.has_a(paragraph).reversed],
    map_key='paragraph_encodings',
    default_dim=512
)

document['representation'] = LSTMLearner(
    document['paragraph_features'],
    input_dim=512,
    hidden_dim=256,
    bidirectional=True
)

# Classification
document['category'] = FullyConnectedLearner(
    document['representation'],
    input_dim=512,
    output_dim=20,  # 20 categories
    loss=nn.CrossEntropyLoss()
)

document['gold_category'] = LabelReaderSensor(keyword='label')
```

### Example 4: Multi-Task Learning

```python
# Graph structure
graph = Graph('multitask')
sentence = Concept('sentence')
word = Concept('word')
sentence.has_a(word)

# Shared representations
word['text'] = ReaderSensor(keyword='words')
word['embedding'] = ModuleLearner(
    word['text'],
    module=shared_embedding
)

word['shared_features'] = ModuleLearner(
    word['embedding'],
    module=shared_encoder
)

# Task 1: POS Tagging
word['pos_tag'] = FullyConnectedLearner(
    word['shared_features'],
    input_dim=256,
    output_dim=45,  # Penn Treebank tagset
    loss=nn.CrossEntropyLoss()
)
word['gold_pos'] = LabelReaderSensor(keyword='pos_labels')

# Task 2: NER
word['ner_tag'] = FullyConnectedLearner(
    word['shared_features'],
    input_dim=256,
    output_dim=9,  # BIO scheme
    loss=nn.CrossEntropyLoss()
)
word['gold_ner'] = LabelReaderSensor(keyword='ner_labels')

# Task 3: Dependency Parsing (as relation)
dependency = Relation('dependency')
word.has_a(word, name='dependency')

def compute_dependency(head_feat, dep_feat, head, dep):
    combined = torch.cat([head_feat, dep_feat], dim=-1)
    return dependency_scorer(combined)

dependency['score'] = CandidateSensor(
    word['shared_features'],
    word['shared_features'],
    relation=dependency,
    forward=compute_dependency
)
dependency['gold'] = LabelReaderSensor(keyword='dependencies')
```

---

## Sensor Patterns Library

### Pattern: Feature Fusion

```python
# Combine multiple feature sources
concept['multi_modal'] = ConcatSensor(
    concept['text_features'],
    concept['image_features'],
    concept['metadata_features']
)

concept['fused'] = ModuleLearner(
    concept['multi_modal'],
    module=fusion_network
)
```

### Pattern: Attention Mechanism

```python
def attention_forward(query, keys, values):
    scores = torch.matmul(query, keys.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, values)

concept['attended'] = FunctionalSensor(
    concept['query'],
    concept['keys'],
    concept['values'],
    forward=attention_forward
)
```

### Pattern: Residual Connection

```python
def residual_forward(input_feat, transformed_feat):
    return input_feat + transformed_feat

concept['residual'] = FunctionalSensor(
    concept['input'],
    concept['transformed'],
    forward=residual_forward
)
```

### Pattern: Gating Mechanism

```python
def gated_forward(input1, input2):
    gate = torch.sigmoid(gate_network(input1))
    return gate * input1 + (1 - gate) * input2

concept['gated'] = FunctionalSensor(
    concept['input1'],
    concept['input2'],
    forward=gated_forward
)
```

### Pattern: Dropout at Inference

```python
class DropoutSensor(FunctionalSensor):
    def __init__(self, *pres, dropout_rate=0.5, **kwargs):
        super().__init__(*pres, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        if self.model.training:
            return self.dropout(x)
        return x

concept['dropout'] = DropoutSensor(
    concept['features'],
    dropout_rate=0.3
)
```

### Pattern: Ensemble Predictions

```python
def ensemble_forward(*predictions):
    # Average multiple model predictions
    return torch.mean(torch.stack(predictions), dim=0)

concept['ensemble'] = FunctionalSensor(
    concept['model1_pred'],
    concept['model2_pred'],
    concept['model3_pred'],
    forward=ensemble_forward
)
```

---

## Summary

The DomiKnows sensor framework provides:

- **Modularity**: Compose complex pipelines from simple sensors
- **Automatic updates**: Dependency-based execution order
- **Caching**: Avoid redundant computation
- **Device management**: Seamless CPU/GPU handling
- **Graph integration**: Natural fit with knowledge graph structure
- **Flexibility**: From simple functions to complex learnable models

**Key Takeaways:**

1. Sensors update DataNode context based on predecessors
2. Use `FunctionalSensor` for most custom logic
3. `TorchLearner` for trainable components
4. `ReaderSensor` for input data
5. Aggregation sensors for hierarchical structures
6. Query sensors for graph-structure-dependent features
7. Relation sensors for linking concepts

For more examples and detailed tutorials, see the DomiKnows documentation and examples directory.