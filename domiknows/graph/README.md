# DataNode and Data Graph Components

This directory contains the runtime data structure components for DomiKnows, enabling instance-level data binding to ontological concepts and logical constraint execution.

---

## Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| **Core Data Structure** | | |
| `DataNode` | `dataNode.py` | Runtime instance data bound to ontological concepts |
| `DataNodeBuilder` | `dataNode.py` | Constructs data graphs from sensor outputs during model execution |
| **Candidate Selection** | | |
| `CandidateSelection` | `candidates.py` | Base class for logical constraint candidate selection |
| `combinationC` | `candidates.py` | Cartesian product candidate generation |
| **Utilities** | | |
| `dataNodeConfig` | `dataNodeConfig.py` | Logging configuration for DataNode components |
| `dataNodeDummy` | `dataNodeDummy.py` | Dummy DataNode generation for testing |

---

## DataNode Class (`dataNode.py`)

### Overview

`DataNode` represents a single data instance in a graph structure, bound to an ontological concept from the knowledge graph. Each DataNode stores:

- **Instance data**: ID, value, attributes
- **Graph links**: Relations to other DataNodes
- **Ontology binding**: Reference to concept/relation type
- **Inference results**: Local and global predictions
```
┌─────────────────────────────────────────────┐
│  Knowledge Graph (Ontology)                 │
│  ┌─────────────────────────────────────┐    │
│  │ Concept: Person                     │    │
│  │ - properties: [age, name]           │    │
│  │ - relations: [work_for]             │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                    │
                    │ binds to
                    └─────────────────
┌─────────────────────────────────────────────┐
│  Data Graph (Runtime)                       │
│  ┌─────────────────────────────────────┐    │
│  │ DataNode: Person                    │    │
│  │ - instanceID: 0                     │    │
│  │ - attributes: {age: 25}             │    │
│  │ - relationLinks: {work_for: [...]}  │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Initialization

```python
from domiknows.graph import DataNode, Concept

# Create concept
person = Concept('person')

# Create DataNode
dn = DataNode(
    myBuilder=builder,           # DataNodeBuilder instance
    instanceID=0,                # Unique ID for this instance
    instanceValue="John",        # Optional value
    ontologyNode=person,         # Concept binding
    graph=graph,                 # Graph reference
    relationLinks={},            # Relations to other nodes
    attributes={}                # Attribute dictionary
)
```

### Key Attributes

```python
dn.id                    # Unique numerical ID
dn.instanceID            # Instance identifier (e.g., token index)
dn.instanceValue         # Optional value (e.g., text, image)
dn.ontologyNode          # Concept/Relation from ontology
dn.graph                 # Containing graph
dn.relationLinks         # Dict: relation_name -> [DataNode]
dn.impactLinks           # Dict: incoming relations
dn.attributes            # Dict: attribute_name -> value
dn.current_device        # 'cpu' or 'cuda'
dn.myBuilder             # DataNodeBuilder reference
```

---

## DataNode Methods

### Instance Information

#### `getInstanceID()`
```python
# Get instance identifier
node_id = dn.getInstanceID()
# Returns: 0, 1, 2, ... (unique per concept type)
```

#### `getInstanceValue()`
```python
# Get instance value
text = dn.getInstanceValue()
# Returns: "John" or other stored value
```

#### `getOntologyNode()`
```python
# Get bound concept
concept = dn.getOntologyNode()
# Returns: Concept instance
```

### Attribute Access

#### `getAttribute(*keys)`
```python
# Simple attribute
age = dn.getAttribute('age')

# Concept attribute (binary)
person_prob = dn.getAttribute('person')
# Returns: tensor([0.2, 0.8])  # [not person, person]

# Inference result
ilp_result = dn.getAttribute('person', 'ILP')
local_result = dn.getAttribute('person', 'local', 'softmax')

# Nested keys
value = dn.getAttribute('sentence', 'token', 'raw')
```

**Attribute Key Formats:**

```python
# Concept predictions: '<concept_name>'
dn.attributes['<person>'] = tensor([0.2, 0.8])

# Inference results: '<concept>/method'
dn.attributes['<person>/ILP'] = tensor(1.0)
dn.attributes['<person>/local/softmax'] = tensor([0.1, 0.9])
dn.attributes['<person>/local/argmax'] = tensor([0., 1.])

# Labels: '<concept>/label'
dn.attributes['<person>/label'] = tensor(1)

# Properties: 'property_name'
dn.attributes['age'] = 25
dn.attributes['name'] = "John"
```

#### `getAttributes()`
```python
# Get all attributes
attrs = dn.getAttributes()
# Returns: dict of all attributes
```

#### `hasAttribute(key)`
```python
# Check if attribute exists
if dn.hasAttribute('<person>/ILP'):
    result = dn.getAttribute('<person>/ILP')
```

### Relation Access

#### `getRelationLinks(relationName=None, conceptName=None)`
```python
# Get all relation links
all_relations = dn.getRelationLinks()
# Returns: {'work_for': [org_dn], 'located_in': [loc_dn]}

# Get specific relation
organizations = dn.getRelationLinks('work_for')
# Returns: [org_dn1, org_dn2, ...]

# Filter by concept type
persons = dn.getRelationLinks('work_for', 'person')
# Returns: [person_dn1, ...]
```

#### `addRelationLink(relationName, dn)`
```python
# Add relation to another DataNode
person_dn.addRelationLink('work_for', org_dn)

# Automatically updates impactLinks:
# org_dn.impactLinks['work_for'] = [person_dn]
```

#### `removeRelationLink(relationName, dn)`
```python
# Remove relation
person_dn.removeRelationLink('work_for', org_dn)
```

#### `getLinks(relationName=None, conceptName=None)`
```python
# Get all links (relationLinks + impactLinks)
all_links = dn.getLinks()

# Get specific relation (both directions)
related = dn.getLinks('work_for')
# Returns: DataNodes where dn is source OR target
```

### Containment Hierarchy

#### `getChildDataNodes(conceptName=None)`
```python
# Get all children
children = sentence_dn.getChildDataNodes()

# Get children of specific type
tokens = sentence_dn.getChildDataNodes('token')
# Returns: [token_dn1, token_dn2, ...]
```

#### `addChildDataNode(dn)`
```python
# Add child (creates 'contains' relation)
sentence_dn.addChildDataNode(token_dn)
```

#### `removeChildDataNode(dn)`
```python
# Remove child
sentence_dn.removeChildDataNode(token_dn)
```

#### `resetChildDataNode()`
```python
# Remove all children
sentence_dn.resetChildDataNode()
```

#### `getRootDataNode()`
```python
# Traverse up to root
root = dn.getRootDataNode()
# Returns: Top-level DataNode with no parent
```

### Equality Relations

#### `getEqualTo(equalName="equalTo", conceptName=None)`
```python
# Get equal DataNodes
equals = dn.getEqualTo()

# Filter by concept type
person_equals = dn.getEqualTo(conceptName='person')
```

#### `addEqualTo(equalDn, equalName="equalTo")`
```python
# Mark as equal
dn1.addEqualTo(dn2)
```

#### `removeEqualTo(equalDn, equalName="equalTo")`
```python
# Remove equality
dn1.removeEqualTo(dn2)
```

---

## DataNode Query Methods

### `findDatanodes(dns=None, select=None, indexes=None, visitedDns=None, depth=0)`

Powerful query method to search the data graph.

**Parameters:**
- `dns`: Starting DataNodes (defaults to [self])
- `select`: Selection criteria
- `indexes`: Filter by related DataNodes
- `visitedDns`: Tracking visited nodes (internal)
- `depth`: Recursion depth (internal)

**Select Formats:**

```python
# 1. By concept name (string)
words = root_dn.findDatanodes(select='word')

# 2. By concept object
words = root_dn.findDatanodes(select=word_concept)

# 3. By instance ID
node = root_dn.findDatanodes(select=5)

# 4. By attribute value (tuple)
john = root_dn.findDatanodes(select=('person', 'name', 'John'))
# Format: (concept, *attribute_keys, value)

# 5. Multiple criteria (nested tuples)
result = root_dn.findDatanodes(select=(
    ('person', 'age', 25),
    ('person', 'name', 'John')
))
```

**Index Filters:**

```python
# Filter by related DataNodes
# Find words containing specific char
words = root_dn.findDatanodes(
    select='word',
    indexes={'contains': ('char', 'raw', 'J')}
)

# Multiple related nodes
words = root_dn.findDatanodes(
    select='word',
    indexes={
        'contains': (
            ('char', 'raw', 'o'),
            ('char', 'raw', 'h')
        )
    }
)

# By relation and instance ID
pairs = root_dn.findDatanodes(
    select='pair',
    indexes={'arg1': 0, 'arg2': 3}
)

# By relation and attribute
pairs = root_dn.findDatanodes(
    select='pair',
    indexes={
        'arg1': ('word', 'raw', 'John'),
        'arg2': ('word', 'raw', 'IBM')
    }
)
```

**Examples:**

```python
# Find all persons
persons = root_dn.findDatanodes(select='person')

# Find person named "John"
john = root_dn.findDatanodes(
    select=('person', 'name', 'John')
)

# Find sentences containing specific token
sentences = root_dn.findDatanodes(
    select='sentence',
    indexes={'contains': ('token', 'text', 'important')}
)

# Find work_for relations between specific entities
relations = root_dn.findDatanodes(
    select='work_for',
    indexes={
        'arg1': ('person', 'name', 'Alice'),
        'arg2': ('organization', 'name', 'Anthropic')
    }
)
```

### `findConcept(conceptName, usedGraph=None)`

Find concept definition in ontology graph.

```python
# Find by name
concept = dn.findConcept('person')
# Returns: (Concept, name, index, multiplicity)

# For EnumConcept
sentiment = dn.findConcept('sentiment')
# Returns: (EnumConcept, 'positive', 0, 3) for specific value
```

### `findRootConceptOrRelation(relationConcept, usedGraph=None)`

Find root concept in hierarchy.

```python
# Get root of subclass
root = dn.findRootConceptOrRelation('employee')
# Returns: person (if employee.is_a(person))

# Traverses is_a hierarchy to root
```

### `isRelation(conceptRelation, usedGraph=None)`

Check if concept is a relation.

```python
# Check if relation
if dn.isRelation('work_for'):
    print("Is a relation")
# Returns: True/False
```

### `collectConceptsAndRelations(conceptsAndRelations=None)`

Collect all concepts/relations in data graph.

```python
# Get all concepts used
concepts = dn.collectConceptsAndRelations()
# Returns: [
#     (Concept, name, index, multiplicity),
#     ...
# ]

# For binary: (Concept, name, None, 1)
# For multiclass: (EnumConcept, value, index, len)
```

---

## Inference Methods

### Local Inference

#### `inferLocal(keys=("softmax", "argmax"), Acc=None)`

Compute local predictions (softmax, argmax) for all concepts.

```python
# Compute softmax and argmax
dn.inferLocal()

# Access results
softmax = dn.getAttribute('<person>/local/softmax')
# Returns: tensor([0.1, 0.9])

argmax = dn.getAttribute('<person>/local/argmax')
# Returns: tensor([0., 1.])

# Custom keys
dn.inferLocal(keys=('softmax',))

# With accuracy weights
dn.inferLocal(Acc={'person': 0.95})
```

**Available Keys:**
- `softmax`: Standard softmax normalization
- `argmax`: Hard assignment (one-hot)
- `normalizedProb`: Entropy-weighted probabilities
- `normalizedProbAcc`: Accuracy-weighted probabilities
- `meanNormalizedProb`: Mean-normalized probabilities

#### `inferGumbelLocal(temperature=1.0, hard=False)`

Apply Gumbel-Softmax for differentiable discrete sampling.

```python
# Soft sampling
dn.inferGumbelLocal(temperature=1.0, hard=False)

# Hard sampling with straight-through estimator
dn.inferGumbelLocal(temperature=0.5, hard=True)

# Results stored in local/softmax attributes
gumbel_probs = dn.getAttribute('<person>/local/softmax')
```

### Global Inference

#### `infer()`

Compute global argmax and softmax.

```python
# Compute global inference
dn.infer()

# Access results
softmax = dn.getAttribute('<person>/softmax')
argmax = dn.getAttribute('<person>/argmax')
```

#### `inferILPResults(*conceptsRelations, key=("local", "softmax"), ...)`

Integer Linear Programming inference with logical constraints.

```python
# ILP inference
dn.inferILPResults(
    person, organization, work_for,
    key=("local", "softmax"),
    epsilon=0.00001,
    minimizeObjective=False
)

# Access ILP results
ilp_person = dn.getAttribute('<person>/ILP')
ilp_work_for = dn.getAttribute('<work_for>/ILP')
```

**Parameters:**
- `*conceptsRelations`: Concepts to include (empty = all)
- `key`: Source probabilities ("local"/"ILP", "softmax"/"argmax")
- `fun`: Custom objective function
- `epsilon`: Tolerance for solver
- `minimizeObjective`: Minimize vs maximize
- `ignorePinLCs`: Ignore fixed constraints
- `Acc`: Accuracy weights per concept

#### `inferGBIResults(*conceptsRelations, model, kwargs)`

Grounded Belief Inference.

```python
# GBI inference
dn.inferGBIResults(
    person, organization,
    model=gbi_model,
    kwargs={'iterations': 10}
)
```

### Constraint Loss

#### `calculateLcLoss(tnorm='P', counting_tnorm=None, sample=False, ...)`

Compute differentiable constraint loss.

```python
# Compute LC loss
lc_losses = dn.calculateLcLoss(
    tnorm='P',              # Product t-norm
    counting_tnorm='L',     # Åukasiewicz for counting
    sample=False,
    sampleSize=0
)

# Returns: dict of losses per constraint
# {
#     'LC0': {
#         'loss': tensor(0.123),
#         'satisfaction': tensor(0.877),
#         ...
#     },
#     ...
# }
```

**T-norms:**
- `'P'`: Product (default)
- `'G'`: Gödel (minimum)
- `'L'`: Åukasiewicz

**Sampling:**
- `sample=True`: Enable sampling for large groundings
- `sampleSize`: Number of samples (-1 = semantic sampling)
- `sampleGlobalLoss`: Compute global loss in sampling mode

### Verification

#### `verifyResultsLC(key="/local/argmax")`

Verify constraint satisfaction.

```python
# Verify constraints
results = dn.verifyResultsLC(key='/ILP')

# Returns: dict per constraint
# {
#     'LC0': {
#         'satisfied': 0.95,  # 95% satisfied
#         'violations': 5,
#         'total': 100
#     },
#     ...
# }
```

---

## Metrics

### `getInferMetrics(*conceptsRelations, inferType='ILP', weight=None, average='binary')`

Calculate precision, recall, F1 for predictions.

```python
# Get metrics for ILP inference
metrics = dn.getInferMetrics(
    person, organization,
    inferType='ILP',
    average='binary'
)

# Returns: dict per concept
# {
#     'person': {
#         'TP': 85, 'FP': 10, 'TN': 90, 'FN': 15,
#         'P': 0.894,   # Precision
#         'R': 0.850,   # Recall
#         'F1': 0.872,  # F1 score
#         'confusion_matrix': [[90, 10], [15, 85]],
#         'labels': array([...]),
#         'preds': array([...])
#     },
#     'organization': {...},
#     'Total': {
#         'TP': 170, 'FP': 20, 'TN': 180, 'FN': 30,
#         'P': 0.895, 'R': 0.850, 'F1': 0.872
#     }
# }
```

**Parameters:**
- `*conceptsRelations`: Concepts to evaluate (empty = all)
- `inferType`: 'ILP', 'local', 'argmax', 'softmax'
- `weight`: Tensor of weights per sample
- `average`: 'binary', 'micro', 'macro', None

### `collectInferredResults(concept, inferKey)`

Collect predictions across all instances.

```python
# Collect ILP predictions for person
preds = dn.collectInferredResults(person, 'ILP')
# Returns: tensor([1, 0, 1, 1, 0, ...])

# Collect labels
labels = dn.collectInferredResults(person, 'label')

# Collect local softmax
probs = dn.collectInferredResults(person, 'local/softmax')
```

---

## Visualization

### `visualize(filename, inference_mode="ILP", include_legend=False, open_image=False)`

Create Graphviz visualization of DataNode.

```python
# Visualize single DataNode
dn.visualize(
    'output/datanode',
    inference_mode='ILP',
    include_legend=True,
    open_image=True
)
```

**Visualization includes:**
- Root node (concept name)
- Attributes (rectangles)
- Decisions (diamonds) with predictions
- Inference results

**Example Output:**
```
┌──────────────────┐
│    Person        │
└───────┬──────────┘
        │
   ┌────┴────┐
   │         │
  ┌┴─────┐  ◊──────┐
  │ age  │  │ work_for │
  │  25  │  │ label=1  │
  └──────┘  │ pred=0.85│
            └──────────┘
```

## DataNodeBuilder Class (`dataNode.py`)

### Overview

`DataNodeBuilder` constructs data graphs dynamically during model execution. It intercepts sensor outputs and builds the DataNode structure automatically.

```python
from domiknows.graph import DataNodeBuilder, Graph

# Create builder
builder = DataNodeBuilder()
builder['graph'] = graph

# Sensors write to builder
builder[sensor] = predictions  # Triggers DataNode creation

# Get resulting DataNode
root_dn = builder.getDataNode()
```

### Key Features

1. **Automatic Graph Construction**: Builds DataNode graph from sensor outputs
2. **Skeleton Mode**: Fast mode that delays full construction
3. **Incremental Building**: Handles streaming data
4. **Relation Discovery**: Automatically creates relation links

### Modes

#### Full Mode (default)
```python
builder = DataNodeBuilder()
# Immediately creates full DataNode structure
```

#### Skeleton Mode
```python
from domiknows.utils import setDnSkeletonMode

setDnSkeletonMode(True)
builder = DataNodeBuilder()
# Creates minimal structure, defers details
```

#### Full Skeleton Mode
```python
from domiknows.utils import setDnSkeletonModeFull

setDnSkeletonModeFull(True)
builder = DataNodeBuilder()
# Records operations, builds on demand
```

### Sensor Integration

```python
from domiknows.sensor.pytorch.sensors import ReaderSensor

# Define sensor
sensor = ReaderSensor(
    concept=person,
    keyword='person_logits',
    build=True  # Enable DataNode building
)

# During forward pass
predictions = model(inputs)

# Sensor writes to builder
sensor_context[sensor] = predictions
# Automatically:
# 1. Creates DataNodes for person concept
# 2. Stores predictions in attributes
# 3. Links to related DataNodes
```

### Builder Methods

#### `__setitem__(key, value)`

Core method for DataNode construction (automatically called by sensors).

```python
# Called automatically by sensors
builder[person_sensor] = person_predictions
builder[work_for_sensor] = relation_predictions

# Triggers:
# 1. Concept identification
# 2. DataNode creation/update
# 3. Relation link creation
# 4. Attribute storage
```

#### `getDataNode(context="interference", device='auto')`

Retrieve root DataNode.

```python
# Get root DataNode
root = builder.getDataNode(
    context="interference",
    device='cuda'
)

# Returns first root DataNode
# Sets device for all tensor operations
```

#### `getBatchDataNodes()`

Retrieve all root DataNodes (for batch processing).

```python
# Get all roots
roots = builder.getBatchDataNodes()
# Returns: [root_dn1, root_dn2, ...]
```

#### `createBatchRootDN()`

Create batch-level root DataNode.

```python
# Combine multiple roots into batch
builder.createBatchRootDN()

# Creates 'batch' concept as root
# All previous roots become children
```

#### `createFullDataNode(rootDataNode)`

Build full DataNode from skeleton.

```python
# In skeleton mode, build on demand
builder.createFullDataNode(root_dn)

# Processes queued operations
# Populates all attributes and links
```

#### `findDataNodesInBuilder(select=None, indexes=None)`

Query builder's DataNodes.

```python
# Find specific DataNodes
persons = builder.findDataNodesInBuilder(select='person')
```

### Builder Attributes

```python
builder['graph']          # Graph reference
builder['dataNode']       # List of root DataNodes
builder['variableSet']    # Set of variable keys (skeleton mode)
builder['propertySet']    # Set of property keys (skeleton mode)
builder['KeysInOrder']    # Ordered list of sensor keys (full skeleton)
builder['Counter_setitem']  # Number of __setitem__ calls
builder['DataNodeTime']   # Time per operation (ns)
```

---

## Candidate Selection (`candidates.py`)

### `CandidateSelection`

Base class for selecting candidates when grounding logical constraints.

```python
from domiknows.graph.candidates import CandidateSelection

class MySelection(CandidateSelection):
    def __call__(self, candidates_list, keys=None):
        # candidates_list: list of candidate lists per variable
        # keys: variable names
        # Return: dict mapping keys to selected candidates
        pass
```

### `combinationC`

Cartesian product of all candidates.

```python
from domiknows.graph.candidates import combinationC

# Use in logical constraint
selection = combinationC()

# Generates all combinations of candidates
# For variables x, y, z with candidates:
# x: [x1, x2]
# y: [y1, y2]
# z: [z1]
# Returns: {
#     'x': [x1, x1, x2, x2],
#     'y': [y1, y2, y1, y2],
#     'z': [z1, z1, z1, z1]
# }
```

### `getCandidates(dn, e, variable, lcVariablesDns, lc, logger, integrate=False)`

Internal function to collect candidates for constraint variables.

**Process:**
1. Identify variable concept
2. Find path if specified
3. Collect matching DataNodes
4. Filter by path constraints
5. Return candidate lists

**Path Handling:**
```python
# Simple variable (no path)
# Variable: person(x)
# Returns: all person DataNodes

# Variable with path
# Variable: organization(y, path=(x, work_for))
# Returns: organizations related to x via work_for

# Multiple paths (intersection)
# Variable: person(x, path=((x, rel1), (x, rel2)))
# Returns: persons satisfying both paths
```

---

## Configuration (`dataNodeConfig.py`)

```python
import logging

dnConfig = {
    'ifLog': True,                          # Enable logging
    'log_name': 'dataNode',                 # Logger name
    'log_level': logging.INFO,              # Log level
    'log_filename': 'logs/datanode.log',    # Log file
    'log_filesize': 5*1024*1024*1024,       # 5GB max
    'log_backupCount': 0,                   # No rotation
    'log_fileMode': 'w'                     # Overwrite mode
}
```

**Usage:**
```python
from domiknows.graph.dataNodeConfig import dnConfig

# Modify configuration
dnConfig['log_level'] = logging.DEBUG

# Access logger
from domiknows.graph.dataNode import _DataNode__Logger

_DataNode__Logger.info("Custom log message")
```

---

## Dummy DataNode Generation (`dataNodeDummy.py`)

### `createDummyDataNode(graph)`

Generate dummy DataNode structure for testing.

```python
from domiknows.graph.dataNodeDummy import createDummyDataNode

# Create test DataNode
dummy_dn = createDummyDataNode(graph)

# Returns:
# - Root DataNode with random predictions
# - Full hierarchy based on graph structure
# - Relations populated
# - Random probability distributions
```

**Generated Structure:**
```python
# For each concept in graph:
# - Creates dataSizeInit * level instances
# - Assigns sequential instanceIDs
# - Creates containment hierarchy
# - Populates relation links
# - Generates random predictions

# Example:
# Root: 1 instance
# Level 1 (children): 5 instances each
# Level 2 (grandchildren): 25 instances each
```

### `satisfactionReportOfConstraints(dn)`

Generate detailed constraint satisfaction report.

```python
from domiknows.graph.dataNodeDummy import satisfactionReportOfConstraints

# Generate report
report = satisfactionReportOfConstraints(dn)

# Returns: dict per constraint
# {
#     'LC0': {
#         'lcResult': tensor([True, False, ...]),
#         'lcVariables': {'x': [...], 'y': [...]},
#         'lcs': {...},
#         'lcSatisfactionMsgs': {
#             'Satisfied': ["LC0 is satisfied because..."],
#             'NotSatisfied': ["LC0 is Not satisfied because..."]
#         }
#     }
# }
```

**Report Format:**
```
LC0 is satisfied (True) because:
    person(x) -> 1.0
    entity(x, path='work_for') -> 1.0

LC1 is Not satisfied (False) because:
    work_for(pair) -> 1.0
    person(x, path='pair, work_for') -> 0.0
    When in ifL the premise is True, the conclusion should also be True
```

---

## Common Workflows

### 1. Basic DataNode Creation

```python
from domiknows.graph import Graph, Concept, DataNode

# Create graph
with Graph('kg') as graph:
    person = Concept('person')
    organization = Concept('organization')

# Create DataNodes
person_dn = DataNode(
    instanceID=0,
    instanceValue="Alice",
    ontologyNode=person
)

org_dn = DataNode(
    instanceID=0,
    instanceValue="Anthropic",
    ontologyNode=organization
)

# Add attributes
person_dn.attributes['age'] = 25
person_dn.attributes['<person>'] = torch.tensor([0.1, 0.9])

# Create relation
person_dn.addRelationLink('work_for', org_dn)
```

### 2. Hierarchical DataNode Structure

```python
# Create sentence DataNode
sentence_dn = DataNode(
    instanceID=0,
    ontologyNode=sentence
)

# Create token DataNodes
for i, token_text in enumerate(tokens):
    token_dn = DataNode(
        instanceID=i,
        instanceValue=token_text,
        ontologyNode=token
    )
    
    # Add to hierarchy
    sentence_dn.addChildDataNode(token_dn)
    
    # Add attributes
    token_dn.attributes['<token>'] = predictions[i]

# Query children
all_tokens = sentence_dn.getChildDataNodes()
specific_token = sentence_dn.getChildDataNodes('token')
```

### 3. DataNode Queries

```python
# Find all persons
persons = root_dn.findDatanodes(select='person')

# Find person by attribute
alice = root_dn.findDatanodes(
    select=('person', 'name', 'Alice')
)

# Find with relation filter
employees = root_dn.findDatanodes(
    select='person',
    indexes={
        'work_for': ('organization', 'name', 'Anthropic')
    }
)

# Complex query
senior_employees = root_dn.findDatanodes(
    select=(
        ('person', 'age', lambda age: age > 30),
        ('person', 'role', 'senior')
    ),
    indexes={
        'work_for': 'organization'
    }
)
```

### 4. Inference Pipeline

```python
# 1. Compute local predictions
root_dn.inferLocal(keys=('softmax', 'argmax'))

# 2. Access local results
for person_dn in root_dn.findDatanodes(select='person'):
    softmax = person_dn.getAttribute('<person>/local/softmax')
    argmax = person_dn.getAttribute('<person>/local/argmax')
    print(f"Person {person_dn.instanceID}: {softmax}")

# 3. Apply ILP inference
root_dn.inferILPResults(
    person, organization, work_for,
    key=('local', 'softmax')
)

# 4. Get ILP results
for person_dn in root_dn.findDatanodes(select='person'):
    ilp_pred = person_dn.getAttribute('<person>/ILP')
    print(f"ILP prediction: {ilp_pred}")

# 5. Verify constraints
verification = root_dn.verifyResultsLC(key='/ILP')
for lc_name, results in verification.items():
    print(f"{lc_name}: {results['satisfied']:.2%} satisfied")

# 6. Calculate metrics
metrics = root_dn.getInferMetrics(
    person, organization,
    inferType='ILP'
)
print(f"Person F1: {metrics['person']['F1']:.3f}")
```

### 5. Constraint Loss Training

```python
# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch)
        
        # Get DataNode (created by sensors)
        root_dn = batch['_dataNode']
        
        # Compute local inference
        root_dn.inferLocal()
        
        # Calculate constraint loss
        lc_losses = root_dn.calculateLcLoss(
            tnorm='P',
            counting_tnorm='L',
            sample=True,
            sampleSize=100
        )
        
        # Aggregate losses
        data_loss = criterion(predictions, labels)
        constraint_loss = sum(
            lc['loss'] for lc in lc_losses.values()
        )
        
        total_loss = data_loss + 0.5 * constraint_loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 6. DataNodeBuilder with Sensors

```python
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.graph import DataNodeBuilder

# Create graph and sensors
with Graph('kg') as graph:
    person = Concept('person')
    
    with person:
        person_sensor = ReaderSensor(
            keyword='person_logits',
            build=True
        )

# Create builder
builder = DataNodeBuilder()
builder['graph'] = graph

# Model forward pass
logits = model(inputs)

# Write to builder (triggers DataNode creation)
builder[person_sensor] = logits

# Get DataNode
root_dn = builder.getDataNode()

# Access predictions
for dn in root_dn.findDatanodes(select='person'):
    pred = dn.getAttribute('<person>')
    print(pred)
```

### 7. Skeleton Mode for Efficiency

```python
from domiknows.utils import setDnSkeletonMode

# Enable skeleton mode
setDnSkeletonMode(True)

# Create builder
builder = DataNodeBuilder()
builder['graph'] = graph

# Fast building during forward pass
builder[sensor1] = predictions1
builder[sensor2] = predictions2
# ... (minimal work done)

# Get skeleton DataNode
root_dn = builder.getDataNode()

# Build full structure when needed
builder.createFullDataNode(root_dn)

# Now all attributes populated
full_pred = root_dn.getAttribute('<person>')
```

### 8. Batch Processing

```python
# Process batch of DataNodes
roots = []

for item in batch:
    # Create builder per item
    builder = DataNodeBuilder()
    builder['graph'] = graph
    
    # Build DataNode
    # ... (sensor writes)
    
    root = builder.getDataNode()
    roots.append(root)

# Create batch root
batch_builder = DataNodeBuilder()
batch_builder['graph'] = graph
batch_builder['dataNode'] = roots
batch_builder.createBatchRootDN()

# Get batch DataNode
batch_root = batch_builder.getDataNode()

# Process batch inference
batch_root.inferILPResults(person, organization)
```

### 9. Relation Discovery

```python
# DataNodeBuilder automatically discovers relations

# 1. Sensor provides relation predictions
# work_for_predictions[i, j] = 1 means person i works for org j

builder[work_for_sensor] = work_for_predictions
# Shape: [num_persons, num_orgs]

# 2. Builder creates relation DataNodes
relation_dns = builder.findDataNodesInBuilder(select='work_for')

# 3. Builder links to source/destination
for rel_dn in relation_dns:
    # Get related DataNodes
    person = rel_dn.getRelationLinks('arg1')[0]
    org = rel_dn.getRelationLinks('arg2')[0]
    
    print(f"{person.instanceValue} works for {org.instanceValue}")
```

### 10. Custom Candidate Selection

```python
from domiknows.graph.candidates import CandidateSelection

class TopKSelection(CandidateSelection):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
    
    def __call__(self, candidates_list, keys=None):
        # Select top-k candidates by confidence
        result = {}
        
        for key, candidates in zip(keys, candidates_list):
            # Sort by prediction confidence
            scored = [
                (dn, dn.getAttribute('<person>')[1].item())
                for dn in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k
            result[key] = [dn for dn, _ in scored[:self.k]]
        
        return result

# Use in constraint
selection = TopKSelection(k=10)
# Apply during constraint grounding
```

---

## Advanced Features

### 1. Multi-level Hierarchies

```python
# Document > Paragraph > Sentence > Token

# Create hierarchy
doc_dn = DataNode(instanceID=0, ontologyNode=document)

for p_idx in range(num_paragraphs):
    para_dn = DataNode(instanceID=p_idx, ontologyNode=paragraph)
    doc_dn.addChildDataNode(para_dn)
    
    for s_idx in range(num_sentences):
        sent_dn = DataNode(instanceID=s_idx, ontologyNode=sentence)
        para_dn.addChildDataNode(sent_dn)
        
        for t_idx in range(num_tokens):
            token_dn = DataNode(instanceID=t_idx, ontologyNode=token)
            sent_dn.addChildDataNode(token_dn)

# Query across levels
all_tokens = doc_dn.findDatanodes(select='token')
# Returns: all tokens in all sentences in all paragraphs
```

### 2. Dynamic Attribute Access

```python
# Attributes support nested paths
dn.attributes['sentence/token/embedding'] = embeddings

# Access with path
embedding = dn.getAttribute('sentence', 'token', 'embedding')

# Concept-based keys
dn.attributes['<person>'] = person_logits
dn.attributes['<person>/local/softmax'] = softmax_result
dn.attributes['<person>/ILP'] = ilp_result

# Access concept predictions
pred = dn.getAttribute(person, 'ILP')
# Automatically resolves to '<person>/ILP'
```

### 3. Variable Sets (Skeleton Mode)

```python
# In skeleton mode, predictions stored in variableSet
setDnSkeletonMode(True)

# Builder creates root with variableSet
root_dn.attributes['variableSet'] = {
    'person/<person>': person_predictions,  # [batch, 2]
    'organization/<organization>': org_predictions,  # [batch, 2]
    'work_for/<work_for>': relation_predictions  # [batch, concepts]
}

# Individual DataNodes reference root
token_dn.attributes['rootDataNode'] = root_dn

# Access via root
pred = token_dn.getAttribute('person')
# Resolves to: root.variableSet['person/<person>'][token_dn.instanceID]
```

### 4. Equality Chains

```python
# Create equality relations
person_dn1.addEqualTo(person_dn2)
person_dn2.addEqualTo(person_dn3)

# Query equality
all_equal = person_dn1.getEqualTo()
# Returns: [person_dn2, person_dn3]

# Transitive equality handled by graph
# person_dn1 == person_dn2 == person_dn3
```

### 5. Impact Links Tracking

```python
# Relations automatically create impact links

# Forward relation
person_dn.addRelationLink('work_for', org_dn)

# Backward tracking (impact)
# org_dn.impactLinks['work_for'] now contains [person_dn]

# Query both directions
all_related = person_dn.getLinks('work_for')
# Returns: relationLinks + impactLinks
```

### 6. Custom Inference Keys

```python
# Define custom inference methods
dn.inferLocal(keys=('custom_softmax',))

# Implement custom normalization
for concept_dn in dn.findDatanodes(select='person'):
    logits = concept_dn.getAttribute('<person>')
    
    # Custom normalization
    custom_probs = my_normalization(logits)
    
    # Store with custom key
    concept_dn.attributes['<person>/custom_softmax'] = custom_probs

# Use in ILP
dn.inferILPResults(person, key=('custom_softmax',))
```

### 7. Temporal DataNodes

```python
# Create temporal sequence
sequence_dn = DataNode(instanceID=0, ontologyNode=sequence)

for t in range(num_timesteps):
    state_dn = DataNode(
        instanceID=t,
        ontologyNode=state
    )
    sequence_dn.addChildDataNode(state_dn)
    
    # Add temporal attributes
    state_dn.attributes['timestep'] = t
    state_dn.attributes['<state>'] = state_predictions[t]
    
    # Link to previous state
    if t > 0:
        prev_state = sequence_dn.getChildDataNodes()[t-1]
        state_dn.addRelationLink('follows', prev_state)

# Query temporal patterns
first_state = sequence_dn.findDatanodes(
    select=('state', 'timestep', 0)
)
```

### 8. Confidence Weighting

```python
# Weight predictions by confidence
for dn in root_dn.findDatanodes(select='person'):
    probs = dn.getAttribute('<person>/local/softmax')
    
    # Calculate confidence (entropy)
    confidence = 1 - torch.distributions.Categorical(probs).entropy()
    
    # Store confidence
    dn.attributes['confidence'] = confidence

# Use in metrics
metrics = root_dn.getInferMetrics(
    person,
    inferType='ILP',
    weight=torch.tensor([
        dn.attributes['confidence']
        for dn in root_dn.findDatanodes(select='person')
    ])
)
```

### 9. Debugging DataNodes

```python
# Enable detailed logging
from domiknows.graph.dataNodeConfig import dnConfig
import logging

dnConfig['log_level'] = logging.DEBUG

# Visualize DataNode
dn.visualize('debug/datanode', open_image=True)

# Inspect attributes
print("Attributes:", dn.getAttributes().keys())

# Inspect relations
print("Relations:", dn.getRelationLinks().keys())

# Check hierarchy
print("Children:", len(dn.getChildDataNodes()))
print("Parent:", 'contains' in dn.impactLinks)

# Verify structure
root = dn.getRootDataNode()
print("Root concept:", root.getOntologyNode().name)

# Check all concepts used
concepts = dn.collectConceptsAndRelations()
print("Concepts:", [c[1] for c in concepts])
```

### 10. Performance Optimization

```python
# Use skeleton mode for large datasets
setDnSkeletonMode(True)

# Defer full construction
builder = DataNodeBuilder()
# ... (fast building)
root_dn = builder.getDataNode()

# Compute metrics on skeleton
# (works with variableSet)
metrics = root_dn.getInferMetrics(person)

# Build full only when needed
# (e.g., for visualization or detailed analysis)
if need_full_structure:
    builder.createFullDataNode(root_dn)

# Clear caches periodically
if epoch % 10 == 0:
    DataNode.clear()
    DataNodeBuilder.clear()
```

---

## Integration with Other Components

### 1. With Sensors

```python
from domiknows.sensor.pytorch.sensors import ReaderSensor, EdgeSensor

# Define sensors
with graph:
    person = Concept('person')
    organization = Concept('organization')
    
    with person:
        # Node sensor
        person_sensor = ReaderSensor(
            keyword='person_features',
            build=True
        )
    
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
    
    # Relation sensor
    work_for_sensor = EdgeSensor(
        src=person,
        dst=organization,
        relation=work_for,
        keyword='work_for_edges',
        build=True
    )

# Sensors automatically build DataNodes
builder[person_sensor] = person_predictions
builder[work_for_sensor] = edge_predictions

# Retrieve DataNode
root_dn = builder.getDataNode()
```

### 2. With Solvers

```python
from domiknows.solver import ilpOntSolverFactory

# Create solver
solver = ilpOntSolverFactory.getOntSolverInstance(graph)

# ILP inference
solver.calculateILPSelection(
    root_dn,
    person, organization, work_for
)

# Constraint loss
lc_losses = solver.calculateLcLoss(
    root_dn,
    tnorm='P'
)

# Verification
results = solver.verifyResultsLC(root_dn, key='/ILP')
```

### 3. With Models

```python
from domiknows.program.model import SampleLossModel

# Create model
model = SampleLossModel(
    graph=graph,
    sensors=[person_sensor, org_sensor]
)

# Forward pass creates DataNode
output = model(batch)

# Access DataNode
root_dn = model.get_datanode(batch)

# Compute constraint loss
lc_loss = model.get_constraint_loss(batch)
```

### 4. With DataLoader

```python
from torch.utils.data import DataLoader

def collate_with_datanode(batch):
    # Standard collation
    collated = default_collate(batch)
    
    # Build DataNode
    builder = DataNodeBuilder()
    builder['graph'] = graph
    
    # Process batch
    # ... (sensor writes)
    
    # Add DataNode to batch
    collated['_dataNode'] = builder.getDataNode()
    
    return collated

# Use custom collate
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_with_datanode
)

# Training
for batch in dataloader:
    root_dn = batch['_dataNode']
    # ... (use DataNode)
```

---

## Error Handling and Debugging

### Common Errors

#### 1. Missing Ontology Node
```python
# Error: DataNode created without ontology binding
dn = DataNode(instanceID=0)
# AttributeError: 'NoneType' object has no attribute 'name'

# Fix: Always provide ontologyNode
dn = DataNode(
    instanceID=0,
    ontologyNode=person_concept
)
```

#### 2. Attribute Key Mismatch
```python
# Error: Wrong attribute key format
value = dn.getAttribute('person')  # Missing <> for concept
# Returns: None

# Fix: Use correct format
value = dn.getAttribute('<person>')  # Concept
value = dn.getAttribute('age')       # Property
```

#### 3. Relation Not Found
```python
# Error: Querying non-existent relation
orgs = dn.getRelationLinks('work_for')
# Returns: []

# Debug: Check available relations
print(dn.getRelationLinks().keys())

# Fix: Ensure relation was created
dn.addRelationLink('work_for', org_dn)
```

#### 4. Builder Graph Missing
```python
# Error: Builder without graph reference
builder = DataNodeBuilder()
builder[sensor] = predictions
# Error: Cannot find graph

# Fix: Set graph first
builder['graph'] = graph
builder[sensor] = predictions
```

#### 5. Skeleton Mode Mismatch
```python
# Error: Accessing full attributes in skeleton mode
setDnSkeletonMode(True)
builder = DataNodeBuilder()
# ... (build skeleton)
root_dn = builder.getDataNode()

# This returns None:
value = root_dn.findDatanodes(select='person')[0].getAttribute('<person>')

# Fix: Build full DataNode
builder.createFullDataNode(root_dn)
# Now accessible:
value = root_dn.findDatanodes(select='person')[0].getAttribute('<person>')

# Or access via variableSet:
value = root_dn.attributes['variableSet']['person/<person>'][0]
```

### Debugging Tools

```python
# 1. Enable logging
from domiknows.graph.dataNodeConfig import dnConfig
import logging

dnConfig['log_level'] = logging.DEBUG
dnConfig['ifLog'] = True

# 2. Visualize structure
dn.visualize('debug/structure', include_legend=True, open_image=True)

# 3. Print hierarchy
def print_hierarchy(dn, indent=0):
    print("  " * indent + str(dn))
    for child in dn.getChildDataNodes():
        print_hierarchy(child, indent + 1)

print_hierarchy(root_dn)

# 4. Check attributes
print("Attributes:", list(dn.getAttributes().keys()))
print("Relations:", list(dn.getRelationLinks().keys()))
print("Impact:", list(dn.impactLinks.keys()))

# 5. Verify concepts
concepts = dn.collectConceptsAndRelations()
print("Concepts in graph:")
for c in concepts:
    print(f"  {c[1]} (multiplicity: {c[3]})")

# 6. Test queries
result = dn.findDatanodes(select='person')
print(f"Found {len(result)} person DataNodes")

# 7. Check builder state
print("Builder keys:", list(builder.keys()))
print("Builder dataNodes:", builder.get('dataNode', []))
```

---

## Best Practices

### 1. DataNode Naming
```python
# Use descriptive instance values
person_dn = DataNode(
    instanceID=0,
    instanceValue="Alice Johnson",  # Helpful for debugging
    ontologyNode=person
)
```

### 2. Attribute Organization
```python
# Group related attributes
dn.attributes['<person>'] = predictions
dn.attributes['<person>/label'] = label
dn.attributes['<person>/local/softmax'] = softmax
dn.attributes['<person>/local/argmax'] = argmax
dn.attributes['<person>/ILP'] = ilp_result

# Separate properties
dn.attributes['age'] = 25
dn.attributes['name'] = "Alice"
```

### 3. Hierarchy Management
```python
# Always maintain clear hierarchy
# Root -> Level1 -> Level2 -> ...

# Bad: Circular references
child.addChildDataNode(parent)  # Don't!

# Good: Clear parent-child
parent.addChildDataNode(child)

# Check hierarchy
root = dn.getRootDataNode()
assert root.impactLinks.get('contains') is None
```

### 4. Relation Consistency
```python
# Use addRelationLink (handles bidirectional)
person_dn.addRelationLink('work_for', org_dn)

# Don't manually modify:
# person_dn.relationLinks['work_for'] = [org_dn]  # Bad!
# Must also update org_dn.impactLinks

# Remove properly
person_dn.removeRelationLink('work_for', org_dn)
```

### 5. Builder Usage
```python
# One builder per data item
for item in batch:
    builder = DataNodeBuilder()
    builder['graph'] = graph
    
    # Process item
    # ... (sensor writes)
    
    item['_dataNode'] = builder.getDataNode()

# Or batch root for entire batch
batch_builder = DataNodeBuilder()
batch_builder.createBatchRootDN()
```

### 6. Memory Management
```python
# Clear caches periodically
if epoch % 10 == 0:
    DataNode.clear()

# Use skeleton mode for large datasets
setDnSkeletonMode(True)

# Avoid storing large tensors in attributes
# Store references instead
dn.attributes['embedding_idx'] = idx
# embeddings[idx] accessed elsewhere
```

### 7. Query Optimization
```python
# Cache query results
persons = dn.findDatanodes(select='person')

# Reuse instead of repeated queries
for person_dn in persons:
    # ... process
    pass

# Instead of:
for i in range(n):
    person_dn = dn.findDatanodes(select='person')[i]  # Slow!
```

---

## Performance Considerations

### 1. Skeleton Mode Benefits
```python
# Regular mode: O(n) DataNode creation
# Skeleton mode: O(1) initial + O(n) on demand

# Enable for large datasets
setDnSkeletonMode(True)

# ~10x faster building
# ~5x less memory during construction
```

### 2. Query Complexity
```python
# O(n) where n = total DataNodes
persons = dn.findDatanodes(select='person')

# O(n * m) where m = relation links per node
filtered = dn.findDatanodes(
    select='person',
    indexes={'work_for': 'organization'}
)

# Optimize: Filter after if possible
persons = dn.findDatanodes(select='person')
filtered = [p for p in persons if check_condition(p)]
```

### 3. Attribute Storage
```python
# Efficient: Store tensors
dn.attributes['<person>'] = tensor([0.1, 0.9])

# Inefficient: Store lists
dn.attributes['predictions'] = [0.1, 0.9]  # Convert to tensor!

# Very inefficient: Store large objects
dn.attributes['model'] = entire_model  # Don't!
```

### 4. Relation Links
```python
# Efficient: Direct links
person_dn.relationLinks = {'work_for': [org1, org2]}

# Less efficient: Many small relations
for org in orgs:
    person_dn.addRelationLink('work_for', org)
# Better: Batch if possible
```

---

## Further Reading

- **Logical Constraints**: See [`../solver/README.md`](../solver/README.md) for constraint solving
- **Sensors**: See [`../sensor/README.md`](../sensor/README.md) for data binding
- **Models**: See [`../program/README.md`](../program/README.md) for neural-symbolic models
- **Graph Components**: See [`README_GRAPH.md`](README_GRAPH.md) for ontology structure

---

## Requirements

- **Python**: 3.7+
- **PyTorch**: 1.7+ for tensor operations
- **NumPy**: For numpy array support
- **scikit-learn**: For metrics computation
- **Graphviz**: For visualization (optional)

---

## Summary

The DataNode system provides:

1. **Runtime Binding**: Connect data instances to ontological concepts
2. **Graph Structure**: Maintain hierarchical and relational structure
3. **Inference Support**: Store and compute predictions at multiple levels
4. **Constraint Integration**: Enable logical constraint checking and satisfaction
5. **Flexible Queries**: Powerful graph traversal and filtering
6. **Performance Modes**: Skeleton mode for efficiency
7. **Automatic Construction**: DataNodeBuilder creates graphs from sensors

Together, these components form the runtime data layer that bridges neural predictions with symbolic reasoning in the DomiKnows framework.