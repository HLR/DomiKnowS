# DomiKnows Graph Components

This directory contains the core graph structure and logical constraint system for the DomiKnows framework, enabling knowledge representation and neural-symbolic reasoning.

---

## Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| **Core Graph** | | |
| `Graph` | `graph.py` | Main graph container with concepts, relations, and constraints |
| `Concept` | `concept.py` | Nodes representing entity types and their instances |
| `Relation` | `relation.py` | Edges connecting concepts (IsA, HasA, Contains, Equal) |
| **Logical Constraints** | | |
| `LogicalConstrain` | `logicalConstrain.py` | First-order logic constraints over graph structures |
| `LcElement` | `logicalConstrain.py` | Base class for logical constraint elements |
| **Graph Utilities** | | |
| `Property` | `property.py` | Named properties attached to concepts |
| `Trial` | `trial.py` | Hierarchical data management for experiments |
| `DataNode` | `dataNode.py` | Runtime data instances bound to graph structures |
| **Equality Support** | | |
| `EqualityMixin` | `equality_mixin.py` | Equivalence relations between concepts |
| **Executable Logic** | | |
| `LogicDataset` | `executable.py` | Dataset wrapper for constraint-aware training |

---

## Core Concepts

### Graph Structure

The DomiKnows graph is a hierarchical knowledge representation combining:

1. **Ontological Knowledge**: Concepts (entity types) and their relationships
2. **Logical Constraints**: First-order logic rules over concepts
3. **Runtime Data**: Instance-level bindings through DataNodes

```python
from domiknows.graph import Graph, Concept

# Create graph
with Graph('knowledge_base') as graph:
    # Define concepts
    person = Concept('person')
    organization = Concept('organization')
    
    # Add relationships
    employee = person()  # employee is-a person
    company = organization()
```

### Three-Layer Architecture

```
┌─────────────────────────────────────┐
│   Logical Constraints (Rules)      │  ← First-order logic over concepts
├─────────────────────────────────────┤
│   Graph Structure (Ontology)       │  ← Concepts, relations, properties
├─────────────────────────────────────┤
│   DataNodes (Runtime Instances)     │  ← Actual data bound to graph
└─────────────────────────────────────┘
```

---

## Graph Class (`graph.py`)

### `Graph`
Main container for knowledge representation.

**Initialization:**
```python
graph = Graph(
    name='my_graph',
    ontology=('http://example.org/onto', 'local_namespace'),
    auto_constraint=True,    # Auto-generate graph constraints
    reuse_model=True         # Cache solver models
)
```

**Key Attributes:**
```python
graph.concepts           # OrderedDict of concepts
graph.relations          # OrderedDict of relations  
graph.logicalConstrains  # OrderedDict of constraints
graph.subgraphs          # Nested subgraphs
graph.constraint         # Special constraint concept
```

**Context Manager:**
```python
with graph:
    # Concepts defined here auto-attach to graph
    person = Concept('person')
    # Variables captured from local scope
    # Constraint validation performed on exit
```

**Key Methods:**

#### `findConcept(conceptName)`
```python
# Find concept by name (searches subgraphs)
person_concept = graph.findConcept('person')
```

#### `findConceptInfo(concept)`
```python
# Get detailed concept information
info = graph.findConceptInfo(person)
# Returns:
# {
#     'concept': person,
#     'relation': bool,           # Has has_a relations?
#     'has_a': [...],            # List of has_a relations
#     'relationAttrs': {...},     # Relation attributes
#     'contains': [...],          # Contained concepts
#     'containedIn': [...],       # Container concepts
#     'is_a': [...],             # Parent concepts
#     'root': bool                # Is root concept?
# }
```

#### `visualize(filename, open_image=False)`
```python
# Generate graph visualization using Graphviz
graph.visualize('my_graph', open_image=True)
# Creates my_graph.png
```

#### `compile_executable(data, logic_keyword='constraint', ...)`
```python
# Compile string-based logical expressions into executable constraints
dataset = graph.compile_executable(
    data=[
        {'constraint': 'ifL(person("x"), entity("x"))', 'label': 1},
        {'constraint': 'andL(work_for("x", "y"), person("x"))', 'label': 0}
    ],
    extra_namespace_values={'custom_var': value}
)
# Returns LogicDataset for constraint-aware training
```

**Variable Name Tracking:**

When exiting graph context, automatically captures variable names:

```python
with graph:
    person = Concept('person')
    work_for = Relation('work_for')
    
# After exit:
# person.var_name == 'person'
# work_for.var_name == 'work_for'
# graph.varNameReversedMap['person'] == person
```

**Constraint Validation:**

On context exit, validates all logical constraints:
- Finds all variables defined in constraints
- Checks paths are well-formed
- Verifies variable types match concept types
- Reports detailed errors with fix suggestions

---

## Concept Class (`concept.py`)

### `Concept`
Represents entity types in the knowledge graph.

**Initialization:**
```python
person = Concept(
    name='person',
    batch=False  # Is this the batch concept?
)
```

**Defining Concept Hierarchies:**

#### `is_a(parent_concept, auto_constraint=None)`
```python
# Subclass relationship
employee = Concept('employee')
employee.is_a(person)
# Generates constraint: IF(employee(x), person(x))
```

#### `has_a(*child_concepts, auto_constraint=None, **named_children)`
```python
# Compositional relationship (≥2 children required)
work_for = Concept('work_for')
work_for.has_a(person, organization)
# Creates relation with argument_name for each child
```

#### `contains(*child_concepts, auto_constraint=None)`
```python
# Containment relationship (exactly 1 child)
sentence = Concept('sentence')
sentence.contains(token)
```

**Calling Concepts:**

Concepts are callable for different purposes:

```python
# 1. Create subclass without name
employee = person()  # Auto-named

# 2. Create named subclass  
manager = person(name='manager')

# 3. Use in logical constraint with string variable
person('x')  # Variable named 'x' of type person

# 4. Use with path (string-based)
person('x', path=('x', work_for, organization))
```

**Relation Access:**

```python
# Access relations by type
person.is_a()      # List of IsA relations
person.has_a()     # List of HasA relations  
person.contains()  # List of Contains relations

# Relations stored in _in and _out
person._out['is_a']     # Outgoing is_a relations
person._in['has_a']     # Incoming has_a relations
```

**Properties:**

```python
# Attach properties to concepts
from domiknows.graph import Property

with person:
    age = Property('age')
    name = Property('name')

# Access properties
person['age']  # Returns Property instance
```

**Utility Methods:**

#### `relate_to(concept, *tests)`
```python
# Find relations between concepts
relations = person.relate_to(
    organization,
    HasA  # Filter by relation type
)
```

#### `getOntologyGraph()`
```python
# Get containing graph
graph = person.getOntologyGraph()
```

#### `candidates(root_data, query=None, logger=None)`
```python
# Get candidate instances from DataNode
candidates = person.candidates(root_datanode)
# Returns iterator of possible DataNode instances
```

### `EnumConcept`
Concept with fixed enumeration of values.

```python
color = EnumConcept('color', values=['red', 'green', 'blue'])

# Access enum values
color.enum  # ['red', 'green', 'blue']
color.attributes  # [(color, 'red', 0, 3), (color, 'green', 1, 3), ...]

# Get index/value
color.get_index('red')    # 0
color.get_value(1)        # 'green'

# Use in constraints
color.red('x')  # Variable 'x' with color=red
```

---

## Relation Classes (`relation.py`)

### `Relation`
Base class for relationships between concepts.

**Hierarchy:**
```
Relation
├── OTORelation (One-to-One)
│   ├── IsA       (Subclass)
│   ├── NotA      (Disjoint)
│   └── Equal     (Equivalence)
├── OTMRelation (One-to-Many)
│   ├── HasMany
│   └── Contains
├── MTORelation (Many-to-One)
│   └── HasA
└── MTMRelation (Many-to-Many)
```

**Automatic Creation:**

Relations are created through concept methods:

```python
# IsA relation
employee.is_a(person)
# Creates: IsA(src=employee, dst=person)

# HasA relation  
work_for.has_a(person, organization)
# Creates: HasA relations for each argument

# Contains relation
sentence.contains(token)
# Creates: Contains(src=sentence, dst=token)
```

**Relation Properties:**

```python
rel = person.is_a()[0]

rel.src         # Source concept
rel.dst         # Destination concept
rel.reversed    # Reverse relation
rel.mode        # 'forward' or 'backward'
rel.graph       # Containing graph
rel.var_name    # Python variable name (auto-captured)
```

**Special Relations:**

#### `IsA` - Subclass
```python
employee.is_a(person)
# Generates: IF(employee(x), person(x))
```

#### `NotA` - Disjoint
```python
person.not_a(organization)
# Generates: NAND(person(x), organization(x))
```

#### `disjoint(*concepts)` - Pairwise Disjoint
```python
from domiknows.graph.relation import disjoint

disjoint(person, organization, location)
# Creates NotA between all pairs
```

#### `HasA` - Composition
```python
work_for.has_a(person, organization)
# Requires ≥2 arguments
# Generates domain/range constraints
```

#### `Contains` - Containment
```python
sentence.contains(token)
# Requires exactly 1 argument
# Represents hierarchical containment
```

#### `Equal` - Equivalence
```python
# See equality_mixin.py for full API
person1.equal(person2)
```

---

## Logical Constraints (`logicalConstrain.py`)

### `LogicalConstrain`
First-order logic constraints over graph concepts.

**Base Structure:**
```python
class LogicalConstrain(LcElement):
    def __init__(self, *e, p=100, active=True, sampleEntries=False, name=None):
        """
        Args:
            *e: Constraint elements (concepts with string variables, relations)
            p: Priority (0-100, higher = more important)
            active: Enable/disable constraint
            sampleEntries: Use sampling for large groundings
            name: Constraint name (auto-generated if None)
        """
```

**Constraint Elements:**

Constraints are built from:
1. **Concepts with string variables**: `person('x')`, `organization('y')`
2. **Relations with string variables**: `work_for('x', 'y')`
3. **Path expressions**: `person('y', path=('x', rel_name))`
4. **Nested Constraints**: Other `LogicalConstrain` instances
5. **Cardinality**: Trailing integer for counting

### Variable Syntax

Variables are plain strings passed as positional arguments to concept/relation calls. For single-variable concepts use one string; for binary relations (defined with `has_a`) use two strings.

```python
# Single variable
person('x')           # concept 'person' with variable named 'x'
entity('x')           # same variable 'x' — refers to same candidate

# Binary relation variable (work_for has_a person, organization)
work_for('x', 'y')    # 'x' → arg1 (person), 'y' → arg2 (organization)

# Path from variable to related concept
organization('y', path=('x', rel_pair_word2.name))
# 'y' is the org reached from 'x' via rel_pair_word2
```

### Logical Operators

#### Single-Variable Operators

##### `notL` - Negation
```python
from domiknows.graph import notL

# ¬person(x)
notL(person('x'))
```

#### Binary/N-ary Operators

##### `andL` - Conjunction
```python
from domiknows.graph import andL

# person(x) ∧ organization(y)
andL(person('x'), organization('y'))

# Can nest
andL(person('x'), orL(student('x'), employee('x')))
```

##### `orL` - Disjunction
```python
# person(x) ∨ organization(x)
orL(person('x'), organization('x'))
```

##### `nandL` - NAND
```python
# ¬(person(x) ∧ organization(x))
nandL(person('x'), organization('x'))
```

##### `norL` - NOR
```python
# ¬(person(x) ∨ organization(x))
norL(person('x'), organization('x'))
```

##### `xorL` - Exclusive OR
```python
# person(x) ⊕ organization(x)
xorL(person('x'), organization('x'))
```

##### `ifL` - Implication
```python
# person(x) → entity(x)
ifL(person('x'), entity('x'))

# work_for(x,y) → person(x) ∧ organization(y)
ifL(
    work_for('x', 'y'),
    andL(person('x'), organization('y'))
)
```

##### `equivalenceL` - Bi-conditional
```python
# person(x) ↔ entity(x)
equivalenceL(person('x'), entity('x'))
```

##### `iffL` - If and Only If (alias)
```python
# Exact alias of equivalenceL
iffL(person('x'), entity('x'))
```

##### `forAllL` - Universal Quantifier
```python
# ∀x: person(x) → entity(x)
forAllL(person('x'), entity('x'))
# Currently implemented as ifL
```

### Counting Constraints

#### Element-wise Counting

##### `existsL` - Exists (≥1)
```python
# ∃x: person(x)
existsL(person('x'))
# Equivalent to: atLeastL(person('x'), 1)
```

##### `atLeastL` - At Least K
```python
# At least 2 persons
atLeastL(person('x'), 2)

# At least 3 tokens per sentence
atLeastL(token('t'), 3)
```

##### `atMostL` - At Most K
```python
# At most 1 CEO per company
atMostL(ceo('x', path=('x', work_for, 'y')), 1)
```

##### `exactL` - Exactly K
```python
# Exactly 3 directors
exactL(director('x'), 3)
```

#### Global (Accumulated) Counting

For batch-level constraints:

##### `existsAL` - Global Exists
```python
# At least 1 person across all instances
existsAL(person('x'))
```

##### `atLeastAL` - Global At Least
```python
# At least 10 persons total
atLeastAL(person('x'), 10)
```

##### `atMostAL` - Global At Most
```python
# At most 100 organizations total
atMostAL(organization('x'), 100)
```

##### `exactAL` - Global Exactly
```python
# Exactly 5 managers total
exactAL(manager('x'), 5)
```

#### Comparative Counting

Compare counts between two variable sets:

##### `greaterL` - Count Greater
```python
# count(person) > count(organization)
greaterL(person('x'), organization('y'))

# With offset: count(person) > count(organization) + 5
greaterL(person('x'), organization('y'), 5)
```

##### `greaterEqL` - Count Greater or Equal
```python
# count(employee) ≥ count(manager)
greaterEqL(employee('x'), manager('y'))
```

##### `lessL` - Count Less
```python
# count(manager) < count(employee)
lessL(manager('x'), employee('y'))
```

##### `lessEqL` - Count Less or Equal
```python
# count(intern) ≤ count(employee)
lessEqL(intern('x'), employee('y'))
```

##### `equalCountsL` - Equal Counts
```python
# count(input) == count(output)
equalCountsL(input('x'), output('y'))
```

##### `notEqualCountsL` - Unequal Counts
```python
# count(success) ≠ count(failure)
notEqualCountsL(success('x'), failure('y'))
```

### Path Expressions

Paths are expressed as tuples of strings (variable names) interleaved with relation names or relation objects. The first element is the source variable name; subsequent elements are relations or concept filters.

```python
# concept reached from variable 'x' via relation rel_name
organization('y', path=('x', rel_name))

# Multi-hop path: x → rel1 → intermediate → rel2 → destination
manager('m', path=('x', rel_reports_to.name, 'y', rel_manages.name))

# Reversed relation in path
person('x', path=('y', rel_work_for.reversed.name))

# Path union (multiple paths to the same variable)
organization('y', path=(('x', rel_work_for.name), ('x', rel_volunteer_at.name)))

# Path with equality filter
person('x', path=('x', rel_work_for.name, eqL(organization, 'name', 'Anthropic')))
```

**Key rules:**
- String items in a path are variable names (they must be defined earlier in the constraint)
- Relation items are the relation name string or the relation object
- A path must start from an already-defined variable name

### Constraint Priority (`p`)

```python
# Critical (always enforced if possible)
critical = ifL(person('x'), entity('x'), p=100)

# Important
important = atLeastL(employee('x'), 1, p=80)

# Optional
optional = exactL(manager('x'), 3, p=50)

# Solver satisfies highest priority first
# Falls back to lower if infeasible
```

### Auxiliary Constraints

#### `eqL` - Path Equality Filter
```python
# Filter by specific value
eqL(organization, 'instanceID', 'ORG-1')

# Used in paths
person('x', path=('x', rel_work_for.name, eqL(organization, 'name', 'Anthropic')))
```

#### `fixedL` - Fix to Ground Truth
```python
# Fix variables to known values (for debugging/testing)
fixedL(person('x'))
```

#### `sumL` - Summation
```python
# Sum of variables (for numeric constraints)
sumL(salary('x'))
```

### Definite Description & Query Constraints

These constraints go beyond boolean satisfaction — they **select entities** and **query attributes**, enabling question-answering style reasoning within the constraint framework.

#### `iotaL` - Definite Description (Select THE Unique Entity)

Based on Russell's theory of definite descriptions: ιx.φ(x) denotes "the unique x such that φ(x)".

Unlike `existsL` which returns a boolean (does any entity satisfy?), `iotaL` **returns the entity itself** — a selection distribution over entities that can be composed with other constraints.

```python
from domiknows.graph import iotaL, andL, existsL, eqL

# Select THE sphere in the scene (assuming exactly one)
iotaL(sphere('x'))

# Select THE person who works for Microsoft
iotaL(person('x', path=('x', rel_work_for.name, eqL(organization, 'name', 'Microsoft'))))

# Nest inside other constraints:
# "Is there something left of THE blue sphere?"
existsL(left('x', iotaL(andL(blue('y'), sphere('y')))))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*e` | elements | — | Constraint elements defining the selection condition |
| `p` | int | 100 | Priority (0–100) |
| `temperature` | float | 1.0 | Softmax temperature for differentiable selection (lower = harder) |
| `active` | bool | True | Enable/disable |
| `sampleEntries` | bool | False | Use sampling for large groundings |
| `name` | str | None | Constraint name (auto-generated if None) |

**Behavior across execution modes:**

| Mode | `onlyConstrains=True` | `onlyConstrains=False` |
|------|----------------------|----------------------|
| **ILP** | Adds uniqueness constraints to model | Returns binary selection variables `[s₀, …, sₙ]` |
| **Loss** | Returns scalar loss tensor | Returns entity distribution tensor `[N]` via softmax |
| **Sample** | Returns boolean violation tensor | Returns tensor of selected entity indices |
| **Verify** | Returns `1` if valid, `0` if violated | Returns index of selected entity, or `-1` if invalid |

**Presuppositions enforced:**

1. **Existence** — at least one entity must satisfy the condition (Σ cᵢ ≥ 1)
2. **Uniqueness** — exactly one entity is selected (Σ sᵢ = 1, sᵢ ≤ cᵢ)

#### `queryL` - Query Multiclass Attribute of Selected Entity

Given a multiclass concept (parent with subclasses via `is_a`, or `EnumConcept`) and an entity selection (typically from `iotaL`), returns which subclass the selected entity belongs to.

This is the constraint-level equivalent of asking *"What is the \<attribute\> of THE \<entity\>?"*

```python
from domiknows.graph import Concept, EnumConcept, queryL, iotaL, andL

# --- Setup: define a multiclass concept via is_a ---
material = Concept('material')
metal = Concept('metal')
rubber = Concept('rubber')
metal.is_a(material)
rubber.is_a(material)

# Query: "What material is THE big sphere?"
answer = queryL(
    material,
    iotaL(andL(big('x'), sphere('x')))
)

# --- Or with EnumConcept ---
color = EnumConcept('color', values=['red', 'green', 'blue'])

# Query: "What color is THE small cube?"
answer = queryL(
    color,
    iotaL(andL(small('x'), cube('x')))
)
```

**First argument requirements** — the concept passed to `queryL` must be one of:

1. A `Concept` that has subclasses defined via `is_a()` relationships.
2. An `EnumConcept` with explicit values.

The graph validates this at context-exit time and raises a clear error otherwise:

```
queryL constraint in LC3 has concept 'material' without subclasses.
The concept used in queryL must be a multiclass concept with subclasses defined via is_a().
Example: metal.is_a(material), rubber.is_a(material)
Alternatively, use EnumConcept: material = EnumConcept('material', values=['value1', 'value2'])
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concept` | Concept/EnumConcept | — | The multiclass concept to query (first positional arg) |
| `*e` | elements | — | Entity selection constraint (typically `iotaL(…)`) |
| `p` | int | 100 | Priority |
| `temperature` | float | 1.0 | Softmax temperature |
| `active` | bool | True | Enable/disable |

**Composing `iotaL` + `queryL` for VQA-style reasoning:**

```python
# "What shape is the large red object?"
queryL(
    shape,
    iotaL(andL(large('x'), red('x')))
)

# "What is left of the blue cylinder?"
queryL(
    object_type,
    iotaL(left('x', iotaL(andL(blue('y'), cylinder('y')))))
)
```

---

### Executable Constraints

Executable constraints are dynamically compiled from string expressions at runtime, enabling data-driven or per-sample constraint definitions. They are stored separately from standard logical constraints and support constraint-aware training where different samples may have different active constraints.

#### `execute` - Mark a Constraint as Executable

Wrapping any `LogicalConstrain` with `execute()` moves it from `graph.logicalConstrains` to `graph.executableLCs`. This separation allows the training loop to process executable constraints differently — for example, switching which constraint is active per data item.

```python
from domiknows.graph import execute, andL, ifL

# Manually wrap a constraint
with graph:
    constraint1 = execute(andL(person('x'), entity('x')))
    # Now in graph.executableLCs as "ELC0", NOT in graph.logicalConstrains
```

Executable constraints are named `ELC0`, `ELC1`, etc., and delegate all operations (ILP, loss, verification) to their wrapped inner constraint.

#### `graph.compile_executable()` - Compile Constraints from Data

The primary way to create executable constraints is from a dataset of string expressions. Each data item contains a constraint string and a label indicating whether it should be satisfied (`True`) or violated (`False`).

```python
data = [
    {'constraint': 'ifL(person("x"), entity("x"))', 'label': True},
    {'constraint': 'andL(work_for("x", "y"), person("x"))', 'label': False},
]

# Compile into executable constraints and get a LogicDataset
logic_dataset = graph.compile_executable(
    data,
    logic_keyword='constraint',
    logic_label_keyword='label',
    extra_namespace_values={},  # additional variables for eval namespace
    verbose=False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | iterable of dicts | — | Each dict must contain keys for the constraint expression and label |
| `logic_keyword` | str | `'constraint'` | Key in data items containing the constraint string |
| `logic_label_keyword` | str | `'label'` | Key in data items containing the expected label |
| `extra_namespace_values` | dict | `{}` | Additional variables available during `eval()` |
| `verbose` | bool | `False` | Print debug information during compilation |

**Returns:** a `LogicDataset` wrapping the original data with constraint metadata.

**What happens during compilation:**

1. Each constraint string is auto-wrapped with `execute()` if not already.
2. Constraint function names are resolved to fully qualified paths (e.g., `andL` → `domiknows.graph.logicalConstrain.andL`).
3. The expression is compiled and evaluated in a namespace containing the graph's concepts, relations, and variables.
4. The resulting executable constraint is stored in `graph.executableLCs`.
5. Labels are stored in `graph.executableLCsLabels`.

#### `LogicDataset` - Dataset Wrapper for Constraint-Aware Training

`LogicDataset` wraps the original dataset and adds per-item metadata for constraint switching during training. When iterated, each item includes additional keys that tell the model which executable constraint is currently active.

```python
from domiknows.graph.executable import LogicDataset

for item in logic_dataset:
    # Original data keys still present
    # Plus constraint metadata:
    #   _constraint_<ELC_NAME>: label (True/False)
    #   _constraint_curr_lc_name: name of the active constraint
    #   _constraint_do_switch: flag indicating constraint switching
    pass
```

This integrates with the model's `inference()` method (in `PoiModel` / `SolverModel`), which checks for `LogicDataset.curr_lc_key` and skips non-active executable constraints during forward passes.

**Typical training workflow:**

```python
with Graph('vqa') as graph:
    # ... define concepts and relations ...
    pass

# Compile constraints from data
logic_dataset = graph.compile_executable(qa_data)

# Use in training — model automatically handles constraint switching
for epoch in range(num_epochs):
    for batch in DataLoader(logic_dataset):
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

---

## Variable Syntax

### Path-Based Syntax (Recommended)

Variables are plain strings. The first time a string is used without a `path=` argument it **defines** that variable (iterates over all candidates of that concept). Subsequent uses with `path=` **navigate** from an already-defined variable via a relation.

```python
from domiknows.graph import ifL, andL

# Basic: person who works for organization
ifL(
    work_for('x', 'y'),          # defines 'x' (person arg) and 'y' (org arg)
    andL(person('x'), organization('y'))
)

# Equivalent longer form with explicit paths:
ifL(
    work_for('pair'),
    andL(
        person('x', path=('pair', rel_pair_word1.name)),
        organization('y', path=('pair', rel_pair_word2.name))
    )
)

# Multiple hops: employee -> reports_to -> manager -> belongs_to -> department
ifL(
    employee('x'),
    department('d', path=('x', rel_reports_to.name, 'm', rel_belongs_to.name))
)
```

**How It Works:**

1. String arguments in concept/relation calls create VarMaps internally.
2. On graph context exit, VarMaps are processed.
3. They are converted to the internal path representation.
4. Original variable connections are validated.

---

## Equality Mixin (`equality_mixin.py`)

### `EqualityMixin`
Methods for handling equivalence relations between concepts.

**Apply to Concept class:**
```python
from domiknows.graph.equality_mixin import apply_equality_mixin

apply_equality_mixin(Concept)
# Now all Concepts have equality methods
```

**Key Methods:**

#### `get_equal_concepts(transitive=False)`
```python
# Get directly equal concepts
equals = person1.get_equal_concepts()

# Get full equivalence class
equiv_class = person1.get_equal_concepts(transitive=True)
```

#### `is_equal_to(other_concept)`
```python
if person1.is_equal_to(person2):
    print("Directly equal")
```

#### `is_equal_to_transitive(other_concept)`
```python
if person1.is_equal_to_transitive(person3):
    print("In same equivalence class")
```

#### `get_equivalence_class()`
```python
equiv_class = person1.get_equivalence_class()
# Returns: [person1, person2, person3, ...]
```

#### `get_canonical_concept()`
```python
canonical = person1.get_canonical_concept()
```

#### `merge_equal_concepts(property_merge_strategy='first')`
```python
merged = person1.merge_equal_concepts(
    property_merge_strategy='all'  # 'first', 'last', 'all'
)
```

---

## Property Class (`property.py`)

### `Property`
Named attributes attached to concepts.

```python
from domiknows.graph import Property

with person:
    age = Property('age')
    name = Property('name')
    
# Access
person['age']  # Returns Property instance

# Attach sensors
from domiknows.sensor.pytorch.sensors import ReaderSensor

person['age'].attach(ReaderSensor(keyword='age_value'))
```

**Key Methods:**

#### `find(*sensor_tests)`
```python
sensors = person['age'].find(
    ReaderSensor,
    lambda s: s.keyword == 'age_value'
)
```

#### `__call__(data_item)`
```python
value = person['age'](data_item)
```

---

## Trial Class (`trial.py`)

### `Trial`
Hierarchical data management for experiments.

```python
from domiknows.graph import Trial

trial = Trial(name='experiment_1')

with trial:
    trial['key'] = value
    
    sub_trial = Trial(name='sub_experiment')
    with sub_trial:
        parent_value = sub_trial['key']  # From trial
        sub_trial['key'] = new_value

trial['key']  # Returns value

for key, value in trial.items():
    print(f"{key}: {value}")
```

**Key Features:**
- Hierarchical data inheritance
- Context manager support
- Obsolescence tracking (deleted keys)
- Weak references to prevent memory leaks

---

## Executable Logic (`executable.py`)

### `LogicDataset`
Wrapper for datasets with executable logical constraints.

```python
from domiknows.graph.executable import LogicDataset

data = [
    {
        'constraint': 'ifL(person("x"), entity("x"))',
        'label': 1,
        'text': 'John is a person'
    },
    {
        'constraint': 'andL(work_for("x", "y"), person("x"))',
        'label': 0,
        'text': 'Works at Anthropic'
    }
]

logic_dataset = graph.compile_executable(
    data,
    logic_keyword='constraint',
    logic_label_keyword='label'
)

for item in logic_dataset:
    # item contains:
    # - Original data
    # - _constraint_<LC_NAME>: label
    # - _constraint_curr_lc_name: active constraint
    # - _constraint_do_switch: switching flag
    pass
```

**Utility Functions:**

#### `add_keyword(expr_str, kwarg_name, kwarg_value)`
```python
expr = "andL(x, y)"
new_expr = add_keyword(expr, 'name', 'my_constraint')
# Returns: "andL(x, y, name='my_constraint')"
```

#### `get_full_funcs(expr_str)`
```python
expr = "andL(x, y)"
full_expr = get_full_funcs(expr)
# Returns: "domiknows.graph.logicalConstrain.andL(x, y)"
```

---

## Base Classes (`base.py`)

### `BaseGraphTreeNode`
Base class for graph nodes with auto-naming.

```python
@NamedTreeNode.localize_context
class BaseGraphTreeNode(AutoNamed, NamedTreeNode):
    """
    Features:
    - Automatic name generation
    - Context attachment
    - Hierarchical structure
    - Copy protection (returns self)
    """
```

### `BaseGraphTree`
Base class for graph containers.

```python
@BaseGraphTreeNode.share_context
class BaseGraphTree(AutoNamed, NamedTree):
    """
    Features:
    - OrderedDict-based storage
    - Path-based queries (concept/property)
    - Tree traversal
    - Context management
    """
```

**Key Methods:**

#### `traversal_apply(func, filter_fn, order='pre', first='depth')`
```python
results = graph.traversal_apply(
    lambda node: node if isinstance(node, Concept) else None,
    filter_fn=lambda x: x is not None,
    order='pre',
    first='depth'
)
```

#### Query Methods
```python
sub = graph.get_sub('subgraph', 'concept', 'property')
graph.set_sub('subgraph', 'concept', sub=new_concept)
graph.del_sub('subgraph', 'concept')
```

### `BaseGraphShallowTree`
Flat graph structure — used for `Property` (properties cannot contain sub-properties).

---

## Common Workflows

### 1. Basic Graph Creation

```python
from domiknows.graph import Graph, Concept

with Graph('knowledge_graph') as graph:
    entity = Concept('entity')
    person = Concept('person')
    organization = Concept('organization')
    
    person.is_a(entity)
    organization.is_a(entity)
    
    person.not_a(organization)
    
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
```

### 2. Logical Constraints

```python
from domiknows.graph import ifL, andL, atLeastL

with graph:
    # Type constraints
    ifL(person('x'), entity('x'))
    
    # Domain/range constraints for binary relation
    ifL(
        work_for('x', 'y'),
        andL(person('x'), organization('y'))
    )
    
    # Counting constraints
    atLeastL(employee('x'), 1, p=80)
```

### 3. Path-Based Constraints

```python
# Person who works for an organization must have a role
ifL(
    person('x'),
    atLeastL(
        role('r', path=('x', rel_work_for.name, 'y', rel_has_role.name)),
        1
    )
)

# Manager's manager must be executive
ifL(
    manager('x'),
    executive('z', path=('x', rel_reports_to.name, 'y', rel_reports_to.name))
)
```

### 4. Enumeration Concepts

```python
from domiknows.graph import EnumConcept

sentiment = EnumConcept('sentiment', values=['positive', 'negative', 'neutral'])

ifL(
    review('x'),
    orL(
        sentiment.positive('x'),
        sentiment.negative('x'),
        sentiment.neutral('x')
    )
)
```

### 5. Properties and Sensors

```python
from domiknows.graph import Property
from domiknows.sensor.pytorch.sensors import ReaderSensor

with person:
    age = Property('age')
    age.attach(ReaderSensor(
        keyword='age_value',
        dtype=torch.float
    ))

age_value = person['age'](data_item)
```

### 6. Compiled Logic Dataset

```python
train_data = [
    {
        'text': 'John works at Anthropic',
        'constraint': 'ifL(work_for("x", "y"), andL(person("x"), organization("y")))',
        'label': 1
    },
]

logic_dataset = graph.compile_executable(
    train_data,
    logic_keyword='constraint',
    logic_label_keyword='label',
    extra_namespace_values={'custom_var': value}
)

for batch in DataLoader(logic_dataset):
    # Constraint label in batch['_constraint_LC0']
    # Active constraint in batch['_constraint_curr_lc_name']
    pass
```

### 7. Equivalence Relations

```python
from domiknows.graph.equality_mixin import apply_equality_mixin

apply_equality_mixin(Concept)

with graph:
    person1 = Concept('person1')
    person2 = Concept('person2')
    person3 = Concept('person3')
    
    person1.equal(person2)
    person2.equal(person3)
# person1, person2, person3 are in the same equivalence class
```

### 8. Graph Visualization
```python
graph.visualize('output/knowledge_graph', open_image=True)
# Generates: output/knowledge_graph.png
```

### 9. Hierarchical Trials
```python
from domiknows.graph import Trial

experiment = Trial(name='main_experiment')

with experiment:
    experiment['hyperparams'] = {'lr': 0.001, 'batch_size': 32}
    experiment['results'] = []
    
    for fold in range(5):
        fold_trial = Trial(name=f'fold_{fold}')
        with fold_trial:
            lr = fold_trial['hyperparams']['lr']
            fold_trial['fold_results'] = train_fold(fold, lr)
            experiment['results'].append(fold_trial['fold_results'])

all_results = experiment['results']
```

### 10. Complex Path Constraints
```python
# Path union: person connected via work_for OR volunteer_at
ifL(
    person('x'),
    organization('y', path=(
        ('x', rel_work_for.name),
        ('x', rel_volunteer_at.name)
    ))
)

# Complex nested path
ifL(
    project('p'),
    atLeastL(
        person('x', path=(
            'p',
            rel_managed_by.name,
            'm',
            rel_reports_to.name,
            'exec',
            rel_has_authority_over.name
        )),
        1
    )
)
```

### 11. Variable Validation

Graph automatically validates constraints on context exit:
```python
with graph:
    person = Concept('person')
    organization = Concept('organization')
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
    
    # This will be validated:
    ifL(
        work_for('x', 'y'),
        andL(person('x'), organization('y'))
    )

# If validation fails, detailed error message:
# "The Path 'x work_for' from variable pair, defined in LC0 is not valid.
#  The relation work_for is from person to organization, but you have 
#  used it from work_for to person. You can change 'work_for' to 
#  'work_for.reversed' to go from organization to person."
```

---

## Advanced Features

### 1. Auto-Constraint Generation
```python
graph = Graph('kg', auto_constraint=True)

with graph:
    person = Concept('person')
    entity = Concept('entity')
    
    # Automatically generates: IF(person(x), entity(x))
    person.is_a(entity)
    
    organization = Concept('organization')
    
    # Automatically generates: NAND(person(x), organization(x))
    person.not_a(organization)
    
    work_for = Concept('work_for')
    
    # Automatically generates domain/range constraints
    work_for.has_a(person, organization)
```

### 2. Batch Concepts
```python
sentence = Concept('sentence', batch=True)
token = Concept('token')
sentence.contains(token)

graph.batch  # Returns: sentence
```

### 3. Constraint Priority Strategies
```python
PRIORITY_CRITICAL = 100
PRIORITY_HIGH = 80
PRIORITY_MEDIUM = 60
PRIORITY_LOW = 40
PRIORITY_HINT = 20

ifL(person('x'), entity('x'), p=PRIORITY_CRITICAL)
atLeastL(employee('x'), 1, p=PRIORITY_MEDIUM)
```

### 4. Model Reuse
```python
graph = Graph('kg', reuse_model=True)

# First inference builds complete ILP model
solver.calculateILPSelection(datanode1, *concepts)

# Subsequent calls reuse model structure (~10x faster)
solver.calculateILPSelection(datanode2, *concepts)
```

### 5. Graph Queries
```python
concepts = list(graph.traversal_apply(
    lambda node: node if isinstance(node, Concept) else None,
    filter_fn=lambda x: x is not None
))

root_concepts = [
    c for c in concepts 
    if not c._out.get('is_a') and not c._in.get('contains')
]

has_a_relations = []
for concept in concepts:
    has_a_relations.extend(concept.has_a())

concepts_with_age = [c for c in concepts if 'age' in c]
```

### 6. Constraint Introspection
```python
for lc_name, lc in graph.allLogicalConstrains:
    print(f"Constraint: {lc_name}")
    print(f"  Type: {type(lc).__name__}")
    print(f"  Priority: {lc.p}")
    print(f"  Active: {lc.active}")
    print(f"  Elements: {lc.strEs()}")
    concepts_used = lc.getLcConcepts()
    print(f"  Concepts: {concepts_used}")
```

### 7. Dynamic Constraint Activation
```python
constraint.active = False  # Disable
constraint.active = True   # Re-enable
# Useful for ablation studies, debugging, curriculum learning
```

### 8. Graph Merging
```python
with Graph('main') as main_graph:
    with Graph('domain1') as sub1:
        person = Concept('person')
    
    with Graph('domain2') as sub2:
        organization = Concept('organization')
    
    person_concept = main_graph['domain1']['person']
    org_concept = main_graph['domain2']['organization']
    
    work_for = Concept('work_for')
    work_for.has_a(person_concept, org_concept)
```

---

## Error Handling and Debugging

### Common Errors

#### 1. Undefined Variable in Constraint
```python
# Wrong: 'y' used in path before being defined
with graph:
    person = Concept('person')
    ifL(person('y', path=('x', rel_work_for.name)), organization('x'))

# Error message:
# "Variable y found in LC0 is not defined. You should first use y
#  without putting it in a path to define it."

# Fix: define 'y' first (no path)
ifL(andL(person('y'), organization('x', path=('y', rel_work_for.name))))
```

#### 2. Invalid Path Direction
```python
# Wrong: work_for goes person→org, not org→person
with graph:
    ifL(
        organization('y'),
        person('x', path=('y', rel_work_for.name))
    )

# Error message:
# "The relation work_for is from person to organization, but you have
#  used it from organization to person. You can change 'work_for' to
#  'work_for.reversed' to go from organization to person."

# Fix:
ifL(
    organization('y'),
    person('x', path=('y', rel_work_for.reversed.name))
)
```

#### 3. Concept Not in Graph
```python
# Wrong: defined outside graph context
person = Concept('person')

with graph:
    work_for = Concept('work_for')
    work_for.has_a(person, organization)  # Error!

# Fix: define inside graph context
with graph:
    person = Concept('person')
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
```

#### 4. Invalid Relation Cardinality
```python
sentence.contains(token, word)  # Error! Contains requires exactly 1 dst

# Fix:
sentence.contains(token)
```

#### 5. HasA with Too Few Destinations
```python
work_for.has_a(person)  # Error! HasA requires ≥2 destinations

# Fix:
work_for.has_a(person, organization)
```

### Debugging Tools

#### 1. Logging
```python
import logging
from domiknows.graph import ilpConfig

ilpConfig['log_level'] = logging.DEBUG
ilpConfig['ifLog'] = True
# Logs to: logs/ilpOntSolver.log
```

#### 2. Constraint String Representation
```python
print(constraint.strEs())   # "[person(x), entity(x)]"
print(constraint.name)      # "LC0"
print(repr(constraint))     # "LC0(ifL)"
```

#### 3. Graph Visualization
```python
graph.visualize('debug/graph', open_image=True)
```

#### 4. Concept Information
```python
info = graph.findConceptInfo(person)
print(f"Concept: {info['concept'].name}")
print(f"Is relation: {info['relation']}")
print(f"Has_a relations: {info['has_a']}")
print(f"Is root: {info['root']}")
```

#### 5. Variable Tracking
```python
print(graph.varNameReversedMap)
# {'person': <Concept>, 'work_for': <Relation>, ...}

print(person.var_name)            # 'person'
print(work_for.var_name)          # 'work_for'
print(work_for.reversed.var_name) # 'work_for.reversed'
```

---

## Performance Considerations

### 1. Memory Management
```python
Concept.clear()  # Clears name counters and object storage
Trial.clear()    # Clears trial trees

for epoch in range(100):
    train(...)
    if epoch % 10 == 0:
        Trial.clear()
```

### 2. Constraint Complexity
```python
# Simple constraint (fast)
ifL(person('x'), entity('x'))

# Break complex multi-hop constraints into simpler ones
ifL(person('x'), employee('x'))
ifL(employee('x'), entity('x'))
```

### 3. Model Reuse
```python
graph = Graph('kg', reuse_model=True)

# First call ~1s (builds model); subsequent calls ~0.1s (reuses model)
for batch in batches:
    solver.calculateILPSelection(batch, *concepts)
```

### 4. Constraint Sampling
```python
large_constraint.sampleEntries = True
# Solver samples instead of full grounding — trades accuracy for speed
```

---

## Integration with Other Components

### 1. With Sensors
```python
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor

with graph:
    person = Concept('person')
    
    with person:
        age = Property('age')
        age.attach(ReaderSensor(keyword='age_value'))
        
        adult = Property('adult')
        adult.attach(FunctionalSensor(
            formula=lambda age: age >= 18,
            dependencies=['age']
        ))
```

### 2. With DataNodes
```python
from domiknows.graph import DataNode

root = DataNode(graph=graph)

person_dn = DataNode(ontologyNode=person, parent=root)
person_dn['age'] = 25
person_dn['name'] = 'Alice'

org_dn = DataNode(ontologyNode=organization, parent=root)
org_dn['name'] = 'Anthropic'
```

### 3. With Solvers
```python
from domiknows.solver import ilpOntSolverFactory

solver = ilpOntSolverFactory.getOntSolverInstance(graph)

solver.calculateILPSelection(datanode, person, organization)
lcLosses = solver.calculateLcLoss(datanode, tnorm='P')
results = solver.verifyResultsLC(datanode)
```

### 4. With Models
```python
from domiknows.model import SampleLossModel

model = SampleLossModel(
    graph=graph,
    sensors=[person['age'], person['name']]
)

outputs = model(batch)
loss = model.get_constraint_loss(batch)
```

---

## Best Practices

### 1. Graph Organization
```python
with Graph('main') as main:
    with Graph('entities') as entities:
        person = Concept('person')
        organization = Concept('organization')
    
    with Graph('relations') as relations:
        work_for = Concept('work_for')
        work_for.has_a(
            main['entities']['person'],
            main['entities']['organization']
        )
```

### 2. Constraint Naming
```python
critical_constraint = ifL(
    person('x'),
    entity('x'),
    name='person_is_entity'
)
```

### 3. Priority Assignment
```python
PRIORITY_CRITICAL = 100
PRIORITY_HIGH = 80
PRIORITY_MEDIUM = 60
PRIORITY_LOW = 40
PRIORITY_HINT = 20

ifL(person('x'), entity('x'), p=PRIORITY_CRITICAL)
atLeastL(employee('x'), 1, p=PRIORITY_MEDIUM)
```

### 4. Variable Naming
```python
# Descriptive names are clearer
ifL(
    employee('emp'),
    manager('mgr', path=('emp', rel_reports_to.name))
)
```

### 5. Path Clarity
```python
# Document complex paths with a comment
ifL(
    project('proj'),
    # path: project → managed_by → manager → reports_to → executive
    executive('exec', path=('proj', rel_managed_by.name, 'mgr', rel_reports_to.name))
)
```

### 6. Error Recovery
```python
with graph:
    try:
        person = Concept('person')
        work_for = Concept('work_for')
        work_for.has_a(person, organization)
    except Exception as e:
        print(f"Error creating graph: {e}")
        Concept.clear()
        raise
```

---

## Migration from Older Versions

### Breaking Changes

1. **Constraint variable syntax**: use plain strings `'x'` instead of `V.x` or `V('x')`
2. **Path syntax**: `path=('x', rel_name)` tuples with strings instead of `V`-based paths
3. **Equal relation**: Now requires `equality_mixin.py` to be applied
4. **Comparative counting**: New constraint types (`greaterL`, `lessL`, etc.)

### Migration Steps
```python
# Old code (V syntax — internal/legacy only):
from domiknows.graph import V, ifL
ifL(person(V.x), entity(V.x))

# New code (path syntax):
ifL(person('x'), entity('x'))
```
```python
# Old path syntax:
person(V.x, path=(V.x, work_for))

# New path syntax:
person('x', path=('x', rel_work_for.name))
```
```python
# Enable equality (still required):
from domiknows.graph.equality_mixin import apply_equality_mixin
apply_equality_mixin(Concept)

person1.equal(person2)
```

---

## Further Reading

- **Logical Constraints**: See `logicalConstrain.py` for full constraint API
- **Solvers**: See `domiknows/solver/README.md` for constraint solving
- **Sensors**: See `domiknows/sensor/README.md` for data binding
- **Models**: See `domiknows/program/README.md` for neural-symbolic models

---

## Requirements

- **Python**: 3.7+
- **PyTorch**: For differentiable constraints
- **Graphviz**: For visualization (optional)
- **owlready2**: For ontology loading (optional)

---

## Example: Complete Application
```python
from domiknows.graph import Graph, Concept, ifL, andL, atLeastL
from domiknows.solver import ilpOntSolverFactory

# 1. Create knowledge graph
with Graph('nlp_knowledge') as graph:
    entity = Concept('entity')
    person = Concept('person')
    organization = Concept('organization')
    location = Concept('location')
    
    person.is_a(entity)
    organization.is_a(entity)
    location.is_a(entity)
    
    person.not_a(organization)
    person.not_a(location)
    organization.not_a(location)
    
    work_for = Concept('work_for')
    (rel_wf_person, rel_wf_org) = work_for.has_a(person, organization)
    
    located_in = Concept('located_in')
    (rel_li_org, rel_li_loc) = located_in.has_a(organization, location)
    
    # Domain/range constraints
    ifL(
        work_for('x', 'y'),
        andL(person('x'), organization('y')),
        p=100
    )
    
    # Transitivity: person works_for org → org located_in loc → person located_in loc
    ifL(
        andL(work_for('x', 'y'), located_in('y', 'z')),
        located_in('x', 'z'),
        p=80
    )
    
    # At least one person per organization
    ifL(
        organization('y'),
        atLeastL(person('x', path=('x', rel_wf_person.name, 'y')), 1),
        p=60
    )

# 2. Create solver
solver = ilpOntSolverFactory.getOntSolverInstance(graph)

# 3. Training
for batch in train_loader:
    predictions = model(batch)
    
    lcLosses = solver.calculateLcLoss(
        batch,
        tnorm='P',
        counting_tnorm='L'
    )
    
    data_loss = criterion(predictions, labels)
    constraint_loss = sum(lc['loss'] for lc in lcLosses.values())
    total_loss = data_loss + 0.5 * constraint_loss
    total_loss.backward()
    optimizer.step()

# 4. Inference
for batch in test_loader:
    predictions = model(batch)
    solver.calculateILPSelection(
        batch,
        person, organization, location, work_for, located_in
    )
    person_predictions = batch['person']['ILP']
    work_for_predictions = batch['work_for']['ILP']

# 5. Verification
verification = solver.verifyResultsLC(test_batch, key='/ILP')
for lc_name, results in verification.items():
    print(f"{lc_name}: {results['satisfied']:.2f}% satisfied")
```