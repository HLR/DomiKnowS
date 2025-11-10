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

#### `compile_logic(data, logic_keyword='constraint', ...)`
```python
# Compile string-based logical expressions into executable constraints
dataset = graph.compile_logic(
    data=[
        {'constraint': 'ifL(person(x), entity(x))', 'label': 1},
        {'constraint': 'andL(work_for(x, y), person(x))', 'label': 0}
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

# 3. Use in logical constraint with variable
from domiknows.graph import V
person(V.x)  # Variable x of type person

# 4. Use with path
person(V.x, path=(V.x, work_for, organization))
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
color.red(V.x)  # Variable x with color=red
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
            *e: Constraint elements (concepts, relations, variables)
            p: Priority (0-100, higher = more important)
            active: Enable/disable constraint
            sampleEntries: Use sampling for large groundings
            name: Constraint name (auto-generated if None)
        """
```

**Constraint Elements:**

Constraints are built from:
1. **Concepts**: `person`, `organization`
2. **Relations**: `work_for`, `is_a`
3. **Variables**: `V(name='x')`, `V(name='y', v=path)`
4. **Nested Constraints**: Other `LogicalConstrain` instances
5. **Cardinality**: Trailing integer for counting

### Logical Operators

#### Single-Variable Operators

##### `notL` - Negation
```python
from domiknows.graph import notL

# ¬person(x)
notL(person(V.x))
```

#### Binary/N-ary Operators

##### `andL` - Conjunction
```python
from domiknows.graph import andL

# person(x) ∧ organization(y)
andL(person(V.x), organization(V.y))

# Can nest
andL(person(V.x), orL(student(V.x), employee(V.x)))
```

##### `orL` - Disjunction
```python
# person(x) ∨ organization(x)
orL(person(V.x), organization(V.x))
```

##### `nandL` - NAND
```python
# ¬(person(x) ∧ organization(x))
nandL(person(V.x), organization(V.x))
```

##### `norL` - NOR
```python
# ¬(person(x) ∨ organization(x))
norL(person(V.x), organization(V.x))
```

##### `xorL` - Exclusive OR
```python
# person(x) ⊕ organization(x)
xorL(person(V.x), organization(V.x))
```

##### `ifL` - Implication
```python
# person(x) → entity(x)
ifL(person(V.x), entity(V.x))

# work_for(x,y) → person(x) ∧ organization(y)
ifL(
    work_for(V.pair),
    andL(person(V.pair[0]), organization(V.pair[1]))
)
```

##### `equivalenceL` - Bi-conditional
```python
# person(x) ↔ entity(x)
equivalenceL(person(V.x), entity(V.x))
```

##### `forAllL` - Universal Quantifier
```python
# ∀x: person(x) → entity(x)
forAllL(person(V.x), entity(V.x))
# Currently implemented as ifL
```

### Counting Constraints

#### Element-wise Counting

##### `existsL` - Exists (≥1)
```python
# ∃x: person(x)
existsL(person(V.x))
# Equivalent to: atLeastL(person(V.x), 1)
```

##### `atLeastL` - At Least K
```python
# At least 2 persons
atLeastL(person(V.x), 2)

# At least 3 tokens per sentence
atLeastL(token(V.t), 3)
```

##### `atMostL` - At Most K
```python
# At most 1 CEO per company
atMostL(ceo(V.x, path=(V.x, work_for, V.y)), 1)
```

##### `exactL` - Exactly K
```python
# Exactly 3 directors
exactL(director(V.x), 3)
```

#### Global (Accumulated) Counting

For batch-level constraints:

##### `existsAL` - Global Exists
```python
# At least 1 person across all instances
existsAL(person(V.x))
```

##### `atLeastAL` - Global At Least
```python
# At least 10 persons total
atLeastAL(person(V.x), 10)
```

##### `atMostAL` - Global At Most
```python
# At most 100 organizations total
atMostAL(organization(V.x), 100)
```

##### `exactAL` - Global Exactly
```python
# Exactly 5 managers total
exactAL(manager(V.x), 5)
```

#### Comparative Counting

Compare counts between two variable sets:

##### `greaterL` - Count Greater
```python
# count(person) > count(organization)
greaterL(person(V.x), organization(V.y))

# With offset: count(person) > count(organization) + 5
greaterL(person(V.x), organization(V.y), 5)
```

##### `greaterEqL` - Count Greater or Equal
```python
# count(employee) ≥ count(manager)
greaterEqL(employee(V.x), manager(V.y))
```

##### `lessL` - Count Less
```python
# count(manager) < count(employee)
lessL(manager(V.x), employee(V.y))
```

##### `lessEqL` - Count Less or Equal
```python
# count(intern) ≤ count(employee)
lessEqL(intern(V.x), employee(V.y))
```

##### `equalCountsL` - Equal Counts
```python
# count(input) == count(output)
equalCountsL(input(V.x), output(V.y))
```

##### `notEqualCountsL` - Unequal Counts
```python
# count(success) ≠ count(failure)
notEqualCountsL(success(V.x), failure(V.y))
```

### Path Expressions

#### Variable with Path: `V(name, v=path)`

```python
from domiknows.graph import V

# Simple variable
V.x  # or V(name='x')

# Variable with path
V(name='y', v=(V.x, work_for))
# y is related to x via work_for

# Path union (multiple paths)
V(name='z', v=((V.x, rel1), (V.x, rel2)))

# Path with filter
from domiknows.graph import eqL
V(name='y', v=(V.x, work_for, eqL(organization, 'instanceID', 'ORG-1')))
```

**Path Examples:**

```python
# Person who works for an organization
ifL(
    person(V.x),
    organization(V.y, path=(V.x, work_for))
)

# Manager of a person's manager
ifL(
    employee(V.x),
    manager(V.z, path=(V.x, reports_to, V.y, reports_to))
)

# Reversed relations in paths
ifL(
    organization(V.y),
    person(V.x, path=(V.y, work_for.reversed))
)
```

### Constraint Priority (`p`)

```python
# Critical (always enforced if possible)
critical = ifL(person(V.x), entity(V.x), p=100)

# Important
important = atLeastL(employee(V.x), 1, p=80)

# Optional
optional = exactL(manager(V.x), 3, p=50)

# Solver satisfies highest priority first
# Falls back to lower if infeasible
```

### Auxiliary Constraints

#### `eqL` - Path Equality Filter
```python
# Filter by specific value
eqL(organization, 'instanceID', 'ORG-1')

# Used in paths
person(V.x, path=(V.x, work_for, eqL(organization, 'name', 'Anthropic')))
```

#### `fixedL` - Fix to Ground Truth
```python
# Fix variables to known values (for debugging/testing)
fixedL(person(V.x))
```

#### `sumL` - Summation
```python
# Sum of variables (for numeric constraints)
sumL(salary(V.x))
```

---

## Variable Syntax

### Standard Path Syntax

```python
from domiknows.graph import V, ifL, andL

# Basic: person who works for organization
ifL(
    person(V.x),
    organization(V.y, path=(V.x, work_for))
)

# Multiple hops: employee -> manager -> department
ifL(
    employee(V.x),
    department(V.d, path=(V.x, reports_to, V.m, belongs_to))
)
```

### Simplified Variable Syntax (VarMaps)

For complex constraints, use variable mapping:

```python
# Instead of explicit paths:
ifL(
    work_for(V.pair),
    andL(
        person(V.pair[0]),
        organization(V.pair[1])
    )
)

# Use simplified syntax:
ifL(
    work_for(x, y),  # x, y are strings (variable names)
    andL(person(x), organization(y))
)

# Automatically translated to path syntax on graph.__exit__
```

**How It Works:**

1. String arguments create VarMaps internally
2. On graph context exit, VarMaps are processed
3. Converted to standard path syntax
4. Original variables validated and connected

**Example:**

```python
with graph:
    # Define relation
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
    
    # Use simplified syntax
    ifL(work_for(x, y), andL(person(x), organization(y)))
    
# After exit:
# Internally becomes:
# ifL(
#     work_for(V.pair),
#     andL(
#         person(V(name='x', v=(V.pair, work_for, person))),
#         organization(V(name='y', v=(V.pair, work_for, organization)))
#     )
# )
```

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
# Check direct equality
if person1.is_equal_to(person2):
    print("Directly equal")
```

#### `is_equal_to_transitive(other_concept)`
```python
# Check transitive equality
if person1.is_equal_to_transitive(person3):
    print("In same equivalence class")
```

#### `get_equivalence_class()`
```python
# Get all transitively equal concepts
equiv_class = person1.get_equivalence_class()
# Returns: [person1, person2, person3, ...]
```

#### `get_canonical_concept()`
```python
# Get representative of equivalence class (alphabetically first)
canonical = person1.get_canonical_concept()
```

#### `merge_equal_concepts(property_merge_strategy='first')`
```python
# Merge properties from equal concepts
merged = person1.merge_equal_concepts(
    property_merge_strategy='all'  # 'first', 'last', 'all'
)
# Returns: Dict of merged properties
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
# Find sensors matching criteria
sensors = person['age'].find(
    ReaderSensor,  # Filter by type
    lambda s: s.keyword == 'age_value'  # Custom test
)
```

#### `__call__(data_item)`
```python
# Get property value from data
value = person['age'](data_item)
```

---

## Trial Class (`trial.py`)

### `Trial`
Hierarchical data management for experiments.

```python
from domiknows.graph import Trial

# Create trial
trial = Trial(name='experiment_1')

with trial:
    # Data stored in trial context
    trial['key'] = value
    
    # Nested trial
    sub_trial = Trial(name='sub_experiment')
    with sub_trial:
        # Inherits parent data
        parent_value = sub_trial['key']  # From trial
        
        # Override in child
        sub_trial['key'] = new_value

# Access data
trial['key']  # Returns value

# Iterate
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

# Create dataset with constraints
data = [
    {
        'constraint': 'ifL(person(x), entity(x))',
        'label': 1,
        'text': 'John is a person'
    },
    {
        'constraint': 'andL(work_for(x,y), person(x))',
        'label': 0,
        'text': 'Works at Anthropic'
    }
]

# Compile constraints
logic_dataset = graph.compile_logic(
    data,
    logic_keyword='constraint',
    logic_label_keyword='label'
)

# Use in training
for item in logic_dataset:
    # item contains:
    # - Original data
    # - _constraint_<LC_NAME>: label
    # - _constraint_curr_lc_name: active constraint
    # - _constraint_do_switch: switching flag
    pass
```

**Key Features:**
- Compiles string constraints to executable objects
- Adds constraint metadata to data items
- Enables constraint-aware training
- Supports multiple constraints per dataset

**Utility Functions:**

#### `add_keyword(expr_str, kwarg_name, kwarg_value)`
```python
# Add keyword argument to expression
expr = "andL(x, y)"
new_expr = add_keyword(expr, 'name', 'my_constraint')
# Returns: "andL(x, y, name='my_constraint')"
```

#### `get_full_funcs(expr_str)`
```python
# Convert to fully qualified names
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
# Traverse graph and apply function
results = graph.traversal_apply(
    lambda node: node if isinstance(node, Concept) else None,
    filter_fn=lambda x: x is not None,
    order='pre',    # 'pre' or 'post'
    first='depth'   # 'depth' or 'breadth'
)
```

#### Query Methods
```python
# Get nested element
sub = graph.get_sub('subgraph', 'concept', 'property')
# Equivalent to: graph['subgraph']['concept']['property']

# Set nested element
graph.set_sub('subgraph', 'concept', sub=new_concept)

# Delete nested element
graph.del_sub('subgraph', 'concept')
```

### `BaseGraphShallowTree`
Flat graph structure (no nesting).

Used for `Property` - properties cannot contain sub-properties.

```python
class Property(BaseGraphShallowTree):
    # No __enter__/__exit__
    # No nested queries
    pass
```

---

## Common Workflows

### 1. Basic Graph Creation

```python
from domiknows.graph import Graph, Concept

with Graph('knowledge_graph') as graph:
    # Define concepts
    entity = Concept('entity')
    person = Concept('person')
    organization = Concept('organization')
    
    # Create hierarchy
    person.is_a(entity)
    organization.is_a(entity)
    
    # Disjoint concepts
    person.not_a(organization)
    
    # Compositional concepts
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
```

### 2. Logical Constraints

```python
from domiknows.graph import V, ifL, andL, atLeastL

with graph:
    # Type constraints
    ifL(person(V.x), entity(V.x))
    
    # Domain/range constraints
    ifL(
        work_for(V.pair),
        andL(
            person(V.pair[0]),
            organization(V.pair[1])
        )
    )
    
    # Counting constraints
    atLeastL(employee(V.x), 1, p=80)
```

### 3. Path-Based Constraints

```python
# Person who works for an organization must have a role
ifL(
    person(V.x),
    atLeastL(
        role(V.r, path=(V.x, work_for, V.y, has_role)),
        1
    )
)

# Manager's manager must be executive
ifL(
    manager(V.x),
    executive(V.z, path=(V.x, reports_to, V.y, reports_to))
)
```

### 4. Enumeration Concepts

```python
from domiknows.graph import EnumConcept

# Define enum
sentiment = EnumConcept('sentiment', values=['positive', 'negative', 'neutral'])

# Use in constraints
ifL(
    review(V.x),
    orL(
        sentiment.positive(V.x),
        sentiment.negative(V.x),
        sentiment.neutral(V.x)
    )
)
```

### 5. Properties and Sensors

```python
from domiknows.graph import Property
from domiknows.sensor.pytorch.sensors import ReaderSensor

with person:
    # Define property
    age = Property('age')
    
    # Attach sensor
    age.attach(ReaderSensor(
        keyword='age_value',
        dtype=torch.float
    ))

# Later: read property
age_value = person['age'](data_item)
```

### 6. Compiled Logic Dataset

```python
# Define data with string constraints
train_data = [
    {
        'text': 'John works at Anthropic',
        'constraint': 'ifL(work_for(x, y), andL(person(x), organization(y)))',
        'label': 1
    },
    # ... more examples
]

# Compile constraints
logic_dataset = graph.compile_logic(
    train_data,
    logic_keyword='constraint',
    logic_label_keyword='label',
    extra_namespace_values={'custom_var': value}
)

# Train with constraint loss
for batch in DataLoader(logic_dataset):
    # Constraint label in batch['_constraint_LC0']
    # Active constraint in batch['_constraint_curr_lc_name']
    pass
```

### 7. Equivalence Relations

```python
from domiknows.graph.equality_mixin import apply_equality_mixin

# Enable equality support
apply_equality_mixin(Concept)

with graph:
    # Define equal concepts
    person1 = Concept('person1')
    person2 = Concept('person2')
    person3 = Concept('person3')
    
    person1.equal(person2)
    person2.equal(person3)
# Now person1, person2, person3 are in the same equivalence class
```

# Define### 8. Graph Visualization
```python
# Create visualization
graph.visualize('output/knowledge_graph', open_image=True)
# Generates: output/knowledge_graph.png

# Shows:
# - Concepts as nodes
# - Relations as edges with labels
# - Subgraphs as clusters
```

### 9. Hierarchical Trials
```python
from domiknows.graph import Trial

# Main experiment
experiment = Trial(name='main_experiment')

with experiment:
    experiment['hyperparams'] = {'lr': 0.001, 'batch_size': 32}
    experiment['results'] = []
    
    # Sub-experiments
    for fold in range(5):
        fold_trial = Trial(name=f'fold_{fold}')
        with fold_trial:
            # Inherits hyperparams
            lr = fold_trial['hyperparams']['lr']
            
            # Add fold-specific data
            fold_trial['fold_results'] = train_fold(fold, lr)
            
            # Accumulate results
            experiment['results'].append(fold_trial['fold_results'])

# Access results
all_results = experiment['results']
```

### 10. Complex Path Constraints
```python
# Employee must report to someone in same department
ifL(
    employee(V.x),
    atLeastL(
        manager(V.m, path=(
            V.x,
            reports_to,
            eqL(department, 'id', V.x.department_id)
        )),
        1
    )
)

# Path union: person connected via work_for OR volunteer_at
ifL(
    person(V.x),
    organization(V.y, path=(
        (V.x, work_for),
        (V.x, volunteer_at)
    ))
)

# Complex nested path
ifL(
    project(V.p),
    atLeastL(
        person(V.x, path=(
            V.p,
            managed_by,
            V.m,
            reports_to,
            V.exec,
            has_authority_over
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
        work_for(V.pair),
        andL(
            person(V.x, path=(V.pair, work_for)),  # Checked!
            organization(V.y, path=(V.pair, work_for))  # Checked!
        )
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
# Mark batch-level concept
sentence = Concept('sentence', batch=True)
token = Concept('token')

sentence.contains(token)

# Graph tracks batch concept
graph.batch  # Returns: sentence
```

### 3. Constraint Priority Strategies
```python
# Critical infrastructure constraints (p=100)
ifL(person(V.x), entity(V.x), p=100)

# Domain knowledge (p=80-90)
ifL(work_for(V.pair), andL(person(V.pair[0]), org(V.pair[1])), p=90)

# Statistical preferences (p=50-70)
atLeastL(employee(V.x), 1, p=60)

# Soft preferences (p=10-40)
exactL(manager(V.x), 3, p=30)

# Solver satisfies in priority order:
# 1. All p=100 constraints if possible
# 2. Fall back to p=90 if p=100 infeasible
# 3. Continue until feasible solution found
```

### 4. Model Reuse
```python
graph = Graph('kg', reuse_model=True)

# First inference builds complete ILP model
solver.calculateILPSelection(datanode1, *concepts)

# Subsequent calls reuse model structure
# Only updates objective coefficients
solver.calculateILPSelection(datanode2, *concepts)  # 10x faster!
```

### 5. Custom Concept Suggestions
```python
class MyConcept(Concept):
    @classmethod
    def suggest_name(cls):
        # Custom name generation
        return f"custom_{cls.__name__}"

# Auto-named as "custom_MyConcept-0", "custom_MyConcept-1", etc.
```

### 6. Namespace Localization
```python
# Separate namespaces for different concept hierarchies
@Concept.localize_namespace
class DomainSpecificConcept(Concept):
    pass

# Separate name counters and object storage
# Prevents name conflicts between domains
```

### 7. Graph Queries
```python
# Find all concepts
concepts = list(graph.traversal_apply(
    lambda node: node if isinstance(node, Concept) else None,
    filter_fn=lambda x: x is not None
))

# Find root concepts
root_concepts = [
    c for c in concepts 
    if not c._out.get('is_a') and not c._in.get('contains')
]

# Find all relations of type HasA
has_a_relations = []
for concept in concepts:
    has_a_relations.extend(concept.has_a())

# Find concepts with specific property
concepts_with_age = [
    c for c in concepts
    if 'age' in c
]
```

### 8. Constraint Introspection
```python
# Get all constraints
for lc_name, lc in graph.logicalConstrains.items():
    print(f"Constraint: {lc_name}")
    print(f"  Type: {type(lc).__name__}")
    print(f"  Priority: {lc.p}")
    print(f"  Active: {lc.active}")
    print(f"  Elements: {lc.strEs()}")
    
    # Get concepts used in constraint
    concepts_used = lc.getLcConcepts()
    print(f"  Concepts: {concepts_used}")
```

### 9. Dynamic Constraint Activation
```python
# Disable constraint temporarily
constraint.active = False

# Re-enable
constraint.active = True

# Useful for:
# - Ablation studies
# - Debugging
# - Progressive constraint introduction
```

### 10. Graph Merging
```python
# Create subgraphs
with Graph('main') as main_graph:
    with Graph('domain1') as sub1:
        person = Concept('person')
    
    with Graph('domain2') as sub2:
        organization = Concept('organization')
    
    # Access from main graph
    person_concept = main_graph['domain1']['person']
    org_concept = main_graph['domain2']['organization']
    
    # Create cross-domain relation
    work_for = Concept('work_for')
    work_for.has_a(person_concept, org_concept)
```

---

## Error Handling and Debugging

### Common Errors

#### 1. Undefined Variable in Constraint
```python
# Error: Variable 'y' not defined before use
with graph:
    person = Concept('person')
    
    # Wrong: y used before definition
    ifL(person(V.y, path=(V.x, work_for)), organization(V.x))

# Error message:
# "Variable y found in LC0 is not defined. You should first use y 
#  without putting it in a path to define it."

# Fix: Define y first
ifL(andL(person(V.y), organization(V.x, path=(V.y, work_for))))
```

#### 2. Invalid Path Type
```python
# Error: Wrong source type in path
with graph:
    person = Concept('person')
    organization = Concept('organization')
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
    
    # Wrong: work_for goes from person to org, not org to person
    ifL(
        organization(V.y),
        person(V.x, path=(V.y, work_for))
    )

# Error message:
# "The relation work_for is from person to organization, but you have
#  used it from organization to person. You can change 'work_for' to
#  'work_for.reversed' to go from organization to person."

# Fix: Use reversed relation
ifL(
    organization(V.y),
    person(V.x, path=(V.y, work_for.reversed))
)
```

#### 3. Concept Not in Graph
```python
# Error: Concept defined outside graph context
person = Concept('person')  # Wrong: outside with graph:

with graph:
    organization = Concept('organization')
    work_for = Concept('work_for')
    work_for.has_a(person, organization)  # Error!

# Error message:
# "Logical Element is incorrect - no graph found"

# Fix: Define inside graph context
with graph:
    person = Concept('person')
    organization = Concept('organization')
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
```

#### 4. Invalid Relation Cardinality
```python
# Error: Contains requires exactly 1 destination
sentence = Concept('sentence')
token = Concept('token')
word = Concept('word')

sentence.contains(token, word)  # Error!

# Error message:
# "The Contains relationship defined from sentence concept to concepts
#  token, word is not valid. The contains relationship can only be 
#  between one source and one destination concepts."

# Fix: One destination only
sentence.contains(token)
```

#### 5. HasA with Too Few Destinations
```python
# Error: HasA requires ≥2 destinations
work_for = Concept('work_for')
person = Concept('person')

work_for.has_a(person)  # Error!

# Error message:
# "The HasA relationship defined from work_for concept to concepts person
#  is not valid. The HasA relationship must be between one source and at
#  least two destination concepts."

# Fix: At least 2 destinations
work_for.has_a(person, organization)
```

### Debugging Tools

#### 1. Logging
```python
import logging

# Enable detailed logging
from domiknows.graph import ilpConfig
ilpConfig['log_level'] = logging.DEBUG
ilpConfig['ifLog'] = True

# Logs to: logs/ilpOntSolver.log
```

#### 2. Constraint String Representation
```python
# Get human-readable constraint
print(constraint.strEs())
# Output: "[person(x), entity(x)]"

# Get constraint name
print(constraint.name)
# Output: "LC0"

# Get constraint repr
print(repr(constraint))
# Output: "LC0(ifL)"
```

#### 3. Graph Visualization
```python
# Generate graph diagram
graph.visualize('debug/graph', open_image=True)

# Inspect:
# - All concepts
# - All relations with labels
# - Subgraph structure
```

#### 4. Concept Information
```python
# Get detailed concept info
info = graph.findConceptInfo(person)

print(f"Concept: {info['concept'].name}")
print(f"Is relation: {info['relation']}")
print(f"Has_a relations: {info['has_a']}")
print(f"Relation attributes: {info['relationAttrs']}")
print(f"Contains: {info['contains']}")
print(f"Contained in: {info['containedIn']}")
print(f"Is_a parents: {info['is_a']}")
print(f"Is root: {info['root']}")
```

#### 5. Variable Tracking
```python
# After graph context exit, check captured variables
print(graph.varNameReversedMap)
# Shows: {'person': <Concept>, 'work_for': <Relation>, ...}

# Check concept variable names
print(person.var_name)  # 'person'
print(work_for.var_name)  # 'work_for'
print(work_for.reversed.var_name)  # 'work_for.reversed'
```

---

## Performance Considerations

### 1. Memory Management
```python
# Clear caches periodically
Concept.clear()  # Clears name counters and object storage
Trial.clear()    # Clears trial trees and releases memory

# For long-running processes:
for epoch in range(100):
    train(...)
    if epoch % 10 == 0:
        Trial.clear()  # Free trial data
```

### 2. Constraint Complexity
```python
# Simple constraint (fast)
ifL(person(V.x), entity(V.x))

# Complex constraint (slower - multiple paths)
ifL(
    person(V.x),
    atLeastL(
        manager(V.m, path=(
            V.x, reports_to, V.y, reports_to, V.z, manages
        )),
        1
    )
)

# Optimization: Break into multiple simpler constraints
ifL(person(V.x), employee(V.x))
ifL(employee(V.x), entity(V.x))
```

### 3. Model Reuse
```python
# Enable model caching
graph = Graph('kg', reuse_model=True)

# First call: ~1s (builds model)
solver.calculateILPSelection(batch1, *concepts)

# Subsequent calls: ~0.1s (reuses model)
for batch in batches:
    solver.calculateILPSelection(batch, *concepts)
```

### 4. Lazy Evaluation
```python
# Concepts and relations are created lazily
with graph:
    # Only creates when accessed
    person = Concept('person')
    
    # Relation created only when called
    person.has_a(...)
```

### 5. Constraint Sampling
```python
# For large constraint groundings
large_constraint.sampleEntries = True

# Solver automatically samples instead of full grounding
# Trades accuracy for speed
```

---

## Integration with Other Components

### 1. With Sensors
```python
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor

with graph:
    person = Concept('person')
    
    with person:
        # Reader sensor
        age = Property('age')
        age.attach(ReaderSensor(keyword='age_value'))
        
        # Functional sensor
        adult = Property('adult')
        adult.attach(FunctionalSensor(
            formula=lambda age: age >= 18,
            dependencies=['age']
        ))
```

### 2. With DataNodes
```python
from domiknows.graph import DataNode

# Create DataNode from graph
with graph:
    person = Concept('person')
    organization = Concept('organization')

# Bind data to graph
root = DataNode(graph=graph)

# Add instances
person_dn = DataNode(ontologyNode=person, parent=root)
person_dn['age'] = 25
person_dn['name'] = 'Alice'

org_dn = DataNode(ontologyNode=organization, parent=root)
org_dn['name'] = 'Anthropic'
```

### 3. With Solvers
```python
from domiknows.solver import ilpOntSolverFactory

# Create solver from graph
solver = ilpOntSolverFactory.getOntSolverInstance(graph)

# ILP inference
solver.calculateILPSelection(datanode, person, organization)

# Constraint loss
lcLosses = solver.calculateLcLoss(datanode, tnorm='P')

# Verification
results = solver.verifyResultsLC(datanode)
```

### 4. With Models
```python
from domiknows.model import SampleLossModel

# Create model with graph
model = SampleLossModel(
    graph=graph,
    sensors=[person['age'], person['name']]
)

# Forward pass
outputs = model(batch)

# Constraint loss automatically computed
loss = model.get_constraint_loss(batch)
```

---

## Best Practices

### 1. Graph Organization
```python
# Use hierarchical subgraphs for modularity
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
# Name important constraints for debugging
critical_constraint = ifL(
    person(V.x),
    entity(V.x),
    name='person_is_entity'
)

# Easier to identify in logs and verification results
```

### 3. Priority Assignment
```python
# Use consistent priority levels
PRIORITY_CRITICAL = 100   # Must be satisfied
PRIORITY_HIGH = 80        # Important domain knowledge
PRIORITY_MEDIUM = 60      # Typical constraints
PRIORITY_LOW = 40         # Preferences
PRIORITY_HINT = 20        # Weak suggestions

ifL(person(V.x), entity(V.x), p=PRIORITY_CRITICAL)
atLeastL(employee(V.x), 1, p=PRIORITY_MEDIUM)
```

### 4. Variable Naming
```python
# Use descriptive variable names
ifL(
    employee(V.emp),
    manager(V.mgr, path=(V.emp, reports_to))
)

# Better than:
ifL(
    employee(V.x),
    manager(V.y, path=(V.x, reports_to))
)
```

### 5. Path Clarity
```python
# Document complex paths
ifL(
    project(V.proj),
    # Path: project -> managed_by -> manager -> reports_to -> executive
    executive(V.exec, path=(V.proj, managed_by, V.mgr, reports_to))
)
```

### 6. Testing Constraints
```python
# Test constraints incrementally
with graph:
    person = Concept('person')
    entity = Concept('entity')
    
    # Test simple constraint first
    ifL(person(V.x), entity(V.x))

# Verify before adding more
solver = ilpOntSolverFactory.getOntSolverInstance(graph)
results = solver.verifyResultsLC(test_data)

# Then add more complex constraints
with graph:
    organization = Concept('organization')
    person.not_a(organization)
```

### 7. Error Recovery
```python
# Wrap constraint creation in try-except
with graph:
    try:
        person = Concept('person')
        work_for = Concept('work_for')
        work_for.has_a(person, organization)
    except Exception as e:
        print(f"Error creating graph: {e}")
        # Clean up if needed
        Concept.clear()
        raise
```

---

## Migration from Older Versions

### Breaking Changes

1. **Constraint syntax**: `V.name` instead of `V('name')`
2. **Path syntax**: `V(name='x', v=path)` instead of older formats
3. **Equal relation**: Now requires `equality_mixin.py` to be applied
4. **Comparative counting**: New constraint types (greaterL, lessL, etc.)

### Migration Steps
```python
# Old code (v1.x):
from domiknows.graph import V, ifL
ifL(person(V('x')), entity(V('x')))

# New code (v2.x):
from domiknows.graph import V, ifL
ifL(person(V.x), entity(V.x))
# or
ifL(person(V(name='x')), entity(V(name='x')))
```
```python
# Old path syntax:
person(V('x'), path=[V('x'), work_for])

# New path syntax:
person(V.x, path=(V.x, work_for))
```
```python
# Enable equality (new requirement):
from domiknows.graph.equality_mixin import apply_equality_mixin
apply_equality_mixin(Concept)

# Now equality works:
person1.equal(person2)
```

---

## Further Reading

- **Logical Constraints**: See `logicalConstrain.py` for full constraint API
- **Solvers**: See `domiknows/solver/README.md` for constraint solving
- **Sensors**: See `domiknows/sensor/README.md` for data binding
- **Models**: See `domiknows/program/README.md` for neural-symbolic models
- **Examples**: See `examples/` directory for complete applications

---

## Requirements

- **Python**: 3.7+
- **PyTorch**: For differentiable constraints
- **Graphviz**: For visualization (optional)
- **owlready2**: For ontology loading (optional)

---

## Example: Complete Application
```python
from domiknows.graph import Graph, Concept, V, ifL, andL, atLeastL
from domiknows.solver import ilpOntSolverFactory

# 1. Create knowledge graph
with Graph('nlp_knowledge') as graph:
    # Define concepts
    entity = Concept('entity')
    person = Concept('person')
    organization = Concept('organization')
    location = Concept('location')
    
    # Create hierarchy
    person.is_a(entity)
    organization.is_a(entity)
    location.is_a(entity)
    
    # Disjoint concepts
    person.not_a(organization)
    person.not_a(location)
    organization.not_a(location)
    
    # Relations
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
    
    located_in = Concept('located_in')
    located_in.has_a(organization, location)
    
    # Logical constraints
    # 1. Type constraints (auto-generated by is_a)
    
    # 2. Domain/range constraints
    ifL(
        work_for(V.pair),
        andL(person(V.pair[0]), organization(V.pair[1])),
        p=100
    )
    
    # 3. Transitivity: if person works for org in location, 
    #    person is in location
    ifL(
        andL(
            work_for(V.x, V.y),
            located_in(V.y, V.z)
        ),
        located_in(V.x, V.z),
        p=80
    )
    
    # 4. At least one person per organization
    ifL(
        organization(V.y),
        atLeastL(person(V.x, path=(V.x, work_for, V.y)), 1),
        p=60
    )

# 2. Create solver
solver = ilpOntSolverFactory.getOntSolverInstance(graph)

# 3. Use in training
for batch in train_loader:
    # Forward pass
    predictions = model(batch)
    
    # Compute constraint loss
    lcLosses = solver.calculateLcLoss(
        batch,
        tnorm='P',  # Product t-norm
        counting_tnorm='L'  # Łukasiewicz for counting
    )
    
    # Aggregate losses
    data_loss = criterion(predictions, labels)
    constraint_loss = sum(lc['loss'] for lc in lcLosses.values())
    
    total_loss = data_loss + 0.5 * constraint_loss
    total_loss.backward()
    optimizer.step()

# 4. Use in inference
for batch in test_loader:
    predictions = model(batch)
    
    # Apply ILP inference
    solver.calculateILPSelection(
        batch,
        person, organization, location, work_for, located_in
    )
    
    # Extract results
    person_predictions = batch['person']['ILP']
    work_for_predictions = batch['work_for']['ILP']

# 5. Verify constraints
verification = solver.verifyResultsLC(test_batch, key='/ILP')

for lc_name, results in verification.items():
    print(f"{lc_name}: {results['satisfied']:.2f}% satisfied")
```

---
