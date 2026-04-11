# DomiKnowS Graph Builder Skill

## Purpose

Build DomiKnowS neural-symbolic AI framework graph definitions and executable constraints. DomiKnowS represents domain knowledge as a graph of **concepts** (entity types and attributes) connected by **relations**, with **logical constraints** that encode questions as differentiable losses for training.

## Two-Phase Architecture

The workflow is split into two independent phases:

- **Phase A: Domain Modeling** — Analyze a dataset or domain description to produce a graph of concepts and relations. No questions or constraints involved.
- **Phase B: Question Encoding** — Given an existing graph, convert natural language questions into executable constraints.

These phases are decoupled. Phase A produces a graph; Phase B consumes one. The user may provide a dataset now and questions later, supply both at once, or bring an existing graph and skip directly to Phase B.

| Scenario | Action |
|---|---|
| User provides a dataset, no questions yet | Phase A only |
| User provides a dataset and questions together | Phase A → Phase B |
| User provides an existing graph + questions | Phase B only |
| User wants to extend an existing graph with new concepts | Modify Phase A graph, re-validate |
| User wants to add questions to an existing graph+constraints | Phase B only, append new `execute()` calls |

---

# Phase A: Domain Modeling

## A.1 — Analyze the Dataset

Examine the dataset (JSON, CSV, schema, sample records, or domain description) to identify:

- **Entity types** — the core "things" in this domain (objects, people, documents, products, events, etc.)
- **Attributes** — properties entities have with a fixed set of values (color, shape, size, status, role, category, etc.)
- **Relations** — how entities connect to each other (spatial, semantic, hierarchical, temporal, causal, etc.)
- **Containment hierarchies** — parent-contains-child structures (scene contains objects, document contains sentences, batch contains samples, etc.)

### Modeling Decisions

Before generating the graph, decide how to represent each part of the domain:

**`is_a` subtypes vs `EnumConcept`:**
- Use **manual `is_a` subtypes** when attribute values will be referenced as standalone variables in constraints. Each value becomes its own Python variable: `red = Concept('red'); red.is_a(color)`. In constraints: `red('x')`.
- Use **`EnumConcept`** for closed value sets where you don't need standalone variables. Values are accessed only via dot notation: `color = EnumConcept('color', values=['red', 'blue'])`. In constraints: `color.red('x')`.
- Either approach supports `queryL` — both provide the required parent-with-children structure.

**Relation structure:**
- Define a **parent relation type** with `has_a()` when multiple relations share the same connection shape (e.g., all spatial relations connect object→object).
- Specific relations can be subtypes (`left_of.is_a(spatial_rel)`) or instances (`left_of = spatial_rel(name='left_of')`).
- Standalone relations that connect unique concept pairs can define their own `has_a()` directly.

**Containment:**
- Use `parent.contains(child)` for hierarchical nesting.
- The top-level container concept often uses `batch=True` (e.g., `Concept('image', batch=True)`).

---

## A.2 — Extract Concepts & Relations

Identify domain concepts and relations from the dataset. Output as JSON:

```json
{
  "concepts": [
    {"name": "snake_case_name", "description": "what this represents in the domain"}
  ],
  "relations": [
    {"name": "snake_case_name", "source": "concept_name", "target": "concept_name", "description": "what this connects"}
  ]
}
```

### Extraction Rules

- Extract ONLY domain concepts: entity types, attribute types, attribute values, and relations
- Do NOT extract concepts for logical operations — `existsL`, `andL`, `queryL`, counting, comparison, existence checks are **framework operators**, not domain concepts
- Do NOT create "has_property" relations (e.g., `has_color`, `has_size`) — model these as attributes via subtyping or `EnumConcept`
- All relation `source` and `target` must reference concepts defined in the `concepts` list
- Include both attribute parents (e.g., `color`, `size`) and their values (e.g., `red`, `large`)

---

## A.3 — Generate Graph Definition

### Imports

```python
from domiknows.graph import Graph, Concept, EnumConcept
```

### Structure

All definitions MUST be inside a `with Graph('name') as graph:` block.

### Concept Types

```python
with Graph('my_domain') as graph:
    # Containment hierarchy
    container = Concept('container', batch=True)
    item = Concept('item')
    container.contains(item)

    # Entity subtypes
    person = Concept('person')
    person.is_a(item)

    # Attributes as manual is_a subtypes
    color = Concept('color')
    red = Concept('red')
    red.is_a(color)
    blue = Concept('blue')
    blue.is_a(color)

    # Attributes as EnumConcept
    status = EnumConcept('status', values=['active', 'inactive', 'pending'])
    # Access: status.active, status.inactive, status.pending (dot notation ONLY)
```

### Relations

Binary relations are themselves Concepts that use `has_a()` to declare which two concepts they connect. `has_a()` returns two relation variables.

```python
    # Parent relation type (defines connection shape)
    pair = Concept(name='pair')
    (rel_arg1, rel_arg2) = pair.has_a(arg1=item, arg2=item)

    # Specific relation as subtype
    left_of = Concept('left_of')
    left_of.is_a(pair)

    # Specific relation as instance
    right_of = pair(name='right_of')

    # Standalone relation with its own has_a
    work_for = Concept('work_for')
    (rel_wf_person, rel_wf_org) = work_for.has_a(person, organization)
```

### Graph Generation Rules

1. All code inside `with Graph('name') as graph:` block
2. Every concept must be defined BEFORE it is referenced
3. `has_a()` for binary relations — returns two relation variables
4. `is_a()` for subtyping (`child.is_a(parent)`)
5. `contains()` for containment hierarchies
6. **No logical constraints at this stage** — no `ifL`, `andL`, `orL`, `existsL`, `execute`, or any constraint operator
7. Code must be self-contained and executable

---

## A.4 — Validate Graph

After generating, validate that:

1. **Coverage** — Every extracted concept and relation appears in the code
2. **Execution** — The code runs without errors with DomiKnowS imports

### Auto-Imports (prepended automatically at execution time)

```python
from domiknows.graph import Graph, Concept, EnumConcept
from domiknows.graph import Relation
from domiknows.graph import (
    ifL, andL, orL, nandL, norL, xorL, notL, equivalenceL,
    eqL, fixedL, forAllL,
    existsL, atLeastL, atMostL, exactL,
    existsAL, atLeastAL, atMostAL, exactAL,
    greaterL, greaterEqL, lessL, lessEqL, equalCountsL,
    sumL, iotaL, queryL, sameL, differentL,
    execute,
)
from domiknows.graph import Property
from domiknows.graph.relation import disjoint
```

### Framework Validation

Run `./scripts/validate.sh graph_file.py` to:
- Parse and execute the graph
- Introspect concepts, relations, and predicates via `graph.getAllConceptNames()`, `graph.relations`, `graph.print_predicates()`
- Run `checkLcCorrectness(graph)` if constraints are present

Save the validated graph. It is now ready for Phase B.

---
---

# Phase B: Question Encoding

**Prerequisite:** A validated DomiKnowS graph from Phase A or provided by the user.

## B.1 — Inspect the Graph

Before encoding questions, understand the available graph:
- What concepts exist? (`graph.getAllConceptNames()`)
- Which concepts have `is_a` children? (Eligible for `queryL`, `sameL`, `differentL`)
- Which concepts are `EnumConcept`? (Must use dot notation in constraints; eligible for `sameL`/`differentL` if connected via `is_a`)
- Which relations exist and what do they connect? (Determines variable arity)
- What are the predicate shapes? (`graph.print_predicates()`)

---

## B.2 — Question Type → Constraint Pattern

### Existence: "Is there a [property] [object]?"

```python
execute(existsL(andL(property('x'), object('x'))))
# label: True/False
```

### Relation: "Is [X] [relation] [Y]?"

```python
execute(existsL(relation(iotaL(X_description), iotaL(Y_description))))
# label: True/False
```

### Counting: "How many [X]...?" (answer = N)

```python
execute(exactL(andL(property('x'), object('x')), N))
# label: True
```

### At Least / More Than: "Are there at least N [X]?"

```python
execute(atLeastL(andL(property('x'), object('x')), N))
# label: True/False
```

### Comparative: "Are there more [X] than [Y]?"

```python
execute(greaterL(X_filter('x'), Y_filter('y')))
# label: True/False
```

### Attribute Query: "What [attribute] is THE [description]?"

```python
execute(queryL(attribute_parent, iotaL(description_filter)))
# label: integer index into subclass list (NOT True/False)
```

### Implication: "If [X] then [Y]"

```python
execute(ifL(X_condition, Y_condition))
# label: True/False
```

### Same Attribute: "Do [X] and [Y] have the same [attribute]?"

```python
execute(sameL(attribute_parent, iotaL(X_description), iotaL(Y_description)))
# label: True/False
```

### Different Attribute: "Do [X] and [Y] have different [attribute]?"

```python
execute(differentL(attribute_parent, iotaL(X_description), iotaL(Y_description)))
# label: True/False
```

---

## B.3 — Constraint Operators Reference

### Boolean Logic

| Operator | Meaning | Example |
|---|---|---|
| `andL(a, b)` | a AND b | `andL(red('x'), cube('x'))` |
| `orL(a, b)` | a OR b | `orL(metal('x'), rubber('x'))` |
| `notL(a)` | NOT a | `notL(large('x'))` |
| `ifL(a, b)` | a → b (implication) | `ifL(sphere('x'), round('x'))` |
| `nandL(a, b)` | NOT(a AND b) | `nandL(red('x'), blue('x'))` |
| `equivalenceL(a, b)` | a ↔ b | `equivalenceL(big('x'), large('x'))` |

### Counting — Element-wise (within a single sample)

| Operator | Meaning | Example |
|---|---|---|
| `existsL(a)` | ∃x: at least 1 | `existsL(red('x'))` |
| `atLeastL(a, k)` | count ≥ k | `atLeastL(sphere('x'), 3)` |
| `atMostL(a, k)` | count ≤ k | `atMostL(cube('x'), 2)` |
| `exactL(a, k)` | count == k | `exactL(green('x'), 4)` |

### Counting — Batch-level (across all instances)

| Operator | Meaning |
|---|---|
| `existsAL(a)` | At least 1 across batch |
| `atLeastAL(a, k)` | count ≥ k across batch |
| `atMostAL(a, k)` | count ≤ k across batch |
| `exactAL(a, k)` | count == k across batch |

### Comparative Counting

| Operator | Meaning | Example |
|---|---|---|
| `greaterL(a, b)` | count(a) > count(b) | `greaterL(red('x'), blue('y'))` |
| `greaterEqL(a, b)` | count(a) ≥ count(b) | |
| `lessL(a, b)` | count(a) < count(b) | |
| `lessEqL(a, b)` | count(a) ≤ count(b) | |
| `equalCountsL(a, b)` | count(a) == count(b) | |

### Entity Selection — `iotaL` (Definite Description)

`iotaL` selects THE unique entity satisfying a condition. Unlike `existsL` (boolean), `iotaL` **returns the entity itself** (a selection distribution).

```python
iotaL(sphere('x'))                          # THE sphere
iotaL(andL(large('x'), red('x')))           # THE large red object

# Nested: "Is something left of THE blue cube?"
existsL(left_of('x', iotaL(andL(blue('y'), cube('y')))))
```

### Attribute Query — `queryL`

`queryL` asks "What is the \<attribute\> of THE \<entity\>?"

```python
queryL(material, iotaL(andL(big('x'), sphere('x'))))  # "What material is the big sphere?"
queryL(color, iotaL(andL(small('x'), cube('x'))))      # "What color is the small cube?"
```

**Requirements:**
- First argument must be a Concept with `is_a()` subclasses OR an `EnumConcept`
- Label is an **integer index** into the subclass list (not True/False)

### Same/Different Attribute — `sameL`, `differentL`

`sameL` checks whether all referenced entities share the same value of a multiclass attribute.
`differentL` is its negation — true when at least one entity differs.

```python
sameL(color, 'x', 'y')           # "Do x and y have the same color?"
differentL(size, 'x', 'y')       # "Do x and y have different sizes?"
```

Semantics:
- `sameL(concept, 'x', 'y')` = OR_j( AND_i( entity_i has subclass_j ) )
- `differentL(concept, 'x', 'y')` = NOT( sameL(...) )

**Requirements:**
- First argument must be a Concept with `is_a()` subclasses OR an `EnumConcept`
- The concept MUST be structurally connected to its parent via `is_a` (use `parent(name='attr', ConceptClass=EnumConcept, values=[...])` instead of standalone `EnumConcept(...)`)
- Works with any number of entity variables (2 or more)
- Can be combined with other constraints: `ifL(pair('x', 'y'), sameL(color, 'x', 'y'))`

---

## B.4 — Variable Syntax

Variables are **plain quoted strings** passed as positional arguments.

```python
person('x')            # single-variable concept
work_for('x', 'y')     # binary relation (two variables)
```

- Same variable string = same entity binding
- Different variable names = different entities (independent iteration)
- First use of a variable **defines** it; reuse = same entity

### Nesting with `iotaL`

Inner variables are selected first, then outer constraints use that selection:

```python
# "x is left of THE blue sphere"
left_of('x', iotaL(andL(blue('y'), sphere('y'))))
```

---

## B.5 — EnumConcept vs Manual `is_a` in Constraints

**Manual `is_a`** — children are standalone Python variables:

```python
# Graph:
color = Concept('color')
red = Concept('red'); red.is_a(color)

# Constraints: use variable directly
red('x')
```

**`EnumConcept`** — children accessed ONLY via dot notation:

```python
# Graph:
color = EnumConcept('color', values=['red', 'blue'])

# Constraints: MUST use dot notation
color.red('x')
```

**Attached `EnumConcept`** — same dot notation:

```python
# Graph:
nli_class = pair(name='nli_class', ConceptClass=EnumConcept,
                 values=['entailment', 'contradiction', 'neutral'])

# Constraints:
nli_class.entailment('x')
```

---

## B.6 — Path Expressions

Path expressions navigate from one concept through a relation to reach another concept. They are required when accessing an attribute connected via a relation.

```python
# Graph:
has_color = Concept('has_color')
(rel_hc_ball, rel_hc_color) = has_color.has_a(ball, color)

# Constraint: "x is a ball that is green (via has_color relation)"
andL(ball('x'), color.green('x', path=('x', rel_hc_color.name)))
```

**When to use paths vs direct arguments:**
- Attribute connected via a relation → MUST use path: `color.green('x', path=('x', rel_hc_color.name))`
- Binary relation with `has_a(A, B)` → two variables directly: `left_of('x', 'y')`
- Direct `is_a` child (no relation involved) → use directly: `ball('x')`

---

## B.7 — Generating Constraints

### Option 1: Inline `execute()` calls (appended to graph in a `with graph:` block)

```python
with graph:
    # [Question type]: [question text]
    execute([constraint_expression])
```

### Option 2: String-based `compile_executable` (runtime API)

```python
qa_data = [
    {"constraint": '[constraint_string]', "label": [label_value]},
]

logic_dataset = graph.compile_executable(
    qa_data,
    logic_keyword='constraint',
    logic_label_keyword='label'
)
```

At compile time:
1. The string is auto-wrapped with `execute()` if needed
2. Function names are resolved to fully-qualified paths
3. The expression is `eval()`-ed in a namespace containing the graph's concepts
4. The constraint is stored in `graph.executableLCs`; labels in `graph.executableLCsLabels`
5. During training, each sample activates its own constraint as a differentiable loss

### Constraint Generation Rules

1. Only reference concepts and relations that exist in the graph — do NOT invent concepts
2. Variables are plain single-quoted strings: `concept('x')` not `concept("x")`
3. Binary relations take two variable arguments: `left_of('x', 'y')`
4. For manual `is_a` children, use the variable directly (`red('x')`); for `EnumConcept`, use dot notation (`color.red('x')`)
5. `iotaL` selects an entity; `existsL` checks existence — do NOT confuse them
6. `queryL` first argument must have `is_a()` children or be `EnumConcept`; label is an integer index
7. Always provide explicit count for counting operators: `exactL(green('x'), 3)` not `exactL(green('x'))`
8. Each question produces EXACTLY ONE `execute()` call
9. Add a comment indicating the question type: `# Existence:`, `# Counting:`, `# Relation:`, `# Query:`, `# Comparative:`, `# Same:`, `# Different:`
10. For `sameL`/`differentL`, the attribute concept must be connected to its parent via `is_a` (use `parent(name='attr', ConceptClass=EnumConcept, values=[...])` rather than standalone `EnumConcept(...)`)

---

## B.8 — Validate Full Code

Run `./scripts/validate.sh output_file.py` to verify:
- Graph + constraints execute successfully
- `checkLcCorrectness(graph)` passes — validates all concepts in constraints exist, `queryL` has multiclass arguments, counting constraints have valid cardinality, relations have proper `has_a` structure, variables are consistent
- If `qa_data` is present, `graph.compile_executable()` succeeds

---
---

# Prompt Engineering Principles

- **NEVER include incorrect patterns as warnings or negative examples** — mentioning bad patterns risks the LLM reproducing them. Remove incorrect patterns entirely from prompts.
- Keep prompts focused on what TO do, not what NOT to do (with rare exceptions for critical mistakes).
- Each pipeline step has a dedicated concern — do not mix graph generation with constraint encoding.

---

# Error Handling

- Graph execution errors are enriched: `ELC0`, `ELC1` references are mapped back to original constraint expressions for debugging
- Up to 5 fix attempts per step — the error message and current code are sent back for correction
- Execute count is validated after fixes to ensure no constraints were dropped
- `checkLcCorrectness` provides detailed error messages with location info and concept suggestions for typos