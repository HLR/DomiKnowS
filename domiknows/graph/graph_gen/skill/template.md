# DomiKnowS Graph Builder — Template

This template covers the two phases of building a DomiKnowS graph:

- **Phase A: Domain Modeling** — Analyze the dataset to define a graph of concepts and relations. No questions or constraints yet.
- **Phase B: Question Encoding** — Given an existing graph, convert natural language questions into executable constraints.

These phases are independent. Phase A produces a graph; Phase B consumes it. The user may provide a dataset now and questions later, or supply both together, or bring an existing graph and skip Phase A entirely.

---

## Phase A: Domain Modeling

### A.1 — Understand the Dataset

**Dataset source:** `[path or description]`

**Dataset format:** `[JSON / CSV / description / schema / sample records]`

Analyze the dataset to identify:
- What **entities** exist (the "things" — objects, people, documents, products, etc.)
- What **attributes** those entities have (properties with a fixed set of values — color, type, status, role, etc.)
- What **relations** connect entities to each other (spatial, semantic, hierarchical, temporal, etc.)
- What **containment** hierarchies exist (a scene contains objects, a document contains sentences, a batch contains samples, etc.)

**Domain summary:**

```
Entity types:     [list the core things in this domain]
Attributes:       [list properties with their possible values]
Relations:        [list how entities connect to each other]
Containment:      [list parent-contains-child hierarchies]
```

**Key modeling decisions:**

- [ ] Which attributes should use `is_a` subtypes (separate Python variables, use when attributes appear as concepts in constraints)?
- [ ] Which attributes should use `EnumConcept` (dot-notation access, use for closed value sets that won't appear individually in constraints)?
- [ ] Which relations need a shared parent type (e.g., all spatial relations share a `pair` parent with `has_a`)?
- [ ] Is there a batch-level container concept (e.g., `image`, `document`, `scene` with `batch=True`)?

---

### A.2 — Extract Concepts & Relations

```json
{
  "concepts": [
    {"name": "[snake_case_name]", "description": "[what this represents in the domain]"}
  ],
  "relations": [
    {"name": "[snake_case_name]", "source": "[concept]", "target": "[concept]", "description": "[what this connects]"}
  ]
}
```

**Checklist:**
- [ ] Only domain concepts — no logical operators (exists, count, query, comparison are framework operators)
- [ ] No "has_property" relations — model attributes via subtyping (`is_a`) or `EnumConcept`
- [ ] All relation sources and targets exist in the concepts list
- [ ] Attribute values are listed as individual concepts (e.g., `red`, `large`, `employee`)
- [ ] Attribute parents are listed for grouping (e.g., `color`, `size`, `role`)

---

### A.3 — Generate Graph Definition

```python
from domiknows.graph import Graph, Concept, EnumConcept

with Graph('[domain_name]') as graph:
    # ── Containment hierarchy ──
    [container] = Concept('[container]', batch=True)
    [item] = Concept('[item]')
    [container].contains([item])

    # ── Entity types (is_a subtypes of item, if applicable) ──
    [entity_type] = Concept('[entity_type]')
    [entity_type].is_a([item])

    # ── Attributes as manual is_a subtypes ──
    # (Use when attribute values will be referenced directly in constraints)
    [attr_parent] = Concept('[attr_parent]')
    [attr_value] = Concept('[attr_value]')
    [attr_value].is_a([attr_parent])

    # ── Attributes as EnumConcept ──
    # (Use for closed value sets; access via dot notation: attr.value)
    [attr_enum] = EnumConcept('[attr_enum]', values=['[val1]', '[val2]', '[val3]'])

    # ── Relations ──
    # Parent relation type with has_a (defines the connection shape)
    [rel_parent] = Concept('[rel_parent]')
    (rel_[src], rel_[tgt]) = [rel_parent].has_a([source_concept], [target_concept])

    # Specific relation subtypes
    [specific_rel] = [rel_parent](name='[specific_rel]')
    # OR as standalone:
    [standalone_rel] = Concept('[standalone_rel]')
    [standalone_rel].is_a([rel_parent])
```

**Checklist:**
- [ ] All code inside `with Graph('...') as graph:` block
- [ ] Every concept defined before it is referenced
- [ ] **No logical constraints** — no `ifL`, `andL`, `existsL`, `execute`, or any constraint operator
- [ ] All extracted concepts and relations are present
- [ ] `has_a()` returns two relation variables for every binary relation
- [ ] `is_a()` used for subtyping, `contains()` for containment
- [ ] `EnumConcept` values accessed only via dot notation (no standalone variables created)
- [ ] Code is self-contained and executable

---

### A.4 — Validate Graph

Run `./scripts/validate.sh [graph_file.py]` to verify:

```
Coverage:  ✅ All [N] concepts and [M] relations present
Execution: ✅ Graph executed successfully
Structure: ✅ [N] concepts, [M] relations, [P] predicates
```

Save the validated graph. It is now ready to accept questions in Phase B.

---
---

## Phase B: Question Encoding

**Prerequisite:** A validated DomiKnowS graph from Phase A (or provided by the user).

### B.1 — Understand the Graph

Before encoding questions, inspect the available graph:
- What concepts exist? (Use `graph.getAllConceptNames()` or read the code)
- Which concepts have `is_a` children? (These can be used with `queryL`, `sameL`, `differentL`)
- Which concepts are `EnumConcept`? (Must use dot notation: `color.red('x')`; eligible for `sameL`/`differentL` if connected via `is_a`)
- Which relations exist and what do they connect? (Determines variable arity)
- What are the predicates? (Use `graph.print_predicates()` for variable shapes)

**Available concepts:** `[list from graph]`
**Available relations:** `[list from graph with source→target]`
**Multiclass parents (for queryL, sameL, differentL):** `[concepts that have is_a children or are EnumConcept]`

---

### B.2 — Classify Each Question

For each question, determine its type and map to the appropriate constraint pattern:

| Question Pattern | Constraint Template | Label Type |
|---|---|---|
| "Is there a [X]?" | `existsL(X_filter)` | `True`/`False` |
| "Is [X] [relation] [Y]?" | `existsL(relation(iotaL(X), iotaL(Y)))` | `True`/`False` |
| "How many [X]...?" (answer=N) | `exactL(X_filter, N)` | `True` |
| "At least / more than N [X]?" | `atLeastL(X_filter, N)` | `True`/`False` |
| "Are there more [X] than [Y]?" | `greaterL(X_filter, Y_filter)` | `True`/`False` |
| "What [attr] is THE [X]?" | `queryL(attr_parent, iotaL(X_filter))` | `int` (index) |
| "Do [X] and [Y] have the same [attr]?" | `sameL(attr_parent, iotaL(X_filter), iotaL(Y_filter))` | `True`/`False` |
| "Do [X] and [Y] have different [attr]?" | `differentL(attr_parent, iotaL(X_filter), iotaL(Y_filter))` | `True`/`False` |
| "If [X] then [Y]" | `ifL(X_condition, Y_condition)` | `True`/`False` |

---

### B.3 — Generate Executable Constraints

#### Option 1: Inline `execute()` calls (appended to graph)

```python
with graph:
    # [Question type]: [question text]
    execute([constraint_expression])

    # [Question type]: [question text]
    execute([constraint_expression])
```

#### Option 2: String-based `compile_executable` (runtime API)

```python
qa_data = [
    {"constraint": '[constraint_string]', "label": [label_value]},
    {"constraint": '[constraint_string]', "label": [label_value]},
]

logic_dataset = graph.compile_executable(
    qa_data,
    logic_keyword='constraint',
    logic_label_keyword='label'
)
```

**Checklist:**
- [ ] Each question produces exactly one `execute()` call or one `qa_data` entry
- [ ] Only concepts from the graph are used — no invented concepts
- [ ] Variables are quoted strings: `'x'`, `'y'`
- [ ] `iotaL` for "THE [specific thing]" (entity selection), `existsL` for "is there" (boolean check)
- [ ] `queryL` first argument has `is_a()` children or is `EnumConcept`
- [ ] `queryL` label is an integer index, not `True`/`False`
- [ ] Counting operators have explicit count argument: `exactL(..., 3)` not `exactL(...)`
- [ ] Question type comment above each constraint
- [ ] Binary relations use two variables: `left_of('x', 'y')`
- [ ] `EnumConcept` values use dot notation: `color.red('x')` not `red('x')`
- [ ] Manual `is_a` children use direct variables: `red('x')` not `color.red('x')`
- [ ] Path expressions used when accessing attributes through a relation
- [ ] For `sameL`/`differentL`, attribute concept is connected to parent via `is_a` (not standalone `EnumConcept`)

---

### B.4 — Validate Full Code

Run `./scripts/validate.sh [output_file.py]` to verify constraints pass DomiKnowS validation:

```
Execution:   ✅ Graph with constraints executed successfully
Constraints: ✅ checkLcCorrectness passed
  - All concepts in constraints exist in the graph
  - queryL has proper multiclass first argument
  - Counting constraints have valid cardinality
  - Relations have proper has_a structure
  - Variables are consistently defined and used
```

---
---

## Choosing Phase A vs Phase B

| Scenario | What to do |
|---|---|
| User provides a dataset, no questions yet | Phase A only — produce the graph |
| User provides a dataset and questions together | Phase A → Phase B |
| User provides an existing graph + questions | Phase B only — encode against existing graph |
| User provides an existing graph, wants to add more concepts | Phase A.3 modification, then re-validate |
| User wants to add more questions to an existing graph+constraints | Phase B only — append new `execute()` calls |

---

## Common Pitfalls

| Mistake | Consequence | Fix |
|---|---|---|
| Adding constraints in graph definition (Phase A) | Execution error or mixed concerns | Keep Phase A constraint-free; constraints go in Phase B only |
| Using `red('x')` when `red` is from `EnumConcept` | `NameError` — `red` is not a standalone variable | Use `color.red('x')` (dot notation) |
| Using `color.red('x')` when `red` is manual `is_a` | Incorrect — `red` IS a standalone variable | Use `red('x')` directly |
| `existsL` instead of `iotaL` for "THE blue cube" | Returns boolean instead of entity | `iotaL` selects an entity; `existsL` checks existence |
| `queryL(red, ...)` instead of `queryL(color, ...)` | `red` has no children to query over | First argument must be the parent with `is_a` children |
| `exactL(green('x'))` without count | Defaults to 1, probably wrong | Always provide explicit count: `exactL(green('x'), N)` |
| Inventing concepts not in the graph | `NameError` at execution | Only use concepts defined in the graph |
| Forgetting path expression for relation-connected attributes | Constraint ignores the relation path | Use `path=('x', rel_name.name)` syntax |
| `sameL(color, ...)` with standalone `EnumConcept` | Empty variable resolution, no results | Connect via `is_a`: `color = parent(name='color', ConceptClass=EnumConcept, values=[...])` |
| Using `andL` to check same attribute across entities | Checks properties on one entity, not cross-entity | Use `sameL(attribute, 'x', 'y')` for cross-entity comparison |