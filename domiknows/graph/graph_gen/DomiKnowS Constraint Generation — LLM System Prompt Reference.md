# DomiKnowS Executable Constraint Generation — LLM Reference

## Purpose

This document teaches an LLM how to convert natural language questions into DomiKnowS executable constraint expressions. These string expressions are compiled at runtime via `graph.compile_executable()` and become per-sample differentiable losses during training.

---

## 1. What Are Executable Constraints?

Each data item contains a **constraint string** and a **label**. The constraint string is a Python expression using DomiKnowS logical operators. The label indicates the expected truth value.

```python
qa_data = [
    {"constraint": 'existsL(andL(green("x"), left_of("x", iotaL(brown("y")))))',
     "label": True},
]

# Compiled into executable constraints:
logic_dataset = graph.compile_executable(
    qa_data,
    logic_keyword='constraint',
    logic_label_keyword='label'
)
```

**What happens at compile time:**
1. The string is auto-wrapped with `execute()` if needed
2. Function names are resolved to fully-qualified paths (`andL` → `domiknows.graph.logicalConstrain.andL`)
3. The expression is `eval()`-ed in a namespace containing the graph's concepts and relations
4. The resulting constraint is stored in `graph.executableLCs`
5. During training, each sample activates its own constraint — different samples can have different constraints

---

## 2. Available Constraint Operators

### 2.1 Boolean Logic

| Operator | Meaning | Example |
|----------|---------|---------|
| `andL(a, b)` | a AND b | `andL(red("x"), cube("x"))` |
| `orL(a, b)` | a OR b | `orL(metal("x"), rubber("x"))` |
| `notL(a)` | NOT a | `notL(large("x"))` |
| `ifL(a, b)` | a → b (implication) | `ifL(sphere("x"), round("x"))` |
| `nandL(a, b)` | NOT(a AND b) | `nandL(red("x"), blue("x"))` |

### 2.2 Counting Constraints (Element-wise)

These count how many entities satisfy a condition **within a single sample**.

| Operator | Meaning | Example |
|----------|---------|---------|
| `existsL(a)` | ∃x: a(x) — at least 1 | `existsL(red("x"))` |
| `atLeastL(a, k)` | count ≥ k | `atLeastL(sphere("x"), 3)` |
| `atMostL(a, k)` | count ≤ k | `atMostL(cube("x"), 2)` |
| `exactL(a, k)` | count == k | `exactL(green("x"), 4)` |

### 2.3 Global (Accumulated) Counting

Batch-level constraints (across all instances):

| Operator | Meaning |
|----------|---------|
| `existsAL(a)` | At least 1 across batch |
| `atLeastAL(a, k)` | count ≥ k across batch |
| `atMostAL(a, k)` | count ≤ k across batch |
| `exactAL(a, k)` | count == k across batch |

### 2.4 Comparative Counting

Compare counts between two variable sets:

| Operator | Meaning | Example |
|----------|---------|---------|
| `greaterL(a, b)` | count(a) > count(b) | `greaterL(red("x"), blue("y"))` |
| `greaterEqL(a, b)` | count(a) ≥ count(b) | |
| `lessL(a, b)` | count(a) < count(b) | |
| `lessEqL(a, b)` | count(a) ≤ count(b) | |
| `equalCountsL(a, b)` | count(a) == count(b) | |

### 2.5 Summation

| Operator | Meaning |
|----------|---------|
| `sumL(a)` | Sum of variable probabilities (soft count) |

### 2.6 Entity Selection — `iotaL` (Definite Description)

**`iotaL` selects THE unique entity satisfying a condition** — based on Russell's ι-operator. Unlike `existsL` which returns a boolean, `iotaL` **returns the entity itself** (a selection distribution).

```python
# Select THE sphere in the scene
iotaL(sphere("x"))

# Select THE large red object
iotaL(andL(large("x"), red("x")))

# Can be nested: "Is something left of THE blue cube?"
existsL(left_of("x", iotaL(andL(blue("y"), cube("y")))))
```

**Key properties:**
- Presupposes **existence** (at least one entity satisfies)
- Presupposes **uniqueness** (exactly one entity is selected)
- Returns a soft selection (probability distribution via softmax) during training
- Gradient flows through softmax for differentiable entity selection

### 2.7 Attribute Query — `queryL`

**`queryL` asks "What is the \<attribute\> of THE \<entity\>?"** — combines with `iotaL` to answer "what" questions.

```python
# "What material is the big sphere?"
queryL(material, iotaL(andL(big("x"), sphere("x"))))

# "What color is the small cube?"
queryL(color, iotaL(andL(small("x"), cube("x"))))

# "What is left of the blue cylinder?"
queryL(object_type, iotaL(left_of("x", iotaL(andL(blue("y"), cylinder("y"))))))
```

**Requirements for the first argument:**
- Must be a `Concept` with subclasses via `is_a()`, OR
- An `EnumConcept` with explicit values

**Label for queryL:** The label is an integer index into the subclass list (not True/False).

---

## 3. Variable Syntax

Variables are **plain strings** in constraint expressions. Each string variable iterates over all candidates of the concept it's used with.

```python
# "x" iterates over all objects
green("x")

# "x" and "y" iterate independently
left_of("x", "y")

# Same variable reused = same entity
andL(green("x"), cube("x"))  # x is green AND x is a cube
```

**Rules:**
- First use of a variable **defines** it (iterates over that concept's candidates)
- Reuse of the same variable name = same entity binding
- Different variable names = different entities (independent iteration)

### Relations with Two Arguments

When a relation has two arguments (e.g., `left_of` connecting two objects via `has_a`):

```python
# left_of("x", "y") means: object x is left of object y
# "x" binds to the first has_a destination, "y" to the second
left_of("x", "y")
```

### Nesting Variables with `iotaL`

When `iotaL` is nested, the inner variable is selected first, then the outer constraint uses that selection:

```python
# "x" is left of THE blue sphere (selected by iotaL)
left_of("x", iotaL(andL(blue("y"), sphere("y"))))
```

---

## 4. Question Type → Constraint Pattern Mapping

### 4.1 Existence Questions (Boolean, label=True/False)

**Pattern:** "Is there a [property] [object]?" → `existsL(...)`

| Question | Constraint | Label |
|----------|-----------|-------|
| "Is there a red cube?" | `existsL(andL(red("x"), cube("x")))` | `True` |
| "Is there a blue sphere?" | `existsL(andL(blue("x"), sphere("x")))` | `False` |

### 4.2 Relation Questions (Boolean, label=True/False)

**Pattern:** "Is X [relation] Y?" → `existsL(relation(filter_x, filter_y))`

| Question | Constraint | Label |
|----------|-----------|-------|
| "Is the red cube left of the blue sphere?" | `existsL(left_of(iotaL(andL(red("x"), cube("x"))), iotaL(andL(blue("y"), sphere("y")))))` | `True` |
| "Is there anything behind the green cylinder?" | `existsL(behind("x", iotaL(andL(green("y"), cylinder("y")))))` | `True` |

### 4.3 Counting Questions (Boolean framed, label=True/False)

**Pattern:** "How many [property] [objects] are [relation] [reference]?" → counting constraint with answer as the limit

| Question | Answer | Constraint | Label |
|----------|--------|-----------|-------|
| "How many green balls are left of the brown ball?" | 3 | `exactL(andL(green("x"), left_of("x", iotaL(brown("y")))), 3)` | `True` |
| "Are there at least 2 red cubes?" | yes | `atLeastL(andL(red("x"), cube("x")), 2)` | `True` |
| "Are there more red objects than blue?" | yes | `greaterL(red("x"), blue("y"))` | `True` |

**Key insight for counting questions:** The numeric answer becomes the limit parameter of the counting constraint. The label is `True` because the constraint with that exact count should be satisfied.

### 4.4 Query Questions (Multiclass, label=index)

**Pattern:** "What [attribute] is THE [description]?" → `queryL(attribute, iotaL(description))`

| Question | Answer | Constraint | Label |
|----------|--------|-----------|-------|
| "What color is the large sphere?" | "red" | `queryL(color, iotaL(andL(large("x"), sphere("x"))))` | `2` (index of "red") |
| "What material is the small cube?" | "metal" | `queryL(material, iotaL(andL(small("x"), cube("x"))))` | `0` (index of "metal") |
| "What shape is left of the blue thing?" | "cube" | `queryL(shape, iotaL(left_of("x", iotaL(blue("y")))))` | `1` (index of "cube") |

### 4.5 Complex / Compound Questions

Combine patterns for multi-step reasoning:

| Question | Constraint |
|----------|-----------|
| "Is the material of the big sphere the same as the small cube?" | `andL(queryL(material, iotaL(andL(big("x"), sphere("x")))), queryL(material, iotaL(andL(small("y"), cube("y")))))` — or use `same_material` relation if available |
| "How many cubes are behind the thing that is left of the red sphere?" | `exactL(andL(cube("x"), behind("x", iotaL(left_of("z", iotaL(andL(red("y"), sphere("y"))))))), N)` |

---

## 5. Critical Rules for Constraint Generation

### MUST follow:
1. **Only reference concepts and relations that exist in the graph.** If the graph has no `color` concept, you cannot use `color("x")`.
2. **Variables are strings.** Always quote them: `red("x")` not `red(x)`.
3. **Use the correct label type.** Boolean questions → `True`/`False`. Query questions → integer index.
4. **Match the answer to the constraint.** For counting, the numeric answer becomes the limit. For existence, the boolean answer becomes the label.
5. **Nest `iotaL` for definite references.** "THE red cube" → `iotaL(andL(red("x"), cube("x")))`, not just `andL(red("x"), cube("x"))`.

### MUST NOT do:
1. **Don't invent concepts.** If `green` isn't in the graph, don't use it.
2. **Don't confuse `iotaL` with `existsL`.** `iotaL` selects an entity; `existsL` checks if any entity satisfies.
3. **Don't use `queryL` without a multiclass concept.** The first argument must have subclasses or be an EnumConcept.
4. **Don't omit the count for counting questions.** `exactL(green("x"))` defaults to 1 — always provide the explicit count: `exactL(green("x"), 3)`.

---

## 6. How `compile_executable` Processes Your Strings

```python
# Your constraint string:
'exactL(andL(green("x"), left_of("x", iotaL(brown("y")))), 3)'

# Step 1: Auto-wrapped with execute()
'execute(exactL(andL(green("x"), left_of("x", iotaL(brown("y")))), 3))'

# Step 2: Function names resolved to full paths
'domiknows.graph.logicalConstrain.execute(domiknows.graph.logicalConstrain.exactL(...))'

# Step 3: eval() in namespace containing graph concepts
# green, brown, left_of etc. are looked up as graph concept variables

# Step 4: Stored as ELC0, ELC1, etc. in graph.executableLCs
# Step 5: Label stored in graph.executableLCsLabels
```

The namespace includes all concepts defined in `graph.varContext` — these are the Python variables created when concepts are defined in a `with Graph(...) as graph:` block.

---

## 7. Complete Example: CLEVR-style VQA

Given this graph:
```python
with Graph('clevr') as graph:
    image = Concept('image', batch=True)
    object = Concept('object')
    image.contains(object)

    # Attributes (each is binary: is/is-not)
    large = Concept('large');  small = Concept('small')
    red = Concept('red');      blue = Concept('blue');  green = Concept('green')
    cube = Concept('cube');    sphere = Concept('sphere'); cylinder = Concept('cylinder')
    metal = Concept('metal');  rubber = Concept('rubber')

    # Multiclass parents for queryL
    size = Concept('size');    large.is_a(size);  small.is_a(size)
    color = Concept('color');  red.is_a(color);   blue.is_a(color);  green.is_a(color)
    shape = Concept('shape');  cube.is_a(shape);  sphere.is_a(shape); cylinder.is_a(shape)
    material = Concept('material'); metal.is_a(material); rubber.is_a(material)

    # Spatial relations (binary between two objects)
    rel = Concept('spatial_rel')
    rel.has_a(object, object)
    left_of = Concept('left_of');   left_of.is_a(rel)
    right_of = Concept('right_of'); right_of.is_a(rel)
    behind = Concept('behind');     behind.is_a(rel)
    in_front = Concept('in_front'); in_front.is_a(rel)
```

### Generated constraint data:

```python
qa_data = [
    # Existence: "Is there a red cube?"
    {"constraint": 'existsL(andL(red("x"), cube("x")))',
     "label": True},

    # Counting: "How many green spheres are left of the blue cube?" → answer: 2
    {"constraint": 'exactL(andL(green("x"), sphere("x"), left_of("x", iotaL(andL(blue("y"), cube("y"))))), 2)',
     "label": True},

    # Relation: "Is the large metal cube behind the small rubber sphere?"
    {"constraint": 'existsL(behind(iotaL(andL(large("x"), metal("x"), cube("x"))), iotaL(andL(small("y"), rubber("y"), sphere("y")))))',
     "label": True},

    # Query: "What color is the large sphere?" → answer: "red" (index 0)
    {"constraint": 'queryL(color, iotaL(andL(large("x"), sphere("x"))))',
     "label": 0},

    # Query: "What shape is left of the green thing?" → answer: "cube" (index 0)
    {"constraint": 'queryL(shape, iotaL(left_of("x", iotaL(green("y")))))',
     "label": 0},

    # Comparative: "Are there more red things than blue things?"
    {"constraint": 'greaterL(red("x"), blue("y"))',
     "label": True},

    # Complex: "How many cubes are the same color as the large sphere?" → answer: 1
    {"constraint": 'exactL(andL(cube("x"), same_color("x", iotaL(andL(large("y"), sphere("y"))))), 1)',
     "label": True},
]

logic_dataset = graph.compile_executable(
    qa_data,
    logic_keyword='constraint',
    logic_label_keyword='label'
)
```

---

## 8. Decision Flowchart for Question → Constraint

```
Question
  │
  ├─ "Is there..." / "Are there any..."
  │   └─ existsL(property_filter("x"))
  │       label = True/False based on answer
  │
  ├─ "How many..." 
  │   └─ exactL(property_filter("x"), ANSWER_NUMBER)
  │       label = True
  │
  ├─ "Are there at least/more than N..."
  │   └─ atLeastL(property_filter("x"), N)
  │       label = True/False
  │
  ├─ "Are there more X than Y?"
  │   └─ greaterL(X_filter("x"), Y_filter("y"))
  │       label = True/False
  │
  ├─ "Is X [relation] Y?"
  │   └─ existsL(relation(iotaL(X_desc), iotaL(Y_desc)))
  │       label = True/False
  │
  ├─ "What [attribute] is THE [description]?"
  │   └─ queryL(attribute_concept, iotaL(description_filter))
  │       label = index_of_answer_in_subclass_list
  │
  └─ "Is the [attr] of X the same as Y?"
      └─ Use same_[attr] relation if available,
         or compare queryL results
```