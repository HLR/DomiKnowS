# DomiKnowS: Counting Relations with Constraints

This guide explains how to formulate constraints that **count the number of valid relations in your data**.

---

## Purpose

In entity-relation extraction tasks, you often need to answer questions like:
- "How many people work for organizations in this document?"
- "Does this text contain at least 3 location-in-location relations?"
- "Are there more employment relations than violence relations?"

Relation counting constraints let you:
1. **Count instances** of relations that satisfy their type requirements
2. **Enforce requirements** on those counts (e.g., "at least 5", "exactly 10")
3. **Compare counts** between different relation types

---

## The Generic Pattern

### Basic Formula

```python
sumL(
    andL(
        <first_entity_type>('a'),                              # First arg has correct type
        <relation>('b', path=('a', <path_to_relation>)),       # Relation 'b' has 'a' as first arg
        <second_entity_type>('c', path=('b', <path_to_second_arg>))  # Second arg has correct type
    )
)
```

### Components

- **`sumL(...)`**: Counts how many times all conditions inside are simultaneously satisfied
- **`andL(...)`**: All sub-conditions must be true together
- **Path expressions**: Navigate from entities through relations to verify structure

### What Gets Counted

The constraint counts **each relation instance in the data** where all three conditions are simultaneously true:
1. The first argument is classified as the correct entity type
2. A relation connects these arguments  
3. The second argument is classified as the correct entity type

---

## How Paths Work

When you define a relation with `has_a`:

```python
pair = Concept(name='pair')  # Represents a relation pair
(rel_pair_phrase1, rel_pair_phrase2,) = pair.has_a(arg1=phrase, arg2=phrase)
```

This creates two edges:
- **`rel_pair_phrase1`**: From pair → first argument
- **`rel_pair_phrase2`**: From pair → second argument

### Navigation

**From first argument to relation:**
```python
<relation>('b', path=('a', rel_pair_phrase1.reversed))
```
- Start at entity 'a'
- Go **backward** through `rel_pair_phrase1` (because the edge goes FROM pair TO arg1)
- Arrive at relation pair 'b'

**From relation to second argument:**
```python
<second_entity_type>('c', path=('b', rel_pair_phrase2))
```
- Start at relation pair 'b'
- Go **forward** through `rel_pair_phrase2`
- Arrive at entity 'c'

---

## Examples

### Setup: Define Your Domain

```python
from domiknows.graph import Graph, Concept
from domiknows.graph.logicalConstrain import sumL, ifL, andL

with Graph('global') as graph:
    # Text structure
    phrase = Concept(name='phrase')
    pair = Concept(name='pair')
    (rel_pair_phrase1, rel_pair_phrase2,) = pair.has_a(arg1=phrase, arg2=phrase)
    
    # Entity types
    entity = phrase(name='entity')
    people = entity(name='people')
    organization = entity(name='organization')
    location = entity(name='location')
    
    # Relations with their type signatures
    work_for = pair(name='work_for')
    work_for.has_a(people, organization)  # Signature: (person, organization)
    
    live_in = pair(name='live_in')
    live_in.has_a(people, location)  # Signature: (person, location)
    
    located_in = pair(name='located_in')
    located_in.has_a(location, location)  # Signature: (location, location)
```

---

### Example 1: Count "Person Works For Organization"

**Constraint:**
```python
sumL(
    andL(
        people('a'),  # 'a' is a person
        work_for('b', path=('a', rel_pair_phrase1.reversed)),  # 'b' is work_for relation starting from 'a'
        organization('c', path=('b', rel_pair_phrase2))        # 'c' is the organization (second arg)
    )
)
```

**Given text:** "Alice works for Microsoft. Bob works for Google. Paris is in France."

**What gets counted:**
- ✓ (Alice:person, Microsoft:organization) - Valid work_for relation
- ✓ (Bob:person, Google:organization) - Valid work_for relation
- ✗ (Paris:location, France:location) - Not a work_for relation
- **Total count: 2**

**Purpose:** Count employment relationships in the data.

---

### Example 2: Count "Location In Location"

**Constraint:**
```python
sumL(
    andL(
        location('a'),
        located_in('b', path=('a', rel_pair_phrase1.reversed)),
        location('c', path=('b', rel_pair_phrase2))
    )
)
```

**Given text:** "Boston is in Massachusetts. Paris is in France. Alice works for Microsoft."

**What gets counted:**
- ✓ (Boston:location, Massachusetts:location) - Valid nested location
- ✓ (Paris:location, France:location) - Valid nested location
- ✗ (Alice:person, Microsoft:organization) - Wrong relation type
- **Total count: 2**

**Purpose:** Count geographic containment relationships (cities in states, etc.).

---

### Example 3: Count "Person Lives In Location"

**Constraint:**
```python
sumL(
    andL(
        people('a'),
        live_in('b', path=('a', rel_pair_phrase1.reversed)),
        location('c', path=('b', rel_pair_phrase2))
    )
)
```

**Given text:** "John lives in Boston. Mary lives in Chicago. Google is based in Mountain View."

**What gets counted:**
- ✓ (John:person, Boston:location) - Valid residence
- ✓ (Mary:person, Chicago:location) - Valid residence
- ✗ (Google:organization, Mountain View:location) - Wrong first arg type (not person)
- **Total count: 2**

**Purpose:** Count where people reside.

---

## Using Counts in Constraints

Once you can count relations, you can enforce requirements:

### Exact Count
```python
from domiknows.graph.logicalConstrain import exactL

# Require exactly 5 work_for relations
exactL(
    sumL(andL(
        people('a'), 
        work_for('b', path=('a', rel_pair_phrase1.reversed)), 
        organization('c', path=('b', rel_pair_phrase2))
    )),
    5
)
```

### Minimum Count
```python
from domiknows.graph.logicalConstrain import atLeastL

# At least 3 live_in relations
atLeastL(
    sumL(andL(
        people('a'), 
        live_in('b', path=('a', rel_pair_phrase1.reversed)), 
        location('c', path=('b', rel_pair_phrase2))
    )),
    3
)
```

### Maximum Count
```python
from domiknows.graph.logicalConstrain import atMostL

# At most 10 located_in relations
atMostL(
    sumL(andL(
        location('a'), 
        located_in('b', path=('a', rel_pair_phrase1.reversed)), 
        location('c', path=('b', rel_pair_phrase2))
    )),
    10
)
```

### Compare Counts Between Relations
```python
from domiknows.graph.logicalConstrain import greaterL

# More work_for than kill relations
greaterL(
    sumL(andL(
        people('a'), 
        work_for('b', path=('a', rel_pair_phrase1.reversed)), 
        organization('c', path=('b', rel_pair_phrase2))
    )),
    sumL(andL(
        people('a'), 
        kill('b', path=('a', rel_pair_phrase1.reversed)), 
        people('c', path=('b', rel_pair_phrase2))
    ))
)
```

---

## Counting Relations Between Specific Entities

To count relations involving **specific entities** (not just types), use `eqL` to filter by property values:

### Filter by Entity Name

**Question:** "How many organizations does **Alice** work for?"

```python
from domiknows.graph.logicalConstrain import eqL

sumL(
    andL(
        people('a', path=(eqL(people, 'name', 'Alice'),)),  # Only Alice
        work_for('b', path=('a', rel_pair_phrase1.reversed)),
        organization('c', path=('b', rel_pair_phrase2))  # Any organization
    )
)
```

### Filter Both Entities

**Question:** "Does **Alice** work for **Microsoft**?"

```python
sumL(
    andL(
        people('a', path=(eqL(people, 'name', 'Alice'),)),
        work_for('b', path=('a', rel_pair_phrase1.reversed)),
        organization('c', path=('b', rel_pair_phrase2, eqL(organization, 'name', 'Microsoft')))
    )
)
```
- Returns 1 if the relation exists, 0 otherwise

### Filter by Instance ID

**Question:** "How many work_for relations involve entity 'phrase_5'?"

```python
sumL(
    andL(
        people('a', path=(eqL(people, 'instanceID', 'phrase_5'),)),
        work_for('b', path=('a', rel_pair_phrase1.reversed)),
        organization('c', path=('b', rel_pair_phrase2))
    )
)
```

### Filter by Properties

**Question:** "How many people from **Boston** work for organizations in **Seattle**?"

```python
sumL(
    andL(
        people('a', path=(eqL(people, 'city', 'Boston'),)),
        work_for('b', path=('a', rel_pair_phrase1.reversed)),
        organization('c', path=('b', rel_pair_phrase2, eqL(organization, 'headquarters', 'Seattle')))
    )
)
```

### Multiple Relations from Same Entity

**Question:** "How many people work for **Microsoft** AND live in **Seattle**?"

```python
sumL(
    andL(
        people('a'),  # Person 'a'
        work_for('b', path=('a', rel_pair_phrase1.reversed)),
        organization('c', path=('b', rel_pair_phrase2, eqL(organization, 'name', 'Microsoft'))),
        live_in('d', path=('a', rel_pair_phrase1.reversed)),
        location('e', path=('d', rel_pair_phrase2, eqL(location, 'name', 'Seattle')))
    )
)
```

**Explanation:** This counts people where:
1. Person 'a' works for organization 'c' (which must be Microsoft)
2. **Same** person 'a' lives in location 'e' (which must be Seattle)

**Example data:**
- "Alice works for Microsoft and lives in Seattle." → ✓ Counts (1)
- "Bob works for Microsoft and lives in Boston." → ✗ Doesn't count
- "Carol works for Google and lives in Seattle." → ✗ Doesn't count
- **Total count: 1**

---

### Multiple Relations of Same Type

**Question 1:** "How many people work for **at least two companies** (any companies)?"

```python
sumL(
    andL(
        people('a'),  # Person 'a'
        work_for('b1', path=('a', rel_pair_phrase1.reversed)),
        organization('c1', path=('b1', rel_pair_phrase2)),
        work_for('b2', path=('a', rel_pair_phrase1.reversed)),
        organization('c2', path=('b2', rel_pair_phrase2)),
        notL(eqL('c1', 'c2'))  # Ensure c1 ≠ c2
    )
)
```

**Explanation:** This counts people where:
1. Person 'a' works for organization 'c1'
2. **Same** person 'a' works for **a different** organization 'c2'
3. The two organizations must be different entities

**Example data:**
- "Alice works for Microsoft and Google." → ✓ Counts (1)
- "Bob works for Apple and Amazon." → ✓ Counts (1)
- "Carol works for Microsoft." → ✗ Doesn't count (only one employer)
- **Total count: 2**

**Use case:** Finding people with multiple employers, contractors, or part-time workers.

---

**Question 2:** "How many people work for **both Microsoft AND Google** (specific companies)?"

```python
sumL(
    andL(
        people('a'),  # Person 'a'
        work_for('b1', path=('a', rel_pair_phrase1.reversed)),
        organization('c1', path=('b1', rel_pair_phrase2, eqL(organization, 'name', 'Microsoft'))),
        work_for('b2', path=('a', rel_pair_phrase1.reversed)),
        organization('c2', path=('b2', rel_pair_phrase2, eqL(organization, 'name', 'Google')))
    )
)
```

**Explanation:** This counts people where:
1. Person 'a' works for organization 'c1' (which must be Microsoft)
2. **Same** person 'a' works for organization 'c2' (which must be Google)
3. Note: 'b1' and 'b2' are **different work_for relation instances**

**Example data:**
- "Alice works for Microsoft. Alice also works for Google." → ✓ Counts (1)
- "Bob works for Microsoft." → ✗ Doesn't count (only one employer)
- "Carol works for Google and Apple." → ✗ Doesn't count (wrong companies)
- **Total count: 1**

**Use case:** Finding specific people working for two particular companies.

**Note:** `eqL(concept, 'property', 'value')` filters entities where `property == value`. Common properties include `'instanceID'`, `'name'`, `'text'`, or any custom attributes.

---

## Understanding sumL with Multiple Arguments

### `sumL(people('x'), organization('y'))` - Separate Counts

**Natural Language:** "Count the **total number of people and organizations combined**."

**What it does:** Adds the count of people + count of organizations

```python
sumL(people('x'), organization('y'))
```

**Example:** "Alice, Bob, and Carol work for Microsoft and Google."
- People: 3 (Alice, Bob, Carol)
- Organizations: 2 (Microsoft, Google)
- **Total count: 5**

This is equivalent to: `count(people) + count(organizations)`

---

### `sumL(andL(people('x'), organization(path=('x'))))` - Overlap Count

**Natural Language:** "Count entities that are **simultaneously classified as both person AND organization**."

**What it does:** Counts entities satisfying BOTH conditions at the same time

```python
sumL(andL(people('x'), organization(path=('x'))))
```

**Example (correct data):** "Alice works for Microsoft."
- Alice: person only → doesn't count
- Microsoft: organization only → doesn't count
- **Total count: 0** (expected, since people and organizations are disjoint)

**Example (classification error):** Model incorrectly classifies "Alice" as both person (0.8) and organization (0.3)
- Alice satisfies both conditions simultaneously
- **Total count: 1** (indicates a problem!)

**Use case:** Detecting classification errors or enforcing mutual exclusivity

---

## Common Use Cases

1. **Question Answering:** "How many employment relations are in this document?"
2. **Data Validation:** Ensure annotated datasets have expected relation counts
3. **Training Constraints:** Guide models to predict valid relation structures
4. **Inference Enforcement:** Require predictions satisfy domain requirements
5. **Quality Metrics:** Compare relation distributions across different texts

---

## Key Principles

### 1. Type Signatures Must Match

The constraint checks both entity types match the relation's signature:

```python
work_for.has_a(people, organization)  # Signature: (person, organization)

# Counting constraint must match this signature
sumL(andL(
    people('a'),           # ✓ First arg: person
    work_for('b', path=('a', rel_pair_phrase1.reversed)),
    organization('c', path=('b', rel_pair_phrase2))  # ✓ Second arg: organization
))
```

### 2. Path Direction Matters

```python
# CORRECT: Go backward from entity to relation
work_for('b', path=('a', rel_pair_phrase1.reversed))

# WRONG: Forward direction won't connect properly
work_for('b', path=('a', rel_pair_phrase1))  # ✗
```
