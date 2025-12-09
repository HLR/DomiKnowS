# Counting Relations Test

Tests for relation counting constraints using `sumL`, `atLeastL`, and `atMostL`.

## Scenario

**Sentence:** "John works for IBM and Alice works for Google"

**Entities:**
- People: John (idx 0), Alice (idx 5)
- Organizations: IBM (idx 3), Google (idx 8)

**Relations:**
- work_for(John, IBM) - pair index 3
- work_for(Alice, Google) - pair index 53

## Constraints Tested

### 1. Entity Type Mutual Exclusivity
```python
ifL(word('x'), exactL(people, organization, location, other, o))
```
Each word must be exactly one entity type.

### 2. Relation Type Requirements
```python
ifL(work_for('x', 'y'), andL(people('x'), organization('y')))
```
work_for requires (person, organization) arguments.

### 3. Counting: At Least 2 work_for Relations
```python
atLeastL(sumL(andL(people('a'), work_for('b', path=...), organization('c', path=...))), 2)
```

### 4. Counting: At Most 3 work_for Relations
```python
atMostL(sumL(andL(people('a'), work_for('b', path=...), organization('c', path=...))), 3)
```

## Test Verifications

- John and Alice classified as people
- IBM and Google classified as organizations
- work_for count is between 2 and 3
- LC loss calculation works for all t-norms (L, G, P)

## Files

- `graph_counting_relations.py` - Graph with counting constraints
- `config_counting_relations.py` - Model configuration
- `test_counting_relations.py` - Test cases