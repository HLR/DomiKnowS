# Test Unit for Logical Constraints, ILP Solver, Loss Calculation and Datanode Queries in DomiKnowS

This example uses a single sentence from the [CoNLL-2004](https://www.cs.upc.edu/~srlconll/st04/st04.html) dataset with precalculated results to test core DomiKnowS functionalities.

## Overview

The test demonstrates and validates:
- **Graph Definition**: Domain concepts and relations
- **Logical Constraints**: Knowledge encoding with constraints
- **ILP Solver**: Inference with Integer Linear Programming
- **Loss Calculation**: Constraint violation losses using t-norms
- **Datanode Queries**: Structured data access and navigation

## Test Data

**Sentence**: "John works for IBM"

**Components**:
- 4 words: ['John', 'works', 'for', 'IBM']
- 15 characters across all words
- 3 phrases: ['John', 'works for', 'IBM']
- 16 word pairs (all combinations)

## Graph Structure

### Linguistic Graph
- **Concepts**: `sentence`, `word`, `char`, `phrase`, `pair`
- **Relations**: 
  - `sentence.contains(word)`
  - `sentence.contains(phrase)`
  - `phrase.contains(word)`
  - `word.contains(char)`
  - `pair.has_a(arg1=word, arg2=word)`

### Application Graph
- **Entity Types**: `people`, `organization`, `location`, `other`, `O` (outside)
- **Relation Types**: `work_for`, `located_in`, `live_in`, `orgbase_on`, `kill`

## Logical Constraints

### LC0: Mutual Exclusion
```python
nandL(people, organization)
```
People and organization cannot both be true for the same word.

### LC1: Entity Classification (commented out)
```python
ifL(word('x'), exactL(people('x'), organization('x'), location('x'), other('x'), o('x')))
```
Each word must have exactly one entity type.

### LC2: Relation Classification (commented out)
```python
ifL(pair('x'), exactL(work_for('x'), located_in('x'), live_in('x'), orgbase_on('x'), kill('x')))
```

### LC3-LC7: Type Constraints for Relations
- **LC3**: `work_for(x,y)` → `people(x) ∧ organization(y)`
- **LC4**: `located_in(x,y)` → `location(x) ∧ organization(y)`
- **LC5**: `live_in(x,y)` → `people(x) ∧ location(y)`
- **LC6**: `orgbase_on(x,y)` → `organization(x) ∧ location(y)`
- **LC7**: `kill(x,y)` → `people(x) ∧ people(y)`

## Test Structure

### 1. Model Declaration (`model_declaration`)
Configures test sensors for each concept with expected inputs/outputs:
- Word embeddings (2048-dim)
- Entity type predictions (softmax distributions)
- Relation predictions with NaN handling

### 2. Graph Naming Test (`test_graph_naming`)
Validates correct naming of:
- Graphs and subgraphs
- Concepts and subconcepts
- Relations (both default and explicit names)

### 3. Main Test (`test_main_conll04`)

#### Datanode Structure Validation
- Verifies parent-child relationships
- Checks attribute access for embeddings and predictions
- Validates relation links between nodes

#### Query Tests
```python
# Find word containing character 'J'
datanode.findDatanodes(select=word, indexes={"contains": (char, 'raw', 'J')})

# Find pairs with specific arg1
datanode.findDatanodes(select=pair, indexes={"arg1": 0})
```

#### ILP Inference Tests
Runs inference with different concept/relation specifications:
- Empty list (all concepts/relations)
- String names
- Concept objects
- Mixed formats

**Expected Results**:
- Word 0 (John): `people=1`
- Word 3 (IBM): `organization=1`
- Words 1,2 (works, for): `O=1`
- Work_for relation: varies based on constraints

#### Loss Calculation Tests
Computes constraint violations using three t-norms:
- **Łukasiewicz (L)**: `L(a,b) = max(0, a+b-1)`
- **Gödel (G)**: `G(a,b) = min(a,b)`
- **Product (P)**: `P(a,b) = a*b`

Validates against precalculated loss tensors for LC0 and LC2.

#### Sampling Tests
```python
datanode.calculateLcLoss(sample=True, sampleSize=1)
datanode.calculateLcLoss(sample=True, sampleSize=1000)
```

## Special Features

### NaN Handling
The `work_for` relation includes NaN values to test constraint handling with missing data:
```python
'work_for': tensor([[0.60, 0.40], ..., [nan, nan], ...])
```

### Skeleton Mode
```python
setDnSkeletonMode(False)
```
Disables skeleton mode for full datanode functionality.

## Running the Tests

```bash
pytest test_main.py
```

### Individual Tests
```bash
pytest test_main.py::test_graph_naming
pytest test_main.py::test_main_conll04
```

### Requirements
- PyTorch with CUDA support (optional)
- Gurobi optimizer (for ILP tests marked with `@pytest.mark.gurobi`)
- flaky package (for test stability)

## Key Assertions

The test validates:
1. **Graph structure**: 4 words, 15 chars, correct relations
2. **ILP results**: Correct entity assignments per word
3. **Loss computation**: Matches precalculated values within 4 decimal places
4. **Query functionality**: Correct filtering and traversal
5. **Constraint concepts**: Proper LC-to-concept mapping