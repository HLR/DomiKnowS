# NER Test Suite

## Overview
This test suite validates a neural network model that learns logical relationships between people and locations through constraint satisfaction.

## Problem Structure
The model learns two logical conditions:
- **Condition 1**: `(work1 ∧ people1) ∧ (work2 ∧ people2)` - Both person 1 and person 2 are real AND work at their respective locations
- **Condition 2**: `(work2 ∧ people2) ∨ (work3 ∧ people3)` - Either person 2 or person 3 is real AND works at their location

## Components

### Data Generation (`NER_utils.py`)
- Generates synthetic data with 3 people and 3 locations
- Each entity has a 2D tensor embedding
- "Real" entities have tensors where both dimensions share the same sign
- Work relationships determined by first dimension sign matching
- Dataset balanced across all condition combinations

### Training Approaches

#### 1. Direct Neural Training (`test_NER.py`)
- Uses simple feedforward networks
- Direct supervision on conditions
- 200 epochs with Adam optimizer (lr=8e-4)

#### 2. Constraint-Based Training (`PDPlus.py`)
- Uses DomiKnows graph framework
- Encodes logical constraints explicitly
- 30 epochs with AdamW optimizer (lr=1e-3)
- Outputs results to `results.txt`

## Running Tests

```bash
# Run direct neural network test
pytest test_NER.py -v

# Run constraint-based approach
python PDPlus.py
```

## Success Criteria
Both condition accuracies should exceed 50% on test set, demonstrating the model learned the logical relationships.