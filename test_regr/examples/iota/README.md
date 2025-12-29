# Visual QA iotaL and queryL Constraints Test

## Overview
This test validates `iotaL` (definite description / unique selection) and `queryL` (multiclass attribute query) constraints for visual question answering using DomiKnows framework across multiple execution pathways: ILP, verification, differentiable loss, and sample-based loss.

## Graph Structure
- **image**: Container concept (scene/image)
- **object_node**: Visual objects with properties (big, large, brown, cylinder, sphere)
- **pair**: Spatial relation pairs between objects via `rel_arg1` and `rel_arg2`
- **Spatial relations**: `right_of`, `left_of`
- **material**: Multiclass concept with subclasses `metal` and `rubber`

## Query Being Tested
> "What material is the big object that is right of the brown cylinder and left of the large brown sphere?"

## Constraints

### iotaL Constraints (Definite Description)

#### Step 1: THE brown cylinder
```python
the_brown_cylinder = iotaL(
    andL(brown('x'), cylinder(path='x'))
)
```
Expected to select: **Object 1**

#### Step 2: THE large brown sphere
```python
the_large_brown_sphere = iotaL(
    andL(large('y'), brown(path='y'), sphere(path='y'))
)
```
Expected to select: **Object 2**

#### Step 3: THE target object
```python
the_target_object = iotaL(
    andL(
        big('z'),
        right_of('r1', path=('z', rel_arg1.reversed)),
        the_brown_cylinder,
        left_of('r2', path=('z', rel_arg1.reversed)),
        the_large_brown_sphere
    )
)
```
Expected to select: **Object 3** (the big object that is right of object 1 and left of object 2)

### queryL Constraint (Multiclass Attribute Query)

#### Step 4: Query material of target object
```python
the_material_answer = queryL(
    material,          # Parent multiclass concept
    the_target_object  # Entity selection from iotaL
)
```
Expected to return: **metal** (the material of object 3)

## Test Data
| Object ID | Properties | Material | Role |
|-----------|------------|----------|------|
| 1 | brown, cylinder | - | THE brown cylinder |
| 2 | large, brown, sphere | - | THE large brown sphere |
| 3 | big, right_of(1), left_of(2) | metal | THE target object |
| 4 | (distractor) | rubber | distractor |

## Test Cases

### ILP Inference Tests

#### `test_iotaL_target_object_selection`
Runs ILP inference and verifies that the correct objects are selected for each iotaL constraint:
- Object 1 is identified as the brown cylinder
- Object 2 is identified as the large brown sphere
- Object 3 is identified as the big target object

#### `test_iotaL_spatial_relations`
Verifies that spatial relations are correctly evaluated:
- Object 3 is right_of object 1 (brown cylinder)
- Object 3 is left_of object 2 (large brown sphere)

#### `test_queryL_material_selection`
Verifies that queryL correctly identifies the material of the target object:
- Object 3 (target) has material **metal**
- Object 4 (distractor) has material **rubber**

### Verification Tests

#### `test_iotaL_queryL_verifyResultsLC`
Tests `verifyResultsLC(key="/local/argmax")` for all constraints including iotaL and queryL.

#### `test_iotaL_verifySingleConstraint`
Tests `verifySingleConstraint(lcName, key="/local/argmax")` for each iotaL constraint individually.

#### `test_queryL_verifySingleConstraint`
Tests `verifySingleConstraint(lcName, key="/local/argmax")` for the queryL constraint.

### Loss Calculation Tests

#### `test_iotaL_queryL_calculateLcLoss`
Tests `calculateLcLoss(tnorm='P', sample=False)` - differentiable loss calculation for iotaL and queryL constraints. Requires constraint label sensors attached via `graph.constraint[lc]`.

#### `test_iotaL_queryL_calculateLcLoss_sampling`
Tests `calculateLcLoss(tnorm='P', sample=True, sampleSize=10)` - Gumbel-softmax sample-based loss calculation for iotaL and queryL constraints.

## Files
- `graph.py` - Ontology with iotaL and queryL constraints
- `reader.py` - Test data provider with material ground truth
- `sensor.py` - Learners for object properties and material subclasses (MetalLearner, RubberLearner)
- `test_main.py` - Pytest validation across all execution pathways

## Running
```bash
uv run pytest test_main.py -v
```

Requires Gurobi solver license for ILP tests (marked with `@pytest.mark.gurobi`).

## Implementation Notes

### iotaL (Definite Description)
- Selects THE unique entity satisfying a condition
- ILP: Creates selection variables with exactly-one constraint
- Loss: Softmax-based selection with existence/uniqueness penalties

### queryL (Multiclass Attribute Query)
- Queries which subclass an entity belongs to
- Requires multiclass concept defined via subclasses (e.g., `metal.is_a(material)`, `rubber.is_a(material)`)
- ILP: Creates result variables for each subclass with exactly-one constraint
- Loss: Softmax distribution over subclasses

### Execution Pathways Tested
| Pathway | Method | Description |
|---------|--------|-------------|
| ILP | `inferILPResults()` | Integer Linear Programming optimization |
| Verification | `verifyResultsLC()`, `verifySingleConstraint()` | Discrete constraint satisfaction check |
| Differentiable Loss | `calculateLcLoss(sample=False)` | Product t-norm based differentiable loss |
| Sample-based Loss | `calculateLcLoss(sample=True)` | Gumbel-softmax sampling for loss |

### Label Sensors for Loss Calculation
Loss calculation requires label sensors attached to constraints:
```python
graph.constraint[the_brown_cylinder] = ReaderSensor(
    keyword='the_brown_cylinder_label', 
    is_constraint=True, 
    label=True
)
```

### Device Considerations
Loss calculation tests use `device='cpu'` to avoid device mismatch issues in nested constraint evaluation.