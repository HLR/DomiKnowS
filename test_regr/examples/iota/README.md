# Visual QA iotaL Constraints Test

## Overview
This test validates `iotaL` (definite description / unique selection) constraints for visual question answering using DomiKnows framework with ILP solver and LogicalConstraintVerifier.

## Graph Structure
- **image**: Container concept (scene/image)
- **object_node**: Visual objects with properties (big, large, brown, cylinder, sphere, material)
- **pair**: Spatial relation pairs between objects via `rel_arg1` and `rel_arg2`
- **Spatial relations**: `right_of`, `left_of`

## Query Being Tested
> "What material is the big object that is right of the brown cylinder and left of the large brown sphere?"

## iotaL Constraints

### Step 1: THE brown cylinder
```python
the_brown_cylinder = iotaL(
    andL(brown('x'), cylinder(path='x'))
)
```
Expected to select: **Object 1**

### Step 2: THE large brown sphere
```python
the_large_brown_sphere = iotaL(
    andL(large('y'), brown(path='y'), sphere(path='y'))
)
```
Expected to select: **Object 2**

### Step 3: THE target object
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

## Test Data
| Object ID | Properties | Role |
|-----------|------------|------|
| 1 | brown, cylinder | THE brown cylinder |
| 2 | large, brown, sphere | THE large brown sphere |
| 3 | big, right_of(1), left_of(2) | THE target object |
| 4 | (other) | distractor |

## Test Cases

### `test_iotaL_target_object_selection`
Runs ILP inference and verifies that the correct objects are selected for each iotaL constraint.

### `test_iotaL_spatial_relations`
Verifies that spatial relations (right_of, left_of) are correctly evaluated in the nested iotaL constraint.

## Files
- `graph.py` - Ontology with iotaL constraints
- `reader.py` - Test data provider
- `sensor.py` - Learners that return ground-truth predictions for test objects
- `test_main.py` - Pytest validation using verifyResultsLC

## Running
```bash
pytest test_main.py -v -m gurobi
```

Requires Gurobi solver license.