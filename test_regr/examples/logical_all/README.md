# Test README

## Overview
This test validates a logical constraint system for tracking entity locations and actions across process steps using the DomiKnows framework with ILP (Integer Linear Programming) solver.

## Test Scenario
- **Entities**: 4 entities (a, b, c, d)
- **Steps**: 8 time steps (0-7)
- **Locations**: 3 locations (loc1, loc2, loc3)
- **Actions**: 6 action types (create, destroy, exist, move, prior, post)

## Key Components

### Logical Constraints
1. **LC0**: Each entity must be in exactly one location at each step
2. **LC2**: If action is "move", entity's location must differ between consecutive steps

### Verification
The test verifies that:
- ILP solutions match ground truth labels for location decisions
- ILP solutions match ground truth labels for action decisions
- All logical constraints are satisfied

## Running the Test
```bash
pytest test_file.py -m gurobi

## Test Flow

1. **Generate random probabilities** for decisions
2. **Build knowledge graph** with concepts and relations
3. **Apply logical constraints**
4. **Run ILP inference**
5. **Validate ILP results** against ground truth
