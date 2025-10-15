# ILP Constraint Satisfaction Test

This project demonstrates constraint satisfaction problems (CSP) using DomiKnows with ILP (Integer Linear Programming) solver.

## Structure

- `graph.py` - Defines the CSP graph structure with concepts, relations, and logical constraints
- `csp_demo.py` - Main demonstration script with various constraint scenarios
- `test_ILP_relations.py` - Pytest test for validating ILP constraint solving
- `main.py` - Alternative entry point (contains duplicate code from csp_demo.py)

## Key Concepts

- **CSP**: Top-level constraint satisfaction problem container
- **CSP Range**: Intermediate concept representing ranges/bags within the CSP
- **Orbs**: Individual items that can be colored or uncolored
- **Constraints**: Logical rules enforced via ILP solver (exactL, atMostL, atLeastL, etc.)

## Running Tests

```bash
pytest test_ILP_relations.py
```

## Running Demo

```bash
python csp_demo.py --constraint exactL
```

Available constraints:
- `exactL` - Exactly N orbs colored per bag
- `foreach_bag_atLeastAL` - At least N colored orbs per bag
- `foreach_bag_atMostAL` - At most N colored orbs per bag
- `foreach_bag_existsL` - At least one colored orb exists per bag
- And more...

## Parameters

- `--colored` - Set initial orb coloring preference
- `--atmostaL` - Maximum count for atMost constraints (default: 25)
- `--atleastaL` - Minimum count for atLeast constraints (default: 2)