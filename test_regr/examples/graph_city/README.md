# City Service Constraints Test

## Overview
This test validates logical constraints on city services using DomiKnows framework with ILP solver (Gurobi).

## Graph Structure
- **World** contains multiple **Cities**
- Cities are connected via **CityLinks** (neighbor relationships)
- Cities can have various services:
  - Fire stations (main or ancillary)
  - Emergency services
  - Grocery shops

## Constraints Tested

### Count Constraints
- Exactly 1 main firestation
- 2-5 ancillary firestations
- 6-7 emergency services
- 8-9 grocery shops

### Comparison Constraints
- `main < ancillary` (main firestations < ancillary firestations)
- `emergency >= firestations` (emergency services ≥ total firestations)
- `grocery > emergency` (grocery shops > emergency services)
- `emergency ≠ grocery` (different counts)

### Summation Constraints (using `sumL`)
- `sum(emergency, grocery) == 14` - Total emergency + grocery services must equal 14
- `sum(emergency, main, ancillary) >= 8` - Total emergency + all firestations must be at least 8
- `emergency > sum(main, ancillary)` - Emergency services must exceed sum of firestation types
- `if sum(emergency, grocery) >= 6 then exists(firestation)` - Conditional based on service sum

### Logical Constraints
- All service cities must be subset of total cities

## Files
- `graph.py` - Ontology definition with constraints
- `reader.py` - Test data provider (9 cities with predefined connections)
- `sensor.py` - Dummy learners for service classification
- `test_main.py` - Pytest validation of ILP constraint satisfaction

## Running
```bash
pytest test_main.py -v
```

Requires Gurobi solver license.