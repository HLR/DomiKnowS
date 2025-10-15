# ILP Ontology Solver Tests

## Overview

This test suite validates the ILP (Integer Linear Programming) ontology solver functionality for structured prediction tasks with logical constraints.

## Test Files

### `test_ilpOntSolver.py`
Basic ILP solver tests using OWL ontology definitions for Entity-Mention-Relation (EMR) extraction.

**Tests:**
- Entity classification (people, organization, location, other, O)
- Relation extraction (work_for, live_in, located_in)
- Example: "John works for IBM" → identifies entities and work_for relation

### `test_ilpOntSolver_graph.py`
Tests ILP solver with constraints specified directly in the graph structure (not OWL ontology).

**Key Features:**
- Graph-based constraint definition using `is_a`, `not_a`, `has_a`
- Same EMR task as above but with programmatic graph construction

### `test_ilpOntSolver_lc.py`
Tests ILP solver using logical constraint API (andL, nandL, ifL, existsL, notL).

**Key Features:**
- Logical constraint-based definitions instead of OWL or graph relations
- More flexible constraint specification
- Same EMR extraction validation

### `test_ilpOntSolver_lc_verify.py`
Tests the solver's verification functionality for checking solution validity.

**Tests:**
- Verify correct solutions pass validation
- Verify constraint violations are detected (e.g., entity type conflicts)
- Verify relation constraint violations (e.g., work_for requires people→organization)

### `test_ilpOntSolver_sprl.py`
Tests Spatial Role Labeling (SpRL) with triplet relations.

**Task:**
- Identify spatial entities: TRAJECTOR, LANDMARK, SPATIAL_INDICATOR, NONE_ENTITY
- Extract spatial triplets: (landmark, trajector, spatial_indicator)
- Example: "kids waiting on stairs" → (stairs[LANDMARK], kids[TRAJECTOR], on[INDICATOR])

### `test_dataNodei_sprl.py`
DataNode-based SpRL test using the graph structure for inference.

**Key Features:**
- Uses DataNode API for representing input data
- Tests `inferILPResults()` method on DataNode
- Validates entity and relation predictions through graph traversal

### `test_gurobi_solver_emr_compare.py`
Comparative benchmarking of different solver implementations.

**Tests:**
- Compares `mini_wrap` (mini_prob_debug solver) vs `owl_wrap` (OWL-based solver)
- Validates both solvers produce identical results
- Performance benchmarking with parameterized input sizes (1-20 entities)

## Common Patterns

All tests follow similar structure:
1. **Setup**: Create ontology graph with concepts and constraints
2. **Input**: Prepare phrase data with prediction scores
3. **Solve**: Run ILP solver to find optimal consistent solution
4. **Validate**: Assert expected entity types and relations are selected

## Requirements

- `pytest` with markers: `@pytest.mark.gurobi`, `@pytest.mark.slow`, `@pytest.mark.benchmark`
- Gurobi solver installation
- DomiKnows library with graph and solver modules

## Running Tests

```bash
# Run all tests
pytest

# Run only gurobi tests
pytest -m gurobi

# Run specific test file
pytest test_ilpOntSolver.py

# Run with benchmarks
pytest -m benchmark
```

