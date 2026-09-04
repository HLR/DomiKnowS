# Graph Error Tests

## Overview
This test suite validates error detection and reporting in logical constraint parsing for the DomiKnows graph system.

## Test Files

### test_graph_cardinality.py
Validates cardinality constraint syntax - ensures the integer value appears as the last element in `atMostL` operators.

### test_graph_concept_in_path.py
Checks that concepts are not used directly in paths - only relations should connect variables.

### test_graph_nli_1.py
Tests path validation for NLI constraints - verifies correct relationship direction and type matching.

### test_graph_nli_2.py
Validates path destination types match the required concept types in symmetry constraints.

### test_graph_nli_3.py
Similar to nli_2, ensures correct path usage in NLI pair relationships.

### test_graph_path.py
Verifies that relation paths start from the correct source concept type.

### test_graph_path_reverse.py
Tests detection of reversed relationship usage - ensures `.reversed` is used when needed.

### test_graph_variable_reuse.py
Checks that variables cannot be redefined with different concept types within the same logical constraint.

### test_graph_variable_type.py
Validates that variables used in paths match their expected concept types.

## Running Tests
```bash
pytest test_graph_*.py -v
```

Each test has two cases:
- `test_setup_graph_exception()` - Verifies the expected error is raised
- `test_setup_graph_no_exception()` - Confirms the fix resolves the error