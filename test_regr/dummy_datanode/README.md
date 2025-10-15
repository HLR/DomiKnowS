# DomiKnows Graph Testing

## Overview

Test suite for DomiKnows graph inference and logical constraint validation using dummy DataNodes with randomly generated data.

## Test Files

### `test_dummy_datanode.py`
Tests inference operations on randomly generated dummy DataNodes for the basic NER/relation extraction graph (`graph.py`):
- Creates dummy DataNode with random attribute values
- Tests satisfaction report generation with synthetic data
- Validates ILP inference on random data
- Verifies general inference execution

### `test_dummy_datanode_multipy.py`
Tests inference operations on randomly generated dummy DataNodes for the multi-label graph (`graph_multi.py`):
- Creates dummy DataNode with random values for enum-based concepts
- Tests multi-graph constraint satisfaction with synthetic data
- Validates inference on multi-label classification scenarios

### `test_dummy_graph.py`
Unit tests for basic graph construction with dummy DataNodes:
- Creates simple test graphs with random data
- Tests logical constraint application (ifL, notL, nandL) on synthetic data
- Validates constraint consistency and satisfaction reports

## Running Tests

```bash
pytest test_dummy_datanode.py
pytest test_dummy_datanode_multipy.py
pytest test_dummy_graph.py
```

Or run all tests:
```bash
pytest
```

## Graph Definitions

- **`graph.py`**: Basic NER graph with entity types (people, organization, location) and relation types (work_for, located_in, etc.)
- **`graph_multi.py`**: Enhanced graph using EnumConcept for multi-label classification with automatic constraint generation

## Dummy DataNode Testing

These tests use `createDummyDataNode()` which automatically generates:
- Random DataNode instances matching the graph structure
- Synthetic attribute values for all concepts
- Randomly initialized tensors for testing inference
- Complete graph structure with proper relations

This approach validates that inference algorithms work correctly with arbitrary data without requiring real datasets.