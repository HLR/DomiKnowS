# DomiKnows Sensor Tests

This directory contains unit tests for various sensor types in the DomiKnows framework.

## Test Files

### `test_functional_sensor.py`
Tests basic functional sensors that transform input data through a forward function without graph context.

### `test_query_sensor.py`
Tests query sensors that operate on DataNode collections within a graph structure, receiving a list of DataNodes as input.

### `test_datanode_sensor.py`
Tests DataNode sensors that process individual DataNodes, providing access to all node attributes through the DataNode interface.

### `test_candidate_sensor.py`
Tests composition candidate sensors that generate edge relationships between concept pairs based on Cartesian product combinations.

### `test_candidate_reader_sensor.py`
Tests composition candidate reader sensors that read edge data from external sources using keyword-based retrieval.

## Helper Files

### `sensors.py`
Contains test helper classes:
- `BaseTestSensor` - Base class for test sensors with input/output validation
- `TestSensor` - Standard test sensor
- `TestLearner` - Test sensor with learning capabilities
- `TestEdgeSensor` - Test sensor for edge relationships

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_functional_sensor.py

# Run with verbose output
pytest -v
```

## Test Structure

Each test file follows a consistent pattern:
1. **case fixture** - Generates random test data
2. **graph/concept fixture** - Sets up the knowledge graph structure
3. **sensor fixture** - Configures the sensor under test
4. **context fixture** - Prepares the execution context
5. **test function** - Validates sensor behavior