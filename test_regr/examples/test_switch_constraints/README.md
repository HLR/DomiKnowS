# DomiKnows Graph Compilation Test

Test suite for validating graph compilation and training with logical constraints in the DomiKnows framework.

## Test Coverage

### test_main.py
- **Graph Setup**: Creates a computation graph with root concept and two child concepts (x, y)
- **Sensor Configuration**: Attaches ReaderSensor for input and ModuleSensor with dummy neural network models
- **Logic Compilation**: Compiles logical constraints (AND, OR, IF operations) into the graph
- **Training**: Validates the training pipeline with InferenceProgram and SolverModel

### test_multiple_calls.py
- **Multiple Dataset Compilation**: Tests compile_logic with multiple datasets of varying sizes
- **ReaderSensor Validation**: Verifies correct ReaderSensor keyword assignment and label mapping
- **Sensor Count Verification**: Confirms total sensors added matches total samples across datasets
- **DataNode Population**: Validates correct input values, label values, and active LC assignments
- **Sequential Training**: Tests training across multiple transformed datasets without errors

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test_main.py
uv run pytest test_multiple_calls.py
```

## Key Components

- Uses PyTorch for neural network operations
- Implements logical constraints via DomiKnows framework
- Tests both graph compilation and model training in a single workflow
- Validates multi-dataset compilation and sequential training workflows