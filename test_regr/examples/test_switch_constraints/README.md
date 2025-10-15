# DomiKnows Graph Compilation Test

Test suite for validating graph compilation and training with logical constraints in the DomiKnows framework.

## Test Coverage

- **Graph Setup**: Creates a computation graph with root concept and two child concepts (x, y)
- **Sensor Configuration**: Attaches ReaderSensor for input and ModuleSensor with dummy neural network models
- **Logic Compilation**: Compiles logical constraints (AND, OR, IF operations) into the graph
- **Training**: Validates the training pipeline with InferenceProgram and SolverModel

## Running Tests

```bash
pytest test_main.py
```

## Key Components

- Uses PyTorch for neural network operations
- Implements logical constraints via DomiKnows framework
- Tests both graph compilation and model training in a single workflow