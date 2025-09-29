# Linear Regression with DomiKnows - Pytest

This project demonstrates a linear regression implementation using the DomiKnows framework, converted to pytest format.

## Overview

The code implements a simple linear regression model using:
- DomiKnows graph-based knowledge representation
- PyTorch neural network backend
- Sensor-based data reading and learning
- Solver-based program execution

## Installation

```bash
pip install pytest torch domiknows
```

## Running Tests

Run all tests:
```bash
pytest test_linear_regression.py -v
```

Run specific test:
```bash
pytest test_linear_regression.py::test_training -v
```

## Test Structure

- `test_training()`: Tests the model training process
- `test_inference()`: Tests model inference on validation data
- `test_data_generator()`: Validates the data generation function
- `test_linear_regression_module()`: Tests the PyTorch module directly

## Key Components

- **Graph**: Defines the concept structure with `number`, `x`, and `y`
- **LinearRegression**: Simple PyTorch linear model
- **SolverPOIProgram**: Main program for training and inference
- **Data Generator**: Creates synthetic linear regression data with noise