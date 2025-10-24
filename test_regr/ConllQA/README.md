# ConLL04 Named Entity Recognition Test

A neural-symbolic learning system for entity recognition and counting constraints using the DomiKnows framework.

## Overview

This project implements entity recognition (People, Organization, Location) with logical counting constraints on ConLL04 dataset variants.

## Prerequisites

Before running the code, download the required spaCy language model:

```bash
uv run python -m spacy download en_core_web_sm
```

## Usage

### Training
```bash
uv run python main.py --epochs 10 --lr 1e-6 --train_portion entities_only_with_1_things_YN
```

### Evaluation
```bash
uv run python main.py --evaluate --epochs 10 --lr 1e-6
```

### Parameters
- `--epochs`: Number of training epochs (default: 1)
- `--lr`: Learning rate (default: 1e-6)
- `--train_portion`: Dataset variant (default: "entities_only_with_1_things_YN")
- `--counting_tnorm`: T-norm for counting constraints: G/P/L/SP (default: "G")
- `--data_path`: Path to data file (default: "conllQA.json")
- `--evaluate`: Run evaluation mode

## Testing

The test suite validates the neural-symbolic system's ability to learn counting constraints with different T-norm operators.

### Test Execution Modes

The tests support two execution modes controlled by the `USE_SUBPROCESS` environment variable:

```bash
# Subprocess mode (default for CI/CD) - Runs main.py as separate process
USE_SUBPROCESS=true uv run pytest test_conll04.py

# Direct call mode (for local debugging) - Imports and calls main() directly
USE_SUBPROCESS=false uv run pytest test_conll04.py
```

**Subprocess mode** (`USE_SUBPROCESS=true`):
- Runs each test in an isolated process via `subprocess.run()`
- Better isolation, catches import errors and crashes
- Recommended for CI/CD pipelines
- Default behavior if environment variable not set

**Direct call mode** (`USE_SUBPROCESS=false`):
- Imports `main()` function and calls it directly
- Easier debugging with breakpoints and stack traces
- Faster execution
- Recommended for local development

### Test Categories

#### Zero Counting Tests
Tests the model's ability to answer "Are there exactly 0 X entities?" questions:
- `test_zero_counting_godel` - Uses Gödel T-norm (min operator, default)
- `test_zero_counting_lukas` - Uses Łukasiewicz T-norm (bounded difference)
- `test_zero_counting_product` - Uses Product T-norm (multiplication)
- `test_zero_counting_simple_product` - Uses simplified Product T-norm

**Dataset**: `zero_counting_YN` - Examples with questions about absence of entities

#### Over Counting Tests
Tests the model's ability to answer "Are there at least/at most N entities?" questions:
- `test_over_counting_godel` - Uses Gödel T-norm
- `test_over_counting_lukas` - Uses Łukasiewicz T-norm
- `test_over_counting_product` - Uses Product T-norm
- `test_over_counting_simple_produce` - Uses simplified Product T-norm

**Dataset**: `over_counting_YN` - Examples with threshold-based counting questions

#### General Performance Test
- `test_general_run` - Trains for 5 epochs and validates accuracy > 80%

### T-norm Operators
Different fuzzy logic operators for combining logical constraints:
- **G (Gödel)**: min(a, b) - Most lenient
- **L (Łukasiewicz)**: max(0, a + b - 1) - Linear penalty
- **P (Product)**: a × b - Multiplicative penalty
- **SP (Simple Product)**: Simplified product variant

### Running Tests

```bash
# Run all tests (subprocess mode - isolated)
uv run pytest test_conll04.py

# Run with direct call mode (faster, better for debugging)
USE_SUBPROCESS=false uv run pytest test_conll04.py

# Run specific test
uv run pytest test_conll04.py::test_zero_counting_godel

# Debug specific test with direct call mode
USE_SUBPROCESS=false uv run pytest test_conll04.py::test_general_run -v -s

# Run with development dependencies
uv sync --extra dev
uv run pytest test_conll04.py -v
```

## Files

- `graph.py` - Knowledge graph definition (entities, relations, constraints)
- `main.py` - Training/evaluation pipeline with BERT encoder
- `reader.py` - Dataset reader for ConLL04 variants
- `test_conll04.py` - Automated test suite

## Results

Results are saved to `result.txt` with training/testing accuracy for each run.