# PMD Counting Constraints Tests

This module tests counting constraints (exactL, atLeastL, atMostL) using Primal-Dual programming with different t-norm implementations. The experiments validate the framework's ability to enforce logical counting constraints on neural network predictions.

## Overview

The PMD (Primal-Dual) approach combines neural learning with logical constraints by maintaining dual variables that enforce constraint satisfaction during training. This experiment specifically focuses on counting constraints over binary predictions.

## Files

- **`main.py`** - Main experiment runner with PMD and sampling program options
- **`graph.py`** - Graph definition for PMD counting constraints using DomiKnows
- **`utils.py`** - PMD-specific utility functions (dataset creation, training, evaluation)
- **`test_PMD_training.py`** - Comprehensive test suite with 12 test scenarios
- **`testcase.py`** - Generic test runner for parameter combinations

## Constraint Types

### ExactL Constraint
Enforces exactly L predictions of a specific value:
```python
exactL(expected_value, L)
```

### AtLeastL Constraint
Enforces at least L predictions of a specific value:
```python
atLeastL(expected_value, L)
```

### AtMostL Constraint
Enforces at most L predictions of a specific value:
```python
atMostL(expected_value, L)
```

## T-Norm Implementations

The framework supports multiple t-norms for logical operations:

- **G** - Gödel t-norm (min operation)
- **P** - Product t-norm (multiplication)
- **L** - Łukasiewicz t-norm (max(0, a+b-1))
- **SP** - Simple Product t-norm

## Usage

### Quick Test (Verify Installation)
Run a minimal test to ensure the framework is working correctly (1-2 minutes):

```bash
uv run python main.py --model PMD --counting_tnorm G --epoch 50 --N 10 --M 5
```

This minimal test:
- Uses simplified dimensions (N=10 features, M=5 predictions)
- Trains for only 50 epochs
- Tests basic Gödel t-norm
- **Completes quickly to verify setup** (may not satisfy constraints fully)

**Expected output:**
- Training progress bar completes
- Shows gradient norms and predictions
- May show "Test case FAILED" - this is normal for quick test
- Verifies the framework runs without errors

**For full constraint satisfaction**, use the complete parameters below.

### Basic Execution
```bash
uv run python main.py --model PMD --counting_tnorm G --epoch 1000 --expected_value 0 --expected_atLeastL 2
```

### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Program type | sampling | PMD, sampling |
| `--counting_tnorm` | T-norm for counting | SP | G, P, L, SP |
| `--atLeastL` | Use atLeastL constraint | False | True, False |
| `--atMostL` | Use atMostL constraint | False | True, False |
| `--epoch` | Training epochs | 500 | int |
| `--expected_value` | Target prediction value | 0 | 0, 1 |
| `--expected_atLeastL` | Minimum count threshold | 3 | int |
| `--expected_atMostL` | Maximum count threshold | 3 | int |
| `--N` | Input feature dimension | 10 | int |
| `--M` | Number of predictions | 8 | int |
| `--beta` | Constraint weight | 10 | float |
| `--device` | Computing device | auto | auto, cpu, cuda |

### Sampling Program Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--sample_size` | Monte Carlo samples | -1 (adaptive) |

## Test Suite

Run the comprehensive test suite:
```bash
uv run pytest test_PMD_training.py -v
```

The test suite includes 12 scenarios covering:
- All 4 t-norms (G, P, L, SP)
- All 3 constraint types (exactL, atLeastL, atMostL)

### Test Scenarios

1. **ExactL Tests** (4 tests)
   - `test_PMD_exactL_Godel`
   - `test_PMD_exactL_Product`
   - `test_PMD_exactL_SimpProduce`
   - `test_PMD_exactL_Lukas`

2. **AtLeastL Tests** (4 tests)
   - `test_PMD_atLeastL_Godel`
   - `test_PMD_atLeastL_Product`
   - `test_PMD_atLeastL_SimpProduce`
   - `test_PMD_atLeastL_Lukas`

3. **AtMostL Tests** (4 tests)
   - `test_PMD_atMostL_Godel`
   - `test_PMD_atMostL_Product`
   - `test_PMD_atMostL_SimpProduce`
   - `test_PMD_atMostL_Lukas`

## Advanced Testing

### Custom Parameter Combinations
```bash
uv run python testcase.py --gpus 0,1 --max-workers 4
```

### CI/CD Integration
The tests automatically adapt for CI environments:
- Reduced epochs (50 instead of 1000)
- Single worker execution
- Simplified logging

## Architecture

### Neural Component
- 2-layer MLP with ReLU activation
- Input: N-dimensional features
- Output: Binary classification logits

### Constraint Component
- Logical counting constraints over predictions
- Dual variable optimization
- Multiple t-norm support

### Training Process
1. **Warmup Phase** - Standard neural training (2 epochs)
2. **Baseline Evaluation** - Discrete prediction assessment
3. **Constraint Phase** - Constraint-only optimization
4. **Final Evaluation** - Test constraint satisfaction

## Expected Results

Successful tests should show:
- Constraint satisfaction within specified thresholds
- Stable convergence during training
- Appropriate prediction counts after constraint optimization

## Troubleshooting

### Common Issues

1. **Non-finite Loss**
   ```
   [WARN] non-finite loss encountered; skipping step.
   ```
   - Usually resolves automatically with gradient clipping
   - Check learning rates if persistent

2. **Test Failures**
   - Review constraint parameters for feasibility
   - Ensure sufficient training epochs
   - Check device compatibility

3. **Memory Issues**
   - Reduce batch size or model dimensions
   - Use CPU device for large parameter sweeps

### Debug Mode
Enable detailed debugging:
```bash
CUDA_LAUNCH_BLOCKING=1 uv run python main.py --device cpu
```