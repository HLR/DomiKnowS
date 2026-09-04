# DomiKnows Machine Learning Experiments

This repository contains two distinct machine learning experiments using the DomiKnows framework for neural-symbolic learning with logical constraints.

## Project Structure

```
project_root/
├── pmd_counting_tests/
│   ├── main.py                     # PMD experiment runner
│   ├── graph.py                    # PMD graph definition
│   ├── utils.py                    # PMD-specific utilities
│   ├── test_PMD_training.py        # PMD test suite
│   └── testcase.py                 # Generic test runner
├── relation_learning_tests/
│   ├── main_rel.py                 # Relation learning runner
│   ├── graph_rel.py                # Relation graph definition
│   ├── utils.py                    # Relation learning utilities
│   ├── run_curriculum_learning.py  # Curriculum learning
│   └── run_process.py              # Batch processing
└── dataset/                        # Optional: shared datasets
```

## Experiments Overview

### 1. PMD Counting Constraints (`pmd_counting_tests/`)
Tests counting constraints (exactL, atLeastL, atMostL) using different t-norms with Primal-Dual programming. This experiment validates the framework's ability to enforce counting constraints on neural predictions.

**Key Features:**
- Multiple t-norm implementations (Gödel, Product, Łukasiewicz, Simple Product)
- Primal-Dual and Sampling program variants
- Comprehensive test suite with 12 different constraint scenarios

### 2. Relation Learning (`relation_learning_tests/`)
Tests logical constraint learning on object relations using curriculum learning approaches. This experiment focuses on learning complex relational patterns between objects in scenes.

**Key Features:**
- Object property and relation classification
- Curriculum learning with progressive complexity
- ExistL and AndL logical constraints
- Batch processing for hyperparameter exploration

## Quick Start

### Verify Installation
Run minimal tests to ensure the framework is working correctly:

```bash
# Quick PMD test (1-2 minutes) - verifies framework runs
cd pmd_counting_tests
uv run python main.py --model PMD --counting_tnorm G --epoch 50 --N 10 --M 5

# Quick relation learning test (1-2 minutes) - verifies framework runs
cd relation_learning_tests
uv run python main_rel.py --N 100 --lr 1e-4 --epoch 2 --max_relation 1
```

**Note:** These quick tests verify the framework runs without errors. They may not fully satisfy constraints due to limited training time. For complete results, use the full parameters below.

### Running PMD Counting Tests
```bash
cd pmd_counting_tests
uv run python main.py --model PMD --counting_tnorm G --epoch 1000
```

### Running Relation Learning Tests
```bash
cd relation_learning_tests
uv run python main_rel.py --N 1000 --lr 1e-4 --epoch 10
```

## Testing

Each experiment includes comprehensive test suites:

**PMD Tests:**
```bash
cd pmd_counting_tests
uv run pytest test_PMD_training.py -v
```

**Relation Learning Tests:**
```bash
cd relation_learning_tests
uv run python run_process.py
```

## Common Parameters

Both experiments share some common configuration options:
- `--epoch`: Number of training epochs
- `--N`: Dataset size
- `--device`: Computing device (auto/cpu/cuda)
- `--lr`: Learning rate (relation learning only)