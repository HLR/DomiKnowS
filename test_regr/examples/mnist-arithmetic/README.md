# Training MNIST-Arithmetic

## Usage
```bash
python train.py --model_name {Sampling, Semantic, PrimalDual, Explicit, DigitLabel, Baseline, GBI} --epochs 10 --num_train 500
```

Enable CUDA with `--cuda`.

## Testing
```bash
# Run all tests
pytest

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"

# Run with coverage
pytest --cov=.

# Run specific test file
pytest test_data.py
```

## Training Examples
```bash
python train.py --model_name Sampling --epochs 10 --num_train 500
python train.py --model_name Semantic --epochs 10 --num_train 500
python train.py --model_name PrimalDual --epochs 10 --num_train 500
```

## Target Accuracy (500 samples)
| Method | Accuracy |
| --- | --- |
| SamplingLoss | 95.92% |
| SemanticLoss | 95.12% |
| PrimalDual | 95.33% |