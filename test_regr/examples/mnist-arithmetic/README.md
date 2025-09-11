# Training MNIST-Arithmetic
## Usage
```bash
python train.py --model_name {Sampling, Semantic, PrimalDual, Explicit, DigitLabel, Baseline} --epochs 10 --num_train 500
```

Enable CUDA with the `--cuda`.

## Test cases: Sampling loss, Semantic loss, Primal dual
```bash
python train.py --model_name Sampling --epochs 10 --num_train 500
python train.py --model_name Semantic --epochs 10 --num_train 500
python train.py --model_name PrimalDual --epochs 10 --num_train 500
```

Training and validation accuracy is printed every epoch (with `post_epoch_metrics`).

## Target accuracy (from AAAI results)
| Method | Accuracy w/ 500 samples |
| --- | --- |
| SamplingLoss | 95.92% |
| SemanticLoss | 95.12% |
| PrimalDual | 95.33% |
