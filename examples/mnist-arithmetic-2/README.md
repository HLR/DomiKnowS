## Testing
To run all tests:

```
bash download_checkpoints.sh
bash test_all.sh --cuda --log TimeOnly | tee test_out.txt
```

Logs will be stored in `test_out.txt` and the `all_logs/` folder.

Single model test:
```
bash download_checkpoints.sh
python test.py --model_name {Sampling, Semantic, PrimalDual, Explicit, DigitLabel, Baseline} --checkpoint_path checkpoints/<checkpoint>.pth
```

Single model test with ILP:
```
bash download_checkpoints.sh
python test.py --model_name {Baseline, Explicit} --checkpoint_path checkpoints/<checkpoint>.pth --ILP --no_fixedL
```

## Training
```
python train.py --model_name {Sampling, Semantic, PrimalDual, Explicit, DigitLabel, Baseline} --epochs 10
```

CUDA can be enabled for training and testing with the `--cuda` flag.

More command line arguments for `train.py` and `test.py` can be found with the `--help` flag.

## Results

| Method | Performance (Method - Baseline) | Low Data (Method - Baseline) | Constraint Violation |
| --- | --- | --- | --- |
| Baseline | 9.01% | 10.32% | 96.92% |
| PrimalDual | +89.39% | +85.01% | 3.18% |
| SamplingLoss | +89.55% | +85.60% | 2.86% |
| SemanticLoss | +89.61% | +84.80% | 2.76% |
| ILP | -2.11% | 0.00% | -% |
| ExplicitSum + ILP | +89.54% | +84.61% | -% |
| ExplicitSum | +89.54% | +84.61% | 2.88% |
| Supervised | +89.53% | +84.30% | 2.86% |
