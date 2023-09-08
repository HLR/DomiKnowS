# DataSets and Their Results

- [CIFAR100](#cifar100-structural-hierarchical-task)
- [MNISTBinary](#mnist-binary-mutual-exclusion-task)
- [BeliefBank](#beliefbank)
- [MNIST](#mnist-arithmetic-task)


## CIFAR100 Structural Hierarchical Task

| Model      | Average Accuracy | Average Accuracy Over Simple Baseline | Average Accuracy On Low Data |
| ----------- | ----------- | ----------- | ----------- |
| Baseline      | 58.03%      | 52.54% | 31.33% |
| Sampling loss   | +0.39%        | +0.54% | +2.18% |
| ILP            | +2.88%        | +3.18% | +1.90% |
| Sampling loss + ILP   | +2.42%        | +3.52% | +3.82% |

## MNIST Binary Mutual Exclusion Task

| Data Scale      | Baseline | Baseline+ILP | SampleLoss | SampleLoss+ILP | primal-dual | primal-dual+ILP |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| 100% | 94.23 | 94.47 | 93.06 | 93.71 | 94.37 | 94.55 |
| 5% | 88.78 | 90.48 | 91.97 | 93.18 | 93.18 | 93.18 |

## BeliefBank

| Data Usage | 25%  | 100% |
| ----------- | ----------- | ----------- |
| Base Domiknows | 94.36 | 94.90 |
| Base Domiknows + ILP | 93.39 | 95.11 |
| Sample Loss | 91.33 | 94.61 |
| Sample Loss + ILP | 92.05 | 96.0 |
| primal dual | 93.87 | 95.84 |
| primal dual + ILP | 95.43 | 96.22 |

## MNIST Arithmetic Task

| Method | Performance (Method - Baseline) | Low Data (Method - Baseline) | Constraint Violation |
| --- | --- | --- | --- |
| Baseline | 9.01% | 10.32% | 96.92% |
| PrimalDual | +89.39% | +85.01% | 3.18% |
| SamplingLoss | +89.55% | +85.60% | 2.86% |
| SemanticLoss | +89.61% | +84.80% | 2.76% |
| ILP | -2.11% | 0.00% | -% |
| ExplicitSum + ILP | +89.+54% | +84.61% | -% |
| ExplicitSum | +89.54% | +84.61% | 2.88% |
| Supervised | +89.53% | +84.30% | 2.86% |
