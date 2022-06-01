# Natural Language Inference
## Run the original train and test

```
python main.py --cuda 0 --lr 2e-6 --epoch 5 --batch_size 8
```

#### inputs

- cuda: the number of GPU you want to use
- epoch: how many epoch you want to train the program
- lr: learning rate for the AdamW optimiser
- training_sample: number of training sample will be used
- testing_sample: number of testing sample will be used

## Run the adversarially regularising logic bg knowledge testset

```
python main_group_sentences.py --lr 1e-5 --training_sample 100000 --epoch 5 --batch_size 32 --sym_relation True
```
- cuda: the number of GPU you want to use
- epoch: how many epoch you want to train the program
- lr: learning rate for the AdamW optimiser
- training_sample: number of training sample will be used
- testing_sample: number of testing sample will be used
- batch_size: batch size of sample
- sym_relation: using symmetric relation or not
- tran_relation: using transitive relation or not
- pmd: using 
- primaldual model
- iml: using iml model
- beta: beta for pmd or iml model

### Result  
#### Large Train Set  
Training sample size | epoch | learning rate | model | using constrain | Accuracy (%) | Accuracy on Augmented test only (%)
--- | :---: | :---: | :---: | :---: | :---: | ---:
100000 | 5 | 1e-5 | ILP | w/o symmetric | 83.89 (Dev Augmentation included) | Not Test 
100000 | 5 | 1e-5 | ILP | Symmetric | 85.392 (Dev Aug included) | Not Test
100000 | 5 | 1e-5 | PMD(beta = 0.5) | Symmetric | 83.57 (Dev Aug included) | 67.600
100000 | 5 | 1e-5 | SimpleLoss(1) | Symmetric | 84.058 | 67.900
100000 | 5 | 1e-5 | ILP + PMD(beta = 0.5) | Symmetric | 82.99 (Dev Aug included) | Not Test
100000 | 5 | 1e-5 | ILP + PMD(beta = 3) | Symmetric | 83.508 (Dev Aug included) | Not Test
100000 | 5 | 1e-5 | ILP + PMD(beta = 1) | Symmetric | 84.12 (Dev Aug included) | 73.800
100000 | 5 | 1e-5 | ILP + SimpleLoss(1) | Symmetric | 85.650 | 76.800
100000 | 5 | 1e-5 | Pytorch | None | 84.808(88.16) | 68.05
150000 | 5 | 1e-5 | ILP | Symmetric | 85.167 (Dev Aug included) | Not Test
550146 | 3 | 1e-5 | ILP + SimpleLoss(1) | Symmetric | 87.575 | 81.750
550146 | 5 | 1e-5 | ILP + SimpleLoss(1) | Symmetric | 86.367 | 72.650


#### Small Train Set  

Training sample size | epoch | learning rate | model | using constrain | Accuracy (%) | Accuracy on Augmented test only (%)
--- | :---: | :---: | :---: | :---: | :---: | ---:  
10000  | 5 | 1e-5 | Normal | w/o Symmetric | 78.07(80.17) | 57.200(61.000)
10000  | 5 | 1e-5 | Normal | Only Symmetric | 78.280 | 60.600
10000  | 5 | 1e-5 | PMD | Symmetric | 79.290 | 60.300
10000  | 5 | 1e-5 | ILP + PMD(beta = 0.5) | Symmetric | 79.392 | 60.400
10000  | 5 | 1e-5 | ILP | Symmetric | 79.510 | 62.700
10000  | 5 | 1e-5 | ILP + PMD(beta = 1) | Symmetric | 79.783 | 65.550
10000  | 5 | 1e-5 | ILP + PMD(beta = 3) | Symmetric | 80.99 | 64.100
10000  | 5 | 1e-5 | SimpleLoss(1) | Symmetric | 79.583 | 59.250
10000  | 5 | 1e-5 | SimpleLoss(1) + ILP | Symmetric | 81.933 | 70.150
10000  | 5 | 1e-5 | Pytorch | None | 82.1083 | 65.0503


#### Result Testing Augment
#### Large Train Set
Training sample size | epoch | learning rate | model | using constrain | Accuracy (%) | Accuracy on Augmented test only (%)
--- | :---: | :---: | :---: | :---: | :---: | ---:
Paper | ? | 1e-5 | ? | Non-Regularize | 87.25 | 60.78
Paper | ? | 1e-5 | ? | Regularize | 87.55 | 73.32
550146 | 3 | 1e-5 | Pytorch | w/o constrain | 90.01 | 72.900
550146 | 3 | 1e-5 | SampleLoss(1) | w/o constrain | 89.57 | 70.500
550146 | 3 | 1e-5 | SampleLoss(1) | w/ constrain | RUNNING | RUNNING
550146 | 3 | 1e-5 | ILP + SampleLoss(1) (Old one) | Symmetric | 88.74 | 81.750

