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
100000 | 5 | 1e-5 | ILP | Symmetric | 86.76 | Not Test
100000 | 5 | 1e-5 | Pytorch | None |88.16 | 68.05
100000 | 5 | 1e-5 | PMD(beta = 0.5) | Symmetric | 83.57 | 67.600
100000 | 5 | 1e-5 | SimpleLoss(1) | Symmetric | 87.29 | 67.900
100000 | 5 | 1e-5 | ILP + PMD(beta = 1) | Symmetric | 86.184 | 73.800
100000 | 5 | 1e-5 | ILP + SimpleLoss(1) | Symmetric | 87.42| 76.800


#### Result Testing Augment
#### Large Train Set
Training sample size | epoch | learning rate | model | using constrain | Accuracy (%) | Accuracy on Augmented test only (%)
--- | :---: | :---: | :---: | :---: | :---: | ---:
Paper | ? | 1e-5 | ? | ESIM w/ Non-Regularize | 87.25 | 60.78
Paper | ? | 1e-5 | ? | ESIM w/ Regularize | 87.55 | 73.32
550146 | 3 | 1e-5 | Pytorch | w/o constrain | 90.01 | 72.900
550146 | 3 | 1e-5 | POI | w/o constrain | 86.025 | 70.750
550146 | 3 | 1e-5 | PMD | w/ constrain | 87.06 | 68.550
550146 | 3 | 1e-5 | SampleLoss(1) | w/ constrain | 89.46 | 73.550
550146 | 3 | 1e-5 | ILP + SampleLoss(1) | w/ constrain | 88.74 | 81.750
550146 | 3 | 1e-5 | ILP + PMD | w/ constrain | ??? | ???

