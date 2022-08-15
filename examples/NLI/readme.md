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
#### Small Train Set, 10%
Training sample size | epoch | learning rate | model | using constrain | Accuracy (%) | Accuracy on Augmented test only (%)
--- | :---: | :---: | :---: | :---: | :---: | ---:
100000 | 5 | 1e-5 | Pytorch | None |88.16 | 68.05
100000 | 5 | 1e-5 | POI(Same for ILP) | None | 88.160 | 69.900
100000 | 5 | 1e-5 | POI| Included | 88.740 | 70.000
100000 | 5 | 1e-5 | ILP| Included | 88.6504 | 77.350
100000 | 5 | 1e-5 | SimpleLoss(1) | Symmetric | 88.185 | 68.650
100000 | 5 | 1e-5 | ILP + SimpleLoss(1) | Symmetric | 88.26 | 76.300
100000 | 5 | 1e-5 | PMD | Symmetric | 88.38 | 70.050
100000 | 5 | 1e-5 | ILP + PMD(beta = 1) | Symmetric | 88.3804 | 79.900


#### Result Testing Augment
#### Large Train Set
Training sample size | epoch | learning rate | model | using constrain | Accuracy (%) | Accuracy on Augmented test only (%)
--- | :---: | :---: | :---: | :---: | :---: | ---:
Paper | ? | 1e-5 | ? | ESIM w/ Non-Regularize | 87.25 | 60.78
Paper | ? | 1e-5 | ? | ESIM w/ Regularize | 87.55 | 73.32
550146 | 3 | 1e-5 | Pytorch | w/o constrain | 90.01 | 72.900
550146 | 3 | 1e-5 | POI | w/o constrain | 89.65 | 71.600
550146 | 3 | 1e-5 | POI | w/ constrain | 90.255 | 72.900
550146 | 3 | 1e-5 | SampleLoss(1) | w/ constrain | 90.11 | 74.550
550146 | 3 | 1e-5 | ILP | w/ constrains | 90.260 | 80.250
550146 | 3 | 1e-5 | ILP + SampleLoss(1) | w/ constrain | 90.245 | 82.200
550146 | 3 | 1e-5 | PMD | w/ constrain | 90.04 | 73.500
550146 | 3 | 1e-5 | ILP + PMD | w/ constrain | 90.04| 79.050

