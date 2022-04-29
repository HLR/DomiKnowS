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
100000 | 5 | 1e-5 | ILP + PMD(beta = 0.5) | Symmetric | 82.99 (Dev Aug included) | Not Test
100000 | 5 | 1e-5 | ILP + PMD(beta = 3) | Symmetric | 83.508 (Dev Aug included) | Not Test
100000 | 5 | 1e-5 | ILP + PMD(beta = 1) | Symmetric | 84.12 (Dev Aug included) | 73.800
150000 | 5 | 1e-5 | ILP | Symmetric | 85.167 (Dev Aug included) | Not Test


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

