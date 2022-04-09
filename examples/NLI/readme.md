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
- pmd: using primaldual model
- iml: using iml model
- beta: beta for pmd or iml model

### Result
Training sample size | epoch | learning rate | model | using constrain | Accuracy (%)
--- | :---: | :---: | :---: | :---: | ---:
100000 | 5 | 1e-5 | POI | False | 83.89 (Dev Augmentation included),  
100000 | 5 | 1e-5 | POI | Only Symmetric | 85.39 (Dev Aug included)