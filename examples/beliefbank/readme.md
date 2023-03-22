
# BeliefBank

## Requirements
install requirements:
```
pip install -r requirements.txt
```

spacy `en_core_web_sm` model:

```
pip install spacy
python -m spacy download en_core_web_sm
```


## How to run

simply write

```
!python -m beliefbank
```
After running the example for the first time the dataset is downloaded from the AllenAI official website. ( If the link is deprecated, check [AllenAI Website](https://allenai.org/data/beliefbank))

## Arguments

Domiknows program parameters:

+ --pd: whether or not to use primaldual constriant learning
+ --iml: whether or not to use IML constriant learning
+ --sam: whether or not to use sampling learning
+ --beta: primal dual or IML multiplier

Server parameters:

+ --cuda: cuda number to train the models on

AI training parameters:

+ --epoch: number of epochs you want your model to train on
+ --samplenum:sample sizes for low data regime 10,20,40 max 37
+ --simple_model: use the simplet model
+ --batch: batch size for neural network training
+ --lr: learning rate of the adam optimiser

## Results


| Data Usage | 25%  | 100% |
| ----------- | ----------- | ----------- |
| Base Domiknows | 94.36 | 94.90 |
| Base Domiknows + ILP | 93.39 | 95.11 |
| Sample Loss | 91.33 | 94.61 |
| Sample Loss + ILP | 92.05 | 96.0 |
| primal dual | 93.87 | 95.84 |
| primal dual + ILP | 95.43 | 96.22 |
