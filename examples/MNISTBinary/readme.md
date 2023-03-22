
# MNISTBinary Mutual Exclusion Task

## Requirements
install requirements:
```
pip install -r requirements.txt
pip install torchvision
```


## How to run

Simply write

```
!python -m main
```

## Arguments

Domiknows program parameters:

+ --ilp: whether or not to use ilp
+ --pd: whether or not to use primaldual constriant learning
+ --iml: whether or not to use IML constriant learning
+ --sam: whether or not to use sampling learning
+ --beta: primal dual or IML multiplier

Domiknows program parameters:

Server parameters:
+ --cuda: cuda number to train the models on

AI training parameters:

+ --namesave: model name to save
+ --epoch: number of epochs you want your model to train on
+ --lr: learning rate of the adam optimiser
+ --test: dont train just test
+ --simple_model: use a simple baseline
+ --samplenum: number of samples to train the model on
+ --batch: batch size for neural network training

## Results

| Data Scale      | Baseline | Baseline+ILP | SampleLoss | SampleLoss+ILP | primal-dual | primal-dual+ILP |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| 100% | 94.23 | 94.47 | 93.06 | 93.71 | 94.37 | 94.55 |
| 5% | 88.78 | 90.48 | 91.97 | 93.18 | 93.18 | 93.18 |
  


