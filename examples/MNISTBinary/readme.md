
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

+ --namesave: model name to save
+ --cuda: cuda number to train the models on
+ --epoch: number of epochs you want your model to train on
+ --lr: learning rate of the adam optimiser
+ --ilp: whether or not to use ilp
+ --pd: whether or not to use primaldual constriant learning
+ --iml: whether or not to use IML constriant learning
+ --sam: whether or not to use sampling learning
+ --test: dont train just test
+ --simple_model: use a simple baseline
+ --samplenum: number of samples to train the model on
+ --batch: batch size for neural network training
+ --beta: primal dual or IML multiplier

