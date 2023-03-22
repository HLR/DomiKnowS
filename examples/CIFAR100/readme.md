
# CIFAR100 Structural Hierarchical Task

## Requirements
install requirements:
```
pip install -r requirements.txt
pip install torchvision
```


## How to run

simply write

```
!python -m main
```
After running the example for the first time the dataset is downloaded from the CIFAR official website. ( If the link is deprecated, check [CIFAR Website](https://www.cs.toronto.edu/~kriz/cifar.html))

## Arguments

Domiknows graph parameter:
+ --graph_type: type of constraints to be defined (It can be only_exactL which does not define structural dependencies, or exactL_ifLorLbothways, which considers structural dependencies).

Domiknows program parameters:
+ --ilp: whether or not to use ilp
+ --pd: whether or not to use primaldual constriant learning
+ --iml: whether or not to use IML constriant learning
+ --sam: whether or not to use sampling learning
+ --beta: primal dual or IML multiplier

Server parameters:
+ --cuda: cuda number to train the models on
+ --verbose: print improved and damaged examples

AI training paramteres:
+ --samplenum: number of samples to choose from the dataset
+ --resnet: how big the resnetmodel is
+ --epochs: number of training epoch
+ --lr: learning rate of the adam optimiser
+ --test: dont train just test

Loading and saving models parameters:
+ --nameload:model name to load
+ --nameloadprogram: model name to load
+ --namesave: model name to save

## Results

| Model      | Average Accuracy | Average Accuracy Over Simple Baseline | Average Accuracy On Low Data |
| ----------- | ----------- | ----------- | ----------- |
| Baseline      | 58.03%      | 52.54% | 31.33% |
| Sampling loss   | +0.39%        | +0.54% | +2.18% |
| ILP            | +2.88%        | +3.18% | +1.90% |
| Sampling loss + ILP   | +2.42%        | +3.52% | +3.82% |
