
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

## Arguments


+ --nameload:model name to load
+'--nameloadprogram: model name to load
+ --namesave: model name to save
+ --cuda: cuda number to train the models on
+ --ilp: whether or not to use ilp
+ --pd: whether or not to use primaldual constriant learning
+ --iml: whether or not to use IML constriant learning
+ --sam: whether or not to use sampling learning
+ --test: dont train just test
+ --verbose: print improved and damaged examples
+ --resnet: how big the resnetmodel is
+ --samplenum: number of samples to choose from the dataset
+ --epochs: number of training epoch
+ --lambdaValue: value of learning rate
+ --lr: learning rate of the adam optimiser
+ --beta: primal dual or IML multiplier
+ --graph_type: type of constraints to be defined
