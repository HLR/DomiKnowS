<!-- TOC depthto:4 withlinks:true -->

- [1. `TorchSensor`](#TorchSensor)
    - [1.1. `FunctionalSensor`](#FunctionalSensor)
        - [1.2.1 `JointSensor`](#JointSensor)
        - [1.2.2 `ConstantSensor`](#ConstantSensor)
            - [1.2.2.1 `ReaderSensor`](#ReaderSensor)
                - [1.2.2.1.1 `FunctionalReaderSensor`](#FunctionalReaderSensor)
                - [1.2.2.1.2 `JointReaderSensor`](#JointReaderSensor)
                - [1.2.2.1.3 `LabelReaderSensor`](#LabelReaderSensor)
        - [1.2.3 `SpacyTokenizorSensor`](#SpacyTokenizorSensor)
        - [1.2.4 `ModuleSensor`](#ModuleSensor)
        - [1.2.5 `TorchEdgeSensor`](#TorchEdgeSensor)
        - [1.2.6 `BertTokenizorSensor`](#BertTokenizorSensor)
        - [1.2.7 `CacheSensor`](#CacheSensor)
    - [1.2 `PrefilledSensor`](#PrefilledSensor)
        - [1.3.1 `TriggerPrefilledSensor`](#TriggerPrefilledSensor)

<!-- /TOC -->

## 1. `TorchSensor`


## 1.1 `FunctionalSensor`

After ReaderSensor, `FunationalSensor` is the most basic sensor. It inherits directly from `TorchSensor` and It overrides its `update_pre_context` and `update_context` functions. It also implements the `forward` fucntion.

This sensor takes as input multiple of arguments, parse them with a python fucntion given to forward and outputs a single feature to be put in a `property` of a `concept`. The definition of a `FunationalSensor` is as such:

```python
FunctionalSensor(arg1,arg2,..., forward, label)
```
the first few arguments will be inputs from various `concept`s and `property`s. Forward get a python function as input that would parse the inputs. and label is an optional input that would determine wheather or not the output of the `FunationalSensor` is used in loss function.

For examples of this Sensor refer to the following examples:

examples->ACE05->model.py

examples->CIFAR10->Sensors->sensors.py

examples->Email_Spam->Sensors->sensors.py

examples->Propara->Propora-Complete.ipynb

examples->SentimentAnalysis->sensors->tweetGraph.py

examples->WIQA->WIQA_aug.py

examples->demo->main.py

examples->squad->BertQA.py

## 1.1.1 `JointSensor`

## 1.1.2 `ConstantSensor`

## 1.1.2.1 `ReaderSensor`

## 1.1.2.1.1 `FunctionalReaderSensor`

## 1.1.2.1.2 `JointReaderSensor`

## 1.1.2.1.3 `JointReaderSensor`

## 1.1.3 `SpacyTokenizorSensor`

## 1.1.4 `ModuleSensor`

## 1.1.5 `TorchEdgeSensor`

## 1.1.6 `BertTokenizorSensor`

## 1.1.7 `CacheSensor`

## 1.2 `PrefilledSensor`

## 1.2.1 `TriggerPrefilledSensor`


