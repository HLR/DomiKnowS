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
Some sensors will use properties to calculate other properties.`JointSensor` can calculate multiple properties at once.
The first few arguments' properties will be passed to the forward `__call__` method. Result of this calculation depends on the output shape of the forward object's `__call__` method.
```python
node["arg1-co","arg2-co",...] = JointSensor(arg1,arg2,...,forward=RobertaTokenizer())
```
In the code below, the program uses the `question_paragraph` and the `text` properties of the question to create `token_ids` and `Mask` by feeding the input to a RobertaTokenizer. afterward, the sensor saves the newly created properties to be used later.

```python
question["token_ids", "Mask"] = JointSensor( "question_paragraph", 'text',forward=RobertaTokenizer())
```

## 1.1.2 `ConstantSensor`
This sensor's use case is to hold a constant value during the computation. It's value won't change during the optimization step.
for instance, in the below example it holds the value of word indexes.
```python
word['index'] = ConstantSensor(data=['words', 'case', 'quality', 'is'])
```
## 1.1.2.1 `ReaderSensor`
`ReaderSensor` is the most used Sensor in DomiKnows. it is used to read values from the input data. This data can be labels for the training step or input for calculations
The example below shows its example for both input value and one label of an image classification system. `keyword` is the key of value in the input data dictionary.
```python
image['pixels'] = ReaderSensor(keyword='pixels')
image[animal] = ReaderSensor(keyword='animal',label=True)
```

## 1.1.2.1.1 `FunctionalReaderSensor`
`FunctionalReaderSensor` is same as the `ReaderSensor` with the difference that it accepts a `forward` argument which processes value of `keyword`.
Below example shows its basic usage.`
```python
def find_label(label_type):
    def find(data):
        label = torch.tensor([item==label_type for item in data])
        return label
    return find
phrase[people] = FunctionalReaderSensor(keyword='label', forward=find_label('Peop'), label=True)
phrase[organization] = FunctionalReaderSensor(keyword='label', forward=find_label('Org'), label=True)
phrase[location] = FunctionalReaderSensor(keyword='label', forward=find_label('Loc'), label=True)
```
## 1.1.2.1.2 `JointReaderSensor`
## TODO Darius should delete?
## 1.1.2.1.3 `LabelReaderSensor`
This Sensor is `ReaderSensor` that always has `label=True` in its `__init__` method. Below example shows its usage for an image classification system.
```python
image['pixels'] = ReaderSensor(keyword='pixels')
image[animal] = LabelReaderSensor(keyword='animal') # equivalent to ReaderSensor(keyword='animal',label=True)
```
## 1.1.3 `SpacyTokenizorSensor`
## TODO Darius should delete?
## 1.1.4 `ModuleSensor`
It is a type of Sensor that applies a Pytorch module on its inputs. Inputs come from its previous connections.
In this example BERT embedding is getting applied on token ids. 
```python
word['bert'] = ModuleSensor('ids', module=BERT())
```
## 1.1.5 `TorchEdgeSensor`
## TODO Darius should delete?

## 1.1.6 `BertTokenizorSensor`
## TODO Darius should delete?

## 1.1.7 `CacheSensor`
## TODO Darius should delete?

## 1.2 `PrefilledSensor`
## TODO Darius should delete?

## 1.2.1 `TriggerPrefilledSensor`
## TODO Darius should delete?


