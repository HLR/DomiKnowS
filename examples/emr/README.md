# Entity Mention Relation

This is an example of the Entity Mention Relation (EMR) problem.

## Problem description

To showcase the effectiveness of the current framework and the components of our pipeline, we use the entity-mention-relation extraction (EMR) task and validate on [CoNLL data](https://www.clips.uantwerpen.be/conll2003/ner/).
The task is as follow:
> **given** an input text such as 
>> *"Washington works for Associated Press."*,
>
> **find** a model that is able to extract the semantic entity types *(e.g., people, organizations, and locations)* as well as relations between them *(e.g., works for, lives in)*, and
>
> **generate** the following output:
>> *[Washington]*<sub>people</sub> *[works for]*<sub>worksFor</sub> *[Associated Press]*<sub>organization</sub>.

[//]: # (description of the problem to be added here)


## Run the example

### Prepare

#### Tools

* Python 3: Python 3 is required by `allennlp`. Install by package management. Remember to mention the version in installation.
* `pip`: to install other required python packages. Follow the [installation instruction](https://pip.pypa.io/en/stable/installing/) and make sure you install it with your Python 3.
* `pytorch`: model calculation behind `allennlp`. There is a bunch of selection other than the standard pip package.
Follow the [installation instruction](https://pytorch.org/get-started/locally/) and select the correct CUDA version if any available to you.
* Anything else will be installed properly with `pip` (including `allennlp`). No worry here.

#### Setup

Follow [instructions](../../README.md#prerequirements-and-setups) of the project to have `regr` ready.
Install all python dependency specified in `requirements.txt` by
```bash
sudo python3 -m pip install -r requirements.txt
```

### The example

The implement of the example is in package `emr`. Involved data is included in [`data/EntityMentionRelation`](data/EntityMentionRelation).

The example can be run by using the following command.
```bash
python3.7 -m emr
```

There are two sub-examples.
`ner` example concider only one non-composed concept.
Run it by the following command.
```bash
python3.7 -m emr.ner
```

`ners` example concider multiple non-composed concept. They are independent classifiers, sharing the same word embedding.
Run it by the following command.
```bash
python3.7 -m emr.ners
```
