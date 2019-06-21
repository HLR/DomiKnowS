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

## Pipeline

This example follows the pipeline we discussed in our preliminary paper.
1. Ontology Declaration
2. Model Declaration
3. Explicit inference

The steps are broken down into parts of the program.

## Composition

The example consists of several parts.

1. **Data reader** [`emr/data.py`](emr/data.py): raw data from CoNLL data set is located [`data/EntityMentionRelation`](data/EntityMentionRelation). We build on the top of AllenNLP in this example. So the reader presents as a [`allennlp.data.DatasetReader`](emr/data.py#L132) finally.

2. **Graph** [`emr/graph.py`](emr/graph.py): domain knowledge is presented as a graph. This is known as the **Ontology Declaration** step of the pipeline.
Concepts are nodes in the graph and relations are the edges of the nodes. The graph can be compiled from OWL format.
[`emr/graph.py`](emr/graph.py) is the graph used in EMR example showing concepts and relations, such as `work_for` (`people`, `organization`), which means when there is a work for relation, the first argument should be a people and the second argument should be an organization. 

3. **Main** [`emr/emr_full.py`](emr/emr_full.py): the main entrance of the program, where the components are [put into one piece](emr/emr_full.py#L80-L93). [`AllenNlpGraph`](emr/emr_full.py#L76) is a wrapper class providing a set of helper functions to make the model run with AllenNLP and PyTorch, which connect us to GPU. Learning based program [`lbp`](emr/emr_full.py#L88) is constructed in [`model_declaration`](emr/emr_full.py#L23-L77) by connecting sensors and learners to the model. This is know as the **Model Declaration** step of the pipeline.

4. **Solver** [`../../regr/ilpSelectClassification.py`](../../regr/solver/ilpSelectClassification.py): it seems we missed something. **Explicit inference**, is not present explicitly?!
They are done in `AllenNlpGraph` with help of a [base model](../../regr/graph/allennlp/model.py#L225) automatically. Cheers!
We convert the inference into an integer linear programming (ILP) problem and maximize the overall confidence of the truth values of all concepts while satisfying the global constraints.
We derive constraints from the input ontology.
Two types of constraints are considered: the [disjoint constraints for the concepts](../../regr/solver/ilpSelectClassification.py#L114-L316) and [the composed-of constraints for the ralation](../../regr/solver/ilpSelectClassification.py#L318-L613).
By [solving the generated ILP problem](../../regr/solver/ilpSelectClassification.py#L663), we can obtain a set of predictions that considers the structure of the data and the knowledge that is expressed in the domain ontology.


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
python3 -m emr
```

There are two a simpler version with only "people", "organization", and "work for" relationship is also avaliable with additional `--simple` or `-s` option.
```bash
python3 -m emr -s
```
