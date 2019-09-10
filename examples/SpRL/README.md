# Spatial Role Labeling

This is an example of the Spatial Role Labeling (SpRL) task.

## Problem description

To showcase the effectiveness of the current framework and the components of our pipeline, we perform the spatial entity labeling and spatial role relationship extraction task and validate on CLEF 2017 mSpRL datasets.

The task is as follow:
> **given** an input text such as 
>> *"About 20 kids in traditional clothing and hats waiting on stairs."*,
>
> **find** a model that is able to extract the semantic entity types *(e.g., Landmark, Trajector, and SpatialIndicator)* as well as relations between them *(e.g., region, direction, distance)*, and
>
> **generate** the following output:
>> *[About 20 kids]*<sub>Trajector</sub> in tradirional clothing and hats waiting *[on]*<sub>SpatialIndicator</sub> *[stairs]*<sub>Landmark</sub>.


[//]: # (description of the problem to be added here)

## Pipeline

This example follows the pipeline we discussed in our preliminary paper.
1. Ontology Declaration
2. Model Declaration
3. Explicit inference

The steps are broken down into parts of the program.

## Composition

The example consists of several parts.

1. **Data reader** [`SpRL/SpRL_reader.py`](SpRL_new/SpRL_reader.py): raw data from mSpRL data set is located [`data`](data). We build on the top of AllenNLP in this example. So the reader presents as a [`allennlp.data.DatasetReader`] finally.

2. **Graph** [`SpRL/spGraph.py`](SpRL/spGraph.py): domain knowledge is presented as a graph. This is known as the **Ontology Declaration** step of the pipeline.
Concepts are nodes in the graph and relations are the edges of the nodes. The graph can be compiled from OWL format.
[`SpRL/spGraph.py`](SpRL_new/graph.py) is the graph used in SpRL example showing concepts and relations, such as `region`, `direction`, `distance`), which means when there is a region relation for a triple with three arguments (landmark, trajector, spatialindicator). By the way, in terms of "spatial_relation", it is a kind of relation type to identify whether the predicated triplets belong to any relation type (`region`, `direction`, `distance`).

3. **Main** [`SpRL/sprlApp.py`](SpRL/sprl_App.py): the main entrance of the program, where the components are [put into one piece]. [`AllenNlpGraph`] is a wrapper class providing a set of helper functions to make the model run with AllenNLP and PyTorch, which connect us to GPU. Learning based program [`lbp`] is constructed in [`model_declaration`] by connecting sensors and learners to the model. This is know as the **Model Declaration** step of the pipeline.

4. **Solver** [`../../regr/ilpSelectClassification.py`](../../regr/solver/ilpSelectClassification.py): it seems we missed something. **Explicit inference**, is not present explicitly?!
They are done in `AllenNlpGraph` with help of a [base model](../../regr/graph/allennlp/model.py#L225) automatically. Cheers!
We convert the inference into an integer linear programming (ILP) problem and maximize the overall confidence of the truth values of all concepts while satisfying the global constraints.
We derive constraints from the input ontology.
Two types of constraints are considered: the [disjoint constraints for the concepts] and [the composed-of constraints for the ralation].
By [solving the generated ILP problem], we can obtain a set of predictions that considers the structure of the data and the knowledge that is expressed in the domain ontology.


## Run the example

**Be sure to run anything in this instruction from current directory `examples/SpRL`, rather than from the project root.**

### Prepare

#### Tools

* Python 3: Python 3 is required by `allennlp`. Install by package management. Remember to mention the version in installation.
* 'Allennlp': The version of Allennlp should be 0.8.4
* `pytorch`: model calculation behind `allennlp`. There is a bunch of selection other than the standard pip package.
Follow the [installation instruction](https://pytorch.org/get-started/locally/) and select the correct CUDA version if any available to you.

#### Setup

Follow [instructions](../../README.md#prerequirements-and-setups) of the project to have `regr` ready.
Install all python dependency specified in `requirements.txt` by
```bash
sudo python3 -m pip install -r requirements.txt
```

#### Data

The SpRL data is located in [`SpRL/data`](SpRL/data), and the training set is [SpRL/data/sprl2017_train.xml], and the testing set is [SpRL/data/sprl2017_test.xml].
If you want to change the dataset directories, you can update [SpRL/config.py] (SpRL/config.py) to update the train and test directories.

#### Word2vec

We use [GloVe](https://nlp.stanford.edu/projects/glove/) (glove.6B.50d) in this example.
Please download [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip) and extract to `data/glove.6B`.
You can use the following commands
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -j "glove.6B.zip" "glove.6B.50d.txt" -d "data/glove.6B"
```
You can also download and extract other word representation, and setup in [`emr/config.py`](emr/config.py) correspondingly.

### The example

The implement of the example is in package `SpRL`. 

The example can be run by using the following command, or directly run SpRLApp.py
```bash
python3 sprlApp.py
```

#### Logs

Tensorboard style log is generated in `log.<starting date and time>`. To visualize the training process, use Tensorboard:
```bash
# replace log.<starting date and time> with your actural log directory
tensorboard --logdir=log.<starting date and time>
```


Tensorboard service will start at local port `6006` by default. Visit ['http://localhost:6006/'](http://localhost:6006/) using a modern browser.

Some useful filters:
* `(landmark|trajector|spatialindicator|other)-F1` show F1 score on entities.
* `(region|distance|direction|spatial_relation|other)-F1` show F1 score on composed entities (relations).

Solver log can be find in the same `log.<starting date and time>` directory and named `solver.log`. It could be too large to load at once after a long-term training and testing. You can try to show the last one solution by `tail`:
```bash
# replace log.<starting date and time> with your actural log directory
tail -n 1000 log.<starting date and time>/solver.log
```
Usually last 1000 lines, with option `-n 1000`, is enough for one solution.


### Evaluation

To evaluate a trained model with specific dataset,
```bash
python3 SpRL.sprlEval -m log.<starting date and time> -d <path to data corp> -b <batch size>
```
You may want to avoid using GPU because they are used by a training procees. Here is a complete example:
```bash
CUDA_VISIBLE_DEVICES=none python3 SpRL.sprlEval -m ./log.20190724-174229 -d ./data/EntityMentionRelation/conll04.corp_1_test.corp -b=16
```
