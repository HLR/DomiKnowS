# Spatial Role Labeling

This is an example of the Spatial Role Labeling (SpRL) task.

## Problem description

To showcase the effectiveness of the current framework and the components of our pipeline, we perform the spatial role labeling and spatial relation extraction task and validate on CLEF 2017 mSpRL datasets.

The task is as follow:
> **given** an input text such as 
>> *"About 20 kids in traditional clothing and hats waiting on stairs."*,
>
> **find** a model that is able to extract the semantic entity types *(e.g., Landmark, Trajector, and SpatialIndicator)* as well as relations between them *(e.g., region, direction, distance)*, and
>
> **generate** the following output:
>> *[About 20 kids]*<sub>Trajector</sub> in tradirional clothing and hats waiting *[on]*<sub>SpatialIndicator</sub> *[stairs]*<sub>Landmark</sub>.

## Components

The example consists of several parts.

1. **Data reader** [`SpRL/SpRL_reader.py`](SpRL_new/SpRL_reader.py): raw data from mSpRL data set is located [`data`](data). We build on the top of AllenNLP in this example. So the reader presents as a [`allennlp.data.DatasetReader`] finally.

2. **Graph** [`SpRL/spGraph.py`](SpRL/spGraph.py): domain knowledge is presented as a graph. This is known as the **Ontology Declaration** step of the pipeline.
Concepts are nodes in the graph and relations are the edges of the nodes. The graph can be compiled from OWL format.
[`SpRL/spGraph.py`](SpRL_new/graph.py) is the graph used in SpRL example showing concepts and relations, such as `region`, `direction`, `distance`), which means when there is a region relation for a triple with three arguments (landmark, trajector, spatialindicator). By the way, in terms of "spatial_relation", it is a kind of relation type to identify whether the predicated triplets belong to any relation type (`region`, `direction`, `distance`).

3. **Main** [`SpRL/sprlApp.py`](SpRL/sprl_App.py): the main entrance of the program, where the components are [put into one piece]. [`AllenNlpGraph`] is a wrapper class providing a set of helper functions to make the model run with AllenNLP and PyTorch, which connect us to GPU. Learning based program [`lbp`] is constructed in [`model_declaration`] by connecting sensors and learners to the model. This is know as the **Model Declaration** step of the pipeline.

4. **Solver** [`../../regr/solver/gurobiILPOntSolver.py`](../../regr/solver/gurobiILPOntSolver.py): it seems we missed something. **Explicit inference**, is not present explicitly?!
They are done in `AllenNlpGraph` with help of a [base model](../../regr/graph/allennlp/model.py) automatically. Cheers!
We convert the inference into an integer linear programming (ILP) problem and maximize the overall confidence of the truth values of all concepts while satisfying the global constraints.
We derive constraints from the input ontology.
Two types of constraints are considered: the [disjoint constraints for the concepts] and [the composed-of constraints for the ralation].
By [solving the generated ILP problem], we can obtain a set of predictions that considers the structure of the data and the knowledge that is expressed in the domain ontology.


## Run the example

**Be sure to run anything in this instruction from current directory `examples/SpRL`, rather than from the project root.**

### Prepare

#### Setup

Follow [instructions](../../README.md#requirement) of the project to have `regr` ready.
Install `SpaCy` model: use command `python -m spacy download en_core_web_lg` to download SpaCy language model.

#### Data

The SpRL data is located in [`data`](data), and the training set is [data/sprl2017_train.xml], and the testing set is [data/sprl2017_gold.xml].
If you want to change the dataset directories, you can update [config.py] (config.py) to update the train and test directories.

### Run

The example can be run by using the following command:
```bash
python sprlApp.py
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
python sprlEval.py -m log.<starting date and time> -d <path to data corp> -b <batch size> [-M <model_file.th>]
```
You may want to avoid using GPU because they are used by a training procees. Here is a complete example:
```bash
CUDA_VISIBLE_DEVICES= python sprlEval.py -m ./log.20190724-174229 -d ./data/sprl2017_gold.xml -b=16
```
