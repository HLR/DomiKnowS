# Entity Mention Relation

This is an example of the Entity Mention Relation (EMR) problem.

## Problem description

To showcase the effectiveness of the current framework and the components of our pipeline, we use the entity-mention-relation extraction (EMR) task and validate on [CoNLL data](http://cogcomp.org/page/resource_view/43).
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

#### Data

The Conll04 data is located in [`data/EntityMentionRelation/conll.corp`](data/EntityMentionRelation/conll.corp).
Please consider resplit the data set.
A handy tool `conll_split` it provided in this example to split dataset for n-fold corss-validation.
```bash
python3 -m emr.conll_split data/EntityMentionRelation/conll.corp
```

It will generate the datasets for trainging and testing [`data/EntityMentionRelation/conll.corp_<fold_id>_<train/test>.corp`](data/EntityMentionRelation/).
For more options like folds or output format, please use `-h`: `python3 -m emr.conll_split -h`.


Some basic statistics of a data file can be obtained by
```bash
python3 -m emr.conll_stat data/EntityMentionRelation/conll04.corp
```
You will see the something like this
```
- Sentence length / min 1
- Sentence length / max 169
- Sentence length / mean 23.41569978245105
- Sentence length / mid 23.0
- Sentence length / hist
             1.0     17.8    34.6    51.4    68.2    85.0    101.8   118.6   135.4   152.2  169.0
length        1.0    17.8    34.6    51.4    68.2    85.0   101.8   118.6   135.4   152.2  169.0
count      1957.0  2522.0   867.0   113.0    29.0    15.0     9.0     0.0     0.0     4.0    NaN
cum_count  1957.0  4479.0  5346.0  5459.0  5488.0  5503.0  5512.0  5512.0  5512.0  5516.0    NaN
- Labels count / O 114984
- Labels count / Loc 4765
- Labels count / Peop 3918
- Labels count / Other 2995
- Labels count / Org 2499
- Relation count / Live_In 521
- Relation count / OrgBased_In 452
- Relation count / Located_In 406
- Relation count / Work_For 401
- Relation count / Kill 268
```
You can show statistics on splited data sets by replacing the file name in above command.


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

Please also checkout configuration of the program in [`emr/config.py`](emr/config.py).


#### Logs

Tensorboard style log is generated in `log.<starting date and time>`. To visualize the training process, use Tensorboard:
```bash
# replace log.<starting date and time> with your actural log directory
tensorboard --logdir=log.<starting date and time>
```


Tensorboard service will start at local port `6006` by default. Visit ['http://localhost:6006/'](http://localhost:6006/) using a modern browser.

Some useful filters:
* `(people|organization|location|other)-F1` show F1 score on entities.
* `(work_for|live_in|located_in|orgbase_on|kill)-F1` show F1 score on composed entities (relations).

Solver log can be find in the same `log.<starting date and time>` directory and named `solver.log`. It could be too large to load at once after a long-term training and testing. You can try to show the last one solution by `tail`:
```bash
# replace log.<starting date and time> with your actural log directory
tail -n 1000 log.<starting date and time>/solver.log
```
Usually last 1000 lines, with option `-n 1000`, is enough for one solution.


### Evaluation

To evaluate a trained model with specific dataset,
```
python3 -m emr.emr_testing -m log.<starting date and time> -d <path to data corp> -b <batch size>
```
You may want to avoid using GPU because they are used by a training procees. Here is a complete example:
```
CUDA_VISIBLE_DEVICES=none python3 -m emr.emr_testing -m ./log.20190724-174229 -d ./data/EntityMentionRelation/conll04.corp_1_test.corp -b=16
```