# CoNLL04: Entity Mention Relation

## Problem description

The task is as follow:
> **given** an input text such as 
>> *"Washington works for Associated Press."*,
>
> **find** a model that is able to extract the semantic entity types *(e.g., people, organizations, and locations)* as well as relations between them *(e.g., works for, lives in)*, and
>
> **generate** the following output:
>> *[Washington]*<sub>people</sub> *[works for]*<sub>worksFor</sub> *[Associated Press]*<sub>organization</sub>.

## Steps

> **Be sure to install the requirements mentioned in [project root](../../) and add [project root](../../) to python module search.**

> **Be sure to run anything in this instruction from current directory `examples/emr`, rather than from the project root.**

### 1. Data

The Conll04 data is located in [`data/EntityMentionRelation/conll04.corp`](data/EntityMentionRelation/conll04.corp).
Please consider resplit the data set.
A handy tool `conll_split` it provided in this example to split dataset for n-fold corss-validation.
```bash
python -m emr.conll_split data/EntityMentionRelation/conll04.corp
```

It will generate the datasets for trainging and testing [`data/EntityMentionRelation/conll04.corp_<fold_id>_<train/test>.corp`](data/EntityMentionRelation/).
For more options like folds or output format, please use `-h`: `python -m emr.conll_split -h`.

We conducted a five-fold experiment. *The following steps shows instruction of only one fold*. We take average over each fold and report the results. Please refer to our paper for more details.

Some basic statistics of a data file can be obtained by
```bash
python -m emr.conll_stat data/EntityMentionRelation/conll04.corp
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

### 2. Configuration

Edit the configuration file [`emr/config.py`](emr/config.py) if you want to spesify any option(s).

### 3. Training

The implement of the example is in package `emr`. Involved data is included in [`data/EntityMentionRelation`](data/EntityMentionRelation).

The example can be run by using the following command.
```bash
python -m emr
```

### 4. Logs

#### Loss and Metrics logs
Tensorboard style log is generated in `log.<starting date and time>`. To visualize the training process, use Tensorboard:
```bash
# replace log.<starting date and time> with your actural log directory
tensorboard --logdir=log.<starting date and time>
```

Tensorboard service will start at local port `6006` by default. Visit ['http://localhost:6006/'](http://localhost:6006/) using a modern browser.

Some useful filters:
* `(people|organization|location|other)-F1` show F1 score on entities.
* `(work_for|live_in|located_in|orgbase_on|kill)-F1` show F1 score on composed entities (relations).

#### Solver Logs

Solver log can be find in the same `log.<starting date and time>` directory and named `solver.log`. It could be too large to load at once after a long-term training and testing. You can try to show the last one solution by `tail`:
```bash
# replace log.<starting date and time> with your actural log directory
tail -n 1000 log.<starting date and time>/solver.log
```
Usually last 1000 lines, with option `-n 1000`, is enough for one solution.

`split-log.sh` is a small script to chunk the solver log per instance.
```bash
split-log.sh log.<starting date and time>/solver.log
```


### 5. Evaluation

To evaluate a trained model with specific dataset,
```bash
python -m emr.emr_testing -m log.<starting date and time> -d <path to data corp> -b <batch size>
```
Here is a complete example:
```bash
python -m emr.emr_testing -m ./log.20190724-174229 -d ./data/EntityMentionRelation/conll04.corp_1_test.corp -b=16
```
