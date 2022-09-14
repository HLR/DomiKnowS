### 5. Evaluation

#### Metrics
DomiKnows provides some tools to evaluate models on different aspects. First, we encode a set of predefined metrics such as F1, Precision, Recall, and Accuracy over concept classes which can be used as follows: 

```python3
Subclass_of_LearningBasedProgram(...,metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
    'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},...)
```
In which the 'Subclass_of_LearningBasedProgram' can be 'SolverPOIProgram' , 'PrimalDualProgram' or any other subclass of LearningBasedProgram. The input value of metric can be a pythonn dict or a single metric class. In the provided example we give metric a dict with two values. As a result, this program will print the metrics for the program with and without 'ILP'.

But suppose we don't want the ILP results. In that case case we can simply define our metric like:

```python3
Subclass_of_LearningBasedProgram(...,metric=PRF1Tracker(DatanodeCMMetric('local/argmax')),...)
```


The Metric is calculated in two main ways:
##### Binary Metric:

If a concept defined in our graph is binary, meaning it can be either true or false, it is evaluated as a binary metric. For every binary concept, Domiknows prints four values precision, recall, f1, and accuracy. This is done by the PRF1Tracker class that takes every Datanode as input and looks for binary concepts, their labels, and their predicted values.

##### Multiclass Metric:

Multiclass concepts have more evaluation methods. That is why it is useful to define a custom metric for multiclass concepts. However, the default metric will output a comprehensive list of evaluations that includes: 
-the confusion matrix
- micro f1, precision, recall, and accuracy for individual labels
- macro f1, precision, recall, and accuracy for the concept
_________
#### Defining Custom Metrics:

The default classes for metric evaluation that can be overwritten are 'PRF1Tracker' and 'DatanodeCMMetric', which are subclasses of 'MetricTracker' and 'torch.nn.Module', respectively.

#### Constraint Violation

As Domiknow's primary goal is to facilitate research in combining constraints and neural decisions, here, we introduce a new metric for evaluating the violation of constraints given the outputs of a neural network. 

This can be called using the following code:
```python3

ac_,t_=0,0
for datanode in program.populate(reader_data, device="cpu"):
    datanode.inferILPResults()
    verifyResult = datanode.verifyResultsLC()
    verifyResultILP = datanode.verifyResultsLC()
    ac_ += sum([verifyResultILP[lc]['satisfied'] for lc in verifyResultILP])
    t_ +=len(verifyResultILP.keys())

print("constraint accuracy: ", ac_ / t_ )
```

- Descriptions of how the constraint violation in computed
- Describe a particular case for If statements and how to get that number

#### Execution Time
Another essential factor in tracking the model's agility and usability is the time they take to be trained or used during inference. 
DomiKnows provide this information in log files accessible to programmers, which can be reviewed. 
The log files are enabled by default and can be disabled using:
```python3
The code to disable the log
```

- Description of what log file presents which feature
