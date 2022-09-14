


### 5. Evaluation

#### Metrics
DomiKnows provides some tools to evaluate models on different aspects. First, we encode a set of predefined metrics such as F1, Precision, Recall, and Accuracy over concept classes which can be used as follows: 

```python3
Subclass_of_LearningBasedProgram(...,metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
    'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},...)
```
In which the `Subclass_of_LearningBasedProgram` can be `SolverPOIProgram` , `PrimalDualProgram` or any other subclass of LearningBasedProgram. The input value of metric can be a pythonn dict or a single metric class. In the provided example we give metric a dict with two values. As a result, this program will print the metrics for the program with and without 'ILP'.

But suppose we don't want the ILP results. In that case case we can simply define our metric like:

```python3
Subclass_of_LearningBasedProgram(...,metric=PRF1Tracker(DatanodeCMMetric('local/argmax')),...)
```


The Metric is calculated in two main ways:
##### Binary Metric:

If a concept defined in our graph is binary, meaning it can be either true or false, it is evaluated as a binary metric. For every binary concept, Domiknows prints four values precision, recall, f1, and accuracy. This is done by the PRF1Tracker class that takes every Datanode as input and looks for binary concepts, their labels, and their predicted values.

##### Multiclass Metric:

Multiclass concepts have more evaluation methods. That is why it is useful to define a custom metric for multiclass concepts. However, the default metric will output a comprehensive list of evaluations that includes: 
- the confusion matrix
- micro f1, precision, recall, and accuracy for individual labels
- macro f1, precision, recall, and accuracy for the concept
_________
#### Defining Custom Metrics:

The default classes for metric evaluation that can be overwritten are `PRF1Tracker` and `DatanodeCMMetric`, which are subclasses of `MetricTracker` and `torch.nn.Module`, respectively.

#### Constraint Violation

As Domiknow's primary goal is to facilitate research in combining constraints and neural decisions, here, we introduce a new metric for evaluating the violation of constraints given the outputs of a neural network. 

This can be called using the following code:

```python3
program.verifyResultILP = datanode.verifyResultsLC(reader_data,names=None)
```

`program` is a subclass of LearningBasedProgram that we defined earlier. reader_data is the dictionary of data that we need to evaluate, and it is usually the dev or test data. `names` is an optional argument and is a list of names of the constraints we want to evaluate. If names are not given, the function will print the results for every constraint possible. Each constraint will have default names given to it. However, for the important constraints, custom names can be defined in this way:

```python3

with Graph('graph_name') as graph:
    # concept definition
    # ...
    # Logical_constraint(...,name="my_constraint")
    # ex:
    #     ifL(andL(fact_check('x'), existsL(implication('s', path=('x', implication)))), fact_check(path=('s', i_arg2)),name="positive_implication")
    #...

```

Here, Logical_constraint can be any class that is a subclass of `LogicalConstrain` such as `IfL`, `NandL`, `OrL`, and... . The argument name is given to the constraint at the end, and it should be unique, meaning that it should be used in previous concepts or constraints.

The function `verifyResultsLC` would print two accuracy metrics. One is the accuracy of all instances of the constraint, and the other one is the accuracy for `Datanode`s. in this case, the `Datanode` is considered correct if all the instances of a constraint are True in that `Datanode`; otherwise, it is False.

#### Execution Time
Another essential factor in tracking the model's agility and usability is the time they take to be trained or used during inference. 
DomiKnows provide this information in log files accessible to programmers, which can be reviewed. 
The log files are enabled by default and can be disabled using:
```python3
The code to disable the log
```

- Description of what log file presents which feature

