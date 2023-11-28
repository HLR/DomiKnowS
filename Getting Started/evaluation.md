


### 5. Evaluation
#### Logical Constraints Verification

An application graph, which has defined logical constraints, can be executed using the command:
```python3
python graph.py
```
The result of this graph execution is either:
	No output, indicating that the logical constraints are both syntactically and semantically correct.
    An exception, detailing the first error encountered in the logical constraints, whether in syntax or semantics.
    
Additional tools defined in dataNodeDummy.py enhance the testing of logical constraints before further development of the ML model:
	```python3
	def createDummyDataNode(graph):
	```
	This method creates a sample data node based on the provided graph, with randomly assigned probabilities to entities.
	
	```python3
	def satisfactionReportOfConstraints(dn):
	```
	This method takes a data node as input, specifically the dummy data node created using the previous method. It generates a dictionary, with an entry for each logical constraint in the processed graph. Each entry contains information about the candidates gathered from the processed data node used to evaluate the constraint. The information under the key "lcSatisfactionMsgs" is divided into two sections: "Satisfied" and "Not Satisfied". These entries indicate whether the logical constraint was satisfied for a given set of candidates. Each message provides detailed evaluation of the constraint in the following example form:
	```
	ifL is satisfied (True) because:
		word('x') -> True
		atMostL(**) -> True
		ifL premise is True and its conclusion is True
	atMostL(**) is satisfied (True) because:
		people -> 0.0
		organization -> 0.0
		location -> 0.0
		other -> 0.0
		O -> 0.0
	```
	
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
- The confusion matrix
- Micro f1, precision, recall, and accuracy for individual labels
- Macro f1, precision, recall, and accuracy for the concept
_________
#### Defining Custom Metrics:

The default classes for metric evaluation that can be overwritten are `PRF1Tracker` and `DatanodeCMMetric`, which are subclasses of `MetricTracker` and `torch.nn.Module`, respectively.

#### Constraint Violation

As Domiknow's primary goal is to facilitate research in combining constraints and neural decisions, here, we introduce a new metric for evaluating the violation of constraints given the outputs of a neural network. 

This can be called using the following code:

```python3
program.verifyResultsLC(reader_data,names=None)
```

`program` is a subclass of LearningBasedProgram that we defined earlier. `reader_data` is the dictionary of data that we need to evaluate, and it is usually the dev or test data. `names` is an optional argument and is a list of names of the constraints we want to evaluate. If names are not given, the function will print the results for every constraint possible. Each constraint will have default names given to it. However, for the important constraints, custom names can be defined in this way:

```python3

with Graph('graph_name') as graph:
    # concept definition
    # ...
    # Logical_constraint(...,name="my_constraint")
    # ex:
    # ifL(andL(fact_check('x'), existsL(implication('s', path=('x', implication)))), fact_check(path=('s', i_arg2)),name="positive_implication")
    #...

```

Here, `Logical_constraint` can be any class that is a subclass of `LogicalConstrain` such as `IfL`, `NandL`, `OrL`, and... . The argument name is given to the constraint at the end, and it should be unique, meaning that it should be used in previous concepts or constraints.

The function `verifyResultsLC` would print two accuracy metrics. One is the accuracy of all instances of the constraint, and the other one is the accuracy for `Datanode`s. In this case, the `Datanode` is considered correct if all the instances of a constraint are True in that `Datanode`; Otherwise, it is False.

The `program` method `verifyResultsLC` internally uses `datanode` `verifyResultsLC` methods, which returns dictionary with keys for specific satisfaction  type: `satisfied` general satisfaction for any constraint and  `ifSatisfied` only for `ifL` constraint. The value for the key is the percentage of the cases for the constraint which were satisfied. In case if there is no cases for the given constraints then float `nan` is returned. In case of `ifL` it means that there was not constraints with antecedent true. To test for the float `nan` value use:

```python3
	if verifyResult[lc]['ifSatisfied'] == verifyResult[lc]['ifSatisfied']:
```

#### Execution Time
Another essential factor in tracking the model's agility and usability is the time they take to be trained or used during inference. 
DomiKnows provide this information in log files accessible to programmers, which can be reviewed. 
The log files are enabled by default and can be disabled using:
```python3
The code to disable the log
```

- Description of what log file presents which feature

