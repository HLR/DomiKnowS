### 5. Evaluation

#### Metrics
DomiKnows provides some tools to evaluate models on different aspects. First, we encode a set of predefined metrics such as F1, Precision, Recall, Accuracy over concept classes which can be used as follows: 

```python3
Python code indicating how they can specify the metrics
```

metric is calculated in two main ways:
##### Binary Metric:

if a concept defined in our graph is binary, meaing that it can be either true or false, it is evaluated as a binary metric. for every binary concept, Domiknows prints four values precision, recall, f1 and accuracy. this is done by the PRF1Tracker class that takes every datanode as input and looks for binary concepts, their labels and their predicted values.

##### Multiclass Metric:

multiclass concepts have more evaluation methods. that's why it is usefull to define a custom metrik for multiclass concepts. however, the default metric will output a comprehensive list of evaluations that includes: 
-the confusion matrix
- micro f1, precision, recall and accuracy for individual labels
- macro f1, precision, recall and accuracy for the concept
_________
#### Defining Custom Metrics:
TODO

#### Constraint Violation

As Domiknows primary goal is to facilitate research in combining constraints and neural decision, here, we introduce a new metric for evaluating the violation of constraints given the outputs of a neural network. 

This can be called using the following code:
```python3
Python Code to run the constraint violation
```
- Desctiptions over how the constraint violation in computed
- Describe a special case for If statements and how to get that number

#### Execution Time
Another important factor to track the models agility and usability is the time they take whether to be trained or used during inference. 
DomiKnows provide this information in log files accessible to programmers which can be reviewed. 
The log files are enabled by default and can be disabled using:
```python3
The code to disable log
```

- Description of what log file presents which feature




