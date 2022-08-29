### 3. Training Paradigms

Now that the `graph`, the `Property`s of `Concept`s are assigned with different types of `Sensor`s and `Learner`s, We can create a `Program` from the `graph`.

```python
program = LearningBasedProgram(graph, model_helper(primal_dual_model,poi=[question[is_less], question[is_more], question[no_effect],\
                                    symmetric, transitive],loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker()))
```
the inputs to the `LearningBasedProgram` are first the conceptual graph that we defined earlier. next, the type of model that can be a simple poimodel, a model with IML loss, or a primal_dual model. [Here](./apis/program) is a list of different programs available for the uses. these models are different in how they use constraints to produce a loss. the simple poi model simply ignores these constraints. these constraints can later be used during inference and do not necessarily need to be used here. next to our model, we define poi that stands for "Properties of Interest". we add the final (leaf node) properties that we want the program to calculate here that in this case are the properties `is_more`, `is_less`, and `no_effect` of the question, and the symmetric and transitive concepts. the next inputs are the type of our loss function and the metric that we want to calculate for each epoch. one can find explanations about different `loss` function [here](../regr/program/loss.py), and explanations about different `metrics` [here](../regr/program/metric.py).

The supported types of training time constraint integration: 

## Programs

There are diffrent types of programs that can be used to train the data. `SolverPOIProgram` is the simplest program that uses only the input and label of given data to learn its parameters.

However, Domiknows can leverage the defined constraints in the graph to teach the model not to violate those constraints and in the process imrove the overall performance of the mdoel.

### IML [[1]](#1)

In order to use IML, the program should be initialized by `IMLProgram`. IMLProgram extends `LearningBasedProgram` and shares most of the same input arguments.

```python
program = IMLProgram(graph, poi=(image, ), inferTypes=['ILP', 'local/argmax'], loss=..., metric=...})
```

The only new parameter here is `inferTypes` that is set to `['ILP', 'local/argmax']`. These parameter is optional and can be used to calcualte metric.

### Priaml Dual[[2]](#2)

Primal Dual model can also be easily defined by:

```python
prgram = PrimalDualProgram( graph, Model=SolverModel, poi=(sentence, phrase, pair), inferTypes=['local/argmax'],loss=..., metric=...)
            
```

Here instead of `POImodel` we use `SolverModel`.

### Sampling Loss[[3]](#3)
Sampling Loss program relies on a sampling strategy and a statisfaction check to augment the loss value with additional constraint based loss. The description of the method is in [3]. 

To use this, we should simply do the following:

```python3
Code to run the Sampling Loss
```

### Semantic-loss[[4]](#4)
The semantic loss method relis on a search to find the satisfactory cases and push the network to generate such examples with higher joint probability. More description of this method is in [4].

```python3
Code to run Semantic-loss method
```


## References


<a id="1">[1] https://hlr.github.io/publications/Inference-Masked-Loss </a> 
Inference-Masked Loss for Deep Structured Output Learning, Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI 2020).


<a id="2">[2] https://papers.nips.cc/paper/2019/file/cf708fc1decf0337aded484f8f4519ae-Paper.pdf </a> 
A Primal-Dual Formulation for Deep Learning with Constraints, 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.
