### 3. Training Paradigms

Now that the `graph`, the `Property`s of `Concept`s are assigned with different types of `Sensor`s and `Learner`s, We can create a `Program` from the `graph`.

```python
program = LearningBasedProgram(graph, model_helper(primal_dual_model,poi=[question[is_less], question[is_more], question[no_effect],\
                                    symmetric, transitive],loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker()))
```
the inputs to the `LearningBasedProgram` are first the conceptual graph that we defined earlier. next, the type of model that can be a simple poimodel, a model with IML loss, or a primal_dual model. [Here](./apis/program) is a list of different programs available for the uses. these models are different in how they use constraints to produce a loss. the simple poi model simply ignores these constraints. these constraints can later be used during inference and do not necessarily need to be used here. next to our model, we define poi that stands for "Properties of Interest". we add the final (leaf node) properties that we want the program to calculate here that in this case are the properties `is_more`, `is_less`, and `no_effect` of the question, and the symmetric and transitive concepts. the next inputs are the type of our loss function and the metric that we want to calculate for each epoch. one can find explanations about different `loss` function [here](../domiknows/program/loss.py), and explanations about different `metrics` [here](../domiknows/program/metric.py).

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
program = SampleLossProgram(
        graph, SolverModel,
        poi=...,
        inferTypes=...,
        loss=...,
        sample = True,
        sampleSize=2000, 
        sampleGlobalLoss = False,
        beta=1,
        )
```

### Semantic-loss[[4]](#4)
The semantic loss method relis on a search to find the satisfactory cases and push the network to generate such examples with higher joint probability. More description of this method is in [4].

```python3
program = SampleLossProgram(
        graph, SolverModel,
        poi=...,
        inferTypes=...,
        loss=...,
        sample = True,
        sampleSize=-1, 
        sampleGlobalLoss = False,
        beta=1,
        )
```
 Setting `SampleSize` to `-1` will execute the Semantic-loss instead of sampling loss.



## Loss Selection
You can use any pytorch Loss function in the Program just by passing the function to compute the loss in the program property `loss`. 
You can also use the following pre-defined losses.

`NBCrossEntropyLoss` is the equivalant of CrossEntropy in Pytorch. See details in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

`BCEWithLogitsLoss` is the equivalant of BCE loss in Pytorch. See details in https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

`NBCrossEntropyIMLoss` is the equivalant of CrossEntropy Loss for using with the IML model.

You can also use multiple different Losses for each concept. To do so, you have to use the keyword `dictloss` instead of `loss` and use `SolverPOIDictLossProgram` instead of `SolverPOIPRogram`. 
An example of this is:
`dictloss={str(people.name): NBCrossEntropyDictLoss(), str(location.name): NBCrossEntropyDictLoss(), "default": NBCrossEntropyDictLoss()},`

The loss function by default receives the following inputs: `input, target`
In the case of DictLoss the inputs are: `builder, prop, input, target`

## References


<a id="1">[1] https://hlr.github.io/publications/Inference-Masked-Loss </a> 
Inference-Masked Loss for Deep Structured Output Learning, Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI 2020).


<a id="2">[2] https://papers.nips.cc/paper/2019/file/cf708fc1decf0337aded484f8f4519ae-Paper.pdf </a> 
A Primal-Dual Formulation for Deep Learning with Constraints, 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

<a id="3">[3] https://proceedings.mlr.press/v176/ahmed22a</a>
Ahmed, Kareem, et al. "PYLON: A PyTorch framework for learning with constraints." NeurIPS 2021 Competitions and Demonstrations Track. PMLR, 2022.

<a id="4">[4] https://proceedings.mlr.press/v80/xu18h.html</a>
Xu, Jingyi, et al. "A semantic loss function for deep learning with symbolic knowledge." International conference on machine learning. PMLR, 2018.
