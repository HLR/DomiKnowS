# Walkthrough Example

The following steps outline how to use our framework.

- Dataset
- Knowledge Declaration
- Model Declaration
- **Training and Testing**
- Inference


## Training and Testing

Now that the `graph`, the `Property`s of `Concept`s are assigned with different types of `Sensor`s and `Learner`s, We can create a `Program` from the `graph`.

```python
program = LearningBasedProgram(graph, model_helper(primal_dual_model,poi=[question[is_less], question[is_more], question[no_effect],\
                                    symmetric, transitive],loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker()))
```
The inputs to the `LearningBasedProgram` are first the conceptual graph that we defined earlier. Next, the type of model can be a simple Poisson model, a model with IML loss, or a primal-dual model. These models differ in how they utilize constraints to minimize a loss. The simple POI model simply ignores these constraints. These constraints can be used later during inference and are not necessary for use here. Next to our model, we define POI, which stands for "Properties of Interest". We add the final (leaf node) properties that we want the program to calculate here, which in this case are the properties `is_more`, `is_less`, and `no_effect` of the question, and the symmetric and transitive concepts. The next inputs are the type of our loss function and the metric that we want to calculate for each epoch.


With `Reader` and `Program` prepared by the modeling step, the user can now train the program.
Simply do 

```python
program.train(reader, train_epoch_num=10, Optim=lambda param: AdamW(param, lr = args.learning_rate,eps = 1e-8 ), device='cuda:0')
print(program.model.loss)  # last training loss will be printed
print(program.model.metric)  # last training metrics will be printed
```

Here, `program` will check for "Multiple Assignment" of `Property` and generate a loss between each two `Sensor`s and/or `Learner`s where one has `label=True` and the other has `label=False`. The default total loss will be the sum of all "Multiple Assignment" losses, and optimization will be used with `Optim`. Parameters in direct and indirect `Learner`s will be updated towards a lower total loss.

After training, we can test our trained program on another dataset

```python
program.test(test_reader)
print(program.model.loss)  
print(program.model.metric)  
```

Check out for more details about [workflows in the program](../../Main%20Components/Workflow%20%28Training%29.md)

____
[Goto next section (Inference)](Inference.md)


