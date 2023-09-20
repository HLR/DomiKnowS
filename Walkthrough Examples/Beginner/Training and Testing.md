# Walkthrough Example

The followings are the user's steps to using our framework.

- Dataset
- Knowledge Declaration
- Model Declaration
- **Training and Testing**
- Inference


## Training and Testing

With `Reader` and `Program` prepared by modeling step, the user can train the program now.
Simply do 

```python
program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})
program.train(train_dataset, test_set=test_dataset, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')
program.test(test_dataset, device="auto")
```

Here, `program` will check for "Multiple Assignment" of `Property` and generate a loss between each two `Sensor`s and/or `Learner`s where one has `label=True` and the other has `label=False`. The default total loss will be the sum of all "Multiple Assignment" losses, and optimization will be used with `Optim`. Parameters in direct and indirect `Learner`s will be updated towards a lower total loss.

After training, we can test our trained program with another dataset

```python
program.test(test_reader)
print(program.model.loss)  
print(program.model.metric)  
```

Checkout for more details about [workflows in the program](https://github.com/HLR/DomiKnowS/blob/Doc/Main%20Components/Workflow%20(Training).md)

____
[Goto next section (Inference)](Inference.md)


