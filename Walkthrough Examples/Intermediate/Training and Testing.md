# Walkthrough Example

The followings are the user's steps to using our framework.

- Dataset
- Knowledge Declaration
- Model Declaration
- **Training and Testing**
- Inference


## Training and Testing

After connecting all the sensors and learners to the graph, we have to define a Program instance to be able to autoamtically train and test our models. Here, we use the POIProgram, which execute all the properties which have multiple assignment in the graph~(multiple sensors connect) and their dependencies.

```python
#ILP inference
program = SolverPOIProgram(graph,
                           poi=(sentence, phrase),
                           inferTypes=['ILP', 'local/argmax'],
                           loss=MacroAverageTracker(NBCrossEntropyLoss()),
                           metric={'ILP':PRF1Tracker(DatanodeCMMetric()), 'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))}
                           )

#IML inference and training
program1 = IMLProgram(graph, poi=(sentence, phrase), inferTypes=['ILP', 'softmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'softmax':PRF1Tracker(DatanodeCMMetric('softmax'))})

#Primal-Dual inference and training
program2 = PrimalDualProgram(graph, SolverModel, poi=(sentence, phrase), inferTypes=['ILP', 'softmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'softmax':PRF1Tracker(DatanodeCMMetric('softmax'))})

# No ILP
program3 = SolverPOIProgram(graph, poi=(sentence, phrase), inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})
```

To start training we can call the train method on the POIProgram instance

```python
program.train(list(iter(train_reader))[0:50], valid_set=list(iter(test_reader))[25:30], test_set=list(iter(test_reader))[0:5], train_epoch_num=1, Optim=lambda param: torch.optim.SGD(param, lr=.0001), device='cuda:0')
program.test(list(iter(test_reader))[0:10], device="cuda:0")
```

Checkout for more details about [workflows in the program](https://github.com/HLR/DomiKnowS/blob/Doc/Main%20Components/Workflow%20(Training).md)

____
[Goto next section (Inference)](Inference.md)


