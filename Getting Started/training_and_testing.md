## 3. Training and Testing

With `Reader` and `Program` prepared by modeling step, the user can train the program now.
Simply do 

```python
program.train(reader, train_epoch_num=10, Optim=lambda param: AdamW(param, lr = args.learning_rate,eps = 1e-8 ), device='cuda:0')
print(program.model.loss)  # last training loss will be printed
print(program.model.metric)  # last training metrics will be printed
```

Here, `program` will check for "Multiple Assignment" of `Property` and generate a loss between each two `Sensor`s and/or `Learner`s where one has `label=True` and the other has `label=False`. The default total loss will be the sum of all "Multiple Assignment" losses, and optimization will be used with `Optim`. Parameters in direct and indirect `Learner`s will be updated towards a lower total loss.

After training, we can test our trained program with another dataset

```python
program.test(test_reader)
print(program.model.loss)  
print(program.model.metric)  
```

Checkout for more details about [workflows in the program](developer/WORKFLOW.md)


### Production Running 
Describe necessary steps to increase the perfomance for running the production setting.
