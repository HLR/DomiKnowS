# Walkthrough Example

The followings are the user's steps to using our framework.

- Dataset
- Knowledge Declaration
- **Model Declaration**
- Training and Testing
- Inference

## Model Declaration


In the model declaration, the user defines how external resources (raw data), external procedures (preprocessing), and trainable deep learning modules are associated with the concepts and properties in the graph.
We use `Reader`s, `Sensor`s, and `Learner`s accordingly for the model declaration to create a *"full program"* as `Program`.

To create a program, the user needs to first assign `Sensor`s and `Learner`s to `Property`s of `Concept`s in the graph. Then initiate a `Program` with the graph.

There are different [pre-defined sensors](../../Main%20Components/Model%20Declaration%20%28Sensor%29.md#sensor) for basic data operation with PyTorch. Users can also extend [base `Sensor`](../../Main%20Components/Model%20Declaration%20%28Sensor%29.md#sensor) to customize for their task [by overriding `forward()` method](../../Main%20Components/Model%20Declaration%20%28Sensor%29.md#overriding-forward).

```python
email['subject'] = ReaderSensor(keyword='Subject')
email['body'] = ReaderSensor(keyword="Body")
email['forward_subject'] = ReaderSensor(keyword="ForwardSubject")
email['forward_body'] = ReaderSensor(keyword="ForwardBody")

```

In the example above, the first `ReaderSensor` is assigned to the properties `subject`, `Body` , `ForwardSubject` and `ForwardBody` to read these features from the reader.
is the following reader sensors, a list of concatenated properties are read from the reader and later through other sensors are given to individual questions.

```python
email['subject_rep'] = SentenceRepSensor('subject')
email['body_rep'] = SentenceRepSensor('body')
email['forward_presence'] = ForwardPresenceSensor('forward_body')
def concat(*x): 
    return torch.cat(x, dim=-1)
email['features'] = FunctionalSensor('subject_rep', 'body_rep', 'forward_presence', forward=concat)
```

Using our customized `Sensors`, `SentenceRepSensor`, and `ForwardPresenceSensor`, we create embeddings for the text features and finally, `FunctionalSensor` concatenates all these features in a unified property named `features`.

```python
email[Spam] = ModuleLearner('features', module=nn.Linear(601, 2))
email[Regular] = ModuleLearner('features', module=nn.Linear(601, 2))
email[Spam] = ReaderSensor(keyword='Spam', label=True)
email[Regular] = ReaderSensor(keyword='Regular', label=True)
```

`ModuleLearner` in the above code is used to calculate our predicted labels. Being a Learner, this sensor's parameters will change and update itself during training later. in the following. To assign the actaul labels we also use the `ReaderSensor`. Instead of overwriting the assignment, "Multiple Assignment" indicates the consistency of the `Sensor`s and `Learner`s assigned to a single `Property`. See ["Multiple Assignment" semantic](../../Main%20Components/Model%20Declaration%20%28Sensor%29.md#multiple-assigment-convention) for more information.

____
[Goto next section (Training and Testing)](Training%20and%20Testing.md)