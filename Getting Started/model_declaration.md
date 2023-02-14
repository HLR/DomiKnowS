
## 2. Model Declaration

Class reference:

- `domiknows.data.reader.RegrReader`
- `domiknows.sensor.Sensor`
- `domiknows.sensor.Learner`
- `domiknows.program.Program`

In the model declaration, the user defines how external resources (raw data), external procedures (preprocessing), and trainable deep learning modules are associated with the concepts and properties in the graph.
We use `Reader`s, `Sensor`s, and `Learner`s accordingly for the model declaration to create a *"full program"* as `Program`.

To create a program, the user needs to first assign `Sensor`s and `Learner`s to `Property`s of `Concept`s in the graph. Then initiate a `Program` with the graph.

There are different [pre-defined sensors](./apis/sensor/PYTORCH.md) for basic data operation with PyTorch. Users can also extend [base `Sensor`](./apis/SENSORS.md) to customize for their task [by overriding `forward()` method](developer/MODEL.md#overriding-forward).

```python
paragraph['paragraph_intext'] = ReaderSensor(keyword='paragraph_intext')
paragraph['question_list'] = ReaderSensor(keyword='question_list')
paragraph['less_list'] = ReaderSensor(keyword='less_list')
paragraph['more_list'] = ReaderSensor(keyword='more_list')
paragraph['no_effect_list'] = ReaderSensor(keyword='no_effect_list')
paragraph['quest_ids'] = ReaderSensor(keyword='quest_ids')

```

In the example above, the first `ReaderSensor` is assigned to the property `'paragraph_intext'`.
is the following reader sensors, a list of concatenated properties are read from the reader and later through other sensors are given to individual questions.

```python
question[para_quest_contains, "question_paragraph", 'text', "is_more_", "is_less_", "no_effect_", "quest_id"] = JointSensor(
    paragraph['paragraph_intext'], paragraph['question_list'], paragraph['less_list'], paragraph['more_list'],
    paragraph['no_effect_list'], paragraph['quest_ids'],forward=make_questions)
```

A joint sensor is a sensor that outputs multiple properties. here it calculates `para_quest_contains`, `"question_paragraph"`, `'text'`, `"is_more_"`, `"is_less_"`, `"no_effect_"`and `"quest_id"` properties for a question while taking multiple properties of a paragraph. the forward function make_questions takes one instance of a paragraph with its properties in the input and the output, it returns seven lists of questions properties.
the first property `para_quest_contains` is the Containts relation between a paragraph and the questions from the definition of graph earlier and its list is equal to a torch of ones of the shape ( length of questions for a paragraph, 1). this encoding implies that a paragraph contains its questions. the length of questions can be different for different paragraphs.
the rest of the lists are lists of the properties of questions in a list of length equal to the number of questions.

```python
question["token_ids", "Mask"] = JointSensor( "question_paragraph", 'text',forward=RobertaTokenizer())
```
in another joint sensor here we take `question_paragraph` and `text` properties of a question, which are the text of its related paragraph and its own text respectively, feed it to a Roberta tokenizer and as output, we get `token_ids` and `Mask` properties of the question. unlike the previous joint sensor, here the input is a list not a single instance. the reason for that is the internal dynamics of DomiKnows. when questions were created they were created in a list and that's how they can be accessed in sensors. however, later during inference, they can be accessed individually. as a result, the output should also be two lists of the desired properties equal to the size of the input questions.


```python
question[is_more] = FunctionalSensor("is_more_", forward=label_reader, label=True)
question[is_less] = FunctionalSensor("is_less_", forward=label_reader, label=True)
question[no_effect] = FunctionalSensor("no_effect_", forward=label_reader, label=True)
```

the properties that we want to train our program on must be calculated in a sensor with `label=True`. joint sensor deals with multiple properties. so instead we use a functional sensor, which here takes the `is_more_`, `is_less_` and `no_effect_` properties of questions and with the function, `label_reader` returns them exactly as they are ( in a list) but with `label=True` in the sensor. the `is_more`, `is_less`, and `no_effect` properties here are also not a string but the variable assigned to these concepts in our graph. so we can calculate these properties here and also define constraints on them in the graph.

`Learner`s, are similar to `Sensor`s. The only difference is that `Learner`s have trainable parameters. The `Program` will update the parameters in `Learner`s based on model performance. we can assign `Learner`s to `Property`s of `Concept`s. `ModuleLearner` is specifically useful to plugin PyTorch modules.


```python
question["robert_emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model)
question[is_more] = ModuleLearner("robert_emb", module=RobertaClassificationHead(roberta_model.last_layer_size))
question[is_less] = ModuleLearner("robert_emb", module=RobertaClassificationHead(roberta_model.last_layer_size))
question[no_effect] = ModuleLearner("robert_emb", module=RobertaClassificationHead(roberta_model.last_layer_size))
```

`ModuleLearner` in the above code is used first to calculate an embedding for a question given its `token_ids` and `Mask` properties. being a Learner, this sensor's parameters will change and update itself during training later. in the following three lines, this `embedding` property is used to calculate the binary labels for `is_more`, `is_less`, and `no_effect`. these learners will also learn from predictions after calculating loss given the actual values of these properties.

It should be noted that we have assigned `ReaderSensor`s to the same `Property`s of `is_more`, `is_less`, and `no_effect`.
This is the ["Multiple Assignment" semantic](MODEL.md#multiple-assigment-convention) of the framework.
Instead of overwriting the assignment, "Multiple Assignment" indicates the consistency of the `Sensor`s and `Learner`s assigned to a single `Property`.

we should also define the sensors for symmetric and transitive concepts. these concepts have arguments and their definition is a little different from previous sensors. for these concepts, we use Edge Sensors.

```python
symmetric[s_arg1.reversed, s_arg2.reversed] = CompositionCandidateSensor(question['quest_id'],relations=(s_arg1.reversed, s_arg2.reversed),forward=guess_pair)
transitive[t_arg1.reversed, t_arg2.reversed, t_arg3.reversed] = CompositionCandidateSensor(question['quest_id'],relations=(t_arg1.reversed,t_arg2.reversed,t_arg3.reversed),forward=guess_triple)
```

`CompositionCandidateSensor` is an Edge sensor that takes two questions ( `quest_id` property of them is this case) and returns True or False determining whether or not they have symmetric relation. `CompositionCandidateSensor` is an Edge sensor that creates the relation Tensors for us but these tensors can be defined and output manually.
the same process also goes for the transitive concept with the difference being the input that is three questions and their `quest_id`s.
