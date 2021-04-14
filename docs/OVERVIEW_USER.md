# Overview (User perspective)

There are several components in the framework representing different design steps and programming workflows.
Please refer to each page for details and usages of the components.

- [Overview (User perspective)](#overview-user-perspective)
  - [How it works](#how-it-works)
  - [How to program](#how-to-program)
    - [Knowledge Declaration](#knowledge-declaration)
    - [Model Declaration](#model-declaration)
    - [Program](#program)
    - [Query and Access](#query-and-access)
    - [Inference](#inference)

## How it works

In DomiKnowS, the program connects concepts' properties in the knowledge with sensors (data accessing procedure) and learners (statistical models), handles the process of training, testing and inference.

First, a graph of concepts must be defined. this graph contains the concepts, their relationship with each other, and logical constraints defined on them. so far the graph is merely conceptual and it doesn't have actual data.

```python
with Graph('WIQA_graph') as graph:
    paragraph = Concept(name='paragraph')
    question = Concept(name='question')
    para_quest_contains, = paragraph.contains(question)
    ...

```

In the same graph we also define the constraints that we wish to apply on these concepts in addition to the constraints that are inherently defined with regards to the Concepts definition. in the follwing code we assert that the labels for is_more, is_less and disjoint must be True one at a time. the next line asserts that if the label for a question is is_more, the label for a questions is has a symmertic relation with it, should be is_less.

```python
with Graph('WIQA_graph') as graph:
    disjoint( is_more, is_less, no_effect)
    ifL(is_more, V(name='x'), is_less, V(name='y', v=('x', symmetric.name, s_arg2.name)))
```

Here enters the reader. the reader is a python iterable and each instance of it is a dictionary with the preliminary information for a single datapoint (preliminary means that these initial properties will be used later to produce other properties).
```python
reader = make_reader(file_address="data/WIQA_AUG/train.jsonl")
>>> print(reader[0])
{'paragraph_intext': 'A tree produces seeds The seeds..., 
'question_list': 'suppose there will be more new trees happe...
...}
```


Then we define sensors. initially sensors will read the properties from the reader.
```python
paragraph['paragraph_intext'] = ReaderSensor(keyword='paragraph_intext')
paragraph['question_list'] = ReaderSensor(keyword='question_list')
...
```

Then some others sensors will use this properties to calculate other properties. in the code below, the program uses the question_paragraph and the text property of the question to create token_ids and Mask by feeding the input to a RobertaTokenizer.
```python
question["token_ids", "Mask"] = JointSensor(para_quest_contains, "question_paragraph", 'text',forward=RobertaTokenizer())
question[is_more] = FunctionalSensor("is_more_", forward=label_reader, label=True)
```
The learner is a type of sensor. while it does use the properties of a concept to calculate a new property, it changes itself and improves its calculation.
```python
question["robert_emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model)
```
Now another learner can use this predicted property of robert_emb infre another property.
```python
question[is_more] = ModuleLearner("robert_emb", module=RobertaClassificationHead(roberta_model.last_layer_size))
```

The program, later when we define it, starts from the `properties of interest` and it recursively calls all the needed sensors until the final `property of interest` is inferred.

In the end, we define and train our program.

```python
program = LearningBasedProgram(graph,...,loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker()))
program.train(reader_train_aug, train_epoch_num=10)

```
After our program is trained we can do inferece by creating datanodes. in the code below we get the result for the `is_more` property before and after `ILP` inference.
```python
for paragraph_ in program.populate(reader):
        paragraph_.inferILPResults(is_more)
        for question_ in paragraph_.getChildDataNodes():
            print(question_.getAttribute(is_more))
            print(question_.getAttribute(is_more, "ILP"))
```

## How to program

DomiKnowS aims to regularize the design to be focused on knowledge.
The following sections introduce the major concepts.
For a complete pipeline of programming with DomiKnowS, please referer to the [pipeline](PIPELINE.md).

### Knowledge Declaration

Class reference:

- `regr.graph.Graph`
- `regr.graph.Concept`
- `regr.graph.Property`
- `regr.graph.Relation`
- `regr.graph.LogicalConstrain`

In knowledge declaration, the user defines a collection of `Concept`s and the way they are related to each other (`Relation`), representing the domain knowledge.
We provide a graph language based on python for knowledge declaration with the notation of `Graph`, `Concept`, `Property`, `Relation`, `LogicalConstrain`.

### Model Declaration

Class reference:

- `regr.sensor.Sensor`
- `regr.sensor.Learner`

In the model declaration, the user defines how external resources (raw data), external procedures (preprocessing), and trainable deep learning modules are associated with the concepts and properties in the graph.
`Sensors` are procedures that access external resources and other sensors. For example, reading from raw data, feature engineering staffs, and preprocessing procedures.
`Learners` are computational modules that predict the unknown representations or probabilities of properties based on learning. `Learners` are differentiable components and come with parameters that can be tuned by gradient-based methods.

### Program

Class reference:

- `regr.program.Program`

The graph, attached with sensors and learners, can be turned into a learning-based program. The program can learn parameters from data, doing tests, or simply making predictions.

### Query and Access

Class reference:

- `regr.graph.DataNode`

The program uses and returns a data structure known as the datanode. They are an instance of the conceptual graph and are created using sensor inputs and learner predictions.

### Inference

Class reference:

- `regr.graph.DataNode`

One of the advantages of the framework is to do global inference over local predictions (made by learners or read by sensors).
The user can invoke inference from `DataNode` by `DataNode.inferILPResults()` and augment the attributes with the best combination of local predictions (configuration) that satisfies all the explicit constraints, as well as those implied by the relations.
