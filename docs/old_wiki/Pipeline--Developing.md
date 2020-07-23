This document is about the **details of framework implementation**.

🚧 This is the **DRAFT** version document for a discussion of the next version of the framework. The functionalities are not yet implemented this way.

🚥 For the step-by-step guide to writing your learning based program, please refer to [Declarative Programming](https://github.com/kordjamshidi/RelationalGraph/wiki/Pipeline-%7C-Programming).

## Building the Program

We take a subset of [entity-mention-relation](https://github.com/kordjamshidi/RelationalGraph/tree/master/examples/emr) (EMR) problem here for example. We consider recognizing "people" mention, "organization" mention, and the "work for" relation from a sentence.
Here are the step-by-step guide and explanation of the implementation details.

### Knowledge Declaration

* **Input**: ontology file (`.OWL` file) in case you want to compile or nothing if you want to write in our graph language.
* **Output**: a graph (`Graph` object) containing subgraphs (`Graph` objects), concepts (`Concept` objects), and relations (`Relation` objects).
* **Example**:
```python
with Graph() as graph:
    sentence = Concept()
    word = Concept()
    sentence.contains(word)
    with Graph() as sub_graph():
        people = Concept()
        people.is_a(word)
        organization = Concept()
        organization.is_a(word)
        work_for = Concept()
        work_for.has_a(people)
        work_for.has_a(organization)
```

A graph is constructed either by manually coding or compiled from `OWL` file(s).
Each `Graph` object can contain other `Graph` objects as sub-graphs. Not cyclic reference in graph hierarchy is allowed.
Each `Graph` object can contain certain `Concept` objects. All the `Concept` objects can be retrieved from a graph (or sub-graph) with a (relative) pathname. For example, `graph['application/people']` and `app_graph['people']`.

The graph is a partial program, and there is no property, sensor, or learner connected. There is no behavior associated. It is only a data structure to express domain knowledge.

`{\pk: could you give a link to an actual toy example code which you bring pieces of code from and explain here? }`
`{\quan: I added a link to EMR in the beginning. However, this is still a draft. Not exactly any of the current versions. I added this declaration too.}`

### Data Declaration

* **Input**: a graph (`Graph` object).
* **Output**: a graph (`Graph` object) with its concepts (`Concept` objects) having properties (`Property` objects accessed as items of concepts) connected to raw data sensors (`Sensor` objects).
* **Example**:
```python
reader = Reader()
sentence['raw'] = ReaderSensor(reader, 'sentence')
sentence.contains()[word] = SimpleTokenizorSensor('raw')
people['label'] = LabelReaderSensor(reader, 'people')
organization['label'] = LabelReaderSensor(reader, 'organization')
work_for['label'] = LabelReaderSensor(reader, 'work_for')
```

### Learning Declaration

* **Input**: a graph (`Graph` object).
* **Output**: a graph (`Graph` object) with its concepts (`Concept` objects) having properties (`Property` objects accessed as items of concepts) connected to learners (`Learner` objects) and pass through sensors (`Sensor` objects). Now the graph is considered a *full program*.
* **Example**:
```python
sentence['embed'] = BertLearner('raw')
people['label'] = LogisticRegression('embed')
organization['label'] = LogisticRegression('embed')
work_for['label'] = LogisticRegression('embed')
```

## Interaction with the Program

### Training

* **Input**: a full program (`Graph` object), data (input-output pairs), and hyper-parameters for training algorithm.
* **Output**: a full program (`Graph` object) with its associated model (torch.Module)'s parameter updated according to training algorithm.
* **Example**:
```python
graph.train('path/to/data', weight_decay=0.01, epochs=50)
```

Training is a loop that consists ["Model Forward Calculation"](#model-forward-calculation), ["Inference"](#inference), ["Loss Function"](#loss-function), and ["Model Backward Calculation and Update"](#model-backward-calculation-and-update).

### Testing

* **Input**: a full program (`Graph` object), data (input-output pairs or input only).
* **Output**: prediction based on the associated model of the graph and inference process regarding the input. If the target output is provided, the result statistics also returned.
* **Example**:
```python
graph.test('path/to/test_data')
```

Testing is a loop that consists ["Model Forward Calculation"](#model-forward-calculation) and ["Inference"](#inference).
The results should be logged.

### Model Forward Calculation

* **Input**: a full program (`Graph` object), a (empty) context (`Context` object, not yet in-place).
* **Output**: an updated context (`dict`-like object) base on all the calculation involved, whose keys are the `.fullname` of sensors and/or learners and values are their corresponding output values.
* **Example**:
```python
with Context() as context:
    for sensor in people['embedding']:
        embedding = sensor()
    work_for['label']()
```

The forward calculation of the model, triggered by a sensor in the graph, involves retrieving raw data from data sensors and a serial of forward-calculations of learners. One can trigger the call from multiple concepts with the same context in one forward calculation.

All updates will go to the same context, and they can share results from common dependency without repeat calculation.
The dependency of calculation could be complex. We use a flattened data structure, namely a `dict`-like object (mutable map), to cache calculation for efficiency. Please refer to [`context`](https://github.com/kordjamshidi/RelationalGraph/wiki/Dev-%7C-Sensor-%7C-context) for more details.

The default exchangeable object in this phrase is `torch.Tensor` object.
The desired output of each learner (as returned by `.forward()` function) is a `torch.Tensor` object. In this case, we can make sure to `backward()` through all components in the network in ["Model Backward Calculation and Update"](#model-backward-calculation-and-update) step.

For sensors, however, could be varied. For the reader sensors that are to be created by the users, the output can be any python primitive type or `torch.Tensor` objects. In case it is a primitive numerical type (e.g., int, long, float, etc.), extend the numerical sensor (not yet in-place) superclass, and the output will be converted to `torch.Tensor` objects with specific `dtype`. In case it is a string or other hashable type (e.g., str, tuple, etc.), extend the nominal sensor (not yet in-place) superclass, and the output will be collected in vocabulary and converted to index represented by `torch.LongTensor` objects.

### Inference

* **Input**: a full program (`Graph` object), a root concept (`Concept` objects), a set of concepts (`Concept` objects), and other inference settings.
* **Output**: a root data node (`DataNode` object) after inference.
* **Example**:
```python
with Context() as context:
    sentence_dataNode = graph.get_data(sentence)
    with Context() as infer_context:
        sentence_dataNode_updated = sentence_dataNode.inference(people, work_for, property='label')
# ...
model_reult = people['label'](context=context)
infer_reult = people['label'](context=infer_context)
```

Inference uses the result of the [forward calculation](#model-forward-calculation), populates data of base data types, extracts the predicates (candidates with probability), find the best configuration concerning constraints derived from the graph.

### Loss Function

### Model Backward Calculation and Update

### Save Parameters