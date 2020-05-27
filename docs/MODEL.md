# Model Declaration

- [Model Declaration](#model-declaration)
  - [Class Overview](#class-overview)
  - [Sensor](#sensor)
  - [Learner](#learner)
  - [Reader](#reader)
  - [Multiple Assigment Convention](#multiple-assigment-convention)

## Class Overview

* package `regr.sensor`:
* `Sensor`:
* `Learner`:

## Sensor

As our program requires readers as the starting point, you have to write some reader classes that interacts with our dataset.
Readers will have a function for each train, valid and test set that returns a generator over separate parts of the dataset.

a graph (`Graph` object) with its concepts (`Concept` objects) having properties (`Property` objects accessed as items of concepts) connected to raw data sensors (`Sensor` objects).

```python
reader = Reader()
sentence['raw'] = ReaderSensor(reader, 'sentence')
sentence.contains()[word] = SimpleTokenizorSensor('raw')
people['label'] = LabelReaderSensor(reader, 'people')
organization['label'] = LabelReaderSensor(reader, 'organization')
work_for['label'] = LabelReaderSensor(reader, 'work_for')
```

## Learner

The learning declaration of the program is where you will define the properties of your graph nodes and define edge functionalities. This part will be a combination of reader sensor, edge transformer, execution sensor and learners.

a graph (`Graph` object) with its concepts (`Concept` objects) having properties (`Property` objects accessed as items of concepts) connected to learners (`Learner` objects) and pass through sensors (`Sensor` objects). Now the graph is considered a *full program*.
Example:
```python
sentence['embed'] = BertLearner('raw')
people['label'] = LogisticRegression('embed')
organization['label'] = LogisticRegression('embed')
work_for['label'] = LogisticRegression('embed')
```

## Reader

To start a chain of learning algorithm first you have to assign a reader sensor to a property of your root node in the graph. The reader job is to initialize the examples for execution of the learning model. The output of this reader sensor is an example per execution.

## Multiple Assigment Convention
