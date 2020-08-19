The Step by Step guide to **writing your learning based program**.

For the framework implementation details, please refer to [Framework Developing](https://github.com/kordjamshidi/RelationalGraph/wiki/Pipeline-%7C-Developing).

***
# Building the Program
## Knowledge Declaration

A `Graph` object is constructed either by manually coding or compiled from `OWL` file(s).
Each `Graph` object can contain other `Graph` objects as sub-graphs. No cyclic reference in graph hierarchy is allowed.

`We can not have Sentence.contains(word), Sentence.contains(Relation) and Relation.contains(Word)? is that a cycle? what do you mean by graph hierarchy? is that related to is-a relations only?`

The graph in this step is just concepts and relations, There is no behavior associated. It is only a data structure to express domain knowledge.

You can either write an owl file initializing your concepts and relations or to write your graph with our specific python classes. for more information on how to write your domain knowledge please refer to the following tutorials.

1. how to make a simple knowledge graph with python
2. how to make a simple knowledge graph with owl

`Are these a part of this Tutorial? or an external info? please give clear links here.`

## Data Declaration
As our program requires readers as the starting point, you have to write some reader classes that interacts with our dataset. 

Readers will have a function for each train, valid and test set that returns a generator over separate parts of the dataset. To see more on how to write a simple reader class please refer to the following tutorial.
3. how to write your first reader `(an actual link is needed)`

## Learning Declaration

The learning declaration of the program is where you will define the properties of your graph nodes and define edge functionalities. This part will be a combination of reader sensor, edge transformer, execution sensor and learners.

### Define a Reader Sensor

To start a chain of learning algorithm first you have to assign a reader sensor to a property of your root node in the graph. The reader job is to initialize the examples for execution of the learning model. The output of this reader sensor is an example per execution. To learn more about reader sensors please refer to the following tutorial. `(\pk: provide an actual link.)`

4. How to write your first reader sensor.

reader sensors will be assigned to a property of the root node as.
`root['raw'] = ReaderSensor(reader)`


# Interaction with the Program

## Training

Training is a loop that consists ["Model Forward Calculation"](#model-forward-calculation), ["Inference"](#inference), ["Loss Function"](#loss-function), and ["Model Backward Calculation and Update"](#model-backward-calculation-and-update).

## Testing

Testing is a loop that consists ["Model Forward Calculation"](#model-forward-calculation) and ["Inference"](#inference).
The results should be logged.

## Model Forward Calculation

## Inference

## Loss Function

## Model Backward Calculation and Update

## Save Parameters