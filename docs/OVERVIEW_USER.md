# Overview (User perspective)

There are several components in the framework representing different design steps and programming workflows.
Please refer to each individual page for details and usages of the components.

- [Overview (User perspective)](#overview-user-perspective)
  - [Knowledge Declaration](#knowledge-declaration)
  - [Model Declaration](#model-declaration)
  - [Program](#program)
  - [Query and Access](#query-and-access)
  - [Inference](#inference)

## Knowledge Declaration

Class reference: `Graph`, `Concept`, `Property`, `Relation`, `Constraint`.

In knowledge declaration, the user defines a collection of concepts and the way they are related to each other, representing the domain knowledge a the task.
We provide a graph language based on python for knowledge declaration with notation of "graph", "concept", "property", "relation", and "constraints".

## Model Declaration

Class reference: `Sensor` and `Learner`.

In model declaration, the user defines how external resources (raw data), external procedures (preprocessing), and trainable deep learning modules are associated with the concepts and properties in the graph.
"Sensors" are procedures to access external resources and procedures. For example, reading from raw data, feature engineering staffs, and preprocessing procedures.
"Sensors" are looked at as blackboxes in a program.
"Learners" are computational modules that predict the unknown representations or probabilities of properties based on learning. "Learners", unlike blackbox sensors, are differentiable components and usually comes with parameters that can be tune by gradient based methods.

## Program

Class reference: `Program`.

The graph, attached with sensors and learners, can be turned into a learning based program. The program can learn parameters from data, doing tests, or simply makeing predictions.

## Query and Access

Class reference: `DataNode`, `DataNodeBuilder`.

The program uses and returns a data structure know as the datanode. They are instanitiate of the concept based graph based on the sensor inputs and learner predictions.

## Inference

Class reference: `ilpOntSolver`.

One of the advantage of the framework is to do global inference over local predictions (made by learners or read by sensors). Solvers are the core component that looks for the best combination of local predictions that satisfies all the explicit constraints, as well as those implied by the relations.
