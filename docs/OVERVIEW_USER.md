# Overview (User perspective)

There are several components in the framework representing different design steps and programming workflows.
Please refer to each individual page for details and usages of the components.

- [Overview (User perspective)](#overview-user-perspective)
  - [How it works](#how-it-works)
  - [How to program](#how-to-program)
    - [Knowledge Declaration](#knowledge-declaration)
    - [Model Declaration](#model-declaration)
    - [Program](#program)
    - [Query and Access](#query-and-access)
    - [Inference](#inference)

## How it works

In DomiKnowS, the program connects concepts' properties in the knowledge with sensors (data accessing procedure) and learners (statistical models), handles the process of training and testing, and generate datanodes that one can query and inference with.

The basic routine in the program is reading a data item form the data reader, looking for a property of interesting and trigger the calculation, which recurcively trigger the calculation and retrieval of data value from data item based on the dependency of sensors and learners. The caluclated values can be used for loss in training, metric in testing, or datanodes for users to query and inference with.

To simplify the design of the above routine and allow users to focus on knowledge design, DomiKnowS encapsulates the above routine and regularize the users to provive domain knowledge by Knowledge Declaration and knowledge centered model by Model Declaration.

The user will need to declare the concepts and their relationship based on domain knowledge ([Knowledge Declaration](#knowledge-declaration)), specify how the properties of the concepts are aquired or calculated ([Model Declaration](#model-declaration)). Then, the [program](#program) can be train and test. The results (datanodes) can be [queried and inference](#query-and-access).

Programming with DomiKnowS for a simple classification output problem may addes a lot of overhead. However, when the output space is highly structured and there are domain konwledge one want to incorporate into the program, DomiKnowS becomes helpful.

## How to program

The aim of DomiKnowS is to regularize the design to be focused on knowledge.
The following sections introduce the major concepts.
For a complete pipeline of programming with DomiKnowS, please referer to the [pipeline](PIPELINE.md).

### Knowledge Declaration

Class reference:

- `regr.graph.Graph`
- `regr.graph.Concept`
- `regr.graph.Property`
- `regr.graph.Relation`
- `regr.graph.LogicalConstrain`

In knowledge declaration, the user defines a collection of `Concept`s and the way they are related to each other (`Relation`), representing the domain knowledge a the task.
We provide a graph language based on python for knowledge declaration with notation of `Graph`, `Concept`, `Property`, `Relation`, `LogicalConstrain`.

### Model Declaration

Class reference:

- `regr.sensor.Sensor`
- `regr.sensor.Learner`

In model declaration, the user defines how external resources (raw data), external procedures (preprocessing), and trainable deep learning modules are associated with the concepts and properties in the graph.
"Sensors" are procedures to access external resources and procedures. For example, reading from raw data, feature engineering staffs, and preprocessing procedures.
"Sensors" are looked at as blackboxes in a program.
"Learners" are computational modules that predict the unknown representations or probabilities of properties based on learning. "Learners", unlike blackbox sensors, are differentiable components and usually comes with parameters that can be tune by gradient based methods.

### Program

Class reference:

- `regr.program.Program`

The graph, attached with sensors and learners, can be turned into a learning based program. The program can learn parameters from data, doing tests, or simply makeing predictions.

### Query and Access

Class reference:

- `regr.graph.DataNode`

The program uses and returns a data structure know as the datanode. They are instanitiate of the concept based graph based on the sensor inputs and learner predictions.

### Inference

Class reference:

- `regr.graph.DataNode`

One of the advantage of the framework is to do global inference over local predictions (made by learners or read by sensors).
The user can invoke inference from `DataNode` by `DataNode.inferILPConstrains()` and augument the attributes with the best combination of local predictions (configuration) that satisfies all the explicit constraints, as well as those implied by the relations.
