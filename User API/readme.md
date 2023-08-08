# DomiKnows

### Table Of Content:
- [Introduction](Introduction.md)
- [Install DomiKnows](Install%20DomiKnows.md)
- [Getting Started](GettingStarted.md)
- [Example Task](ExampleTask.md)
- [Run With Jupyter](Run%20With%20Jupyter/)
- [Datasets and Their Results](Datasets%20and%20Their%20Results.md)



### Main Components:

- [Knowledge Declaration (Graph)](Main%20Components/Graph.md)
- [Model Declaration (Sensor)](Main%20Components/Sensor.md)
- [DataNode](Main%20Components/Datanode.md)
- [Training](Main%20Components/Training.md)
- [Inference](Main%20Components/Inference.md)
- [Solver](Main%20Components/Solver.md)



### Technical API:

- [Graph](Technical%20API/Graph/)
  - [Class Graph](Technical%20API/Graph/Class%20Graph.md)
    - [Method Visualize](Technical%20API/Graph/Class%20Graph.md#Method-Visualize)
  - [Class DataNode](Technical%20API/Graph/Class%20DataNode.md)
    - [Class DataNodeBuilder](Technical%20API/Graph/Class%20DataNode.md#Class-DataNodeBuilder)
  - [Class LogicalConstrain](Technical%20API/Graph/Class%20LogicalConstrain.md)
    - [andL](Technical%20API/Graph/Class%20LogicalConstrain.md#andL)
    - [orL](Technical%20API/Graph/Class%20LogicalConstrain.md#orL)
    - [nandL](Technical%20API/Graph/Class%20LogicalConstrain.md#nandL)
    - [ifL](Technical%20API/Graph/Class%20LogicalConstrain.md#ifL)
    - [norL](Technical%20API/Graph/Class%20LogicalConstrain.md#norL)
    - [xorL](Technical%20API/Graph/Class%20LogicalConstrain.md#xorL)
    - [epqL](Technical%20API/Graph/Class%20LogicalConstrain.md#epqL)
    - [notL](Technical%20API/Graph/Class%20LogicalConstrain.md#notL)
    - [exactL](Technical%20API/Graph/Class%20LogicalConstrain.md#exactL)
    - [existL](Technical%20API/Graph/Class%20LogicalConstrain.md#existL)
    - [existsL](Technical%20API/Graph/Class%20LogicalConstrain.md#existsL)
    - [atLeastL](Technical%20API/Graph/Class%20LogicalConstrain.md#atLeastL)
    - [atMostL](Technical%20API/Graph/Class%20LogicalConstrain.md#atMostL)
    - [exactI](Technical%20API/Graph/Class%20LogicalConstrain.md#exactI)
    - [existsI](Technical%20API/Graph/Class%20LogicalConstrain.md#atLeastI)
    - [atLeastI](Technical%20API/Graph/Class%20LogicalConstrain.md#atLeastI)
    - [atMostI](Technical%20API/Graph/Class%20LogicalConstrain.md#atMostI)
  - [Class Property](Technical%20API/Graph/Class%20Property.md)
  - [Class Transformed](Technical%20API/Graph/Class%20Transformed.md)
  - [Class Relation](Technical%20API/Graph/Class%20Relation.md)
    - [OTORelation](Technical%20API/Graph/Class%20Relation.md#OTORelation)
    - [OTMRelation](Technical%20API/Graph/Class%20Relation.md#OTMRelation)
    - [MTORelation](Technical%20API/Graph/Class%20Relation.md#MTORelation)
    - [MTMRelation](Technical%20API/Graph/Class%20Relation.md#MTMRelation)
    - [IsA](Technical%20API/Graph/Class%20Relation.md#IsA)
    - [NotA](Technical%20API/Graph/Class%20Relation.md#NotA)
    - [HasA](Technical%20API/Graph/Class%20Relation.md#HasA)
    - [HasMany](Technical%20API/Graph/Class%20Relation.md#HasMany)
    - [Contains](Technical%20API/Graph/Class%20Relation.md#Contains)
    - [Equal](Technical%20API/Graph/Class%20Relation.md#Equal)
  - [Class TrialTree](Technical%20API/Graph/Class%20TrialTree.md)
  - [Class Trial](Technical%20API/Graph/Class%20Trial.md)
- [Program](Technical%20API/Program/)
- [Sensor](Technical%20API/Sensor/)
  - [Class Sensor](Technical%20API/Sensor/Class%20Sensor.md)
  - [Class Learner](Technical%20API/Sensor/Class%20Learner.md)
- [Solver](Technical%20API/Solver/)
