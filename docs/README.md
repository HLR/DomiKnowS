# DomiKnows

### Table Of Content:

- [Introduction](introduction.md)
- [Install DomiKnows](InstallDomiKnows.md)
- [Getting Started](Getting Started.md)
  - [Simple Example](Getting Started.md#Simple Example)
  - [Complex Example](Getting Started.md#Complex Example)
- [Run With Jupyter](Run With Jupyter.md)
- [Datasets and Their Results](Datasets and Their Results.md)



### Main Components:

- [Data]()
- [Graph]()
- [Reader]()
- [Sensor]()
  - [preprocess]()
  - [Learner]()
- [Program]()
  - [pytorch]()
  - [ILP]()
  - [IML]()
  - [Primal-Dual]()
- [Datanode]()



### Technical API:

- Graph
  - Class Graph
    - Method Visualize
  - Class DataNode
    - Class DataNodeBuilder
  - Class LogicalConstrain
    - andL
    - orL
    - nandL
    - ifL
    - norL
    - xorL
    - epqL
    - notL
    - exactL
    - existL
    - existsL
    - atLeastL
    - atMostL
    - exactI (?)
    - existsI (?)
    - atLeastI (?)
    - atMostI (?)
  - Class Property
  - Class Transformed
  - Class Relation
    - OTORelation (?)
    - OTMRelation (?)
    - MTORelation (?)
    - MTMRelation (?)
    - IsA
    - NotA (?)
    - HasA
    - HasMany
    - Contains
    - Equal
  - Class TrialTree
  - Class Trial
- Program
- Sensor (LearnerModels remove)
  - Class Sensor
  - Class Learner
- Solver

