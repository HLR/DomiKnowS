# `regr`

## [`regr.graph`](./graph)

The package contains elements of a conceptual graph and a data graph, including `Graph`, `Concept`, `Relation`, `Property`, `LogicalConstrain` for conceptual graph, and `DataNode` as well as `DataNodeBuilder` for data graph.

## [`regr.sensor`](./sensor)

The package contains generic `Sensor`s and `Learner`s, and ones specific to different computational framework, for example, [`TorchSensor`](./sensor/pytorch) and subclasses for PyTorch.

## [`regr.program`](./program)

The package contains base class of learning based program and excutive models that are connected to specific computational frameworks.

## `regr.solver`

The package contains interface to constraint linear integer optimization solvers and methods to translate logical constraints into linear constraints.

## `regr.data`

The package contains optional interface to data reader.
