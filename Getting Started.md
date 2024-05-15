# DomiKnowS: Getting Started

Welcome to DomiKnowS, a library for integrating symbolic domain knowledge in deep learning! This readme provides a quick overview of DomiKnowS and step-by-step instructions on how to get started with the library.

## Installation

To install DomiKnowS, you can use `pip`, the Python package manager. Open a terminal and run the following command:

```bash
pip install DomiKnowS
```

## Requirements

The DomiKnowS library is integrated with the ILP solver [Gurobi](https://www.gurobi.com/). The library provides necessary capability to find solutions for logical constraints encoded as an ILP model.

The [requirements](https://github.com/HLR/DomiKnowS/blob/main/requirements.txt) file for the library includes the Gurobi Python package [gurobipy](https://pypi.org/project/gurobipy/).
This package comes with a trial license that allows solving problems of limited size. 
As a student or staff member of an academic institution, you qualify for a free, full product license. For more information, see:

- [https://www.gurobi.com/academia/academic-program-and-licenses/](https://www.gurobi.com/academia/academic-program-and-licenses/)

For a commercial evaluation, you can request an evaluation license.

## Verification

To ensure that Domiknows works correctly,we can verify the installation by running sample Domiknows example code.

Following is an example of a simple classifier:

```python
import sys, torch
from transformers import AdamW
from domiknows.graph import Graph, Concept, Relation
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    x = Concept(name='x')
    y = x(name='y')

reader = [{"value": [[10, 10]], "y": [0, 1]}, {"value": [[-1, -1]], "y": [0, 1]}, {"value": [[-20, -20]], "y": [1, 0]}]
x["value"] = ReaderSensor(keyword="value")
x[y] = ReaderSensor(keyword="y", label=True)
x[y] = ModuleLearner("value", module=torch.nn.Linear(2, 2))
program = SolverPOIProgram(graph, poi=[y, ], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
program.train(reader, train_epoch_num=1, Optim=lambda param: AdamW(param, lr=1e-4, eps=1e-8))
```

## Examples

To help you get started quickly, we provide a [walkthrough example](ExampleTask.md). This example demonstrates how to use DomiKnowS step by step.

Refer to documentation for detailed information!
