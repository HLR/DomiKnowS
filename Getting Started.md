# DomiKnowS: Getting Started

Welcome to DomiKnowS, a library for integrating symbolic domain knowledge in deep learning! This readme provides a quick overview of DomiKnowS and step-by-step instructions on how to get started with the library.

## Installation

To install DomiKnowS, you can use `pip`, the Python package manager. Open a terminal and run the following command:

```bash
pip install DomiKnowS
```

## Requirements

The DomiKnowS library is integrated with the ILP solver [Gurobi](https://www.gurobi.com/). The library provides the necessary capability to find solutions for logical constraints encoded as an ILP model.

The [requirements](https://github.com/HLR/DomiKnowS/blob/main/requirements.txt) file for the library includes the Gurobi Python package [gurobipy](https://pypi.org/project/gurobipy/).
This package comes with a trial license that allows solving problems of limited size.
As a student or staff member of an academic institution, you qualify for a free, full product license. For more information, see:

- [https://www.gurobi.com/academia/academic-program-and-licenses/](https://www.gurobi.com/academia/academic-program-and-licenses/)

For a commercial evaluation, you can request a license.

## Verification

To ensure that Domiknows works correctly, we can verify the installation by running a sample Domiknows example code.

Following is an example of a simple classifier:

```python
import torch, logging
from transformers import AdamW

from domiknows.graph import Graph, Concept, Relation
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    xcon = Concept(name='xcon')
    ycon = xcon(name='ycon')

reader = [{"value": [[10.0, 10.0]], "y": [[1]]}, {"value": [[-1.0, -1.0]], "y": [[0]]}, {"value": [[-20.0, -20.0]], "y": [[0]]}]
xcon["value"] = ReaderSensor(keyword="value")
xcon[ycon] = ModuleLearner("value", module=torch.nn.Linear(2, 2),device=torch.device("cpu"))
xcon[ycon] = ReaderSensor(keyword="y", label=True)
program = SolverPOIProgram(graph,poi=[xcon,ycon],inferTypes=['local/softmax'], 
                           loss=MacroAverageTracker(NBCrossEntropyLoss()),
                           Optim=lambda param: AdamW(param, lr = 1e-2 ))
program.train(reader, train_epoch_num=10)
```

## Examples

To help you get started quickly, we provide a [walkthrough example](Walkthrough%20Examples). This example demonstrates how to use DomiKnowS step by step.

Refer to documentation for detailed information!
