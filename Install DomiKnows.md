# Install DomiKnows

Clone the repository and cd into Domiknows folder. Install the requirements by running:

```
python -m pip install -r requirements.txt
```

### Verification

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

sys.path.append("Domiknows/domiknows")
sys.path.append("../..")

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

### Dependencies

- [Ubuntu](https://ubuntu.com/) 18.04
- [Python](https://www.python.org/) 3.7
- [PyTorch](https://pytorch.org/) 1.4.0
- [Gurobi](https://gurobi.com/) 8.0
- [Graphviz](https://graphviz.org/)
- Other dependencies specified in [`requirements.txt`](https://github.com/HLR/DomiKnowS/blob/develop_newLC/requirements.txt), that are installed by `pip`.
