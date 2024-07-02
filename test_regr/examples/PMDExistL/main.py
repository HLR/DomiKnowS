import sys

sys.path.append('../../../../domiknows/')

import torch, argparse, time
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor, JointSensor
from domiknows.program import SolverPOIProgram

from domiknows.sensor.pytorch.learners import TorchLearner
from graph import get_graph

from domiknows.sensor import Sensor

Sensor.clear()

graph, a, b, a_contain_b, b_answer = get_graph()


class DummyLearner(TorchLearner):
    def __init__(self, *pre):
        TorchLearner.__init__(self, *pre)

    def forward(self, x):
        # Dummy result always return 1
        result = torch.stack((torch.zeros(len(x)), torch.ones(len(x)) * 10000), dim=-1)
        return result


def return_contain(b, _):
    return torch.ones_like(b).unsqueeze(-1)


dataset = [{"a": [0], "b": [1, 2, 3]}]

a["index"] = ReaderSensor(keyword="a")
b["index"] = ReaderSensor(keyword="b")
b[a_contain_b] = EdgeSensor(b["index"], a["index"], relation=a_contain_b, forward=return_contain)

b[b_answer] = DummyLearner('index')

from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel

# program = SolverPOIProgram(graph, poi=[aa, b, b[b_answer]])
program = PrimalDualProgram(graph, SolverModel, poi=[a, b, b_answer],
                            inferTypes=['local/argmax'],
                            loss=MacroAverageTracker(NBCrossEntropyLoss()),
                            beta=0.1,
                            device='cpu')

# Checking inference
for datanode in program.populate(dataset=dataset):
    for child in datanode.getChildDataNodes():
        pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
        assert (pred == 1)

for data in dataset:
    mloss, metric, *output = program.model(data)  # output = (datanode, builder)
    closs, *_ = program.cmodel(output[1])

    # Learner always return [0, 1] and constraint is ifL(a, returnOne). Then, closs should be zero
    assert closs.item() == 0
