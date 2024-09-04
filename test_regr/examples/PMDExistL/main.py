import sys

sys.path.append('../../../../domiknows/')

import torch, argparse, time
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor, JointSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner, LSTMLearner
from domiknows.program import SolverPOIProgram
from tqdm import tqdm
from domiknows.program.model.base import Mode

from domiknows.sensor.pytorch.learners import TorchLearner
from graph import get_graph

from domiknows.sensor import Sensor

Sensor.clear()

parser = argparse.ArgumentParser()
parser.add_argument("--test_train", default=True, type=bool)
parser.add_argument("--atLeastL", default=False, type=bool)
parser.add_argument("--atMostL", default=False, type=bool)
parser.add_argument("--epoch", default=200, type=int)
args = parser.parse_args()


graph, a, b, a_contain_b, b_answer = get_graph(args)

class DummyLearner(TorchLearner):
    def __init__(self, *pre):
        TorchLearner.__init__(self, *pre)

    def forward(self, x):
        # Dummy result always return 1
        result = torch.stack((torch.ones(len(x)) * 4, torch.ones(len(x)) * 6), dim=-1)
        print(torch.nn.functional.softmax(result))
        return result


class TestTrainLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 2)
        with torch.no_grad():
            self.linear1.weight.copy_(torch.tensor([[8.0], [2.0]]))

    def forward(self, _, x):
        return self.linear1(x.unsqueeze(-1))


def return_contain(b, _):
    return torch.ones_like(b).unsqueeze(-1)


dataset = [{"a": [0], "b": [1.0, 2.0, 3.0], "label": [1, 1, 1]}]

a["index"] = ReaderSensor(keyword="a")
b["index"] = ReaderSensor(keyword="b")
b["temp_answer"] = ReaderSensor(keyword="label")
b[a_contain_b] = EdgeSensor(b["index"], a["index"], relation=a_contain_b, forward=return_contain)

if args.test_train:
    b[b_answer] = ModuleLearner(a_contain_b, "index", module=TestTrainLearner(), device="cpu")
    b[b_answer] = FunctionalSensor(a_contain_b, "temp_answer", forward=lambda _, label: label, label=True)
else:
    b[b_answer] = DummyLearner('index')

from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel

# program = SolverPOIProgram(graph, poi=[aa, b, b[b_answer]])r
program = PrimalDualProgram(graph, SolverModel, poi=[a, b, b_answer],
                            inferTypes=['local/argmax'],
                            loss=MacroAverageTracker(NBCrossEntropyLoss()),
                            beta=0,
                            device='cpu')

# # Checking inference
for datanode in program.populate(dataset=dataset):
    for child in datanode.getChildDataNodes():
        pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()

# Constraint is checking if there is answer => exactL(one), so if the answer is 1. This should be zero.
with torch.no_grad():
    for data in dataset:
        mloss, metric, *output = program.model(data)
        closs, *_ = program.cmodel(output[1])
        print(closs)

# Training

if args.test_train:
    print("Before Train: ")
    for datanode in program.populate(dataset=dataset):
        for child in datanode.getChildDataNodes():
            pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
            # print(child.getAttribute(b_answer, 'label'), pred, end="\n")
            print(pred, end=" ")
    print()


    program.model.train()
    program.model.reset()
    program.cmodel.train()
    program.cmodel.reset()
    copt = torch.optim.Adam(program.cmodel.parameters(), lr=1e-3)
    opt = torch.optim.Adam(program.model.parameters(), lr=1e-2)
    program.model.mode(Mode.TRAIN)
    # program.cmodel.mode(Mode.TRAIN)
    for epoch in tqdm(range(args.epoch)):
        for data in dataset:
            opt.zero_grad()
            copt.zero_grad()
            mloss, metric, *output = program.model(data)  # output = (datanode, builder)
            closs, *_ = program.cmodel(output[1])
            loss = mloss * 0 + closs
            # print(loss)
            if loss:
                loss.backward()
                opt.step()
                copt.step()
        for datanode in program.populate(dataset=dataset):
            for child in datanode.getChildDataNodes():
                pred = child.getAttribute(b_answer, 'local/softmax')[1].item()*100//1/100
                print(pred, end=" ")
        print(closs)
        print()

    print("Expected all values to be 0.")
    print("After Train: ")
    for datanode in program.populate(dataset=dataset):
        for child in datanode.getChildDataNodes():
            pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
            print(pred, end=" ")
    print()
