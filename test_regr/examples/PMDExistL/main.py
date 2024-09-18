import sys

sys.path.append('../../../../domiknows/')

import torch, argparse, time
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor, JointSensor, \
    FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner, LSTMLearner
from domiknows.program import SolverPOIProgram
from tqdm import tqdm
from domiknows.program.model.base import Mode

from domiknows.sensor.pytorch.learners import TorchLearner
from graph import get_graph

from domiknows.sensor import Sensor
import numpy as np

Sensor.clear()

parser = argparse.ArgumentParser()
parser.add_argument("--test_train", default=True, type=bool)
parser.add_argument("--atLeastL", default=False, type=bool)
parser.add_argument("--atMostL", default=False, type=bool)
parser.add_argument("--epoch", default=500, type=int)
parser.add_argument("--expected_atLeastL", default=3, type=int)
parser.add_argument("--expected_atMostL", default=3, type=int)
parser.add_argument("--expected_value", default=0, type=int)
parser.add_argument("--N", default=10, type=int)
parser.add_argument("--M", default=8, type=int)
args = parser.parse_args()

N = args.N  # Setting the b parameters
M = args.M
graph, a, b, a_contain_b, b_answer = get_graph(args)


class DummyLearner(TorchLearner):
    def __init__(self, *pre):
        TorchLearner.__init__(self, *pre)

    def forward(self, x):
        # Dummy result always return 1
        result = torch.stack((torch.ones(len(x)) * 4, torch.ones(len(x)) * 6), dim=-1)
        # print(torch.nn.functional.softmax(result))
        return result


class TestTrainLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N, N),
            # torch.nn.ReLU(),
            torch.nn.Linear(N, 2)
        )
        # self.layers = torch.nn.Linear(N, 2)
        # with torch.no_grad():
        #     self.linear1.weight.copy_(torch.tensor([[8.0], [2.0]]))

    def forward(self, _, x):
        # print("input:", x)
        output = self.layers(x)
        # print(output.softmax(dim=-1))
        return output


def return_contain(b, _):
    return torch.ones(len(b)).unsqueeze(-1)


np.random.seed(0)
torch.manual_seed(0)

dataset = [{"a": [0],
            "b": [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
            "label": [0] * M}]

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
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
from domiknows.program.model.pytorch import SolverModel

# program = SolverPOIProgram(graph, poi=[aa, b, b[b_answer]])

program = PrimalDualProgram(graph, SolverModel, poi=[a, b, b_answer],
                            inferTypes=['local/argmax'],
                            loss=MacroAverageTracker(NBCrossEntropyLoss()),
                            beta=10,
                            device='cpu')

# program = SampleLossProgram(graph, SolverModel, poi=[a, b, b_answer],
#                                     inferTypes=['local/argmax'],
#                                     loss=MacroAverageTracker(NBCrossEntropyLoss()),
#                                     sample=True,
#                                     sampleSize=100,
#                                     sampleGlobalLoss=False,
#                                     beta=0,
#                                     device='cpu')

# # Checking inference
for datanode in program.populate(dataset=dataset):
    for child in datanode.getChildDataNodes():
        pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()

# Constraint is checking if there is answer => exactL(one), so if the answer is 1. This should be zero.
# with torch.no_grad():
#     for data in dataset:
#         mloss, metric, *output = program.model(data)
#         closs, *_ = program.cmodel(output[1])
#         print(closs)

# Training

if args.test_train:
    program.model.train()
    program.model.reset()
    program.cmodel.train()
    program.cmodel.reset()
    program.model.mode(Mode.TRAIN)
    print("Starting the test case experiment")
    print("Before Train: ")
    # for datanode in program.populate(dataset=dataset):
    #     for child in datanode.getChildDataNodes():
    #         pred = child.getAttribute(b_answer, 'local/softmax')
    #         # print(child.getAttribute(b_answer, 'label'), pred, end="\n")
    #         print(pred, end=" ")
    # print()
    for datanode in program.populate(dataset=dataset):
        for child in datanode.getChildDataNodes():
            pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
            # print(child.getAttribute(b_answer, 'label'), pred, end="\n")
            print(pred, end=" ")

    # Initial train the model to have the predicted label
    program.model.train()
    program.model.reset()
    program.cmodel.train()
    program.cmodel.reset()
    program.model.mode(Mode.TRAIN)

    opt = torch.optim.Adam(program.model.parameters(), lr=1e-2)
    copt = torch.optim.Adam(program.cmodel.parameters(), lr=1e-3)

    for each in tqdm(range(20), desc="Training without PMD"):
        for data in dataset:
            loss, metric, *output = program.model(data)
            loss.backward()
            opt.step()
    print()
    program.model.eval()
    program.model.reset()
    program.cmodel.eval()
    program.cmodel.reset()
    program.model.mode(Mode.TEST)

    print("After Train without PMD: ")
    for datanode in program.populate(dataset=dataset):
        for child in datanode.getChildDataNodes():
            pred = child.getAttribute(b_answer, 'local/softmax')
            print(pred, end=" ")

    print()
    print("After Train without PMD: ")
    for datanode in program.populate(dataset=dataset):
        for child in datanode.getChildDataNodes():
            pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
            print(pred, end=" ")

    print()

    program.model.train()
    program.model.reset()
    program.cmodel.train()
    program.cmodel.reset()
    program.model.mode(Mode.TRAIN)
    # program.cmodel.mode(Mode.TRAIN)
    for epoch in tqdm(range(args.epoch), desc="Training with PMD"):
        for data in dataset:
            opt.zero_grad()
            copt.zero_grad()
            mloss, metric, *output = program.model(data)  # output = (datanode, builder)
            closs, *_ = program.cmodel(output[1])

            if torch.is_tensor(closs):
                loss = mloss * 0 + closs
            else:
                loss = mloss * 0

            # print(loss)
            if loss.item() < 0:
                break
            if loss:
                loss.backward()
                opt.step()
                copt.step()

    program.model.eval()
    program.model.reset()
    program.cmodel.eval()
    program.cmodel.reset()
    program.model.mode(Mode.TEST)
    # for datanode in program.populate(dataset=dataset):
    #     for child in datanode.getChildDataNodes():
    #         pred = child.getAttribute(b_answer, 'local/softmax')
    #         print(pred, end=" ")
    #
    # print()

    final_result_count = {}
    print("After Train with PMD: ")
    for datanode in program.populate(dataset=dataset):
        for child in datanode.getChildDataNodes():
            pred = child.getAttribute(b_answer, 'local/argmax').argmax().item()
            print(pred, end=" ")
            final_result_count[pred] = final_result_count.get(pred, 0) + 1
    print()

    pass_test_case = True

    expected_value = args.expected_value
    if args.atLeastL:
        pass_test_case = pass_test_case and (final_result_count.get(expected_value, 0) >= args.expected_atLeastL)

    if args.atMostL:
        pass_test_case = pass_test_case and (final_result_count.get(expected_value, 0) <= args.expected_atMostL)

    if not (args.atLeastL or args.atMostL):
        # print(f"Testing exactL of {expected_value}: {final_result_count.get(expected_value, 0)}")
        pass_test_case = pass_test_case and (final_result_count.get(expected_value, 0) == args.expected_atLeastL)

    print("It PASSES this case" if pass_test_case else "It FAILS this case")