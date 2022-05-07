import torch
from torch import nn
import torch.nn.functional as F
from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor, ConstantSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program import LearningBasedProgram
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.program import POIProgram, IMLProgram, SolverPOIProgram
from regr.program.model.ilpu import ILPUModel
from regr.program.metric import ValueTracker, MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, MultiClassCMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss

from graph import *

import config


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)

        x = x.reshape(2, 1, 28, 28)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(2, -1)

        x = self.lin1(x)
        x = self.relu(x)

        x = self.drop(x)

        y_digit = self.lin2(x)

        return torch.unsqueeze(y_digit, dim=0)


class SumLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(20, 64)
        self.lin2 = nn.Linear(64, 19)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(torch.flatten(x), dim=0)

        x = self.lin1(x)
        x = self.relu(x)

        y_sum = self.lin2(x)

        #return torch.zeros((1, 19), requires_grad=True)
        return y_sum

def print_and_output(x, f=lambda x: x.shape, do_print=False):
    if do_print:
        print(prefix + str(f(x)))
    return x

def build_program():
    # (1, 2, 784)
    images['pixels'] = ReaderSensor(keyword='pixels')

    # (1, 2, 784) -> (2, 784) -> (2, 10) -> (1, 2, 10)
    images['logits'] = ModuleLearner('pixels', module=Net())

    # (1, 2, 10) -> (1, 10) to digit enums
    images[d0] = FunctionalSensor('logits', forward=lambda x: print_and_output(x[:, 0]))
    images[d0] = ReaderSensor(keyword='digit0', label=True)

    # (1, 2, 10) -> (1, 10) to digit enums
    images[d1] = FunctionalSensor('logits', forward=lambda x: print_and_output(x[:, 1]))
    images[d1] = ReaderSensor(keyword='digit1', label=True)

    # (1, 2, 10) -> (2, 10) -> (19,) -> (1, 19) to summation enums
    images[s] = ModuleLearner('logits', module=SumLayer())

    for d_nm, d_c in zip(digits_0, digits_0_c):
        d_number = name_to_number(d_nm)

        images[d_c] = FunctionalSensor('probs', d_number,
                                       forward=lambda x, n: I_pr(probs_to_digit(x, 0, n),
                                                              prefix=f"d0_{n} ", do_print=False))

    for d_nm, d_c in zip(digits_1, digits_1_c):
        d_number = name_to_number(d_nm)

        images[d_c] = FunctionalSensor('probs', d_number,
                                       forward=lambda x, n: I_pr(probs_to_digit(x, 1, n),
                                                                 prefix=f"d1_{n} ", do_print=False))

    for s_nm, s_c in zip(summations, summations_c):
        s_number = name_to_number(s_nm)

        images[s_c] = FunctionalSensor('sum_probs', s_number,
                                       forward=lambda x, n: I_pr(probs_to_sum(x, n),
                                                              prefix=f"s_{n} ", do_print=False))

        def label_to_binary(lbl, n):
            if lbl[0] == n:
                return torch.tensor([1])
            return torch.tensor([0])

        images[s_c] = FunctionalSensor('label', s_number,
                                       forward=lambda x, n: I_pr(label_to_binary(x, n),
                                                                prefix=f"s'_{n} ", do_print=False), label=True)

    program = SolverPOIProgram(graph,
                               poi=(images,),
                               inferTypes=['ILP', 'local/argmax'],
                               loss=MacroAverageTracker(NBCrossEntropyLoss()),
                               metric={
                                   'ILP': PRF1Tracker(DatanodeCMMetric()),
                                   'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}
                               )

    return program




