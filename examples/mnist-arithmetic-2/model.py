import torch
from torch import nn
import torch.nn.functional as F
from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor, ConstantSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.program import POIProgram, IMLProgram, SolverPOIProgram, LearningBasedProgram
from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.model.ilpu import ILPUModel
from regr.program.model.iml import IMLModel
from regr.program.metric import ValueTracker, MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, MultiClassCMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, BCEWithLogitsIMLoss

from graph import *

from digit_label import digit_labels
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

    def forward(self, digit0, digit1):
        x = torch.cat((digit0, digit1), dim=1)

        x = self.lin1(x)
        x = self.relu(x)

        y_sum = self.lin2(x)

        #return torch.zeros((1, 19), requires_grad=True)
        return y_sum


class SumLayerExplicit(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, digit0, digit1):
        #x = F.softmax(x, dim=2)

        #digit0, digit1 = torch.squeeze(x, dim=0)

        digit0 = F.softmax(digit0, dim=1)
        digit1 = F.softmax(digit1, dim=1)

        digit0 = torch.reshape(digit0, (10, 1))
        digit1 = torch.reshape(digit1, (1, 10))
        d = torch.matmul(digit0, digit1)
        d = d.repeat(1, 1, 1, 1)
        f = torch.flip(torch.eye(10), dims=(0,)).repeat(1, 1, 1, 1)
        conv_diag_sums = F.conv2d(d, f, padding=(9, 0), groups=1)[..., 0]

        return torch.squeeze(conv_diag_sums, dim=0)


class NBSoftCrossEntropyLoss(NBCrossEntropyLoss):
    def __init__(self, prior_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prior_weight = prior_weight

    def forward(self, input, target, *args, **kwargs):
        if target.dim() == 1:
            return super().forward(input, target, *args, **kwargs)

        epsilon = 1e-5
        input = input.view(-1, input.shape[-1])
        input = input.clamp(min=epsilon, max=1-epsilon)

        logprobs = F.log_softmax(input, dim=1)
        return self.prior_weight * -(target * logprobs).sum() / input.shape[0]


class NBSoftCrossEntropyIMLoss(BCEWithLogitsIMLoss):
    def __init__(self, prior_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prior_weight = prior_weight

    def forward(self, input, inference, target, weight=None):
        if target.dim() == 1:
            num_classes = input.shape[-1]
            target = target.to(dtype=torch.long)
            target = F.one_hot(target, num_classes=num_classes)

            return super().forward(input, inference, target, weight=weight)

        return super().forward(input, inference, target, weight=weight) * self.prior_weight


def print_and_output(x, f=lambda x: x.shape, do_print=False):
    if do_print:
        print(prefix + str(f(x)))
    return x


def build_program():
    # (1, 2, 784)
    images['pixels'] = ReaderSensor(keyword='pixels')

    # (1, 2, 784) -> (2, 784) -> (2, 10) -> (1, 2, 10)
    images['logits'] = ModuleLearner('pixels', module=Net())

    images['digit0_label'] = ReaderSensor(keyword='digit0')
    images['digit1_label'] = ReaderSensor(keyword='digit1')
    images['summation_label'] = ReaderSensor(keyword='summation')

    # (1, 2, 10) -> (1, 10) to digit enums
    #images[d0] = ModuleLearner('logits', module=Net())
    images[d0] = FunctionalSensor('logits', forward=lambda x: x[:, 0])
    #images[d0] = ReaderSensor(keyword='digit0', label=True)
    #images[d0] = FunctionalSensor('summation_label',
    #                              forward=lambda x: torch.unsqueeze(digit_labels[x[0]], dim=0), label=True)

    # (1, 2, 10) -> (1, 10) to digit enums
    #images[d1] = ModuleLearner('logits', 1, module=Net())
    images[d1] = FunctionalSensor('logits', forward=lambda x: x[:, 1])
    #images[d1] = ReaderSensor(keyword='digit1', label=True)
    #images[d1] = FunctionalSensor('summation_label',
    #                              forward=lambda x: torch.unsqueeze(digit_labels[x[0]], dim=0), label=True)

    # (1, 2, 10) -> (2, 10) -> (19,) -> (1, 19) to summation enums
    images[s] = ModuleLearner(images[d0], images[d1], module=SumLayer())
    images[s] = ReaderSensor(keyword='summation', label=True)

    '''program = IMLProgram(graph,
                       poi=(images,),
                       inferTypes=['local/argmax'],
                       loss=MacroAverageTracker(NBSoftCrossEntropyIMLoss(prior_weight=0.1, lmbd=0.5)))'''

    '''program = SolverPOIProgram(graph,
                         poi=(images,),
                         inferTypes=['local/argmax'],
                         loss=MacroAverageTracker(NBSoftCrossEntropyLoss(prior_weight=1.0)))'''

    program = PrimalDualProgram(graph,
                                IMLModel,
                                poi=(images,),
                                inferTypes=['local/argmax'],
                                loss=MacroAverageTracker(NBSoftCrossEntropyIMLoss(prior_weight=0.1, lmbd=0.5)))

    return program




