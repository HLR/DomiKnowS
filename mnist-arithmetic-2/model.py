import torch
from torch import nn
import torch.nn.functional as F
from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor, ConstantSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.program import POIProgram, IMLProgram, SolverPOIProgram, LearningBasedProgram
from regr.program.model.ilpu import ILPUModel
from regr.program.model.iml import IMLModel
from regr.program.metric import ValueTracker, MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, MultiClassCMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, BCEWithLogitsIMLoss

from graph import *

import time

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

        self.drop = nn.Dropout(p=0.2)

        self.norm = nn.LayerNorm(256)

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

        x = self.norm(x)

        x = self.lin1(x)
        x = self.relu(x)

        x = self.drop(x)

        y_digit = self.lin2(x)

        return y_digit


time_sum = 0.0
time_iter = 0


def get_avg_time():
    return time_sum/time_iter

class SumLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(20, 64)
        self.lin2 = nn.Linear(64, 19)

        self.relu = nn.ReLU()

    def forward(self, digits, do_time=True):
        if do_time:
            t0 = time.time()

        digit0 = torch.unsqueeze(digits[0, :], dim=0)
        digit1 = torch.unsqueeze(digits[1, :], dim=0)

        x = torch.cat((digit0, digit1), dim=1)

        x = self.lin1(x)
        x = self.relu(x)

        y_sum = self.lin2(x)

        if do_time:
            global time_sum
            global time_iter
            time_iter += 1
            time_sum += (time.time() - t0) * 1000

        #return torch.zeros((1, 19), requires_grad=True)
        return y_sum


class SumLayerExplicit(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device

    def forward(self, digits, do_time=True):
        if do_time:
            t0 = time.time()

        digit0 = torch.unsqueeze(digits[0, :], dim=0)
        digit1 = torch.unsqueeze(digits[1, :], dim=0)

        #x = F.softmax(x, dim=2)

        #digit0, digit1 = torch.squeeze(x, dim=0)

        digit0 = F.softmax(digit0, dim=1)
        digit1 = F.softmax(digit1, dim=1)

        digit0 = torch.reshape(digit0, (10, 1))
        digit1 = torch.reshape(digit1, (1, 10))
        d = torch.matmul(digit0, digit1)
        d = d.repeat(1, 1, 1, 1)
        f = torch.flip(torch.eye(10), dims=(0,)).repeat(1, 1, 1, 1)
        conv_diag_sums = F.conv2d(d, f.to(self.device), padding=(9, 0), groups=1)[..., 0]

        out = torch.squeeze(conv_diag_sums, dim=0)

        if do_time:
            global time_sum
            global time_iter
            time_iter += 1
            time_sum += (time.time() - t0) * 1000

        return out


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


def build_program(sum_setting=None, digit_labels=False, device='cpu', use_fixedL=True, test=False):
    image['pixels'] = ReaderSensor(keyword='pixels')

    def make_batch(pixel):
        return pixel.flatten().unsqueeze(0), torch.ones((1, len(pixel)))
    image_batch['pixels', image_contains.reversed] = JointSensor(image['pixels'], forward=make_batch)

    image['logits'] = ModuleLearner('pixels', module=Net())

    def make_pairs(*inputs):
        return torch.tensor([[True, False]]), torch.tensor([[False, True]])

    image_pair[pair_d0.reversed, pair_d1.reversed] = JointSensor(image['pixels'], forward=make_pairs)

    image_pair['summation_label'] = ReaderSensor(keyword='summation')

    image['digit_label'] = ReaderSensor(keyword='digit')

    image[digit] = FunctionalSensor('logits', forward=lambda x: x)

    if digit_labels:
        image[digit] = FunctionalSensor('digit_label', forward=lambda x: x, label=True)

    if use_fixedL and test:
        # during test time, set model output to be the summation label
        def manual_fixedL(s):
            res = torch.zeros((1, 19))
            res[0, s] = 1
            return res

        image_pair[s] = FunctionalSensor('summation_label', forward=manual_fixedL)
    else:
        if sum_setting == 'explicit':
            image_pair[s] = ModuleLearner(image['logits'], module=SumLayerExplicit(device=device))
        elif sum_setting == 'baseline':
            image_pair[s] = ModuleLearner(image['logits'], module=SumLayer())
        else:
             image_pair[s] = FunctionalSensor(forward=lambda: torch.ones(1, config.summationRange))  # dummy values to populate

    if use_fixedL:
        image_pair[s] = ReaderSensor(keyword='summation', label=True)
        image_pair['summationEquality'] = FunctionalSensor(forward=lambda: torch.ones(1,1))

    return graph, image, image_pair, image_batch




