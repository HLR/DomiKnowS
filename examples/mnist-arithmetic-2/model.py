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
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.recognition = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                    #   nn.LogSoftmax(dim=1)
                      )
    def forward(self, x):
        y = self.recognition(torch.squeeze(x, dim=0))

        #print("net out", y[0], y[1])

        '''y = torch.zeros((2, 10))

        y[0, 3] = 1.0
        y[1, 5] = 1.0'''

        return torch.unsqueeze(y, dim=0)


def sum_func(d_distr):
    # given P(d_1) and P(d_2), find P(d_1 + d_2)
    
    #print("sum in", d_1_distr, d_2_distr)
    
    #print(d_1_distr)

    Pd_1, Pd_2 = d_distr
    
    #print(Pd_1, Pd_1.shape)
    
    Pd_sum = torch.zeros((config.summationRange,))
    
    for i in range(config.digitRange):
        for j in range(config.digitRange):
            Pd_sum[i + j] += Pd_1[i] * Pd_2[j]
    
    #print(Pd_sum.shape)
    
    return Pd_sum


def I_pr(x, f=lambda x: x, do_print=True, prefix=""):
    if do_print:
        print(prefix + str(f(x)))
    return x


def probs_to_digit(probs, digit_idx, digit):
    # input: (1, 2, 10)
    # prob distribution -> prob of single digit
    # output: (2,)
    p_digit = probs[0, digit_idx, digit]
    return torch.unsqueeze(torch.tensor([1 - p_digit, p_digit], requires_grad=True), dim=0)


def probs_to_sum(sum_probs, digit_sum):
    # input: (1, 19)
    # output: (2,)
    p_sum = sum_probs[0, digit_sum]
    return torch.unsqueeze(torch.tensor([1 - p_sum, p_sum], requires_grad=True), dim=0)


def build_program():
    # (1, 2, 784)
    images['pixels'] = ReaderSensor(keyword='pixels')

    # (1, 2, 784) -> (2, 784) -> (2, 10) -> (1, 2, 10)
    images['logits'] = ModuleLearner('pixels', module=Net(config.input_size, config.hidden_sizes, config.digitRange))

    # (1, 2, 10) -> (1, 2, 10)
    images['probs'] = FunctionalSensor('logits', forward=lambda x: F.softmax(x, dim=2))

    # (1, 2, 10) -> (2, 10) -> (19,) -> (1, 19)
    images['sum_probs'] = FunctionalSensor('probs', forward=lambda x: torch.unsqueeze(sum_func(x[0]), dim=0))

    # [lbl] -> label
    images['label'] = ReaderSensor(keyword='summation')

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




