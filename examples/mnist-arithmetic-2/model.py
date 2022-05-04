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

        return torch.unsqueeze(y, dim=0)

def sum_func(d_distr, prob_func = lambda d: F.softmax(d, dim=0)):
    # given d_1 and d_2 logits, get P(d_1) and P(d_2)
    # using P(d_1) and P(d_2), find P(d_1 + d_2)
    
    #print("sum in", d_1_distr, d_2_distr)
    
    #print(d_1_distr)

    d_1_distr, d_2_distr = d_distr

    Pd_1 = prob_func(d_1_distr)
    Pd_2 = prob_func(d_2_distr)
    
    #print(Pd_1, Pd_1.shape)
    
    Pd_sum = torch.zeros((config.summationRange,))
    
    for i in range(config.digitRange):
        for j in range(config.digitRange):
            Pd_sum[i + j] += Pd_1[i] * Pd_2[j]
    
    #print(Pd_sum.shape)
    
    return Pd_sum

class SumFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_distr):
        return sum_func(d_distr)


def print_and_output(x, f=lambda x: x.shape, do_print=True):
    if do_print:
        print(f(x))
    return x


def build_program():
    # (1, 2, 784)
    images['pixels'] = ReaderSensor(keyword='pixels')

    # (1, 2, 784) -> (2, 784) -> (2, 10) -> (1, 2, 10)
    images['logits'] = ModuleLearner('pixels', module=Net(config.input_size, config.hidden_sizes, config.digitRange))

    # (1, 2, 10) -> (1, 10) to digit enums
    images[d0] = FunctionalSensor('logits', forward=lambda x: print_and_output(x[:, 0]))

    # (1, 2, 10) -> (1, 10) to digit enums
    images[d1] = FunctionalSensor('logits', forward=lambda x: print_and_output(x[:, 1]))

    # (1, 2, 10) -> (2, 10) -> (19,) -> (1, 19) to summation enums
    images[s] = FunctionalSensor('logits', forward=lambda x: print_and_output(torch.unsqueeze(sum_func(torch.squeeze(x, dim=0)), dim=0)))

    # [lbl] -> summation enums
    images[s] = ReaderSensor(keyword='summation', label=True)

    program = SolverPOIProgram(graph,
                               poi=(images,),
                               inferTypes=['ILP', 'local/argmax'],
                               loss=MacroAverageTracker(NBCrossEntropyLoss()))

    return program




