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
        y = self.recognition(x)
        return y

def sum_func(d_1_distr, d_2_distr, prob_func = lambda d: F.softmax(d, dim=1)):
    # given d_1 and d_2 logits, get P(d_1) and P(d_2)
    # using P(d_1) and P(d_2), find P(d_1 + d_2)
    
    #print(d_1_distr.shape)
    
    #print(d_1_distr)
    
    Pd_1 = prob_func(d_1_distr)[0]
    Pd_2 = prob_func(d_2_distr)[0]
    
    #print(Pd_1, Pd_1.shape)
    
    Pd_sum = torch.zeros((config.summationRange,))
    
    for i in range(config.digitRange):
        for j in range(config.digitRange):
            Pd_sum[i + j] += Pd_1[i] * Pd_2[j]
    
    #print(Pd_sum.shape)
    
    return Pd_sum

def build_program():
    class ConstantEdgeSensor(ConstantSensor, EdgeSensor): pass

    image['pixels'] = ReaderSensor(keyword='pixels')

    addition[operand1.reversed] = ConstantEdgeSensor(image['pixels'], data=[[1,0]], relation=operand1.reversed)
    addition[operand2.reversed] = ConstantEdgeSensor(image['pixels'], data=[[0,1]], relation=operand1.reversed)

    image['logits'] = ModuleLearner('pixels', module=Net(config.input_size, config.hidden_sizes, config.digitRange))

    image[digit] = FunctionalSensor('logits', forward=lambda x: x)

    def test(x1, x2):
        print(x1.shape, x2.shape)
        
        return torch.zeros(summationRange)

    addition[summation] = ReaderSensor(keyword='summation', label=True)
    addition[summation] = FunctionalSensor(operand2.reversed('logits'), operand2.reversed('logits'), forward=sum_func)

    program = SolverPOIProgram(graph,
                               poi=(image, addition, summation),
                               inferTypes=['ILP', 'local/argmax'],
                               loss=MacroAverageTracker(NBCrossEntropyLoss()))

    return program




