from functools import partial

from owlready2.annotation import label
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
import torch
import torch.nn as nn

from domiknows.program import LearningBasedProgram
from domiknows.program.model.ilpu import ILPUModel
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, MultiClassCMWithLogitsMetric
from domiknows.sensor.pytorch.sensors import ConstantSensor, FunctionalReaderSensor, FunctionalSensor, ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner

from graph_u import graph, T1, image, digit, addition, summation, operand1, operand2, tci
from graph_u import digitRange, summationVal


class Model(ILPUModel):
    def __init__(self, graph):
        super().__init__(
            graph,
            poi=(T1, image, addition),
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={
                'ILP': PRF1Tracker(DatanodeCMMetric()),
                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
            inferTypes=['ILP', 'local/argmax'])


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
        return self.recognition(x)


def model_declaration(config):
    class ConstantEdgeSensor(ConstantSensor, EdgeSensor): pass
    graph.detach()

    image['pixels'] = ReaderSensor(keyword='pixels')
    addition[summation] = ConstantSensor(data=[summationVal])

    T1[tci.reversed] = ConstantEdgeSensor(image['pixels'], data=[[1,1]], relation=tci.reversed)
    addition[operand1.reversed] = ConstantEdgeSensor(image['pixels'], data=[[1,0]], relation=operand1.reversed)
    addition[operand2.reversed] = ConstantEdgeSensor(image['pixels'], data=[[0,1]], relation=operand2.reversed)

    image[digit] = ModuleLearner('pixels', module=Net(config.input_size, config.hidden_sizes, digitRange))
    image[digit] = ReaderSensor(keyword='vals', label=True)
    program = LearningBasedProgram(graph, Model)

    return program
