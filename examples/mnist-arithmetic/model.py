import torch
import torch.nn as nn

from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import NBCrossEntropyLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, MultiClassCMWithLogitsMetric
from regr.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor, ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner

from graph import graph, image, digit, addition, operand1, operand2, summation

class Model(PoiModel):
    def __init__(self, graph):
        super().__init__(
            graph,
            poi=(image, addition),
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric=PRF1Tracker(MultiClassCMWithLogitsMetric(10)))


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
    graph.detach()

    image['pixels'] = ReaderSensor(keyword='pixels')
    image[digit] = ModuleLearner('pixels', module=Net(config.input_size, config.hidden_sizes, config.output_size))

    addition[operand1.reversed, operand2.reversed] = FunctionalSensor(forward=lambda : ([1,0],[0,1]))
    addition['pixels'] = FunctionalSensor(
        operand1.reversed('pixels'), operand2.reversed('pixels'),
        forward=lambda op1, op2: torch.cat((op1, op2), dim=-1))
    addition[summation] = ModuleLearner('pixel', module=Net(config.input_size*2, config.hidden_sizes*2, 18))
    addition[summation] = ReaderSensor(keyword='summation', label=True)
    program = LearningBasedProgram(graph, Model)
    return program
