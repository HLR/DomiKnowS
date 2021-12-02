from functools import partial
from regr.sensor.pytorch.relation_sensors import EdgeSensor
import torch
import torch.nn as nn

from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import NBCrossEntropyIMLoss, NBCrossEntropyLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, MultiClassCMWithLogitsMetric
from regr.sensor.pytorch.sensors import ConstantSensor, FunctionalReaderSensor, FunctionalSensor, ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner

from graph import graph, T1, image, digit, addition, summation, operand1, operand2, tci

class Model(PoiModel):
    def __init__(self, graph):
        super().__init__(
            graph,
            poi=(T1, image, addition),
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric=PRF1Tracker(MultiClassCMWithLogitsMetric(19)))


class IMLModel(IMLModel):
    def __init__(self, graph):
        super().__init__(
            graph,
            poi=(T1, image, addition),
            loss=MacroAverageTracker(NBCrossEntropyIMLoss(lmbd=0.5)),
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
    addition[summation] = ReaderSensor(keyword='summation', label=True)

    T1[tci.reversed] = ConstantEdgeSensor(image['pixels'], data=[[1,1]], relation=tci.reversed)
    addition[operand1.reversed] = ConstantEdgeSensor(image['pixels'], data=[[1,0]], relation=operand1.reversed)
    addition[operand2.reversed] = ConstantEdgeSensor(image['pixels'], data=[[0,1]], relation=operand2.reversed)

    image[digit] = ModuleLearner('pixels', module=Net(config.input_size, config.hidden_sizes, len(digit.enum)))
    addition['pixels'] = FunctionalSensor(
        operand1.reversed('pixels'), operand2.reversed('pixels'),
        forward=lambda operand1, operand2: torch.cat((operand1, operand2), dim=-1))
    addition[summation] = ModuleLearner('pixels', module=Net(config.input_size*2, config.hidden_sizes, len(summation.enum)))

    program = LearningBasedProgram(graph, IMLModel)

    return program
