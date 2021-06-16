import torch
import torch.nn as nn

from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import NBCrossEntropyLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric, MultiClassCMWithLogitsMetric


class MyModel(PoiModel):
    def __init__(self, graph):
        super().__init__(
            graph, 
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
