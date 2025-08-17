import torch
import torch.nn as nn

from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, ValueTracker
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory

def prediction_softmax(pr, gt):
    return torch.softmax(pr.data, dim=-1)

# class MyIMLModel(IMLModel):
#     def __init__(self, graph):
#         super().__init__(
#             graph, 
#             loss=MacroAverageTracker(nn.CrossEntropyLoss()),
#             metric=ValueTracker(prediction_softmax))

class MyModel(PoiModel):
    def __init__(self, graph):
        super().__init__(
            graph, 
            loss=MacroAverageTracker(nn.NLLLoss()),
            metric=ValueTracker(prediction_softmax))

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
                      nn.LogSoftmax(dim=1))
    def forward(self, x):
        return self.recognition(x)
