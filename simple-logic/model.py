import torch

from domiknows.program.model.pytorch import PoiModel, IMLModel
from domiknows.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
from domiknows.program.metric import MacroAverageTracker, ValueTracker

def prediction_softmax(pr, gt, data_item = None, prop = None):
    return torch.softmax(pr.data, dim=-1)

class MyIMLModel(IMLModel):
    def __init__(self, graph):
        super().__init__(
            graph, 
            loss=MacroAverageTracker(BCEWithLogitsIMLoss(0.)),
            metric=ValueTracker(prediction_softmax))

class MyModel(PoiModel):
    def __init__(self, graph):
        super().__init__(
            graph, 
            loss=MacroAverageTracker(BCEWithLogitsLoss()),
            metric=ValueTracker(prediction_softmax))

class Net(torch.nn.Module):
    def __init__(self, w=None):
        super().__init__()
        if w is not None:
            self.w = torch.nn.Parameter(torch.tensor(w).float().view(1, 2))
        else:
            self.w = torch.nn.Parameter(torch.randn(1, 2))

    def forward(self, x):
        return x.matmul(self.w)
