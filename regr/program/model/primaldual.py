import logging

import torch


class PrimalDualModel(torch.nn.Module):
    logger = logging.getLogger(__name__)

    def __init__(self, graph):
        super().__init__()
        nconstr = len(graph.logicalConstrains)
        self.lmbd = torch.nn.Parameter(torch.empty(nconstr))
        self.lmbd_p = torch.empty(nconstr) # none parameter
        self.lmbd_index = {}
        for i, (key, lc) in enumerate(graph.logicalConstrains.items()):
            self.lmbd_index[key] = i
            self.lmbd_p[i] = float(lc.p) / 100.
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.lmbd, 0.5)

    def get_lmbd(self, key):
        return self.lmbd[self.lmbd_index[key]].clamp(max=self.lmbd_p[self.lmbd_index[key]])

    def forward(self, datanode):
        # call the loss calculation
        # returns a dictionary, keys are matching the constraints
        constr_loss = datanode.calculateLcLoss()
        lmbd_loss = [self.get_lmbd[key] * loss.clamp(min=0).sum() for key, loss in constr_loss]
        # lmbd_loss = torch.cat(lmbd_loss, dim=1).sum(dim=1)
        lmbd_loss = torch.cat(lmbd_loss, dim=1).sum(dim=1)
        return lmbd_loss
