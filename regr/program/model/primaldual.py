import logging

import torch
from torch.utils import data

from ..model.pytorch import PoiModel, SolverModel


class PrimalDualModel(SolverModel):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, *args, **kwargs):
        super().__init__(graph, *args, **kwargs)
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

    def pd_loss(self, datanode):
        # call the loss calculation
        # returns a dictionary, keys are matching the constraints
        constr_loss = datanode.calculateLcLoss()
        if not constr_loss:
            return 0
        lmbd_loss = [self.get_lmbd(key) * loss['lossTensor'].clamp(min=0).sum() for key, loss in constr_loss.items()]
        # lmbd_loss = torch.cat(lmbd_loss, dim=1).sum(dim=1)
        # NB: there is no batch-dim in this loss
        lmbd_loss = torch.stack(lmbd_loss).sum()
        return lmbd_loss

    def populate(self, builder, run=True):
        loss, *outputs = super().populate(builder, run=True)
        datanode = builder.getDataNode()
        loss += self.pd_loss(datanode)
        return (loss, *outputs)
