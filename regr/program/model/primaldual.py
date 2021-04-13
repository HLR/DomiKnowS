import logging
import warnings

import torch
from torch.utils import data

from ...graph import DataNodeBuilder
from ..model.pytorch import TorchModel
from ..metric import MetricTracker, MacroAverageTracker


class PrimalDualModel(TorchModel):
    logger = logging.getLogger(__name__)

    def __init__(self, graph):
        super().__init__(graph)
        nconstr = len(graph.logicalConstrains)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'PrimalDualModel will not generate any constraint loss.')
        self.lmbd = torch.nn.Parameter(torch.empty(nconstr))
        self.lmbd_p = torch.empty(nconstr)  # none parameter
        self.lmbd_index = {}
        for i, (key, lc) in enumerate(graph.logicalConstrains.items()):
            self.lmbd_index[key] = i
            self.lmbd_p[i] = float(lc.p) / 100.
        self.reset_parameters()
        self.loss = MacroAverageTracker(lambda x:x)

    def reset_parameters(self):
        torch.nn.init.constant_(self.lmbd, 0.5)

    def reset(self):
        if isinstance(self.loss, MetricTracker):
            self.loss.reset()

    def get_lmbd(self, key):
        return self.lmbd[self.lmbd_index[key]].clamp(max=self.lmbd_p[self.lmbd_index[key]])

    def forward(self, builder, build=None):
        if build is None:
            build = self.build
        if not build or not isinstance(builder, DataNodeBuilder):
            raise ValueError('PrimalDualModel must be invoked with `build` on.')
        datanode = builder.getDataNode()
        # call the loss calculation
        # returns a dictionary, keys are matching the constraints
        constr_loss = datanode.calculateLcLoss()
        lmbd_loss = []
        for key, loss in constr_loss.items():
            loss_ = self.get_lmbd(key) * loss['lossTensor'].clamp(min=0).sum()
            self.loss[key](loss_)
            lmbd_loss.append(loss_)
        lmbd_loss = sum(lmbd_loss)
        return lmbd_loss, datanode, builder
