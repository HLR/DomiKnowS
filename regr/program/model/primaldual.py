import logging
import warnings
from collections import OrderedDict

import numpy as np
import torch
from torch.utils import data

from ...graph import DataNodeBuilder
from ..model.pytorch import TorchModel
from ..metric import MetricTracker, MacroAverageTracker


class PrimalDualModel(torch.nn.Module):
    logger = logging.getLogger(__name__)

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.build = True
        constr = OrderedDict(graph.logicalConstrainsRecursive)
        nconstr = len(constr)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'PrimalDualModel will not generate any constraint loss.')
        self.lmbd = torch.nn.Parameter(torch.empty(nconstr))
        self.lmbd_p = torch.empty(nconstr)  # none parameter
        self.lmbd_index = {}
        for i, (key, lc) in enumerate(constr.items()):
            self.lmbd_index[key] = i
            p = float(lc.p) / 100.
            self.lmbd_p[i] = -np.log(1 - p)  # range: [0, inf)
        self.reset_parameters()
        self.loss = MacroAverageTracker(lambda x:x)

    def reset_parameters(self):
        torch.nn.init.constant_(self.lmbd, 1.)

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
            loss_value = loss['lossTensor'].clamp(min=0)
            loss_nansum = loss_value[loss_value==loss_value].sum()
            loss_ = self.get_lmbd(key) * loss_nansum
            self.loss[key](loss_)
            lmbd_loss.append(loss_)
        lmbd_loss = sum(lmbd_loss)
        return lmbd_loss, datanode, builder
