import logging
import warnings
from collections import OrderedDict

import numpy as np
import torch

from ...graph import DataNodeBuilder, DataNode
from ..metric import MetricTracker, MacroAverageTracker

from regr.program.model.pytorch import PoiModel

class PrimalDualModel(PoiModel):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, poi = (), loss=None, metric=None, tnorm=DataNode.tnormsDefault, sample = False, sampleSize = 0, sampleGlobalLoss = False):
        super().__init__(graph, poi=poi, loss=loss, metric=metric)
        self.tnorm = tnorm
        self.sample = sample
        self.sampleSize = sampleSize
        self.sampleGlobalLoss = sampleGlobalLoss
        
        self.constr = OrderedDict(graph.logicalConstrainsRecursive)
        nconstr = len(self.constr)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'PrimalDualModel will not generate any constraint loss.')
            
        self.lmbd = torch.nn.Parameter(torch.empty(nconstr))
        self.lmbd_p = torch.empty(nconstr)  # none parameter
        self.lmbd_index = {}
        
        for i, (key, lc) in enumerate(self.constr.items()):
            self.lmbd_index[key] = i
            p = float(lc.p) / 100.
            if p == 1:
                p = 0.999999999999999
            self.lmbd_p[i] = -np.log(1 - p)  # range: [0, inf)
            
        self.reset_parameters()
        #self.loss = MacroAverageTracker(lambda x:x)

    def reset_parameters(self):
        torch.nn.init.constant_(self.lmbd, 1.)

    def reset(self):
        if isinstance(self.loss, MetricTracker):
            self.loss.reset()

    def get_lmbd(self, key):
        return self.lmbd[self.lmbd_index[key]].clamp(max=self.lmbd_p[self.lmbd_index[key]])

    def forward(self, builderOrData, build=None):
        loss, metric, datanode, builder = super().forward(builderOrData, build) 
        
        if build is None:
            build = self.build
            
        if not build and not isinstance(builderOrData, DataNodeBuilder):
            raise ValueError('PrimalDualModel must be invoked with `build` on or with provided DataNode Builder.')
        
        if isinstance(builderOrData, DataNodeBuilder):
            builder = builderOrData
        else:       
            builderOrData.update({"graph": self.graph, 'READER': 0})
            
            builder = DataNodeBuilder(builderOrData)
            from regr.sensor.pytorch.sensors import TorchSensor
            for prop in self.poi:
                for sensor in prop.find(TorchSensor):
                    sensor(builder)

        datanode = builder.getDataNode()
        
        # Call the loss calculation returns a dictionary, keys are matching the constraints
        constr_loss = datanode.calculateLcLoss(tnorm=self.tnorm, sample=self.sample, sampleSize = self.sampleSize)

        if self.sampleGlobalLoss and constr_loss['lossGlobalTensor']:
            lmbd_loss = constr_loss['lossGlobalTensor'].item()
        else:
            lmbd_loss = []
            for key, loss in constr_loss.items():
                if key not in self.constr:
                    continue
                loss_value = loss['lossTensor'].clamp(min=0)
                loss_nansum = loss_value[loss_value==loss_value].sum()
                loss_ = self.get_lmbd(key) * loss_nansum
                #self.loss[key](loss_)
                lmbd_loss.append(loss_)
            lmbd_loss = sum(lmbd_loss)
        
        # (*out, datanode, builder)
        return lmbd_loss, metric, datanode, builder
    
    def populate(self, builder, run=True):
        return super().populate(builder, run=False)
