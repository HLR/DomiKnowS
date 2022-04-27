import logging
import warnings
from collections import OrderedDict

import numpy as np
import torch

from ...graph import DataNodeBuilder, DataNode
from ..metric import MetricTracker, MacroAverageTracker

class LossModel(torch.nn.Module):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, 
                 tnorm=DataNode.tnormsDefault, 
                 sample = False, sampleSize = 0, sampleGlobalLoss = False, device='auto'):
        super().__init__()
        self.graph = graph
        self.build = True
        
        self.tnorm = tnorm
        self.device = device
        
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
            
        if not build and not isinstance(builder, DataNodeBuilder):
            raise ValueError('PrimalDualModel must be invoked with `build` on or with provided DataNode Builder.')
        
        if (builder.needsBatchRootDN()):
            builder.addBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        
        # Call the loss calculation returns a dictionary, keys are matching the constraints
        constr_loss = datanode.calculateLcLoss(tnorm=self.tnorm, sample=self.sample, sampleSize = self.sampleSize)

        lmbd_loss = []
        if self.sampleGlobalLoss and constr_loss['globalLoss']:
            globalLoss = constr_loss['globalLoss']
            self.loss['globalLoss'](globalLoss)
            lmbd_loss = torch.tensor(globalLoss, requires_grad=True)
        else:
            for key, loss in constr_loss.items():
                if key not in self.constr:
                    continue
                loss_value = loss['lossTensor'].clamp(min=0)
                loss_nansum = loss_value[loss_value==loss_value].sum()
                loss_ = self.get_lmbd(key) * loss_nansum
                self.loss[key](loss_)
                lmbd_loss.append(loss_)
            lmbd_loss = torch.tensor(sum(lmbd_loss), requires_grad=True)
        
        # (*out, datanode, builder)
        return lmbd_loss, datanode, builder
    
class PrimalDualModel(LossModel):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, tnorm=DataNode.tnormsDefault, device='auto'):
        super().__init__(graph, tnorm=tnorm, device=device)
        
class SampleLosslModel(torch.nn.Module):
    logger = logging.getLogger(__name__)

    # def __init__(self, graph, sample = False, sampleSize = 0, sampleGlobalLoss = False):
    #     super().__init__(graph, sample=sample, sampleSize=sampleSize, sampleGlobalLoss=sampleGlobalLoss)

    def __init__(self, graph, 
                 tnorm=DataNode.tnormsDefault, 
                 sample = False, sampleSize = 0, sampleGlobalLoss = False, device='auto'):
        super().__init__()
        self.graph = graph
        self.build = True
        
        self.tnorm = tnorm
        self.device = device
        
        self.sample = sample
        self.sampleSize = sampleSize
        self.sampleGlobalLoss = sampleGlobalLoss
        
        self.constr = OrderedDict(graph.logicalConstrainsRecursive)
        nconstr = len(self.constr)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'PrimalDualModel will not generate any constraint loss.')
            
        self.lmbd = torch.nn.Parameter(torch.zeros(nconstr).float())
        self.lmbd_index = {}
        
        for i, (key, lc) in enumerate(self.constr.items()):
            self.lmbd_index[key] = i
            
        self.reset_parameters()
        self.loss = MacroAverageTracker(lambda x:x)

    def reset_parameters(self):
        torch.nn.init.constant_(self.lmbd, 0.0)

    def reset(self):
        if isinstance(self.loss, MetricTracker):
            self.loss.reset()

    def get_lmbd(self, key):
        if self.lmbd[self.lmbd_index[key]] < 0:
            with torch.no_grad():
                self.lmbd[self.lmbd_index[key]] = 0
        return self.lmbd[self.lmbd_index[key]]

    def forward(self, builder, build=None):
        if build is None:
            build = self.build
            
        if not build and not isinstance(builder, DataNodeBuilder):
            raise ValueError('PrimalDualModel must be invoked with `build` on or with provided DataNode Builder.')
        
        if (builder.needsBatchRootDN()):
            builder.addBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        
        # Call the loss calculation returns a dictionary, keys are matching the constraints
        constr_loss = datanode.calculateLcLoss(tnorm=self.tnorm, sample=self.sample, sampleSize = self.sampleSize, sampleGlobalLoss = self.sampleGlobalLoss)
        import math
        lmbd_loss = []
        if self.sampleGlobalLoss and constr_loss['globalLoss']:
            globalLoss = constr_loss['globalLoss']
            globalLoss = -1 * math.log(globalLoss)
            self.loss['globalLoss'](globalLoss)
            lmbd_loss = torch.tensor(globalLoss, requires_grad=True)
        else:
            for key, loss in constr_loss.items():
                if key not in self.constr:
                    continue
                # loss_value = loss['loss']
                epsilon = 0.0
                key_loss = 0
                for i, lossTensor in enumerate(loss['lossTensor']):
                    lcSuccesses = loss['lcSuccesses'][i]
                    if lossTensor.sum().item() != 0:
                        tidx = (lcSuccesses == 1).nonzero().squeeze(-1)
                        true_val = lossTensor[tidx]
                        if true_val.sum().item() != 0: 
                            loss_value = true_val.sum() / lossTensor.sum()
                            loss_value = epsilon - ( -1 * torch.log(loss_value) )
                            # loss_value = -1 * torch.log(loss_value)
                            with torch.no_grad():
                                min_val = 10 * loss_value
                            loss_ = - (self.get_lmbd(key) - min_val) * loss_value
                            key_loss += loss_
                            # loss_ = min_val * loss_value
                        else:
                            loss_ = 0
    
                        # if loss['lossTensor'].nansum().item() <= 1:
                        #     loss_value = loss['lossTensor']
                        #     # loss_value = loss_value[loss_value.nonzero()].squeeze(-1)
                        # else:
                        #     _idx = []
                        #     for i in torch.unique(loss['lossTensor']):
                        #         _idx.append((loss['lossTensor'] == i.item()).nonzero(as_tuple=True)[0][0].item())
                        #     loss_value = loss['lossTensor'][_idx]
                        #     # loss_value = torch.unique(loss['lossTensor'])
    
                        # loss_value = torch.log(loss_value.sum())
                        # loss_ = -1 * (loss_value)
                        # self.loss[key](loss_)
                        # lmbd_loss.append(loss_) 
                        
                    else:
                        loss_ = 0

                if key_loss != 0:
                    self.loss[key](key_loss)
                    lmbd_loss.append(key_loss) 
                
            lmbd_loss = sum(lmbd_loss)
        
        # (*out, datanode, builder)
        return lmbd_loss, datanode, builder
