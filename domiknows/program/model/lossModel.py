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

    def to(self, device):
        super().to(device=device)
        if self.device is not None:
            if hasattr(self, 'lmbd_p'):
                self.lmbd_p = self.lmbd_p.to(device)

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
                
                if loss['lossTensor'] != None:
                    loss_value = loss['lossTensor'].clamp(min=0)
                    loss_nansum = loss_value[loss_value==loss_value].sum()
                    loss_ = self.get_lmbd(key) * loss_nansum
                    self.loss[key](loss_)
                    lmbd_loss.append(loss_)    
               
            lmbd_loss = sum(lmbd_loss)
        
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

        self.iter_step = 0
        self.warmpup = 80
        
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
        self.iter_step += 1
            
        if not build and not isinstance(builder, DataNodeBuilder):
            raise ValueError('PrimalDualModel must be invoked with `build` on or with provided DataNode Builder.')
        
        if (builder.needsBatchRootDN()):
            builder.addBatchRootDN()
        
#         self.loss.reset()

        datanode = builder.getDataNode(device=self.device)
        
        # Call the loss calculation returns a dictionary, keys are matching the constraints
        constr_loss = datanode.calculateLcLoss(tnorm=self.tnorm, sample=self.sample, sampleSize = self.sampleSize, sampleGlobalLoss = self.sampleGlobalLoss)
        import math
        lmbd_loss = []
        replace_mul = False
        
        key_losses = dict()
        for key, loss in constr_loss.items():
            if key not in self.constr:
                continue
            # loss_value = loss['loss']
            epsilon = 0.0
            key_loss = 0
            new_eps = 0.01
            for i, lossTensor in enumerate(loss['lossTensor']):
                lcSuccesses = loss['lcSuccesses'][i]
                if self.sampleSize == -1:
                    sample_info = [val_ for key, val in loss['sampleInfo'].items() for val_ in val if len(val_)]
                    sample_info = [val[i][1] for val in sample_info]
                    sample_info = torch.stack(sample_info).t()
                    unique_output, unique_inverse, counts = torch.unique(sample_info, return_inverse=True, dim=0, return_counts=True)
                    _, ind_sorted = torch.sort(unique_inverse, stable=True)
                    cum_sum = counts.cumsum(0)
                    cum_sum = torch.cat((torch.tensor([0]).to(counts.device), cum_sum[:-1]))
                    first_indicies = ind_sorted[cum_sum]
                    assert lcSuccesses.sum().item() != 0
                    tidx = (lcSuccesses == 1).nonzero().squeeze(-1)
                    unique_selected_indexes = torch.tensor(np.intersect1d(first_indicies.cpu().numpy(), tidx.cpu().numpy()))
                    if unique_selected_indexes.shape:
                        loss_value = lossTensor[unique_selected_indexes].sum()
                        loss_ = -1 * torch.log(loss_value)
                        key_loss += loss_

                    else:
                        loss_ = 0
                    
                else:
                    if constr_loss["globalSuccessCountet"] > 0:
                        lcSuccesses = constr_loss["globalSuccesses"]
                    if lossTensor.sum().item() != 0:
                        tidx = (lcSuccesses == 1).nonzero().squeeze(-1)
                        true_val = lossTensor[tidx]
                        
                        if true_val.sum().item() != 0: 
                            if not replace_mul:
                                loss_value = true_val.sum() / lossTensor.sum()
                                loss_value = epsilon - ( -1 * torch.log(loss_value) )
                                # loss_value = -1 * torch.log(loss_value) 
                                if self.iter_step < self.warmpup:
                                    with torch.no_grad():
                                        min_val = loss_value
                                else:
                                    min_val = -1
                                # min_val = -1
                                # with torch.no_grad():
                                #     min_val = loss_value
                                loss_ = min_val * loss_value
                                key_loss += loss_
                            else:
                                loss_value = true_val.logsumexp(dim=0) - lossTensor.logsumexp(dim=0)
                                key_loss += -1 * loss_value

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

            epsilon = 1e-2
            if self.sampleSize != -1:
                key_loss = max(key_loss - epsilon, 0) 
            if key_loss != 0:  
                key_losses[key] = key_loss
                # self.loss[key](key_loss)
                # lmbd_loss.append(key_loss) 
                    
        all_losses = [key_losses[key] for key in key_losses]
        if all_losses:
            all_losses = torch.stack(all_losses)

            satisfied_num = len( set(constr_loss.keys()) - set(key_losses.keys()) )
            unsatisfied_num = len(set(constr_loss.keys())) - satisfied_num
            #self.logger.info(f'-- number of satisfied constraints are {satisfied_num}')
            #self.logger.info(f'-- number of unstatisfied constraints are {unsatisfied_num}')
            for key in key_losses:
                if self.sampleSize != -1:
                    if replace_mul:
                        loss_val = (key_losses[key] / all_losses.sum()) * key_losses[key]
                    else:
                        loss_val = key_losses[key]
                else:
                    loss_val = key_losses[key]

                self.loss[key](loss_val)
                lmbd_loss.append(loss_val) 
                
            lmbd_loss = sum(lmbd_loss)
        
        # (*out, datanode, builder)
        # self.logger.info(f'-- lmbd_loss is {lmbd_loss}')
        return lmbd_loss, datanode, builder
