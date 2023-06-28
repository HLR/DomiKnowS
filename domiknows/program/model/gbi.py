from collections import OrderedDict
import warnings

import torch
import copy
from torch.optim import SGD

from domiknows.program.metric import MacroAverageTracker, MetricTracker
from domiknows.program.model.base import Mode

from .pytorch import SolverModel
from ...graph.concept import EnumConcept
from domiknows.utils import detuple
import torch.nn.functional as F
import argparse
import numpy

# Gradient-based Inference
class GBIModel(torch.nn.Module):
    def __init__(self, graph, solver_model, gbi_iters = 100, device='auto'):
        super().__init__()
        
        self.server_model= solver_model
        self.gbi_iters = gbi_iters
        self.device = device
        
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
        if hasattr(self, 'loss') and self.loss and isinstance(self.loss, MetricTracker):
            self.loss.reset()

    def get_lmbd(self, key):
        if self.lmbd[self.lmbd_index[key]] < 0:
            with torch.no_grad():
                self.lmbd[self.lmbd_index[key]] = 0
        return self.lmbd[self.lmbd_index[key]]

 # --- GBI methods
    def get_constraints_satisfaction(self, node):
        """
        Get constraint satisfaction from datanode
        Returns number of satisfied constraints and total number of constraints
        """
        verifyResult = node.verifyResultsLC()

        assert verifyResult

        satisfied_constraints = []
        for lc_idx, lc in enumerate(verifyResult):
            satisfied_constraints.append(verifyResult[lc]['satisfied'])

        num_constraints = len(verifyResult)
        num_satisfied = sum(satisfied_constraints) // 100

        return num_satisfied, num_constraints
    
    def get_lambda(self, model, lr=1e-3):
        model_l = copy.deepcopy(model)

        c_opt = SGD(model_l.parameters(), lr=lr)

        return model_l, c_opt

    def reg_loss(self, model_lambda, model, exclude_names=set()):
        orig_params = []
        lambda_params = []

        for (name, w_orig), (_, w_curr) in zip(model.named_parameters(), model_lambda.named_parameters()):
            if name not in exclude_names:
                orig_params.append(w_orig.flatten())
                lambda_params.append(w_curr.flatten())
            else:
                print('skipping %s' % name)

        orig_params = torch.cat(orig_params, dim=0)
        lambda_params = torch.cat(lambda_params, dim=0)

        return torch.linalg.norm(orig_params - lambda_params, dim=0, ord=2)

    def populate_forward(self, model, data_item):
        """
        Forward pass through torch model.
        Returns DataNode and DataNodeBuilder.
        """
        _, _, *output = model(data_item)
        node = detuple(*output[:1])
        return node, output[1]
    # ----
    
    def forward(self, builder, build=None):
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)

        datanode.infer()

        # Get constraint satisfaction for the current DataNode
        num_satisfied, num_constraints = self.get_constraints_satisfaction(datanode)

        # --- Test if to start GBI for this data_item
        if num_satisfied == num_constraints:
            return 0.0, datanode, builder
        
        # ------- Continue with GBI
                
        # -- Make copy of original model
        model_l, c_opt = self.get_lambda(self.server_model, lr=1e-1)
        
        # -- model_l is the model that gets optimized by GBI
        model_l.mode(Mode.TRAIN)
        model_l.train()
        model_l.reset()        

        # Remove "GBI" from the list of inference types if model has it
        if hasattr(model_l, 'inferTypes'):
            if 'GBI' in model_l.inferTypes:
                model_l.inferTypes.remove('GBI')
        
        no_of_not_satisfied = 0
        for c_iter in range(self.gbi_iters):
            c_opt.zero_grad()

            # -- forward pass through model_l
            node_l, builder_l = self.populate_forward(model_l, builder) # data_item

            num_satisfied_l, num_constraints_l = self.get_constraints_satisfaction(node_l)

            if num_satisfied_l == num_constraints_l:
                is_satisfied = 1
                no_of_not_satisfied = 0
            else:
                is_satisfied = 0
                no_of_not_satisfied += 1

            is_satisfied = num_satisfied_l/num_constraints_l
            # -- collect probs from datanode (in skeleton mode) 
            probs = {}
            for var_name, var_val in node_l.getAttribute('variableSet').items():
                if not var_name.endswith('/label') and var_val.requires_grad:
                    probs[var_name] = torch.sum(F.log_softmax(var_val, dim=-1))

            # get total log prob
            log_probs = 0.0
            for c_prob in probs.values():
                eps=1e-7
                t = F.relu(c_prob)
                tLog = torch.log(t+eps)
                log_probs += torch.sum(tLog)

            #  -- Constraint loss: NLL * binary satisfaction + regularization loss
            # reg loss is calculated based on L2 distance of weights between optimized model and original weights
            c_loss = -1 * log_probs * is_satisfied + self.reg_loss(model_l, self.server_model)

            print("iter=%d, c_loss=%d, num_constraints_l=%d, satisfied=%d"%(c_iter, c_loss.item(), num_constraints_l, num_satisfied_l))

            # --- Check if constraints are satisfied
            if num_satisfied_l == num_constraints_l:
                # --- End early if constraints are satisfied
                self.server_model 
                return c_loss, datanode, builder
            elif no_of_not_satisfied > 3: # three consecutive iterations where constraints are not satisfied
                return c_loss, datanode, builder # ? float("nan")
                        
            # --- Backward pass on model_l
            if c_loss.requires_grad:
                c_loss.backward(retain_graph=True)
            #  -- Update model_l
            c_opt.step()
        
        return c_loss, datanode, builder # ? float("nan")
 
    def calculateGBISelection(self, builder, conceptsRelations):
        self.forward(builder)
        
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        
        for c in conceptsRelations:
            cRoot = datanode.findRootConceptOrRelation(c[0])
            dns = datanode.findDatanodes(select = cRoot)
            
            keyArgmax  = "<" + c[0].name + ">/local/argmax"
            keyGBI  = "<" + c[0].name + ">/GBI"

            if not dns:
                continue
                        
            for dn in dns: 
                v = dn.getAttribute(c[0])
                 
                vGBI = torch.zeros(v.size())
                vArgmaxIndex = torch.argmax(v).item()
                vGBI[vArgmaxIndex] = 1
                                
                dn.attributes[keyGBI] = vGBI

