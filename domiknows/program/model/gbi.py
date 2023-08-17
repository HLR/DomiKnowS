from collections import OrderedDict
import warnings
import copy

import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

from domiknows.program.metric import MacroAverageTracker, MetricTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.model.base import Mode

from domiknows.utils import getDnSkeletonMode

# Gradient-based Inference
class GBIModel(torch.nn.Module):
    def __init__(self, graph, solver_model=None, gbi_iters = 100, device='auto'):
        super().__init__()
        
        if solver_model is None:
            self.solver_model = self
        else:
            self.server_model = solver_model
            
        self.gbi_iters = gbi_iters
        self.device = device
        
        self.constr = OrderedDict(graph.logicalConstrainsRecursive)
        nconstr = len(self.constr)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'GBIModel will not generate any constraint loss.')
            
        self.loss = MacroAverageTracker(NBCrossEntropyLoss())
        
    def reset(self):
        if hasattr(self, 'loss') and self.solver_model.loss and isinstance(self.solver_model.loss, MetricTracker):
            self.solver_model.loss.reset()

 # --- GBI methods
    def get_constraints_satisfaction(self, node):
        """
        Get constraint satisfaction from datanode
        Returns number of satisfied constraints and total number of constraints
        """
        verifyResult = node.verifyResultsLC(key = "/local/argmax")

        assert verifyResult

        satisfied_constraints = []
        for lc_idx, lc in enumerate(verifyResult):
            satisfied_constraints.append(verifyResult[lc]['satisfied'])

        num_constraints = len(verifyResult)
        num_satisfied = sum(satisfied_constraints) // 100

        return num_satisfied, num_constraints
    
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
    
    # ----
    
    def forward(self, datanode, build=None):
        # Get constraint satisfaction for the current DataNode
        num_satisfied, num_constraints = self.get_constraints_satisfaction(datanode)

        # --- Test if to start GBI for this data_item
        if num_satisfied == num_constraints:
            return 0.0, datanode, datanode.myBuilder
        
        # ------- Continue with GBI
                
        # Make copy of original model
        model_l = copy.deepcopy(self.server_model)
    
        # Data to be used for inference
        x = datanode.myBuilder["data_item"]
       
        # reset instance-specific weights to pretrained weights 
        # discard the last layer and the second last layer (dense and output) and replace it with random weights
        for modelChild in model_l.children():
            if hasattr(modelChild, 'model'):
                if hasattr(modelChild.model, 'net'):
                    currentNet = modelChild.model.net
                else:
                    # Handle the case where the attribute is not present
                    currentNet = None
            else:
                # Handle the case where the attribute is not present
                if hasattr(modelChild, 'reset_parameters'):
                   modelChild.reset_parameters()
                   
                continue
                
            if currentNet is None:
                continue
                     
            last_layer = None
            second_layer = None
            for layer in currentNet:
                if hasattr(layer, 'reset_parameters'):
                    second_layer = last_layer # optionally discard the second layer as well
                    last_layer = layer
            
            if last_layer is not None:
                #print("Resetting parameters for layer: ", last_layer)
                last_layer.reset_parameters()
                
                if second_layer is not None:
                    #print("Resetting parameters for layer: ", second_layer)
                    second_layer.reset_parameters()
                        
        # -- model_l is the model that gets optimized by GBI
        model_l.reset()        

        modelLParams = model_l.parameters()
        c_opt = Adam(modelLParams, lr=1e-1, betas=[0.9, 0.999], eps=1e-07, amsgrad=False) #SGD(modelLParams, lr=1e-1)
        
        # Remove "GBI" from the list of inference types if model has it
        if hasattr(model_l, 'inferTypes'):
            if 'GBI' in model_l.inferTypes:
                model_l.inferTypes.remove('GBI')
        
        no_of_not_satisfied = 0
        node_l = None
        for c_iter in range(self.gbi_iters):
            # perform inference using weights from model_l
            loss, metric, node_l, _ = model_l(x)

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
                if var_name.endswith('>'):# and var_val.requires_grad:
                    probs[var_name] = torch.sum(F.log_softmax(var_val, dim=-1))

            # print probs with the keys
            print("probs:")
            for key, value in probs.items():
                print(key, value.item())
            
            # get total log prob
            log_probs = 0.0
            for c_prob in probs.values():
                eps = 1e-7
                t = F.relu(c_prob)
                tLog = torch.log(t + eps)
                log_probs += torch.sum(tLog)

            #  -- Constraint loss: NLL * binary satisfaction + regularization loss
            # reg loss is calculated based on L2 distance of weights between optimized model and original weights
            reg_loss =  self.reg_loss(model_l, self.server_model)
            c_loss = -1 * log_probs * is_satisfied + reg_loss

            if c_loss != c_loss:
                continue
            
            print("iter={}, c_loss={:.2f}, c_loss.grad_fn={}, num_constraints_l={}, satisfied={}".format(c_iter, c_loss.item(), c_loss.grad_fn.__class__.__name__, num_constraints_l, num_satisfied_l))
            print("reg_loss={:.2f}, reg_loss.grad_fn={}, log_probs={:.2f}, log_probs.grad_fn={}\n".format(reg_loss.item(), reg_loss.grad_fn.__class__.__name__, log_probs.item(), log_probs.grad_fn.__class__.__name__))
            
            # --- Check if constraints are satisfied
            if num_satisfied_l == num_constraints_l:
                # --- End early if constraints are satisfied
                return c_loss, datanode, datanode.myBuilder
            elif no_of_not_satisfied > 3: # three consecutive iterations where constraints are not satisfied
                return c_loss, datanode, datanode.myBuilder # ? float("nan")
                        
            # --- Backward pass on model_l
            if c_loss.requires_grad:
                # Zero gradients of model_l params before backward pass
                c_opt.zero_grad()
                
                # Compute gradients
                c_loss.backward()
                
                # Print the params of the model parameters which have grad
                print("Params before model step which have grad")
                for name, param in model_l.named_parameters():
                    if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                        print(name, 'param sum ', torch.sum(torch.abs(param)).item())
                                            
                # Update model_l params based on gradients
                c_opt.step()
                
                #  Print the params of the model parameters which have grad
                print("Params after model step which have grad")
                for name, param in model_l.named_parameters():
                    if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                        print(name, 'param sum ', torch.sum(torch.abs(param)).item())

        node_l_builder = None
        if node_l is not None:
            node_l_builder = node_l.myBuilder
            
        return c_loss, node_l, node_l_builder # ? float("nan")
 
    def calculateGBISelection(self, datanode, conceptsRelations):
        c_loss, updatedDatanode, updatedBuilder = self.forward(datanode)
        
        for currentConceptRelation in conceptsRelations:
            cRoot = updatedDatanode.findRootConceptOrRelation(currentConceptRelation[0])
            dns = updatedDatanode.findDatanodes(select = cRoot)
            originalDns = datanode.findDatanodes(select = cRoot)
            
            currentConceptRelationName = currentConceptRelation[0].name
            keyGBI  = "<" + currentConceptRelationName + ">/GBI"
            
            if not dns:
                continue
                        
            gbiForConcept  = None
            if getDnSkeletonMode() and "variableSet" in updatedDatanode.attributes:
                gbiForConcept = torch.zeros([len(dns), currentConceptRelation[3]], dtype=torch.float, device=updatedDatanode.current_device)
                   
            for i, (dn, originalDn) in enumerate(zip(dns, originalDns)): 
                v = dn.getAttribute(currentConceptRelation[0]) # Get ILP results
                if v is None:
                    continue
                
                # Calculate GBI results
                vGBI = torch.zeros(v.size())
                vArgmaxIndex = torch.argmax(v).item()
                vGBI[vArgmaxIndex] = 1
                                
                # Add GBI inference result to the original datanode 
                if gbiForConcept is not None:
                    if currentConceptRelation[2] is None: # binary concept
                        gbiForConcept[i] = vGBI[1]
                    else:
                        gbiForConcept[i] = vGBI
                else:
                    originalDn.attributes[keyGBI] = vGBI
            
            if gbiForConcept is not None:
                rootConceptName = cRoot.name
                keyGBIInVariableSet = rootConceptName + "/" + keyGBI
                updatedDatanode.attributes["variableSet"][keyGBIInVariableSet]=gbiForConcept

