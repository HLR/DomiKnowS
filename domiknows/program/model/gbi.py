from collections import OrderedDict
import warnings
import copy

import torch
from torch import nn
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
    
    def reg_loss(self, model_updated, model, exclude_names=set()):
        orig_params = []
        lambda_params = []

        for key in model.keys():
            w_orig = model[key]
            w_curr = model_updated[key]
            
            if key not in exclude_names:
                orig_params.append(w_orig.flatten())
                lambda_params.append(w_curr.flatten())
            else:
                print('skipping %s' % key)

        orig_params = torch.cat(orig_params, dim=0)
        lambda_params = torch.cat(lambda_params, dim=0)

        return torch.linalg.norm(orig_params - lambda_params, dim=0, ord=2)
    
    # Finds the last layer in each submodel of a PyTorch model
    def find_last_layers_in_submodels(self, model, name=""):
        last_layers = {}
        child_list = list(model.named_children())
        child_names = [name for (name, child) in child_list]

        # Determine if all child names are unique
        are_child_names_unique = len(set(child_names)) == len(child_names)
        
        grandchilds = {}
        for idx, (child_name, child) in enumerate(child_list):
            grandchild_list = list(child.named_children())
            if grandchild_list:
                grandchilds[child_name] = len(grandchild_list)
        
        for idx, (child_name, child) in enumerate(child_list):
            grandchild_list = list(child.named_children())
            if grandchild_list:
                updated_name = f"{name}.{child_name}"
                last_layers.update(self.find_last_layers_in_submodels(child, name=updated_name))
            else:
                if idx == len(child_list) - 1 or grandchilds:
                    updated_name = f"{name}.{child_name}"
                    if hasattr(child, 'bias') and hasattr(child, 'weight'):
                        last_layers[updated_name] = child
                        
        return last_layers

    # Resets the last layer of each submodel of a PyTorch model
    def reset_last_layers_in_submodels(self, model, last_layers):
        for name, last_layer in last_layers.items():
            if isinstance(last_layer, (nn.Linear, nn.Conv2d)):
                last_layer.bias.data += torch.randn_like(last_layer.bias.data) * 1e-4
                last_layer.weight.data += torch.randn_like(last_layer.weight.data) * 1e-4

    # ----
    
    def forward(self, datanode, build=None, print_grads=False):
        # Get constraint satisfaction for the current DataNode
        num_satisfied, num_constraints = self.get_constraints_satisfaction(datanode)
        model_has_GBI_inference = False
        
        # --- Test if to start GBI for this data_item
        if num_satisfied == num_constraints:
            return 0.0, datanode, datanode.myBuilder
        
        # ------- Continue with GBI
        
        optimized_parameters = {name: param for name, param in self.server_model.named_parameters()}
        original_parameters = {name: param.clone().detach() for name, param in self.server_model.named_parameters()}
        
        last_layers = self.find_last_layers_in_submodels(self.server_model)
        self.reset_last_layers_in_submodels(self.server_model, last_layers)
        
        # Print original and cloned parameters to verify they are the same but different in memory location
        for name, param in self.server_model.named_parameters():
            are_equal = torch.equal(param, original_parameters[name])
            same_memory = param.storage().data_ptr() == original_parameters[name].untyped_storage().data_ptr()
            if same_memory or not are_equal:
                print(f"{name} -> Equal: {are_equal}, Same Memory: {same_memory}")
    
        # Data to be used for inference
        x = datanode.myBuilder["data_item"]
                        
        # -- self.server_mode is the model that gets optimized by GBI
        self.server_model.reset()        

        modelLParams = self.server_model.parameters()
        c_opt = Adam(modelLParams, lr=1e-2, betas=[0.9, 0.999], eps=1e-07, amsgrad=False) #SGD(modelLParams, lr=1e-1)
        
        # Remove "GBI" from the list of inference types if model has it
        if hasattr(self.server_model, 'inferTypes'):
            if 'GBI' in self.server_model.inferTypes:
                self.server_model.inferTypes.remove('GBI')
                model_has_GBI_inference = True
        
        no_of_not_satisfied = 0
        node_l = None
        for c_iter in range(self.gbi_iters):
            # perform inference using weights from self.server_mode
            loss, metric, node_l, _ = self.server_model(x)

            num_satisfied_l, num_constraints_l = self.get_constraints_satisfaction(node_l)

            # -- collect probs from datanode (in skeleton mode) 
            probs = []
            for var_name, var_val in node_l.getAttribute('variableSet').items():
                if var_name.endswith('>'):# and var_val.requires_grad:
                    probs.append(F.log_softmax(var_val, dim=-1).flatten())

            log_probs = torch.cat(probs, dim=0).mean()
            if print_grads:
                print('probs mean:')
                print(log_probs)

            if print_grads:
                argmax_vals = [torch.argmax(prob) for prob in probs]
                print('argmax predictions:')
                print(argmax_vals)

            #  -- Constraint loss: NLL * binary satisfaction + regularization loss
            # reg loss is calculated based on L2 distance of weights between optimized model and original weights
            reg_loss =  self.reg_loss(optimized_parameters, original_parameters)
            c_loss = log_probs * ((num_constraints_l - num_satisfied_l) / num_constraints_l) + reg_loss

            if c_loss != c_loss:
                continue
            if print_grads:
                print("iter={}, c_loss={:.2f}, c_loss.grad_fn={}, num_constraints_l={}, satisfied={}".format(c_iter, c_loss.item(), c_loss.grad_fn.__class__.__name__, num_constraints_l, num_satisfied_l))
                print("reg_loss={:.2f}, reg_loss.grad_fn={}, log_probs={:.2f}, log_probs.grad_fn={}\n".format(reg_loss.item(), reg_loss.grad_fn.__class__.__name__, log_probs.item(), log_probs.grad_fn.__class__.__name__))
            
            # --- Check if constraints are satisfied
            if num_satisfied_l == num_constraints_l:
                # --- End early if constraints are satisfied
                if model_has_GBI_inference:
                    self.server_model.inferTypes.append('GBI')
                    
                print(f'Finishing GBI - Constraints are satisfied after {c_iter} iteration')
                return c_loss, node_l, node_l.myBuilder
                        
            # --- Backward pass on self.server_model
            if c_loss.requires_grad:
                # Zero gradients of self.server_model params before backward pass
                c_opt.zero_grad()
                
                # Compute gradients
                c_loss.backward()
                
                if print_grads:
                    # Print the params of the model parameters which have grad
                    print("Params before model step which have grad")
                    for name, param in self.server_model.named_parameters():
                        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                            print(name, 'param sum ', torch.sum(torch.abs(param)).item())
                                            
                # Update self.server_model params based on gradients
                c_opt.step()
                
                if print_grads:
                    # Print the params of the model parameters which have grad
                    print("Params after model step which have grad")
                    for name, param in self.server_model.named_parameters():
                        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                            print(name, 'param sum ', torch.sum(torch.abs(param)).item())
            
        if model_has_GBI_inference:
            self.server_model.inferTypes.append('GBI')
            
        print(f'Finishing GBI - Constraints not are satisfied after {self.gbi_iters} iteration')
        return c_loss, node_l, node_l.myBuilder
 
    def calculateGBISelection(self, datanode, conceptsRelations):
        c_loss, updatedDatanode, updatedBuilder = self.forward(datanode)
        
        for currentConcept, conceptLen in updatedDatanode.rootConcepts:
            cRoot = updatedDatanode.findRootConceptOrRelation(currentConcept)
            dns = updatedDatanode.findDatanodes(select = cRoot)
            originalDns = datanode.findDatanodes(select = cRoot)
            
            currentConceptRelationName = currentConcept.name
            keyGBI  = "<" + currentConceptRelationName + ">/GBI"
            
            if not dns:
                continue
                        
            gbiForConcept  = None
            if getDnSkeletonMode() and "variableSet" in updatedDatanode.attributes:
                gbiForConcept = torch.zeros([len(dns), conceptLen], dtype=torch.float, device=updatedDatanode.current_device)
                   
            for i, (dn, originalDn) in enumerate(zip(dns, originalDns)): 
                v = dn.getAttribute(currentConcept) # Get learned probabilities
                #print(f'Net(depth={i}inGBI); pred: {torch.argmax(v, dim=-1)}')
                if v is None:
                    continue
                
                # Calculate GBI results
                vGBI = torch.zeros(v.size(), dtype=torch.float, device=updatedDatanode.current_device)
                vArgmaxIndex = torch.argmax(v).item()
                #print(f'vArgmaxIndex: {vArgmaxIndex}')
                vGBI[vArgmaxIndex] = 1
                                
                # Add GBI inference result to the original datanode 
                if gbiForConcept is not None:
                    if conceptLen == 1: # binary concept
                        gbiForConcept[i] = vGBI[1]
                    else:
                        gbiForConcept[i] = vGBI
                else:
                    originalDn.attributes[keyGBI] = vGBI
            
            if gbiForConcept is not None:
                rootConceptName = cRoot.name
                keyGBIInVariableSet = rootConceptName + "/" + keyGBI
                datanode.attributes["variableSet"][keyGBIInVariableSet]=gbiForConcept
                updatedDatanode.attributes["variableSet"][keyGBIInVariableSet]=gbiForConcept

        return