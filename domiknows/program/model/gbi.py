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
    def __init__(
            self,
            graph,
            solver_model=None,
            gbi_iters = 50,
            lr=1e-1,
            reg_weight=1,
            reset_params=True,
            device='auto'
        ):
        """
        This class specifies a model that uses gradient-based inference (GBI) for each inference step.
        
        :param graph: The `graph` parameter is an object that represents the logical constraints of a
        graph. It contains information about the nodes, edges, and constraints of the graph
        :param solver_model: Underlying model to perform GBI on. (Optional)
        :param gbi_iters: The maximum number of gradient update steps to perform. GBI will exit early if all constraints are specified. Defaults to 50. (Optional)
        :param lr: The step size of each update step. Defaults to 1e-1. (Optional)
        :param reg_weight: The weight of the regularization loss. Increasing this value will result in parameter updates that are closer to the original, unoptimized parameters. Defaults to 1. (Optional)
        :param reset_params: If set to `True`, the parameters of the model will be reset to the original (non-optimized) parameters after GBI is complete. If set to `False`, the parameters will *only* be reset if the loss becomes `NaN` or the constraints aren't satisfied after `gbi_iters` updates. Set this to `True` if GBI is to only be used during inference. Defaults to `True`. (Optional)
        :param device: The device to use for GBI updates. Defaults to 'auto'. (Optional)
        """

        super().__init__()
        
        if solver_model is None:
            self.solver_model = self
        else:
            self.server_model = solver_model
            
        self.gbi_iters = gbi_iters
        self.lr = lr
        self.reg_weight = reg_weight
        self.reset_params = reset_params

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
        Get constraint satisfaction from datanode. Returns number of satisfied constraints and total number of constraints.

        :params: node: The DataNode to get constraint satisfaction from.
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
        """
        Calculates regularization loss for GBI. The loss is defined as the L2 distance between the original and updated model parameters.

        :param model_updated: The model being optimized.
        :param model: The original, unoptimized model. The parameters for this model should be frozen.
        :param exclude_names: A set of parameter names to exclude from the regularization loss calculation. Defaults to an empty set. (Optional)
        """

        orig_params = {}
        lambda_params = {}

        for key in model.keys():
            key_prefix = '/'.join(key.split('/')[:3])

            # print(key, key_prefix)

            w_orig = model[key]
            w_curr = model_updated[key]

            if key_prefix not in orig_params:
                orig_params[key_prefix] = []
                lambda_params[key_prefix] = []

            if key not in exclude_names:
                orig_params[key_prefix].append(w_orig.flatten())
                lambda_params[key_prefix].append(w_curr.flatten())
            else:
                print('skipping %s' % key)

        result_norm = 0.0

        for key in orig_params.keys():
            orig_params_module = torch.cat(orig_params[key], dim=0)
            lambda_params_module = torch.cat(lambda_params[key], dim=0)

            result_norm += torch.linalg.norm(orig_params_module - lambda_params_module, dim=0, ord=2)

        return result_norm

    def set_pretrained(self, model, orig_params):
        """
        Resets the parameters of a PyTorch model to the original, unoptimized parameters.

        :param model: The PyTorch model to reset the parameters of.
        :param orig_params: The original, unoptimized parameters of the model.
        """
        model.load_state_dict(orig_params)

    def get_argmax_from_node(self, node):
        probs_named = {}
        for var_name, var_val in node.getAttribute('variableSet').items():
            if var_name.endswith('>'):
                probs_named[var_name] = F.log_softmax(var_val, dim=-1).flatten()

        argmax_vals = {name: torch.argmax(param).item() for name, param in probs_named.items()}
        return argmax_vals

    def forward(self, datanode, build=None, verbose=False):
        """
        Performs a forward pass on the model and updates the parameters using gradient-based inference (GBI).

        :param datanode: The DataNode to perform GBI on.
        :param build: Defaults to `None`. (Optional)
        :param verbose: Print intermediate values during inference. Defaults to `False`. (Optional)
        """

        # Get constraint satisfaction for the current DataNode
        num_satisfied, num_constraints = self.get_constraints_satisfaction(datanode)
        model_has_GBI_inference = False

        # --- Test if to start GBI for this data_item
        if num_satisfied == num_constraints:
            return 0.0, datanode, datanode.myBuilder
        # ------- Continue with GBI

        # to be optimized by gbi
        optimized_parameters = {name: param for name, param in self.server_model.named_parameters()}

        # for regularization calculation
        original_parameters = {name: param.clone().detach() for name, param in self.server_model.named_parameters()}

        # for reloading parameters after gbi
        reload_parameters = self.server_model.state_dict()
        for name, param in reload_parameters.items():
            reload_parameters[name] = param.clone().detach()
        
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
        c_opt = SGD(modelLParams, lr=self.lr)
        
        # Remove "GBI" from the list of inference types if model has it
        if hasattr(self.server_model, 'inferTypes'):
            if 'GBI' in self.server_model.inferTypes:
                self.server_model.inferTypes.remove('GBI')
                model_has_GBI_inference = True
        
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

            log_probs_cat = torch.cat(probs, dim=0)
            log_probs = log_probs_cat.mean()

            if verbose:
                print('probs mean:')
                print(log_probs)

            #  -- Constraint loss: NLL * binary satisfaction + regularization loss
            # reg loss is calculated based on L2 distance of weights between optimized model and original weights
            reg_loss =  self.reg_loss(optimized_parameters, original_parameters)
            c_loss = log_probs * ((num_constraints_l - num_satisfied_l) / num_constraints_l) + self.reg_weight * reg_loss

            if c_loss != c_loss:
                # if parameter reset is normally disabled, reset parameters here
                if not self.reset_params:
                    self.set_pretrained(self.server_model, reload_parameters)

                break
            
            if verbose:
                print("iter={}, c_loss={:.4f}, c_loss.grad_fn={}, num_constraints_l={}, satisfied={}".format(c_iter, c_loss.item(), c_loss.grad_fn.__class__.__name__, num_constraints_l, num_satisfied_l))
                print("reg_loss={:.4f}, reg_loss.grad_fn={}, log_probs={:.4f}, log_probs.grad_fn={}\n".format(reg_loss.item(), reg_loss.grad_fn.__class__.__name__, log_probs.item(), log_probs.grad_fn.__class__.__name__))
                
            # --- Check if constraints are satisfied
            if num_satisfied_l == num_constraints_l:
                # --- End early if constraints are satisfied
                if model_has_GBI_inference:
                    self.server_model.inferTypes.append('GBI')

                # reset model parameters
                self.server_model.zero_grad()
                if self.reset_params:
                    self.set_pretrained(self.server_model, reload_parameters)
                
                print(f'Finishing GBI - Constraints are satisfied after {c_iter} iteration')
                return c_loss, node_l, node_l.myBuilder
                        
            # --- Backward pass on self.server_model
            if c_loss.requires_grad:
                # Compute gradients
                c_loss.backward()
                
                if verbose:
                    # Print the params of the model parameters which have grad
                    print("Params before model step which have grad")
                    for name, param in self.server_model.named_parameters():
                        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                            print(name, 'param sum ', torch.sum(torch.abs(param)).item())
                                            
                # Update self.server_model params based on gradients
                c_opt.step()
                
                if verbose:
                    # Print the params of the model parameters which have grad
                    print("Params after model step which have grad")
                    for name, param in self.server_model.named_parameters():
                        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                            print(name, 'param sum ', torch.sum(torch.abs(param)).item())
            
        if model_has_GBI_inference:
            self.server_model.inferTypes.append('GBI')
        
        # reset model parameters
        self.server_model.zero_grad()

        # reset parameters regardless of self.reset_params flag because constraints aren't satisfied
        self.set_pretrained(self.server_model, reload_parameters)

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