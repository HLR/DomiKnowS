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
            device='auto',
            grad_clip=None,
            early_stop_patience=5,
            loss_plateau_threshold=1e-6,
            optimizer='sgd',
            momentum=0.0
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
        :param grad_clip: Maximum gradient norm for clipping. If None, no clipping is applied. (Optional)
        :param early_stop_patience: Number of iterations to wait for improvement before early stopping. Defaults to 5. (Optional)
        :param loss_plateau_threshold: Minimum loss improvement to reset patience counter. Defaults to 1e-6. (Optional)
        :param optimizer: Optimizer type ('sgd' or 'adam'). Defaults to 'sgd'. (Optional)
        :param momentum: Momentum for SGD optimizer. Defaults to 0.0. (Optional)
        """

        super().__init__()
        
        # Fixed typo: should be solver_model consistently
        if solver_model is None:
            self.solver_model = self
        else:
            self.solver_model = solver_model
            
        self.gbi_iters = gbi_iters
        self.lr = lr
        self.reg_weight = reg_weight
        self.reset_params = reset_params
        self.grad_clip = grad_clip
        self.early_stop_patience = early_stop_patience
        self.loss_plateau_threshold = loss_plateau_threshold
        self.optimizer_type = optimizer.lower()
        self.momentum = momentum

        self.device = device
        
        self.constr = OrderedDict(graph.logicalConstrainsRecursive)
        nconstr = len(self.constr)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'GBIModel will not generate any constraint loss.')
            
        self.loss = MacroAverageTracker(NBCrossEntropyLoss())
        
        # Statistics tracking
        self.stats = {
            'total_iterations': 0,
            'early_stops': 0,
            'constraint_satisfied_stops': 0,
            'nan_resets': 0,
            'avg_iterations': 0
        }
        
    def reset(self):
        if hasattr(self, 'loss') and self.solver_model.loss and isinstance(self.solver_model.loss, MetricTracker):
            self.solver_model.loss.reset()

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
    
    def reg_loss(self, model_updated, model_original):
        """
        Calculates regularization loss for GBI using efficient parameter grouping.

        :param model_updated: The model being optimized.
        :param model_original: The original, unoptimized model parameters (frozen).
        """
        result_norm = 0.0
        
        # Efficient computation without intermediate lists
        for (name_updated, param_updated), (name_orig, param_orig) in zip(
            model_updated.items(), model_original.items()
        ):
            # Ensure we're comparing the same parameters
            assert name_updated == name_orig, f"Parameter mismatch: {name_updated} vs {name_orig}"
            
            # Compute L2 norm directly
            result_norm += torch.linalg.norm(param_orig - param_updated, ord=2)

        return result_norm

    def set_pretrained(self, model, orig_params):
        """
        Resets the parameters of a PyTorch model to the original, unoptimized parameters.

        :param model: The PyTorch model to reset the parameters of.
        :param orig_params: The original, unoptimized parameters of the model.
        """
        model.load_state_dict(orig_params)
        
    def _create_optimizer(self, parameters):
        """Create optimizer based on configuration."""
        if self.optimizer_type == 'adam':
            return Adam(parameters, lr=self.lr)
        elif self.optimizer_type == 'sgd':
            return SGD(parameters, lr=self.lr, momentum=self.momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def _apply_gradient_clipping(self):
        """Apply gradient clipping if configured."""
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.solver_model.parameters(), 
                self.grad_clip
            )

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
            if verbose:
                print('All constraints already satisfied, skipping GBI')
            return 0.0, datanode, datanode.myBuilder
        
        # ------- Continue with GBI
        if verbose:
            print(f'Starting GBI: {num_satisfied}/{num_constraints} constraints satisfied')

        # Store original parameters for regularization and potential reset
        original_parameters = {
            name: param.clone().detach() 
            for name, param in self.solver_model.named_parameters()
        }
        
        # Store state dict for reset (more memory efficient than cloning all)
        reload_parameters = {
            name: param.clone().detach()
            for name, param in self.solver_model.state_dict().items()
        }
        
        # Data to be used for inference
        x = datanode.myBuilder["data_item"]
                        
        # Reset solver model
        self.solver_model.reset()        

        # Create optimizer
        c_opt = self._create_optimizer(self.solver_model.parameters())
        
        # Remove "GBI" from the list of inference types if model has it
        if hasattr(self.solver_model, 'inferTypes'):
            if 'GBI' in self.solver_model.inferTypes:
                self.solver_model.inferTypes.remove('GBI')
                model_has_GBI_inference = True
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        best_iteration = 0
        
        node_l = None
        for c_iter in range(self.gbi_iters):
            # Perform inference using weights from solver_model
            loss, metric, node_l, _ = self.solver_model(x)

            num_satisfied_l, num_constraints_l = self.get_constraints_satisfaction(node_l)

            # Collect probabilities from datanode
            probs = []
            for var_name, var_val in node_l.getAttribute('variableSet').items():
                if var_name.endswith('>'):
                    probs.append(F.log_softmax(var_val, dim=-1).flatten())

            log_probs_cat = torch.cat(probs, dim=0)
            log_probs = log_probs_cat.mean()

            if verbose:
                print(f'Iter {c_iter}: log_probs mean = {log_probs.item():.6f}')

            # Constraint loss: NLL * binary satisfaction + regularization loss
            optimized_parameters = {name: param for name, param in self.solver_model.named_parameters()}
            reg_loss_val = self.reg_loss(optimized_parameters, original_parameters)
            c_loss = log_probs * ((num_constraints_l - num_satisfied_l) / num_constraints_l) + self.reg_weight * reg_loss_val

            # Check for NaN loss
            if torch.isnan(c_loss):
                if verbose:
                    print(f'NaN loss detected at iteration {c_iter}, resetting parameters')
                
                self.stats['nan_resets'] += 1
                # Always reset on NaN
                self.set_pretrained(self.solver_model, reload_parameters)
                if model_has_GBI_inference:
                    self.solver_model.inferTypes.append('GBI')
                break
            
            if verbose:
                print(f"iter={c_iter}, c_loss={c_loss.item():.6f}, "
                      f"reg_loss={reg_loss_val.item():.6f}, "
                      f"satisfied={num_satisfied_l}/{num_constraints_l}")
                
            # --- Check if constraints are satisfied
            if num_satisfied_l == num_constraints_l:
                if verbose:
                    print(f'Constraints satisfied after {c_iter} iterations')
                
                self.stats['constraint_satisfied_stops'] += 1
                self.stats['total_iterations'] += c_iter
                
                # Reset model parameters if configured
                self.solver_model.zero_grad()
                if self.reset_params:
                    self.set_pretrained(self.solver_model, reload_parameters)
                
                if model_has_GBI_inference:
                    self.solver_model.inferTypes.append('GBI')
                
                return c_loss, node_l, node_l.myBuilder
            
            # Early stopping based on loss plateau
            if c_loss.item() < best_loss - self.loss_plateau_threshold:
                best_loss = c_loss.item()
                patience_counter = 0
                best_iteration = c_iter
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stop_patience:
                if verbose:
                    print(f'Early stopping at iteration {c_iter} '
                          f'(best was {best_iteration} with loss {best_loss:.6f})')
                
                self.stats['early_stops'] += 1
                self.stats['total_iterations'] += c_iter
                break
                        
            # --- Backward pass on solver_model
            if c_loss.requires_grad:
                # Zero gradients
                c_opt.zero_grad()
                
                # Compute gradients
                c_loss.backward()
                
                # Apply gradient clipping if configured
                self._apply_gradient_clipping()
                
                if verbose:
                    # Print gradient statistics
                    total_norm = 0.0
                    for name, param in self.solver_model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f'  Total gradient norm: {total_norm:.6f}')
                                            
                # Update solver_model params based on gradients
                c_opt.step()
        else:
            # Loop completed without break (max iterations reached)
            self.stats['total_iterations'] += self.gbi_iters
            if verbose:
                print(f'Max iterations ({self.gbi_iters}) reached without satisfying constraints')
        
        # Update statistics
        if self.stats['constraint_satisfied_stops'] + self.stats['early_stops'] > 0:
            self.stats['avg_iterations'] = (
                self.stats['total_iterations'] / 
                (self.stats['constraint_satisfied_stops'] + self.stats['early_stops'])
            )
        
        if model_has_GBI_inference:
            self.solver_model.inferTypes.append('GBI')
        
        # Reset model parameters
        self.solver_model.zero_grad()
        
        # Always reset parameters if constraints not satisfied
        self.set_pretrained(self.solver_model, reload_parameters)

        if verbose:
            print(f'Finishing GBI - Constraints not satisfied after {self.gbi_iters} iterations')
        
        return c_loss, node_l, node_l.myBuilder
 
    def get_stats(self):
        """Return GBI optimization statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics tracking."""
        self.stats = {
            'total_iterations': 0,
            'early_stops': 0,
            'constraint_satisfied_stops': 0,
            'nan_resets': 0,
            'avg_iterations': 0
        }
 
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
                v = dn.getAttribute(currentConcept)
                if v is None:
                    continue
                
                # Calculate GBI results
                vGBI = torch.zeros(v.size(), dtype=torch.float, device=updatedDatanode.current_device)
                vArgmaxIndex = torch.argmax(v).item()
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
                datanode.attributes["variableSet"][keyGBIInVariableSet] = gbiForConcept
                updatedDatanode.attributes["variableSet"][keyGBIInVariableSet] = gbiForConcept

        return