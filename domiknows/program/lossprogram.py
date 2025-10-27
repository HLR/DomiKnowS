import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from domiknows.program.model.pytorch import SolverModel
#from domiknows.graph.LeftLogic import LeftLogicElementOutput

from .program import LearningBasedProgram, get_len
from ..utils import consume, entuple, detuple
from .model.lossModel import PrimalDualModel, SampleLossModel, InferenceModel
from .model.base import Mode

from .model.gbi import GBIModel

# Primal-dual need multiple backward through constraint loss.
# It requires retain_graph=True.
# Memory leak problem with `backward(create_graph=True)`
# https://github.com/pytorch/pytorch/issues/4661#issuecomment-596526199
# workaround with following functions based on `grad()`
def backward(loss, parameters):
    parameters = list(parameters)
    grads = torch.autograd.grad(outputs=loss, inputs=parameters, retain_graph=False)
    assert len(grads) == len(parameters)
    for grad, parameter in zip(grads, parameters):
        parameter.grad = grad

def unset_backward(parameters):
    for parameter in parameters:
        parameter.grad = None

def reverse_sign_grad(parameters, factor=-1.):
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad = factor * parameter.grad

def gumbel_softmax(logits, temperature=1.0, hard=False, dim=-1):
    """
    Gumbel-Softmax sampling for differentiable discrete sampling.
    
    Args:
        logits: [..., num_classes] unnormalized log probabilities
        temperature: controls sharpness (lower = more discrete, higher = more smooth)
        hard: if True, returns one-hot but backprops through soft (straight-through estimator)
        dim: dimension to apply softmax
    
    Returns:
        Sampled probabilities (soft or hard)
    """
    # Sample from Gumbel(0, 1)
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    y_soft = F.softmax(gumbels, dim=dim)
    
    if hard:
        # Straight-through estimator: forward = one-hot, backward = soft
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    
    return y_soft

class LossProgram(LearningBasedProgram):
    DEFAULTCMODEL = PrimalDualModel

    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, CModel=None, beta=1, **kwargs):
        super().__init__(graph, Model, **kwargs)
        if CModel is None:
            CModel = self.DEFAULTCMODEL
            
        from inspect import signature
        cmodelSignature = signature(CModel.__init__)
        
        cmodelKwargs = {}
        for param in cmodelSignature.parameters.values():
            paramName = param.name
            if paramName in kwargs:
                cmodelKwargs[paramName] = kwargs[paramName]
        self.cmodel = CModel(graph, **cmodelKwargs)
        self.copt = None
        self.beta = beta

    def to(self, device):
        super().to(device=device)
        if self.device is not None:
            self.model.to(self.device)
            self.cmodel.device = self.device
            self.cmodel.to(self.device)

    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        # COptim=None,  # SGD only
        c_lr=0.05,
        c_momentum=0.9,
        c_warmup_iters=10,  # warmup
        c_freq=10,
        c_freq_increase=5,  # d
        c_freq_increase_freq=1,
        c_lr_decay=4,  # strategy
        c_lr_decay_param=1,  # param in the strategy
        batch_size = 1,
        dataset_size = None, # provide dataset_size if dataset does not have len implemented
        print_loss = True, # print loss on each grad update
        warmup_epochs=0,  # NEW: number of epochs for warmup phase
        constraint_epochs=0,  # NEW: number of epochs for constraint-only phase
        constraint_only=False,  # NEW: if True, use constraint-only training for main phase
        constraint_loss_scale=1.0,  # NEW: scale factor for constraint loss in constraint-only mode
        **kwargs):
        
        # if COptim is None:
        #     COptim = Optim
        # if COptim is not None and list(self.model.parameters()):
        #     self.copt = COptim(self.model.parameters())
        if list(self.cmodel.parameters()):
            self.copt = torch.optim.Adam(self.cmodel.parameters(), lr=c_lr)
        else:
            self.copt = None
            
        # To provide a session cache for cross-epoch variables like iter-count
        c_session = {'iter':0, 'c_update_iter':0, 'c_update_freq':c_freq, 'c_update':0}  
        
        # NEW: Phase-based training - handle both cases
        if warmup_epochs > 0 or constraint_epochs > 0:
            # Custom phased training with manual epoch loop
            self.stop = False
            epoch_counter = 0
            
            if warmup_epochs > 0:
                self.logger.info(f"[Phase 1] Warmup training for {warmup_epochs} epochs")
                for i in range(warmup_epochs):
                    if self.stop:
                        break
                    epoch_counter += 1
                    self.epoch = epoch_counter
                    self.logger.info(f'Epoch: {self.epoch}')
                    self.call_epoch(
                        'Training',
                        training_set,
                        self.train_epoch,
                        c_lr=c_lr,
                        c_warmup_iters=warmup_epochs,
                        c_freq_increase=c_freq_increase,
                        c_freq_increase_freq=c_freq_increase_freq,
                        c_lr_decay=c_lr_decay,
                        c_lr_decay_param=c_lr_decay_param,
                        c_session=c_session,
                        batch_size=batch_size,
                        dataset_size=dataset_size,
                        print_loss=print_loss,
                        training_mode='warmup',
                        **kwargs
                    )
                    if valid_set is not None:
                        self.call_epoch('Validation', valid_set, self.test_epoch, **kwargs)
            
            if constraint_epochs > 0 and not self.stop:
                mode = 'constraint_only' if constraint_only else 'standard'
                self.logger.info(f"[Phase 2] {mode} training for {constraint_epochs} epochs")
                for i in range(constraint_epochs):
                    if self.stop:
                        break
                    epoch_counter += 1
                    self.epoch = epoch_counter
                    self.logger.info(f'Epoch: {self.epoch}')
                    self.call_epoch(
                        'Training',
                        training_set,
                        self.train_epoch,
                        c_lr=c_lr,
                        c_warmup_iters=0,
                        c_freq_increase=c_freq_increase,
                        c_freq_increase_freq=c_freq_increase_freq,
                        c_lr_decay=c_lr_decay,
                        c_lr_decay_param=c_lr_decay_param,
                        c_session=c_session,
                        batch_size=batch_size,
                        dataset_size=dataset_size,
                        print_loss=print_loss,
                        training_mode=mode,
                        constraint_loss_scale=constraint_loss_scale,
                        **kwargs
                    )
                    if valid_set is not None:
                        self.call_epoch('Validation', valid_set, self.test_epoch, **kwargs)
            
            # Final evaluation if test set provided
            if test_set is not None:
                self.call_epoch('Testing', test_set, self.test_epoch, **kwargs)
            
            # Reset epoch and stop after training
            self.epoch = None
            self.stop = None
        else:
            # No phases specified, use standard training
            return super().train(
                training_set,
                valid_set=valid_set,
                test_set=test_set,
                c_lr=c_lr,
                c_warmup_iters=c_warmup_iters,
                c_freq_increase=c_freq_increase,
                c_freq_increase_freq=c_freq_increase_freq,
                c_lr_decay=c_lr_decay,
                c_lr_decay_param=c_lr_decay_param,
                c_session=c_session,
                batch_size=batch_size,
                dataset_size=dataset_size,
                print_loss=print_loss,
                **kwargs)

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        if dataset is not None:
            self.logger.info(f'{name}:')
            desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'

            consume(tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset), desc=desc))

            if self.model.loss:
                self.logger.info(' - loss:')
                self.logger.info(self.model.loss)
                    
            if self.cmodel.loss is not None and  repr(self.cmodel.loss) == "'None'":
                desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'
                self.logger.info(' - Constraint loss:')
                self.logger.info(self.cmodel.loss)

            if self.model.metric:
                self.logger.info(' - metric:')
                for key, metric in self.model.metric.items():
                    self.logger.info(f' - - {key}')
                    self.logger.info(metric)
                    try:
                        self.f.write(f' - - {name}')
                        self.f.write(f' - - {key}')
                        self.f.write("\n")
                        self.f.write(str(metric))
                        self.f.write("\n")
                    except:
                        pass    
    
    def train_epoch(
        self, dataset,
        c_lr=1,
        c_warmup_iters=10,  # warmup
        c_freq_increase=1,  # d
        c_freq_increase_freq=1,
        c_lr_decay=0,  # strategy
        c_lr_decay_param=1,
        c_session={},
        batch_size = 1,
        dataset_size = None, # provide dataset_size if dataset does not have len implemented
        print_loss = True, # print loss on each grad update
        training_mode='standard',  # NEW: 'standard', 'warmup', 'constraint_only'
        constraint_loss_scale=1.0,  # NEW: scale factor for constraint-only mode
        **kwargs):

        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, but got batch_size={batch_size}')

        assert c_session
        from torch import autograd
        torch.autograd.set_detect_anomaly(False)
        self.model.mode(Mode.TRAIN)
        iter = c_session['iter']
        c_update_iter = c_session['c_update_iter']
        c_update_freq = c_session['c_update_freq']
        c_update = c_session['c_update']
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()

        # try to get the number of iterations
        num_data_iters = dataset_size
        if num_data_iters is None:
            if not hasattr(dataset, '__len__'):
                raise ValueError(f'dataset must have attribute __len__ if dataset_size is not provided')
            num_data_iters = len(dataset)

        # NEW: Track stats for constraint-only mode
        constraint_loss_zero_count = 0
        no_gradient_count = 0
        total_steps = 0

        batch_loss = 0.0 # track accumulated loss across batch for logging
        for data_idx, data in enumerate(dataset):
            total_steps += 1
            
            # forward pass
            mloss, metric, *output = self.model(data)  # output = (datanode, builder)
            
            # NEW: Compute loss based on training mode
            if training_mode == 'warmup':
                # Warmup: only use model loss
                loss = mloss
                
            elif training_mode == 'constraint_only':
                # Constraint-only: only use constraint loss to update model
                # IMPORTANT: Must pass retain_graph or compute closs differently
                closs, *_ = self.cmodel(output[1])
                
                # Validate constraint loss
                if not torch.is_tensor(closs):
                    if total_steps <= 5:
                        self.logger.warning("[constraint_only] Constraint loss is None - constraints may not be active")
                    continue
                
                if not torch.isfinite(closs):
                    if total_steps <= 5:
                        self.logger.warning(f"[constraint_only] Non-finite constraint loss: {closs.item()}")
                    continue
                
                # Log constraint loss in first few steps
                if total_steps <= 5:
                    self.logger.info(f"[constraint_only] Step {total_steps} - Constraint loss: {closs.item():.6f}")
                    # Debug: Check if closs requires grad
                    self.logger.info(f"[constraint_only] closs.requires_grad: {closs.requires_grad}")
                
                # Check if constraint loss is non-zero
                if abs(closs.item()) < 1e-8:
                    constraint_loss_zero_count += 1
                    if constraint_loss_zero_count == 1:
                        self.logger.warning(f"[constraint_only] Constraint loss is near zero: {closs.item()}")
                    continue
                
                # Apply scaling for constraint loss
                # Note: We need mloss in the computation graph for gradients to flow
                # Use a small coefficient for mloss to keep constraint dominant
                loss = mloss * 0.01 + closs * constraint_loss_scale
                
            else:  # 'standard' mode
                # Standard: combined mloss + beta*closs
                if iter < c_warmup_iters:
                    loss = mloss
                else:
                    closs, *_ = self.cmodel(output[1])
                    if torch.is_nonzero(closs):
                        loss = mloss + self.beta * closs
                    else:
                        loss = mloss

            if not loss:
                continue

            # get hypothetical position in the batch
            batch_pos = data_idx % batch_size

            # calculate gradients for item
            loss /= batch_size
            batch_loss += loss.item()
            loss.backward()

            # only update if we're the last item in the batch
            # or if we're the last item in the dataset
            do_update = (
                (batch_pos == (batch_size - 1)) or
                (data_idx == (num_data_iters - 1))
            )

            # NEW: Check gradients in constraint-only mode
            if training_mode == 'constraint_only' and do_update and total_steps <= 5:
                self.logger.info(f"[constraint_only] Step {total_steps} - Checking gradients...")
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.abs().sum().item()
                        if grad_norm > 1e-8:
                            self.logger.info(f"  {name}: grad_sum={grad_norm:.6e}")

            # NEW: Verify gradients exist in constraint-only mode
            if training_mode == 'constraint_only' and do_update:
                has_model_grads = any(
                    p.grad is not None and p.grad.abs().sum() > 1e-8
                    for p in self.model.parameters() if p.requires_grad
                )
                
                if not has_model_grads:
                    no_gradient_count += 1
                    if no_gradient_count <= 5:
                        self.logger.warning(f"[constraint_only] Step {total_steps}: No gradients flowing to model parameters")
                    # Zero out and skip update
                    if self.opt is not None:
                        self.opt.zero_grad()
                    if self.copt is not None:
                        self.copt.zero_grad()
                    batch_loss = 0.0
                    continue

            # NEW: Gradient clipping - stronger for constraint-only mode
            if do_update:
                if training_mode == 'constraint_only':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    if self.copt is not None:
                        torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=5.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    if self.copt is not None:
                        torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=10.0)

            # do backwards pass update
            if self.opt is not None and do_update:
                self.opt.step()
            iter += 1

            # NEW: In constraint-only mode, always update copt
            if training_mode == 'constraint_only' and self.copt is not None and do_update:
                self.copt.step()
            elif (
                training_mode != 'constraint_only' and
                self.copt is not None and
                iter > c_warmup_iters and
                do_update and
                iter - c_update_iter > c_update_freq
            ):
                # Standard primal-dual update logic
                reverse_sign_grad(self.model.parameters())
                self.copt.step()
                # update counting
                c_update_iter = iter
                c_update += 1
                # update freq
                if c_freq_increase_freq > 0 and c_update % c_freq_increase_freq == 0:
                    c_update_freq += c_freq_increase
                # update c_lr
                if c_lr_decay == 0:
                    def update_lr(lr):
                        return c_lr * 1. / (1 + c_lr_decay_param * c_update)
                elif c_lr_decay == 1:
                    def update_lr(lr):
                        return lr * np.sqrt(((c_update-1.) / c_lr_decay_param + 1.) / (c_update / c_lr_decay_param + 1.))
                elif c_lr_decay == 2:
                    def update_lr(lr):
                        return lr * (((c_update-1.) / c_lr_decay_param + 1.) / (c_update / c_lr_decay_param + 1.))
                elif c_lr_decay == 3:
                    assert c_lr_decay_param <= 1.
                    def update_lr(lr):
                        return lr * c_lr_decay_param
                elif c_lr_decay == 4:
                    def update_lr(lr):
                        return lr * np.sqrt((c_update+1) / (c_update+2))
                else:
                    raise ValueError(f'c_lr_decay={c_lr_decay} not supported.')
                for param_group in self.copt.param_groups:
                    param_group['lr'] = update_lr(param_group['lr'])
            
            yield (loss, metric, *output[:1])

            # only zero out grads & loss tracker if we've done an update
            if do_update:
                batch_loss = 0.0
                if self.opt is not None:
                    self.opt.zero_grad()
                if self.copt is not None:
                    self.copt.zero_grad()

        # NEW: Summary at end of constraint-only training
        if training_mode == 'constraint_only':
            self.logger.info(f"[constraint_only] Training summary:")
            self.logger.info(f"  Total steps: {total_steps}")
            self.logger.info(f"  Steps with zero loss: {constraint_loss_zero_count}")
            self.logger.info(f"  Steps with no gradients: {no_gradient_count}")
            self.logger.info(f"  Successful steps: {total_steps - constraint_loss_zero_count - no_gradient_count}")

        c_session['iter'] = iter
        c_session['c_update_iter'] = c_update_iter
        c_session['c_update_freq'] = c_update_freq
        c_session['c_update'] = c_update
                  
class PrimalDualProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=PrimalDualModel, beta=beta, **kwargs)

class GumbelPrimalDualProgram(PrimalDualProgram):
    """
    Primal-Dual Program with Gumbel-Softmax support for better discrete optimization.
    
    Backward compatible: when use_gumbel=False or temperature=1.0, behaves like standard PMD.
    """
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, 
                 use_gumbel=False, 
                 initial_temp=1.0, 
                 final_temp=0.1, 
                 anneal_start_epoch=0,
                 anneal_epochs=None,
                 **kwargs):
        """
        Args:
            graph: Knowledge graph
            Model: Model class
            beta: Constraint weight
            use_gumbel: If True, use Gumbel-Softmax instead of standard softmax
            initial_temp: Starting temperature (1.0 = standard softmax)
            final_temp: Ending temperature (0.1 = nearly discrete)
            anneal_start_epoch: Epoch to start annealing (default: 0)
            anneal_epochs: Number of epochs to anneal over (default: total epochs)
            **kwargs: Other arguments passed to parent
        """
        super().__init__(graph, Model, beta=beta, **kwargs)
        self.use_gumbel = use_gumbel
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_start_epoch = anneal_start_epoch
        self.anneal_epochs = anneal_epochs
        self.current_epoch = 0
        self.current_temp = initial_temp
        
        if use_gumbel:
            self.logger.info(f"[Gumbel] Enabled with temp: {initial_temp} → {final_temp}")
    
    def get_temperature(self):
        """Compute current temperature with linear annealing."""
        if not self.use_gumbel:
            return 1.0
        
        if self.current_epoch < self.anneal_start_epoch:
            return self.initial_temp
        
        if self.anneal_epochs is None:
            return self.final_temp
        
        progress = (self.current_epoch - self.anneal_start_epoch) / self.anneal_epochs
        progress = min(1.0, max(0.0, progress))
        
        return self.initial_temp - (self.initial_temp - self.final_temp) * progress
    
    def train(self, training_set, valid_set=None, test_set=None, 
              num_epochs=None, **kwargs):
        """Override train to set anneal_epochs if not provided."""
        if self.use_gumbel and self.anneal_epochs is None and num_epochs is not None:
            self.anneal_epochs = num_epochs
            self.logger.info(f"[Gumbel] Auto-set anneal_epochs to {num_epochs}")
        
        return super().train(training_set, valid_set=valid_set, test_set=test_set, **kwargs)
    
    def train_epoch(self, dataset, **kwargs):
        """Override to update temperature each epoch."""
        self.current_temp = self.get_temperature()
        
        if self.use_gumbel and self.current_epoch % 10 == 0:
            self.logger.info(f"[Gumbel] Epoch {self.current_epoch}: temp={self.current_temp:.3f}")
        
        # Call parent's train_epoch
        yield from super().train_epoch(dataset, **kwargs)
        
        self.current_epoch += 1

class InferenceProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=InferenceModel, beta=beta, **kwargs)

    def evaluate_condition(self, evaluate_data, device="cpu"):
        acc = 0
        total = 0

        for datanode in tqdm(self.populate(evaluate_data, device=device), total=len(evaluate_data), desc="Evaluating"):

            # Finding the label of constraints/condition
            find_constraints_label = datanode.myBuilder.findDataNodesInBuilder(select=datanode.graph.constraint)
            if len(find_constraints_label) < 1:
                self.logger.error("No Constraint Labels found")
                continue
            find_constraints_label = find_constraints_label[0]
            constraint_labels_dict = find_constraints_label.getAttributes()
            active_lc_name = set(x.split('/')[0] for x in constraint_labels_dict)

            # Follow code for set active/non-active before querying
            for lc_name, lc in self.graph.logicalConstrains.items():
                assert lc_name == str(lc)

                if lc_name in active_lc_name:
                    lc.active = True
                else:
                    lc.active = False

            total += 1
            # Inference the final output
            verify_constrains = datanode.verifyResultsLC()
            if not verify_constrains:
                continue
            # Getting label of constraints and convert to 0-1
            verify_constrains = {k: v for k, v in verify_constrains.items() if k in active_lc_name}

            condition_list = [1 if verify_constrains[lc]["satisfied"] == 100.0 else 0 for lc in verify_constrains]
            constraint_labels = [int(constraint_labels_dict[lc + "/label"].item()) for lc in active_lc_name]
            acc += int(constraint_labels == condition_list)

        if total == 0:
            self.logger.error("No Valid Constraint found for this dataset.")
            return None

        return acc / total

class SampleLossProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=SampleLossModel, beta=beta, **kwargs)


    def train(
        self,
        training_set,
        valid_set=None,
        test_set=None,
        # COptim=None,  # SGD only
        c_lr=0.05,
        c_momentum=0.9,
        c_warmup_iters=10,  # warmup
        c_freq=10,
        c_freq_increase=5,  # d
        c_freq_increase_freq=1,
        c_lr_decay=4,  # strategy
        c_lr_decay_param=1,  # param in the strategy
        **kwargs):
        
        return super().train(
            training_set=training_set,
            valid_set=valid_set,
            test_set=test_set,
            c_lr=c_lr,
            c_momentum=c_momentum,
            c_warmup_iters=c_warmup_iters,  # warmup
            c_freq=c_freq,
            c_freq_increase=c_freq_increase,  # d
            c_freq_increase_freq=c_freq_increase_freq,
            c_lr_decay=c_lr_decay,  # strategy
            c_lr_decay_param=c_lr_decay_param,  # param in the strategy
            **kwargs)


    def train_epoch(
        self, dataset,
        c_warmup_iters=0,  # warmup
        c_session={},
        **kwargs):
        self.model.mode(Mode.TRAIN)
#         self.cmodel.mode(Mode.TRAIN)
        assert c_session
        iter = c_session['iter']
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            mloss, metric, *output = self.model(data)  # output = (datanode, builder)
            if iter < c_warmup_iters:
                loss = mloss
            else:
                closs, *_ = self.cmodel(output[1])
                if torch.is_tensor(closs):
                    loss = mloss + self.beta * closs
                else:
                    loss = mloss
                    
                if loss != loss:
                    raise Exception("Calculated loss is nan")
                
            if self.opt is not None and loss:
                loss.backward()
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print (name, param.grad)

                self.opt.step()
                iter += 1
            
            if (
                self.copt is not None and
                loss
            ):
                self.copt.step()
            
            yield (loss, metric, *output[:1])

        c_session['iter'] = iter    
        
class GumbelSampleLossProgram(SampleLossProgram):
    """
    Sample Loss Program with Gumbel-Softmax support for better discrete optimization.
    """
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1,
                 use_gumbel=False,
                 initial_temp=1.0,
                 final_temp=0.1,
                 anneal_start_epoch=0,
                 anneal_epochs=None,
                 hard_gumbel=False,  # Add this parameter
                 **kwargs):
        super().__init__(graph, Model, beta=beta, **kwargs)
        self.use_gumbel = use_gumbel
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_start_epoch = anneal_start_epoch
        self.anneal_epochs = anneal_epochs
        self.hard_gumbel = hard_gumbel  # Store it
        self.current_epoch = 0
        self.current_temp = initial_temp
        
        if use_gumbel:
            self.logger.info(f"[Gumbel] Enabled: temp {initial_temp}→{final_temp}, hard={hard_gumbel}")
    
    def get_temperature(self):
        """Compute current temperature with linear annealing."""
        if not self.use_gumbel:
            return 1.0
        
        if self.current_epoch < self.anneal_start_epoch:
            return self.initial_temp
        
        if self.anneal_epochs is None:
            return self.final_temp
        
        progress = (self.current_epoch - self.anneal_start_epoch) / self.anneal_epochs
        progress = min(1.0, max(0.0, progress))
        
        return self.initial_temp - (self.initial_temp - self.final_temp) * progress
    
    def train(self, training_set, valid_set=None, test_set=None,
              num_epochs=None, **kwargs):
        """Override train to set anneal_epochs if not provided."""
        if self.use_gumbel and self.anneal_epochs is None and num_epochs is not None:
            self.anneal_epochs = num_epochs
            self.logger.info(f"[Gumbel] Auto-set anneal_epochs to {num_epochs}")
        
        return super().train(training_set, valid_set=valid_set, test_set=test_set, **kwargs)
    
    def train_epoch(self, dataset, c_warmup_iters=0, c_session={}, **kwargs):
        """Override to update temperature and pass Gumbel params to model."""
        self.current_temp = self.get_temperature()
        
        if self.use_gumbel and self.current_epoch % 10 == 0:
            self.logger.info(f"[Gumbel] Epoch {self.current_epoch}: temp={self.current_temp:.3f}")
        
        # Modified train loop to pass Gumbel parameters
        self.model.mode(Mode.TRAIN)
        assert c_session
        iter = c_session['iter']
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()
        
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
                
            mloss, metric, *output = self.model(data)
            
            if iter < c_warmup_iters:
                loss = mloss
            else:
                # CRITICAL: Pass Gumbel parameters to cmodel forward
                closs, *_ = self.cmodel.forward(
                    output[1], 
                    use_gumbel=self.use_gumbel,
                    temperature=self.current_temp,
                    hard_gumbel=self.hard_gumbel
                )
                
                if torch.is_tensor(closs):
                    loss = mloss + self.beta * closs
                else:
                    loss = mloss
                    
                if loss != loss:
                    raise Exception("Calculated loss is nan")
                
            if self.opt is not None and loss:
                loss.backward()
                self.opt.step()
                iter += 1
            
            if self.copt is not None and loss:
                self.copt.step()
            
            yield (loss, metric, *output[:1])

        c_session['iter'] = iter
        self.current_epoch += 1
        
class GBIProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, poi, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=GBIModel, beta=beta, poi=poi, **kwargs)
        from domiknows.utils import setDnSkeletonMode
        setDnSkeletonMode(True)