import logging
import warnings
from collections import OrderedDict

import numpy as np
import torch

from ...graph import DataNodeBuilder
from ..metric import MetricTracker, MacroAverageTracker
from domiknows import setup_logger, getProductionModeStatus
from domiknows.graph.logicalConstrain import sumL

try:
    from monitor.constraint_monitor import ( # type: ignore
        log_single_lc, log_memory
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class LossModel(torch.nn.Module):
    """
    Base model for training from constraint loss
    
    Implements the Primal Dual algorithm for constraint loss calculation.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, graph, 
                 tnorm='P',
                 counting_tnorm=None,
                 sample=False, sampleSize=0, sampleGlobalLoss=False, device='auto',
                 use_gumbel=False, temperature=1.0, hard_gumbel=False):
        """
        Initialize LossModel.
        
        :param graph: Graph representing the logical constraints
        :param tnorm: T-norm type for fuzzy logic ('P' for product)
        :param counting_tnorm: T-norm for counting constraints (None uses tnorm)
        :param sample: Whether to use sampling during training
        :param sampleSize: Number of samples per iteration
        :param sampleGlobalLoss: Whether to sample global loss
        :param device: Device for computation ('auto', 'cpu', 'cuda')
        :param use_gumbel: If True, apply Gumbel-Softmax to local inference
        :param temperature: Gumbel-Softmax temperature (lower = more discrete)
        :param hard_gumbel: If True, use straight-through estimator
        """
        super().__init__()
        self.graph = graph
        self.build = True
        
        self.tnorm = tnorm
        self.counting_tnorm = counting_tnorm
        self.device = device
        
        self.sample = sample
        self.sampleSize = sampleSize
        self.sampleGlobalLoss = sampleGlobalLoss
        
        # Gumbel-Softmax parameters
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.hard_gumbel = hard_gumbel
        
        # Extract all logical constraints from the graph recursively
        self.constr = OrderedDict(graph.allLogicalConstrainsRecursive)
        nconstr = len(self.constr)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'PrimalDualModel will not generate any constraint loss.')
            
        # Initialize lambda (Lagrange multipliers) as learnable parameters for Primal-Dual optimization
        # Each constraint gets its own lambda value that balances its contribution to the total loss
        self.lmbd = torch.nn.Parameter(torch.empty(nconstr))
        
        # Penalty terms (upper bounds) for lambda values, derived from constraint priorities
        self.lmbd_p = torch.empty(nconstr)
        
        # Mapping from constraint keys to their index positions in lambda tensors
        self.lmbd_index = {}
        
        # Initialize penalty terms based on constraint priority values (p)
        for i, (key, lc) in enumerate(self.constr.items()):
            self.lmbd_index[key] = i
            
            # Convert percentage priority to probability (0-1 range)
            p = float(lc.p) / 100.
            
            # Avoid log(0) by capping probability just below 1
            if p == 1:
                p = 0.999999999999999
            
            # Compute penalty term: -log(1-p) ensures higher priority constraints have higher penalties
            self.lmbd_p[i] = -np.log(1 - p)
            
        # Initialize lambda values (default: all set to 1.0)
        self.reset_parameters()
        
        # Set up loss tracker for monitoring constraint losses during training
        self.loss = MacroAverageTracker(lambda x:x)
        
        self._setup_lossmodel_logger()

    def _setup_lossmodel_logger(self):
        """Set up dedicated logger for LossModel operations."""
        lossmodel_log_config = {
            'log_name': 'lossModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'lossmodel_operations.log',
            'log_filesize': 50*1024*1024,
            'log_backupCount': 5,
            'log_fileMode': 'a',
            # log_dir intentionally omitted — setup_logger uses _default_log_dir()
            'timestamp_backup_count': 10
        }
        
        self.lossModelLogger = setup_logger(lossmodel_log_config)
        
        if getProductionModeStatus():
            self.lossModelLogger.addFilter(lambda record: False)
        else:
            self.lossModelLogger.info("=== LossModel Operations Logger Initialized ===")

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

    def _apply_gumbel_softmax(self, datanode, temperature=None, hard=None):
        """
        Apply Gumbel-Softmax to softmax predictions in the datanode.
        
        Delegates to datanode.inferGumbelLocal() to avoid code duplication.
        
        Args:
            datanode: The datanode containing predictions
            temperature: Gumbel-Softmax temperature (defaults to self.temperature)
            hard: If True, use straight-through estimator (defaults to self.hard_gumbel)
        """
        temperature = temperature if temperature is not None else self.temperature
        hard = hard if hard is not None else self.hard_gumbel
        
        # Delegate to datanode's inferGumbelLocal method
        datanode.inferGumbelLocal(temperature=temperature, hard=hard)

    def forward(self, builder, build=None, use_gumbel=None, temperature=None, hard_gumbel=None):
        """
        Calculates the constraint loss based on the soft-logic translation.

        :param builder: DataNode builder instance.
        :param build: Whether to build the datanode.
        :param use_gumbel: Override instance use_gumbel setting.
        :param temperature: Override instance temperature setting.
        :param hard_gumbel: Override instance hard_gumbel setting.
        :returns: tuple of the constraint loss, a DataNode instance, and the DataNodeBuilder instance.
        """
        use_gumbel = use_gumbel if use_gumbel is not None else self.use_gumbel
        temperature = temperature if temperature is not None else self.temperature
        hard_gumbel = hard_gumbel if hard_gumbel is not None else self.hard_gumbel
        
        self.lossModelLogger.info("=== LossModel Forward Operation Started ===")
        self.lossModelLogger.info(f"Gumbel settings: use={use_gumbel}, temp={temperature}, hard={hard_gumbel}")
        
        if build is None:
            build = self.build
            
        if not build and not isinstance(builder, DataNodeBuilder):
            raise ValueError('PrimalDualModel must be invoked with `build` on or with provided DataNode Builder.')
        
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        
        # Apply Gumbel-Softmax if enabled
        if use_gumbel:
            self.lossModelLogger.info(f"Applying Gumbel-Softmax: temp={temperature}, hard={hard_gumbel}")
            datanode.inferLocal(keys=("softmax",))
            datanode.inferGumbelLocal(temperature=temperature, hard=hard_gumbel)
        
        constr_loss = datanode.calculateLcLoss(
            tnorm=self.tnorm,
            counting_tnorm=self.counting_tnorm, 
            sample=self.sample, 
            sampleSize=self.sampleSize
        )

        lmbd_loss = []
        if self.sampleGlobalLoss and constr_loss['globalLoss']:
            globalLoss = constr_loss['globalLoss']
            self.loss['globalLoss'](globalLoss)
            dtype = getattr(datanode, 'current_dtype', torch.float32)
            lmbd_loss = torch.tensor(globalLoss, dtype=dtype, requires_grad=True)
        else:
            for key, loss in constr_loss.items():
                if key not in self.constr:
                    continue
                
                if loss['lossTensor'] is not None:
                    loss_value = loss['lossTensor'].clamp(min=0)
                    loss_nansum = loss_value[loss_value == loss_value].sum()
                    loss_ = self.get_lmbd(key) * loss_nansum
                    self.loss[key](loss_)
                    lmbd_loss.append(loss_)
               
            lmbd_loss = sum(lmbd_loss)
        
        self.lossModelLogger.info(f"Total loss: {lmbd_loss.item() if hasattr(lmbd_loss, 'item') else lmbd_loss}")
        return lmbd_loss, datanode, builder

class PrimalDualModel(LossModel):
    """
    Class used to train from the constraint loss, calculated using the Primal Dual method.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, graph, tnorm='P', counting_tnorm=None, device='auto'):
        """
        The above function is the constructor for a class that initializes an object with a graph,
        tnorm, and device parameters.
        
        :param graph: The `graph` parameter is the input graph that the coding assistant is being
        initialized with. It represents the structure of the graph and can be used to perform various
        operations on the graph, such as adding or removing nodes and edges, calculating node and edge
        properties, and traversing the graph
        :param tnorm: The tnorm parameter is used to specify the type of t-norm to be used in the graph.
        A t-norm is a binary operation that generalizes the concept of conjunction (logical AND) to
        fuzzy logic. The 'P' value for tnorm indicates that the product t-norm should, defaults to P
        (optional)
        :param device: The `device` parameter specifies the device on which the computations will be
        performed. It can take the following values:, defaults to auto (optional)
        """
        super().__init__(graph, tnorm=tnorm, counting_tnorm = counting_tnorm, device=device)
        self._setup_primaldual_logger()

    def _setup_primaldual_logger(self):
        """Set up dedicated logger for PrimalDualModel operations."""
        primaldual_log_config = {
            'log_name': 'primalDualModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'primaldual_model_operations.log',
            'log_filesize': 50*1024*1024,
            'log_backupCount': 5,
            'log_fileMode': 'a',
            # log_dir intentionally omitted — setup_logger uses _default_log_dir()
            'timestamp_backup_count': 10
        }
        
        self.primalDualLogger = setup_logger(primaldual_log_config)
        
        if getProductionModeStatus():
            self.primalDualLogger.addFilter(lambda record: False)
        else:
            self.primalDualLogger.info("=== PrimalDualModel Operations Logger Initialized ===")

class InferenceModel(LossModel):
    """
    Class used to train from the program execution loss.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, graph,
                 tnorm='P',
                 loss=torch.nn.BCELoss,
                 counting_tnorm=None,
                 sample=False, sampleSize=0, sampleGlobalLoss=False, device='auto',
                 use_gumbel=False, temperature=1.0, hard_gumbel=False,
                 pos_weight=1.0):
        """
        Initializes an instance of InferenceModel.

        :param graph: The initialized graph either containing the logical expressions to be executed
            and/or called with `.compile_executable` to use the logical expressions in the dataset.
        :param tnorm: Sets the method used to perform the soft-logic translation of the logical expressions.
            Defaults to 'P' (Product).
        :param loss: Loss function to use for binary program outputs.
        :counting_tnorm: Sets the method used to perform the soft-logic translation of the counting logical
            expressions. If set to None, uses the same method as `tnorm`. Defaults to None.
        :param sample: The `sample` parameter is a boolean flag that determines whether to use sampling
        during training. If set to `True`, the model will use sampling to estimate the loss function. If
        set to `False`, the model will not use sampling and will use the exact loss function, defaults
        to False (optional)
        :param sampleSize: The `sampleSize` parameter determines the size of the sample used for
        training. It specifies the number of samples that will be randomly selected from the dataset for
        each training iteration, defaults to 0 (optional)
        :param sampleGlobalLoss: The parameter `sampleGlobalLoss` is a boolean flag that determines
        whether to sample the global loss during training. If `sampleGlobalLoss` is set to `True`, the
        global loss will be sampled. Otherwise, it will not be sampled, defaults to False (optional)
        :param device: The `device` parameter specifies the device (CPU or GPU) on which the model will
        be trained and evaluated. It can take the following values:, defaults to auto (optional)
        """
        self.graph = graph

        super().__init__(graph, tnorm=tnorm, counting_tnorm=counting_tnorm, 
                         sample=sample, sampleSize=sampleSize, 
                         sampleGlobalLoss=sampleGlobalLoss, device=device,
                         use_gumbel=use_gumbel, temperature=temperature, 
                         hard_gumbel=hard_gumbel)

        self.loss_func = loss()
        # pos_weight rebalances BCE against majority-class collapse on existsL
        # constraints. When the dataset's logic_label has a skewed Yes/No ratio
        # the unweighted BCE will drift toward the majority direction — setting
        # pos_weight > 1 up-weights the Yes (label=1) loss contribution.
        self.pos_weight = float(pos_weight)
        # Diagnostic: set DOMIKNOWS_INFER_DIAG=<N> to print (lbl, conversionSigmoid, loss)
        # for the first N forward calls. Used to trace gradient-sign inversions.
        import os
        self._diag_budget = int(os.environ.get('DOMIKNOWS_INFER_DIAG', '0'))
        self._diag_step = 0
        self._setup_inference_logger()

    def _setup_inference_logger(self):
        """Set up dedicated logger for InferenceModel operations."""
        inference_log_config = {
            'log_name': 'inferenceModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'inference_model_operations.log',
            'log_filesize': 50*1024*1024,
            'log_backupCount': 5,
            'log_fileMode': 'a',
            # log_dir intentionally omitted — setup_logger uses _default_log_dir()
            'timestamp_backup_count': 10
        }
        
        self.inferenceLogger = setup_logger(inference_log_config)
        
        if getProductionModeStatus():
            self.inferenceLogger.addFilter(lambda record: False)
        else:
            self.inferenceLogger.info("=== InferenceModel Operations Logger Initialized ===")

    def forward(self, builder, build=None, use_gumbel=None, temperature=None, hard_gumbel=None):
        use_gumbel = use_gumbel if use_gumbel is not None else self.use_gumbel
        temperature = temperature if temperature is not None else self.temperature
        hard_gumbel = hard_gumbel if hard_gumbel is not None else self.hard_gumbel
        
        self.inferenceLogger.info("=== InferenceModel Forward Operation Started ===")
        self.inferenceLogger.info(f"Gumbel settings: use={use_gumbel}, temp={temperature}, hard={hard_gumbel}")
        
        if build is None:
            build = self.build
            
        if not build and not isinstance(builder, DataNodeBuilder):
            raise ValueError('InferenceModel must be invoked with `build` on or with provided DataNode Builder.')
        
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        dtype = getattr(datanode, 'current_dtype', torch.float32)

        if use_gumbel:
            self.inferenceLogger.info(f"Applying Gumbel-Softmax: temp={temperature}, hard={hard_gumbel}")
            datanode.inferLocal(keys=("softmax",))
            datanode.inferGumbelLocal(temperature=temperature, hard=hard_gumbel)

        # read executable constraint labels from datanode
        read_labels = datanode.getExecutableConstraintLabels()
        if len(read_labels) == 0:
            raise ValueError('No active executable constraint labels found in datanode.')

        # Prepare shared context for loss calculation
        lc_context = datanode._prepareLcLossContext(
            tnorm=self.tnorm,
            counting_tnorm=self.counting_tnorm,
        )

        losses = []
        for lcName, lc in self.constr.items():
            if f'{lcName}/label' not in read_labels:
                continue
            
            if not lc.active:
                continue

            # Use datanode method to get the label
            lbl = datanode.getExecutableConstraintLabel(lcName).float()
            
            loss_dict = datanode.calculateSingleLcLoss(
                lcName,
                tnorm=self.tnorm,
                counting_tnorm=self.counting_tnorm,
                _context=lc_context
            )

            if loss_dict.get('loss') is None:
                continue
                
            if MONITORING_AVAILABLE:
                lcRepr = f'{lc.__class__.__name__} {lc.strEs()}'
                log_single_lc(
                    constraint_name=lcName,
                    loss_dict=loss_dict,
                    label_tensor=lbl,
                    lc_formulation=lcRepr
                )
                
            is_sumL = isinstance(lc, sumL)
            if is_sumL:
                lbl = torch.tensor(1.0, dtype=dtype, device=self.device, requires_grad=True)
                
            constr_out = loss_dict['conversionSigmoid']
            #if torch.equal(constr_out, lbl):
            #    print(f"Constraint {lcName}: loss={constr_out}, label={lbl}" + (f", is_sumL={is_sumL}" if is_sumL else ""))
            constraint_loss = self.loss_func(constr_out.float(), lbl)

            if self._diag_step < self._diag_budget:
                try:
                    co = constr_out.detach().float().flatten()
                    lb = lbl.detach().float().flatten()
                    cl = constraint_loss.detach().float().flatten()
                    print(
                        f"[INFER_DIAG step={self._diag_step} lc={lcName}] "
                        f"convSig={co.tolist()} lbl={lb.tolist()} "
                        f"loss={cl.tolist()} is_sumL={is_sumL}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[INFER_DIAG error] {e}", flush=True)

            # Up-weight the positive (label=1) class if pos_weight != 1.
            # BCELoss has no pos_weight param (unlike BCEWithLogitsLoss), so we
            # scale the already-computed loss by the per-sample weight.
            if self.pos_weight != 1.0:
                lbl_scalar = lbl.float().mean()  # lbl is 0-d or 1-d singleton here
                sample_weight = (self.pos_weight - 1.0) * lbl_scalar + 1.0
                constraint_loss = constraint_loss * sample_weight

            losses.append(constraint_loss)

        if len(losses) == 0:
            dtype = getattr(datanode, 'current_dtype', torch.float32)
            loss = torch.tensor(0.0, dtype=dtype, device=self.device, requires_grad=True)
        else:
            loss = sum(losses)
            
        if MONITORING_AVAILABLE:
            log_memory() 
        
        self.inferenceLogger.info(f"Total loss: {loss.item()}")

        if self._diag_step < self._diag_budget:
            try:
                concept_names = []
                for c in getattr(self.graph, 'concepts', {}):
                    concept_names.append(c)
                for cname in concept_names[:3]:
                    for dn in datanode.findDatanodes(select=cname):
                        sm = dn.getAttribute(cname, 'local/softmax')
                        if sm is None:
                            continue
                        sm_t = sm.detach().float().flatten().tolist()
                        print(
                            f"[INFER_DIAG step={self._diag_step} concept={cname}] "
                            f"softmax={sm_t[:4]} (col0=False, col1=True)",
                            flush=True,
                        )
                        break
            except Exception as e:
                print(f"[INFER_DIAG concept error] {e}", flush=True)
            self._diag_step += 1

        return loss, datanode, builder
    
class SampleLossModel(LossModel):
    """
    Class used to train from the constraint loss, calculated using sampling.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, graph, 
                 tnorm='P', 
                 counting_tnorm=None,
                 sample=False, sampleSize=0, sampleGlobalLoss=False, device='auto',
                 use_gumbel=False, temperature=1.0, hard_gumbel=False,
                 temperature_schedule='constant', min_temperature=0.5, anneal_rate=0.0003):
        
        super().__init__(
            graph=graph,
            tnorm=tnorm,
            counting_tnorm=counting_tnorm,
            sample=sample,
            sampleSize=sampleSize,
            sampleGlobalLoss=sampleGlobalLoss,
            device=device,
            use_gumbel=use_gumbel,
            temperature=temperature,
            hard_gumbel=hard_gumbel
        )
        
        # SampleLossModel-specific: temperature annealing
        self.initial_temperature = temperature
        self.temperature_schedule = temperature_schedule
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        self._step_count = 0
        
        # SampleLossModel-specific: iteration tracking
        self.iter_step = 0
        self.warmup = 80
        
        self._setup_sampleloss_logger()

    def _setup_sampleloss_logger(self):
        """Set up dedicated logger for SampleLossModel operations."""
        sampleloss_log_config = {
            'log_name': 'sampleLossModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'sampleloss_model_operations.log',
            'log_filesize': 50*1024*1024,
            'log_backupCount': 5,
            'log_fileMode': 'a',
            # log_dir intentionally omitted — setup_logger uses _default_log_dir()
            'timestamp_backup_count': 10
        }
        
        self.sampleLossLogger = setup_logger(sampleloss_log_config)
        
        if getProductionModeStatus():
            self.sampleLossLogger.addFilter(lambda record: False)
        else:
            self.sampleLossLogger.info("=== SampleLossModel Operations Logger Initialized ===")

    def reset_parameters(self):
        """Override: Initialize lambda to 0.0 instead of 1.0."""
        torch.nn.init.constant_(self.lmbd, 0.0)

    def get_lmbd(self, key):
        """Override: Clamp to min 0 instead of max lmbd_p."""
        if self.lmbd[self.lmbd_index[key]] < 0:
            with torch.no_grad():
                self.lmbd[self.lmbd_index[key]] = 0
        return self.lmbd[self.lmbd_index[key]]
    
    def set_temperature(self, temperature):
        """Update Gumbel-Softmax temperature."""
        self.temperature = max(temperature, self.min_temperature)
    
    def anneal_temperature(self):
        """Anneal temperature according to schedule."""
        if self.temperature_schedule == 'constant':
            return
        
        self._step_count += 1
        
        if self.temperature_schedule == 'exponential':
            new_temp = self.initial_temperature * np.exp(-self.anneal_rate * self._step_count)
        elif self.temperature_schedule == 'linear':
            new_temp = self.initial_temperature - self.anneal_rate * self._step_count
        else:
            new_temp = self.temperature
        
        self.temperature = max(new_temp, self.min_temperature)
    
    def reset_temperature(self):
        """Reset temperature to initial value."""
        self.temperature = self.initial_temperature
        self._step_count = 0

    def forward(self, builder, build=None, use_gumbel=None, temperature=None, hard_gumbel=None):
        """
        Forward pass with sampling-based loss calculation.
        """
        use_gumbel = use_gumbel if use_gumbel is not None else self.use_gumbel
        temperature = temperature if temperature is not None else self.temperature
        hard_gumbel = hard_gumbel if hard_gumbel is not None else self.hard_gumbel
        
        self.sampleLossLogger.info("=== SampleLossModel Forward Operation Started ===")
        self.sampleLossLogger.info(f"Iteration step: {self.iter_step}")
        self.sampleLossLogger.info(f"Gumbel settings: use={use_gumbel}, temp={temperature}, hard={hard_gumbel}")
        
        if build is None:
            build = self.build
        self.iter_step += 1
            
        if not build and not isinstance(builder, DataNodeBuilder):
            raise ValueError('SampleLossModel must be invoked with `build` on or with provided DataNode Builder.')
        
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        
        # Apply Gumbel-Softmax if enabled using datanode's method
        if use_gumbel:
            if self.training and temperature == self.temperature:
                self.anneal_temperature()
                temperature = self.temperature
            
            self.sampleLossLogger.info(f"Applying Gumbel-Softmax: temp={temperature}, hard={hard_gumbel}")
            datanode.inferLocal(keys=("softmax",))
            datanode.inferGumbelLocal(temperature=temperature, hard=hard_gumbel)
        
        # Calculate LC loss
        constr_loss = datanode.calculateLcLoss(
            tnorm=self.tnorm, 
            sample=self.sample, 
            sampleSize=self.sampleSize, 
            sampleGlobalLoss=self.sampleGlobalLoss
        )
        
        lmbd_loss = []
        replace_mul = False
        key_losses = dict()
        
        for key, loss in constr_loss.items():
            if key not in self.constr:
                continue
            key_loss = 0
            
            for i, lossTensor in enumerate(loss['lossTensor']):
                lcSuccesses = loss['lcSuccesses'][i]
                
                if self.sampleSize == -1:
                    sample_info = [val_ for k, val in loss['sampleInfo'].items() for val_ in val if len(val_)]
                    sample_info = [val[i][1] for val in sample_info]
                    sample_info = torch.stack(sample_info).t()
                    unique_output, unique_inverse, counts = torch.unique(
                        sample_info, return_inverse=True, dim=0, return_counts=True
                    )
                    _, ind_sorted = torch.sort(unique_inverse, stable=True)
                    cum_sum = counts.cumsum(0)
                    cum_sum = torch.cat((torch.tensor([0]).to(counts.device), cum_sum[:-1]))
                    first_indicies = ind_sorted[cum_sum]
                    assert lcSuccesses.sum().item() != 0
                    tidx = (lcSuccesses == 1).nonzero().squeeze(-1)
                    unique_selected_indexes = torch.tensor(
                        np.intersect1d(first_indicies.cpu().numpy(), tidx.cpu().numpy())
                    )
                    if unique_selected_indexes.shape:
                        loss_value = lossTensor[unique_selected_indexes].sum()
                        loss_ = -1 * torch.log(loss_value)
                        key_loss += loss_
                else:
                    if constr_loss["globalSuccessCounter"] > 0:
                        lcSuccesses = constr_loss["globalSuccesses"]
                    if lossTensor.sum().item() != 0:
                        tidx = (lcSuccesses == 1).nonzero().squeeze(-1)
                        true_val = lossTensor[tidx]
                        
                        if true_val.sum().item() != 0: 
                            if not replace_mul:
                                loss_value = true_val.sum() / lossTensor.sum()
                                loss_value = -(-1 * torch.log(loss_value))
                                if self.iter_step < self.warmup:
                                    with torch.no_grad():
                                        min_val = loss_value
                                else:
                                    min_val = -1
                                loss_ = min_val * loss_value
                                key_loss += loss_
                            else:
                                loss_value = true_val.logsumexp(dim=0) - lossTensor.logsumexp(dim=0)
                                key_loss += -1 * loss_value

            epsilon = 1e-2
            if self.sampleSize != -1:
                key_loss = max(key_loss - epsilon, 0) 
            if key_loss != 0:  
                key_losses[key] = key_loss
                    
        all_losses = [key_losses[key] for key in key_losses]
        if all_losses:
            all_losses = torch.stack(all_losses)
            
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
        else:
            lmbd_loss = 0
        
        self.sampleLossLogger.info(f"Total loss: {lmbd_loss.item() if hasattr(lmbd_loss, 'item') else lmbd_loss}")
        return lmbd_loss, datanode, builder