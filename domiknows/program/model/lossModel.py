import logging
import warnings
from collections import OrderedDict

import numpy as np
import torch

from ...graph import DataNodeBuilder
from ..metric import MetricTracker, MacroAverageTracker
from domiknows import setup_logger, getProductionModeStatus

try:
    from monitor.constraint_monitor import ( # type: ignore
       next_step, log_single_lc, log_memory
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class LossModel(torch.nn.Module):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, 
                 tnorm='P',
                 counting_tnorm=None,
                 sample = False, sampleSize = 0, sampleGlobalLoss = False, device='auto'):
        """
        This function initializes a LossModel object with the given parameters and sets up the
        necessary variables and constraints.
        
        :param graph: The `graph` parameter is an object that represents the logical constraints of a
        graph. It contains information about the nodes, edges, and constraints of the graph
        :param tnorm: The `tnorm` parameter specifies the type of t-norm to be used in the model.
        T-norms are a family of binary operations that are used to model logical conjunction (AND) in
        fuzzy logic. The default value is 'P', which stands for the product t-norm, defaults to P
        (optional)
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
        super().__init__()
        self.graph = graph
        self.build = True
        
        self.tnorm = tnorm
        self.counting_tnorm = counting_tnorm
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
        
        # Set up dedicated logger for LossModel operations
        self._setup_lossmodel_logger()

    def _setup_lossmodel_logger(self):
        """Set up dedicated logger for LossModel operations."""
        lossmodel_log_config = {
            'log_name': 'lossModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'lossmodel_operations.log',
            'log_filesize': 50*1024*1024,  # 50MB
            'log_backupCount': 5,
            'log_fileMode': 'a',
            'log_dir': 'logs',
            'timestamp_backup_count': 10
        }
        
        self.lossModelLogger = setup_logger(lossmodel_log_config)
        
        # Disable logger if in production mode
        if getProductionModeStatus():
            self.lossModelLogger.addFilter(lambda record: False)
            self.lossModelLogger.info("LossModel logger disabled due to production mode")
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
        """
        The function `get_lmbd` returns a clamped value from a dictionary based on a given key.
        
        :param key: The key parameter is used to access a specific value in the lmbd dictionary
        :return: the value of `self.lmbd[self.lmbd_index[key]]` after clamping it to a maximum value of
        `self.lmbd_p[self.lmbd_index[key]]`.
        """
        return self.lmbd[self.lmbd_index[key]].clamp(max=self.lmbd_p[self.lmbd_index[key]])

    def forward(self, builder, build=None):
        self.lossModelLogger.info("=== LossModel Forward Operation Started ===")
        self.lossModelLogger.info(f"Parameters: build={build}, sample={self.sample}, sampleSize={self.sampleSize}")
        self.lossModelLogger.info(f"T-norm: {self.tnorm}, counting_tnorm: {self.counting_tnorm}")
        self.lossModelLogger.info(f"Device: {self.device}")
        
        if build is None:
            build = self.build
            self.lossModelLogger.debug(f"Using default build value: {build}")
            
        if not build and not isinstance(builder, DataNodeBuilder):
            self.lossModelLogger.error("PrimalDualModel must be invoked with `build` on or with provided DataNode Builder")
            raise ValueError('PrimalDualModel must be invoked with `build` on or with provided DataNode Builder.')
        
        self.lossModelLogger.debug("Creating batch root data node")
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        self.lossModelLogger.info(f"DataNode created on device: {datanode.device if hasattr(datanode, 'device') else 'unknown'}")
        
        # Call the loss calculation returns a dictionary, keys are matching the constraints
        self.lossModelLogger.info("Calculating LC loss...")
        constr_loss = datanode.calculateLcLoss(tnorm=self.tnorm,counting_tnorm=self.counting_tnorm, sample=self.sample, sampleSize = self.sampleSize)
        self.lossModelLogger.info(f"Constraint loss keys: {list(constr_loss.keys())}")

        lmbd_loss = []
        if self.sampleGlobalLoss and constr_loss['globalLoss']:
            globalLoss = constr_loss['globalLoss']
            self.lossModelLogger.info(f"Using global loss: {globalLoss}")
            self.loss['globalLoss'](globalLoss)
            lmbd_loss = torch.tensor(globalLoss, requires_grad=True)
        else:
            self.lossModelLogger.debug("Processing individual constraint losses")
            for key, loss in constr_loss.items():
                if key not in self.constr:
                    self.lossModelLogger.debug(f"Skipping key '{key}' (not in constraints)")
                    continue
                
                if loss['lossTensor'] != None:
                    loss_value = loss['lossTensor'].clamp(min=0)
                    loss_nansum = loss_value[loss_value==loss_value].sum()
                    loss_ = self.get_lmbd(key) * loss_nansum
                    self.lossModelLogger.debug(f"Constraint '{key}': loss_nansum={loss_nansum.item() if hasattr(loss_nansum, 'item') else loss_nansum}, lambda={self.get_lmbd(key).item()}, weighted_loss={loss_.item() if hasattr(loss_, 'item') else loss_}")
                    self.loss[key](loss_)
                    lmbd_loss.append(loss_)
                else:
                    self.lossModelLogger.debug(f"Constraint '{key}': lossTensor is None")
               
            lmbd_loss = sum(lmbd_loss)
            self.lossModelLogger.info(f"Total lambda loss: {lmbd_loss.item() if hasattr(lmbd_loss, 'item') else lmbd_loss}")
        
        self.lossModelLogger.info("=== LossModel Forward Operation Completed ===\n")
        # (*out, datanode, builder)
        return lmbd_loss, datanode, builder


# The `PrimalDualModel` class is a subclass of `LossModel` that implements a primal-dual optimization
# algorithm.
class PrimalDualModel(LossModel):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, tnorm='P',counting_tnorm=None, device='auto'):
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
        
        # Set up dedicated logger for PrimalDualModel operations
        self._setup_primaldual_logger()

    def _setup_primaldual_logger(self):
        """Set up dedicated logger for PrimalDualModel operations."""
        primaldual_log_config = {
            'log_name': 'primalDualModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'primaldual_model_operations.log',
            'log_filesize': 50*1024*1024,  # 50MB
            'log_backupCount': 5,
            'log_fileMode': 'a',
            'log_dir': 'logs',
            'timestamp_backup_count': 10
        }
        
        self.primalDualLogger = setup_logger(primaldual_log_config)
        
        # Disable logger if in production mode
        if getProductionModeStatus():
            self.primalDualLogger.addFilter(lambda record: False)
            self.primalDualLogger.info("PrimalDualModel logger disabled due to production mode")
        else:
            self.primalDualLogger.info("=== PrimalDualModel Operations Logger Initialized ===")


class InferenceModel(LossModel):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, 
                 tnorm='P',
                 loss=torch.nn.BCELoss,
                 counting_tnorm=None,
                 sample = False, sampleSize = 0, sampleGlobalLoss = False, device='auto'):

        # The concept where all the labels for the constraints are stored as properties
        self.constraint_concept = graph.get_constraint_concept()

        self.graph = graph

        super().__init__(graph, tnorm=tnorm, counting_tnorm=counting_tnorm, sample=sample, sampleSize=sampleSize, sampleGlobalLoss=sampleGlobalLoss, device=device)

        # Initialize loss function (needs to be after module initialization)
        self.loss_func = loss()
        
        # Set up dedicated logger for InferenceModel operations
        self._setup_inference_logger()

    def _setup_inference_logger(self):
        """Set up dedicated logger for InferenceModel operations."""
        inference_log_config = {
            'log_name': 'inferenceModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'inference_model_operations.log',
            'log_filesize': 50*1024*1024,  # 50MB
            'log_backupCount': 5,
            'log_fileMode': 'a',
            'log_dir': 'logs',
            'timestamp_backup_count': 10
        }
        
        self.inferenceLogger = setup_logger(inference_log_config)
        
        # Disable logger if in production mode
        if getProductionModeStatus():
            self.inferenceLogger.addFilter(lambda record: False)
            self.inferenceLogger.info("InferenceModel logger disabled due to production mode")
        else:
            self.inferenceLogger.info("=== InferenceModel Operations Logger Initialized ===")

    def forward(self, builder, build=None):
        self.inferenceLogger.info("=== InferenceModel Forward Operation Started ===")
        
        if MONITORING_AVAILABLE:
            next_step()
            self.inferenceLogger.debug("Monitoring next_step() called")
            
        if build is None:
            build = self.build
            self.inferenceLogger.debug(f"Using default build value: {build}")
            
        if not build and not isinstance(builder, DataNodeBuilder):
            self.inferenceLogger.error("InferenceModel must be invoked with `build` on or with provided DataNode Builder")
            raise ValueError('InferenceModel must be invoked with `build` on or with provided DataNode Builder.')
        
        self.inferenceLogger.debug("Creating batch root data node")
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)

        # Try to get the datanode for the constraints concept
        constraint_dn_search = builder.findDataNodesInBuilder(select=self.constraint_concept.name)
        if len(constraint_dn_search) == 0:
            self.inferenceLogger.error(f"Constraint datanode (for concept {self.constraint_concept.name}) not found")
            raise ValueError(f'Constraint datanode (for concept {self.constraint_concept.name}) not found.')
        elif len(constraint_dn_search) > 1:
            self.inferenceLogger.error(f"Multiple constraint datanodes found: {len(constraint_dn_search)}, expected one")
            raise ValueError(f'Multiple constraint datanodes (for concept {self.constraint_concept.name}) found: found {len(constraint_datanode)}, expected one.')

        constraint_datanode = constraint_dn_search[0]
        self.inferenceLogger.info(f"Found constraint datanode: {constraint_datanode}")

        # Get the constraint labels
        # read_labels will be of format: {'LC{n}/label': label_value}
        read_labels = constraint_datanode.getAttributes()
        
        datanode.setActiveLCs()
                
        # Call the loss calculation returns a dictionary, keys are matching the constraints
        # Has the format {'LC{n}': {'lossTensor': tensor, ...}
        self.inferenceLogger.info("Calculating LC loss...")
        constr_loss = datanode.calculateLcLoss(tnorm=self.tnorm,counting_tnorm=self.counting_tnorm, sample=self.sample, sampleSize = self.sampleSize)
        self.inferenceLogger.info(f"Constraint loss keys: {list(constr_loss.keys())}")

        # print('retrieved labels:', read_labels)

        # Compile losses
        losses = []
        for i, (lcName, loss_dict) in enumerate(constr_loss.items()):
            if lcName not in self.constr:
                self.inferenceLogger.debug(f"Skipping constraint '{lcName}' (not in self.constr)")
                continue
            
            lc = self.graph.logicalConstrains[lcName]
            lcRepr = f'{lc.__class__.__name__} {lc.strEs()}'
            self.inferenceLogger.debug(f"Processing constraint '{lcName}' ({i+1}/{len(constr_loss)}) with representation: {lcRepr}")
                      
            # Get the t-norm translated output of the constraint
            constr_out = loss_dict['conversionSigmoid']
            self.inferenceLogger.debug(f"Constraint '{lcName}' conversion (succes) output shape: {constr_out.shape}: {constr_out}")

            # Target for for constraint lcName
            lbl = read_labels[f'{lcName}/label'].float().unsqueeze(0)
            lbl = lbl.squeeze() # remove singleton dimension if present
            self.inferenceLogger.debug(f"Constraint '{lcName}' label shape: {lbl.shape}: {lbl}")

            if MONITORING_AVAILABLE:
                log_single_lc(
                    constraint_name=lcName,
                    loss_dict=loss_dict,
                    label_tensor=lbl,
                    lc_formulation=lcRepr
                )

            # Calcluate loss 
            constraint_loss = self.loss_func(constr_out.float(), lbl)
            self.inferenceLogger.debug(f"Constraint '{lcName}' calculated loss with function {self.loss_func.__class__.__name__}: {constraint_loss.item()}")
            losses.append(constraint_loss) # TODO: match dtypes too?

        loss_scalar = sum(losses)
        self.inferenceLogger.info(f"Total inference loss: {loss_scalar.item() if hasattr(loss_scalar, 'item') else loss_scalar}")
        
        if MONITORING_AVAILABLE:
            log_memory() 

        self.inferenceLogger.info("=== InferenceModel Forward Operation Completed ===\n")
        # (*out, datanode, builder)
        return loss_scalar, datanode, builder

class SampleLossModel(torch.nn.Module):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, 
                 tnorm='P', 
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
        
        # Set up dedicated logger for SampleLossModel operations
        self._setup_sampleloss_logger()

    def _setup_sampleloss_logger(self):
        """Set up dedicated logger for SampleLossModel operations."""
        sampleloss_log_config = {
            'log_name': 'sampleLossModelOperations',
            'log_level': logging.DEBUG,
            'log_filename': 'sampleloss_model_operations.log',
            'log_filesize': 50*1024*1024,  # 50MB
            'log_backupCount': 5,
            'log_fileMode': 'a',
            'log_dir': 'logs',
            'timestamp_backup_count': 10
        }
        
        self.sampleLossLogger = setup_logger(sampleloss_log_config)
        
        # Disable logger if in production mode
        if getProductionModeStatus():
            self.sampleLossLogger.addFilter(lambda record: False)
            self.sampleLossLogger.info("SampleLossModel logger disabled due to production mode")
        else:
            self.sampleLossLogger.info("=== SampleLossModel Operations Logger Initialized ===")

    def reset_parameters(self):
        torch.nn.init.constant_(self.lmbd, 0.0)

    def reset(self):
        if isinstance(self.loss, MetricTracker):
            self.loss.reset()

    def get_lmbd(self, key):
        """
        The function `get_lmbd` returns the value of `self.lmbd` at the index specified by
        `self.lmbd_index[key]`, ensuring that the value is non-negative.
        
        :param key: The `key` parameter is used to access a specific element in the `self.lmbd` list. It
        is used as an index to retrieve the corresponding value from the list
        :return: the value of `self.lmbd[self.lmbd_index[key]]`.
        """
        if self.lmbd[self.lmbd_index[key]] < 0:
            with torch.no_grad():
                self.lmbd[self.lmbd_index[key]] = 0
        return self.lmbd[self.lmbd_index[key]]

    def forward(self, builder, build=None):
        """
        The `forward` function calculates the loss for a PrimalDualModel using a DataNodeBuilder and
        returns the loss value, the DataNode, and the builder.
        
        :param builder: The `builder` parameter is an instance of the `DataNodeBuilder` class. It is
        used to create a batch root data node and retrieve a data node
        :param build: The `build` parameter is an optional argument that specifies whether the
        `DataNodeBuilder` should be invoked or not. If `build` is `None`, then the value of `self.build`
        is used. If `build` is `True`, then the `createBatchRootDN()` method
        :return: three values: lmbd_loss, datanode, and builder.
        """
        self.sampleLossLogger.info("=== SampleLossModel Forward Operation Started ===")
        self.sampleLossLogger.info(f"Iteration step: {self.iter_step}")
        self.sampleLossLogger.info(f"Parameters: build={build}, sample={self.sample}, sampleSize={self.sampleSize}")
        self.sampleLossLogger.info(f"Warmup threshold: {self.warmpup}")
        
        if build is None:
            build = self.build
            self.sampleLossLogger.debug(f"Using default build value: {build}")
        self.iter_step += 1
            
        if not build and not isinstance(builder, DataNodeBuilder):
            self.sampleLossLogger.error("PrimalDualModel must be invoked with `build` on or with provided DataNode Builder")
            raise ValueError('PrimalDualModel must be invoked with `build` on or with provided DataNode Builder.')
        
        self.sampleLossLogger.debug("Creating batch root data node")
        builder.createBatchRootDN()

#       self.loss.reset()

        datanode = builder.getDataNode(device=self.device)
        
        # Call the loss calculation returns a dictionary, keys are matching the constraints
        self.sampleLossLogger.info("Calculating LC loss...")
        constr_loss = datanode.calculateLcLoss(tnorm=self.tnorm, sample=self.sample, sampleSize = self.sampleSize, sampleGlobalLoss = self.sampleGlobalLoss)
        self.sampleLossLogger.info(f"Constraint loss keys: {list(constr_loss.keys())}")
        
        import math
        lmbd_loss = []
        replace_mul = False
        
        key_losses = dict()
        for key, loss in constr_loss.items():
            if key not in self.constr:
                self.sampleLossLogger.debug(f"Skipping key '{key}' (not in constraints)")
                continue
            # loss_value = loss['loss']
            epsilon = 0.0
            key_loss = 0
            new_eps = 0.01
            self.sampleLossLogger.debug(f"Processing constraint '{key}' with {len(loss['lossTensor'])} loss tensors")
            
            for i, lossTensor in enumerate(loss['lossTensor']):
                lcSuccesses = loss['lcSuccesses'][i]
                self.sampleLossLogger.debug(f"Constraint '{key}' tensor {i}: lossTensor sum={lossTensor.sum().item()}, lcSuccesses sum={lcSuccesses.sum().item()}")
                
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
                        self.sampleLossLogger.debug(f"Constraint '{key}' tensor {i}: unique selected, loss_value={loss_value.item()}, loss_={loss_.item()}")

                    else:
                        loss_ = 0
                        self.sampleLossLogger.debug(f"Constraint '{key}' tensor {i}: no unique selected indexes")
                    
                else:
                    if constr_loss["globalSuccessCounter"] > 0:
                        lcSuccesses = constr_loss["globalSuccesses"]
                        self.sampleLossLogger.debug(f"Using global successes: {lcSuccesses.sum().item()}")
                    if lossTensor.sum().item() != 0:
                        tidx = (lcSuccesses == 1).nonzero().squeeze(-1)
                        true_val = lossTensor[tidx]
                        self.sampleLossLogger.debug(f"Constraint '{key}' tensor {i}: true_val sum={true_val.sum().item()}")
                        
                        if true_val.sum().item() != 0: 
                            if not replace_mul:
                                loss_value = true_val.sum() / lossTensor.sum()
                                loss_value = epsilon - ( -1 * torch.log(loss_value) )
                                if self.iter_step < self.warmpup:
                                    with torch.no_grad():
                                        min_val = loss_value
                                    self.sampleLossLogger.debug(f"Constraint '{key}' tensor {i}: warmup phase, min_val set to loss_value")
                                else:
                                    min_val = -1
                                    self.sampleLossLogger.debug(f"Constraint '{key}' tensor {i}: post-warmup phase, min_val=-1")
                                # min_val = -1
                                # with torch.no_grad():
                                #     min_val = loss_value
                                loss_ = min_val * loss_value
                                key_loss += loss_
                                self.sampleLossLogger.debug(f"Constraint '{key}' tensor {i}: loss_value={loss_value.item()}, min_val={min_val}, loss_={loss_.item()}")
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
                self.sampleLossLogger.debug(f"Stacked all_losses tensor: {all_losses}")

                satisfied_num = len( set(constr_loss.keys()) - set(key_losses.keys()) )
                unsatisfied_num = len(set(constr_loss.keys())) - satisfied_num
                self.sampleLossLogger.info(f"Number of satisfied constraints: {satisfied_num}")
                self.sampleLossLogger.info(f"Number of unsatisfied constraints: {unsatisfied_num}")
                
                for key in key_losses:
                    if self.sampleSize != -1:
                        if replace_mul:
                            loss_val = (key_losses[key] / all_losses.sum()) * key_losses[key]
                            self.sampleLossLogger.debug(f"Constraint '{key}': replace_mul mode, loss_val={loss_val.item()}")
                        else:
                            loss_val = key_losses[key]
                            self.sampleLossLogger.debug(f"Constraint '{key}': standard mode, loss_val={loss_val.item()}")
                    else:
                        loss_val = key_losses[key]
                        self.sampleLossLogger.debug(f"Constraint '{key}': sampleSize=-1 mode, loss_val={loss_val.item()}")

                    self.loss[key](loss_val)
                    lmbd_loss.append(loss_val) 
                    
                lmbd_loss = sum(lmbd_loss)
                self.sampleLossLogger.info(f"Final lambda loss: {lmbd_loss.item()}")
        else:
            self.sampleLossLogger.info("No losses calculated - all constraints satisfied or no valid losses")
            lmbd_loss = 0
            self.sampleLossLogger.info(f"Lambda loss set to: {lmbd_loss}")
        
        self.sampleLossLogger.info("=== SampleLossModel Forward Operation Completed ===")
        self.sampleLossLogger.debug(f"Returning: lmbd_loss, datanode, builder")
        
        return lmbd_loss, datanode, builder
