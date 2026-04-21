import logging
from unittest import result
import torch
import numpy as np

from .program import LearningBasedProgram, get_len
from ..utils import consume, setup_logger
from tqdm import tqdm

from .model.lossModel import PrimalDualModel, SampleLossModel, InferenceModel
from .model.base import Mode
from .model.gbi import GBIModel

logger = setup_logger({
    'log_name': 'lossProgram',
    'log_level': logging.INFO,
    'log_filename': 'lossProgram.log',
    'log_filesize': 50*1024*1024,  # 50MB
    'log_backupCount': 5,
    'log_fileMode': 'a',
})


# =============================================================================
# Gradient Utilities for Primal-Dual
# =============================================================================

def reverse_sign_grad(parameters, factor=-1.):
    """Reverse gradient sign for dual step (gradient ascent)."""
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad = factor * parameter.grad

# =============================================================================
# Evaluation Helpers  
# =============================================================================

def _apply_threshold_to_predictions(program, datanode, threshold=0.5):
    """Apply threshold to convert probabilities to binary decisions."""
    from domiknows.utils import getDnSkeletonMode
    
    conceptsRelations = datanode.collectConceptsAndRelations()
    
    for c in conceptsRelations:
        cRoot = datanode.findRootConceptOrRelation(c[0])
        
        if getDnSkeletonMode() and "variableSet" in datanode.attributes:
            vKeyInVariableSet = cRoot.name + "/<" + c[0].name + ">"
            localSoftmaxKey = vKeyInVariableSet + "/local/softmax"
            localDecisionKey = vKeyInVariableSet + "/local/decision"
            
            if localSoftmaxKey in datanode.attributes.get("variableSet", {}):
                softmax_probs = datanode.attributes["variableSet"][localSoftmaxKey]
                if softmax_probs is not None and torch.is_tensor(softmax_probs):
                    thresholded = (softmax_probs >= threshold).float()
                    datanode.attributes["variableSet"][localDecisionKey] = thresholded
        else:
            dns = datanode.findDatanodes(select=cRoot)
            if not dns:
                continue
            
            keySoftmax = "<" + c[0].name + ">/local/softmax"
            keyDecision = "<" + c[0].name + ">/local/decision"
            
            for dn in dns:
                softmax_probs = dn.getAttribute(keySoftmax)
                if softmax_probs is not None and torch.is_tensor(softmax_probs):
                    thresholded = (softmax_probs >= threshold).float()
                    dn.attributes[keyDecision] = thresholded

def _evaluate_condition_impl(program, evaluate_data, device="cpu", threshold=0.0, return_dict=False):
    """
    Unified implementation for evaluating constraints with proper metrics.

    This method evaluates both boolean and counting constraints:
    - Boolean constraints (andL, atLeastAL, exactAL, etc.): Binary accuracy using 'local/argmax'
    - Counting constraints (sumL): Uses thresholded 'local/decision' key for verification

    Uses AnswerSolver as the primary evaluation method — it performs an ILP
    hypothesis search that correctly handles executable-constraint semantics
    (iotaL selection, counting, etc.). verifySingleConstraint runs in parallel
    as a diagnostic cross-check and any disagreement between the two is
    logged. 

    Args:
        evaluate_data: Dataset to evaluate on
        device: Device to run evaluation on (default: "cpu")
        threshold: Threshold for converting probabilities to binary decisions
                  for counting constraints. Defaults to 0.0 - the threshold will not be applied.
        return_dict: If True, return full results dict; if False, return primary_metric float

    Returns:
        dict or float: Full results dictionary or primary metric value
    """
    from domiknows.graph.logicalConstrain import sumL
    from domiknows.solver.answerModule import AnswerSolver
    from domiknows.step_notebook import (
        StepNotebook, write_active_step, reset_vlm_buffer, drain_vlm_buffer,
    )

    boolean_correct = 0
    boolean_total = 0

    counting_correct = 0
    counting_total = 0

    total = 0

    answer_solver = AnswerSolver(program.graph)

    _notebook_active = StepNotebook.active() is not None
    try:
        _evaluate_data_indexable = hasattr(evaluate_data, '__getitem__')
    except Exception:
        _evaluate_data_indexable = False

    def _populate_with_vlm_buffer():
        # Clear once at entry so a step's buffer only contains that step's
        # VLM calls, then drain after every yielded datanode.
        if _notebook_active:
            reset_vlm_buffer()
        for _dn in program.populate(evaluate_data, device=device):
            _calls = drain_vlm_buffer() if _notebook_active else None
            yield _dn, _calls

    for _step_idx, (datanode, _vlm_calls) in enumerate(tqdm(
        _populate_with_vlm_buffer(),
        total=len(evaluate_data),
        desc="Evaluating",
        position=0,
        leave=True
    )):
        _step_results = {} if _notebook_active else None
        constraint_labels_dict = datanode.getExecutableConstraintLabels()
        if not constraint_labels_dict:
            continue

        active_lc_name = datanode.getActiveExecutableConstraintNames()

        for lc_name, lc in program.graph.executableLCs.items():
            lc.active = lc_name in active_lc_name

        total += 1
        datanode.inferLocal()
        if threshold > 0.0:
            _apply_threshold_to_predictions(program, datanode, threshold)
            key = "/local/decision"
        else:
            key = "/local/argmax"

        for lc_name in active_lc_name:
            if lc_name not in program.graph.executableLCs:
                continue

            lc = program.graph.executableLCs[lc_name]
            if not lc.active:
                continue

            label = datanode.getExecutableConstraintLabel(lc_name)
            if label is None:
                continue

            is_counting = isinstance(lc.innerLC, sumL)

            # ── AnswerSolver (primary) ──────────────────────────────
            answer_correct = None
            try:
                answer_result = answer_solver.answer(f"execute({lc_name})", datanode)
            except Exception as e:
                logger.exception("AnswerSolver failed for %s: %s", lc_name, e)
                answer_result = None

            if answer_result is not None:
                if is_counting:
                    # For sumL the label is the expected count
                    expected_count = int(label.item() if torch.is_tensor(label) else label) # Based on label
                    answer_correct = (answer_result == expected_count)
                else:
                    # For boolean constraints the label is 1 (True) or 0 (False)
                    expected_bool = int(label.item() if torch.is_tensor(label) else label) == 1
                    answer_correct = (answer_result == expected_bool)

            # ── verifySingleConstraint (comparison) ─────────────────
            verify_correct = None
            verify_result = None
            try:
                if is_counting:
                    verify_result = datanode.verifySingleConstraint(lc_name, key="/local/argmax", label=label)
                else:
                    verify_result = datanode.verifySingleConstraint(lc_name, key="/local/argmax")

                is_satisfied = verify_result["satisfied"] == 100.0

                if is_counting:
                    verify_correct = (is_satisfied == True)
                else:
                    expected_satisfied = int(label.item() if torch.is_tensor(label) else label) == 1
                    verify_correct = (is_satisfied == expected_satisfied)
            except Exception as e:
                logger.warning(f"verifySingleConstraint failed for {lc_name}: {e}")

            # ── Pick result source ───────────────────────────────────
            if verify_correct is not None:
                use_correct = verify_correct
            elif answer_correct is not None:
                use_correct = answer_correct
            else:
                continue

            if is_counting:
                if use_correct:
                    counting_correct += 1
                counting_total += 1
            else:
                if use_correct:
                    boolean_correct += 1
                boolean_total += 1

            if _step_results is not None:
                _step_results[lc_name] = {
                    'answer_result': answer_result,
                    'verify_result': verify_result,
                    'correct': bool(use_correct),
                    'is_counting': bool(is_counting),
                }

        if _notebook_active:
            _data_item = None
            if _evaluate_data_indexable:
                try:
                    _data_item = evaluate_data[_step_idx]
                except Exception:
                    _data_item = None
            _extras = {'threshold': threshold}
            if _vlm_calls:
                _extras['vlm_calls'] = _vlm_calls
            write_active_step(
                datanode, program,
                data_item=_data_item,
                phase='eval',
                step_idx=_step_idx,
                precomputed_constraints=_step_results,
                extras=_extras,
            )

    # Build results
    results = {
        'boolean_correct': boolean_correct,
        'boolean_total': boolean_total,
        'counting_correct': counting_correct,
        'counting_total': counting_total
    }

    if total == 0:
        logger.error("No Valid Constraint found for this dataset.")
        results.update({
            'boolean_accuracy': 0.0, 'counting_accuracy': 0.0,
            'primary_metric': 0.0
        })
        return results if return_dict else 0.0

    if boolean_total > 0:
        results['boolean_accuracy'] = (boolean_correct / boolean_total) * 100
        print(f"Boolean accuracy: {results['boolean_accuracy']:.2f}% ({boolean_correct}/{boolean_total})")
    else:
        results['boolean_accuracy'] = None

    if counting_total > 0:
        counting_accuracy = (counting_correct / counting_total) * 100
        results['counting_accuracy'] = counting_accuracy
        print(f"Counting accuracy: {counting_accuracy:.2f}% ({counting_correct}/{counting_total})")
    else:
        results['counting_accuracy'] = None

    # Primary metric
    if results['counting_accuracy'] is not None and results['boolean_accuracy'] is not None:
        total_constraints = boolean_total + counting_total
        boolean_weight = boolean_total / total_constraints
        counting_weight = counting_total / total_constraints
        results['accuracy'] = boolean_weight * results['boolean_accuracy'] + counting_weight * results['counting_accuracy']
    elif results['counting_accuracy'] is not None:
        results['accuracy'] = results['counting_accuracy']
    elif results['boolean_accuracy'] is not None:
        results['accuracy'] = results['boolean_accuracy']
    else:
        results['accuracy'] = 0.0

    results["boolean_total"] = boolean_total
    results["counting_total"] = counting_total

    return results if return_dict else results['accuracy']


def _write_training_step_record(program, data, data_idx, builder,
                                 mloss, closs, total_loss,
                                 iter_count, epoch, training_mode,
                                 device='cpu'):
    """Write one step record during training, mirroring the eval hook.

    Captures per-concept softmax/argmax (so the trajectory of a given
    example across epochs is readable), the classification loss
    (``mloss``), the constraint loss (``closs``, may be None before the
    warmup boundary), and enough training state (epoch, global step,
    training mode) to order the records. Silent no-op when no
    StepNotebook is active. All work guarded by try/except so a bad
    datanode extraction can never break training.
    """
    from domiknows.step_notebook import StepNotebook, write_active_step
    if StepNotebook.active() is None:
        return
    try:
        datanode = builder.getDataNode(device=device)
    except Exception:
        return
    try:
        def _scalar(x):
            if x is None:
                return None
            try:
                return float(x.item()) if torch.is_tensor(x) else float(x)
            except Exception:
                return None
        extras = {
            'epoch': epoch,
            'global_step': iter_count,
            'training_mode': training_mode,
            'mloss': _scalar(mloss),
            'closs': _scalar(closs),
            'total_loss': _scalar(total_loss),
        }
        write_active_step(
            datanode, program,
            data_item=data if isinstance(data, dict) else None,
            phase='train',
            step_idx=data_idx,
            extras=extras,
        )
    except Exception:
        pass


################################################################################
# LossProgram Base Class
################################################################################

class LossProgram(LearningBasedProgram):
    """
    Base class for training with constraint loss.
    
    Provides common infrastructure for constraint-based training.
    Subclasses implement specific training algorithms in train_epoch().
    """
    DEFAULTCMODEL = None  # Subclasses should specify

    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, CModel=None, beta=1, **kwargs):
        """
        Initializes an instance of the LossProgram.

        :param graph: Instance of the initialized DomiKnowS graph.
        :param Model: Class used to calculate the regular model loss (e.g., SolverModel).
        :param CModel: Class used to calculate the constraint model loss. If set to None uses
            the `DEFAULTCMODEL`, which is `PrimalDualModel`.
        :param beta: Weight given to the constraint model loss.
        :params kwargs: Keyword arguments are passed to both the parent class and the CModel.
            (if found in the signature).
        """
        super().__init__(graph, Model, **kwargs)
        
        if CModel is None:
            CModel = self.DEFAULTCMODEL
        if CModel is None:
            raise ValueError(f"{self.__class__.__name__} must specify CModel or set DEFAULTCMODEL")
            
        from inspect import signature
        cmodelSignature = signature(CModel.__init__)
        
        cmodelKwargs = {}
        for param in cmodelSignature.parameters.values():
            if param.name in kwargs:
                cmodelKwargs[param.name] = kwargs[param.name]
                
        self.cmodel = CModel(graph, **cmodelKwargs)
        self.copt = None
        self.beta = beta

    def to(self, device):
        super().to(device=device)
        if self.device is not None:
            self.model.to(self.device)
            self.cmodel.device = self.device
            self.cmodel.to(self.device)

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        if dataset is not None:
            logger.info(f'{name}:')
            desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'

            consume(tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset), desc=desc))

            if self.model.loss:
                logger.info(' - loss:')
                logger.info(self.model.loss)
                    
            if self.cmodel.loss is not None and repr(self.cmodel.loss) != "'None'":
                logger.info(' - Constraint loss:')
                logger.info(self.cmodel.loss)

            if self.model.metric:
                logger.info(' - metric:')
                for key, metric in self.model.metric.items():
                    logger.info(f' - - {key}')
                    logger.info(metric)
                    try:
                        self.f.write(f' - - {name}')
                        self.f.write(f' - - {key}')
                        self.f.write("\n")
                        self.f.write(str(metric))
                        self.f.write("\n")
                    except:
                        pass

    def train(self, training_set, valid_set=None, test_set=None,
              batch_size=1, dataset_size=None, print_loss=True,
              warmup_epochs=0, constraint_epochs=0,
              **kwargs):
        """
        Base training loop. Subclasses pass algorithm-specific kwargs.
        """
        c_session = self._init_session()
        
        if warmup_epochs > 0 or constraint_epochs > 0:
            self._phased_training(
                training_set, valid_set, test_set,
                warmup_epochs, constraint_epochs,
                batch_size, dataset_size, print_loss,
                c_session, **kwargs
            )
        else:
            return super().train(
                training_set, valid_set=valid_set, test_set=test_set,
                c_session=c_session, batch_size=batch_size,
                dataset_size=dataset_size, print_loss=print_loss,
                **kwargs
            )

    def _init_session(self):
        """Initialize session state. Override in subclasses for additional state."""
        return {'iter': 0}

    def _phased_training(self, training_set, valid_set, test_set,
                         warmup_epochs, constraint_epochs,
                         batch_size, dataset_size, print_loss,
                         c_session, **kwargs):
        """Execute phased training (warmup -> constraint)."""
        self.stop = False
        epoch_counter = 0
        
        if warmup_epochs > 0:
            logger.info(f"[Phase 1] Warmup training for {warmup_epochs} epochs")
            for i in range(warmup_epochs):
                if self.stop:
                    break
                epoch_counter += 1
                self.epoch = epoch_counter
                logger.info(f'Epoch: {self.epoch}')
                self.call_epoch(
                    'Training', training_set, self.train_epoch,
                    c_session=c_session, batch_size=batch_size,
                    dataset_size=dataset_size, print_loss=print_loss,
                    training_mode='warmup', **kwargs
                )
                if valid_set is not None:
                    self.call_epoch('Validation', valid_set, self.test_epoch, **kwargs)
        
        if constraint_epochs > 0 and not self.stop:
            logger.info(f"[Phase 2] Constraint training for {constraint_epochs} epochs")
            for i in range(constraint_epochs):
                if self.stop:
                    break
                epoch_counter += 1
                self.epoch = epoch_counter
                logger.info(f'Epoch: {self.epoch}')
                self.call_epoch(
                    'Training', training_set, self.train_epoch,
                    c_session=c_session, batch_size=batch_size,
                    dataset_size=dataset_size, print_loss=print_loss,
                    training_mode='standard', **kwargs
                )
                if valid_set is not None:
                    self.call_epoch('Validation', valid_set, self.test_epoch, **kwargs)
        
        if test_set is not None:
            self.call_epoch('Testing', test_set, self.test_epoch, **kwargs)
        
        self.epoch = None
        self.stop = None

    def train_epoch(self, dataset, c_session={}, batch_size=1,
                    dataset_size=None, print_loss=True,
                    training_mode='standard', **kwargs):
        """
        Basic training epoch. Subclasses override for specific algorithms.
        """
        raise NotImplementedError("Subclasses must implement train_epoch()")
    
# =============================================================================
# Gumbel Temperature Mixin
# =============================================================================

class GumbelTemperatureMixin:
    """
    Mixin providing Gumbel-Softmax temperature annealing for training programs.
    
    Provides shared temperature management across GumbelPrimalDualProgram,
    GumbelSampleLossProgram, and GumbelInferenceProgram.
    """
    
    def _init_gumbel(self, use_gumbel=False, initial_temp=1.0, final_temp=0.1,
                     anneal_start_epoch=0, anneal_epochs=None, hard_gumbel=False):
        """Initialize Gumbel-Softmax parameters. Call from subclass __init__."""
        self.use_gumbel = use_gumbel
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_start_epoch = anneal_start_epoch
        self.anneal_epochs = anneal_epochs
        self.hard_gumbel = hard_gumbel
        self.current_epoch = 0
        self.current_temp = initial_temp
        
        if use_gumbel:
            logger.info(f"[Gumbel] Enabled: temp {initial_temp}→{final_temp}, hard={hard_gumbel}")
    
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
    
    def _auto_set_anneal_epochs(self, num_epochs):
        """Auto-set anneal_epochs from num_epochs if not specified."""
        if self.use_gumbel and self.anneal_epochs is None and num_epochs is not None:
            self.anneal_epochs = num_epochs
            logger.info(f"[Gumbel] Auto-set anneal_epochs to {num_epochs}")
    
    def _update_temperature_for_epoch(self):
        """Update temperature and log periodically. Call at start of train_epoch."""
        self.current_temp = self.get_temperature()
        if self.use_gumbel and self.current_epoch % 10 == 0:
            logger.info(f"[Gumbel] Epoch {self.current_epoch}: temp={self.current_temp:.3f}")
    
    def _increment_epoch(self):
        """Increment epoch counter. Call at end of train_epoch."""
        self.current_epoch += 1
    
    def _call_cmodel_with_gumbel(self, output):
        """Call cmodel.forward with Gumbel parameters."""
        return self.cmodel.forward(
            output,
            use_gumbel=self.use_gumbel,
            temperature=self.current_temp,
            hard_gumbel=self.hard_gumbel
        )
        
#=============================================================================
# Primal-Dual Program
#=============================================================================

class PrimalDualProgram(LossProgram):
    """
    Primal-Dual training using Lagrangian relaxation.
    
    Alternates between:
    - Primal step: minimize model loss w.r.t. model parameters
    - Dual step: maximize constraint violation w.r.t. lambda parameters
    """
    DEFAULTCMODEL = PrimalDualModel
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=PrimalDualModel, beta=beta, **kwargs)

    def _init_session(self):
        """Primal-dual specific session state."""
        c_freq = getattr(self, '_c_freq', 10)
        return {
            'iter': 0,
            'c_update_iter': 0,
            'c_update_freq': c_freq,
            'c_update': 0
        }

    def train(self, training_set, valid_set=None, test_set=None,
              c_lr=0.05, c_warmup_iters=10, c_freq=10,
              c_freq_increase=5, c_freq_increase_freq=1,
              c_lr_decay=4, c_lr_decay_param=1,
              batch_size=1, dataset_size=None, print_loss=True,
              warmup_epochs=0, constraint_epochs=0,
              constraint_only=False, constraint_loss_scale=1.0,
              **kwargs):
        """
        Performs training over a single epoch using a combination of the model loss
        and the constraint model loss.

        Optionally performs batched training using gradient accumulation.

        In `standard` training_mode, updates using the constraint loss are scheduled 
        based on the `c_*` parameters.

        In `constraint_only` training_mode, updates on each iteration and rescales
        the constraint loss.
        
        :param dataset: Iterable of data items to train on.
        :param c_lr: Learning rate for updating parameters with the constraint loss.
        :param c_warmup_iters: The number of initial steps to perform where only the
            regular model loss is used to perform updates.
        :param c_freq_increase: Schedules the rate of parameter updates from constraint loss.
        :param c_freq_increase_freq: Schedules the rate of parameter updates from constraint loss.
        :param c_lr_decay: Method for scheduling the learning rate of the constraint loss.
        :param c_lr_decay_param: Parameter used in the learning rate scheduler.
        :param c_session: Saves the constraint update schedule state from epoch to epoch. Gets updated
            at the end of the epoch.
        :param batch_size: If set > 1, batches updates using gradient accumulation.
        :param dataset_size: Used to determine when to update the last batch. If set to None, tries
            to calculate using len(dataset).
        :param training_mode: `standard`, `constraint_only`, or `warmup` (see above).
        :param constraint_loss_scale: Used in `constraint_only` mode; rescales the constraint loss.
        """
        
        # Setup constraint optimizer
        if list(self.cmodel.parameters()):
            self.copt = torch.optim.Adam(self.cmodel.parameters(), lr=c_lr)
        else:
            self.copt = None
        
        # Store c_freq for _init_session to use
        self._c_freq = c_freq
        
        # Let base class create c_session via _init_session()
        return super().train(
            training_set, valid_set=valid_set, test_set=test_set,
            c_lr=c_lr, c_warmup_iters=c_warmup_iters,
            c_freq_increase=c_freq_increase,
            c_freq_increase_freq=c_freq_increase_freq,
            c_lr_decay=c_lr_decay, c_lr_decay_param=c_lr_decay_param,
            batch_size=batch_size, dataset_size=dataset_size,
            print_loss=print_loss, warmup_epochs=warmup_epochs,
            constraint_epochs=constraint_epochs,
            constraint_only=constraint_only,
            constraint_loss_scale=constraint_loss_scale,
            **kwargs
        )

    def train_epoch(self, dataset, c_lr=0.05, c_warmup_iters=10,
                    c_freq_increase=5, c_freq_increase_freq=1,
                    c_lr_decay=4, c_lr_decay_param=1,
                    c_session={}, batch_size=1, dataset_size=None,
                    print_loss=True, training_mode='standard',
                    constraint_only=False, constraint_loss_scale=1.0,
                    **kwargs):
        """
        Primal-Dual training epoch.
        
        Primal step: gradient descent on model params (minimize loss)
        Dual step: gradient ascent on lambda params (maximize constraint violation)
        """
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1')

        assert c_session
        
        self.model.mode(Mode.TRAIN)
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()

        # Session state
        iter_count = c_session['iter']
        c_update_iter = c_session['c_update_iter']
        c_update_freq = c_session['c_update_freq']
        c_update = c_session['c_update']

        # Dataset size
        num_data_iters = dataset_size
        if num_data_iters is None:
            if not hasattr(dataset, '__len__'):
                raise ValueError('dataset must have __len__ if dataset_size not provided')
            num_data_iters = len(dataset)

        batch_loss = 0.0
        
        for data_idx, data in enumerate(dataset):
            try:
                mloss, metric, *output = self.model(data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

            # Compute loss based on training mode
            closs = None
            if training_mode == 'warmup':
                loss = mloss
            elif constraint_only:
                closs, *_ = self.cmodel(output[1])
                if torch.is_tensor(closs) and torch.isfinite(closs) and abs(closs.item()) > 1e-8:
                    loss = mloss * 0.01 + closs * constraint_loss_scale
                else:
                    continue
            else:
                if iter_count < c_warmup_iters:
                    loss = mloss
                else:
                    closs, *_ = self.cmodel(output[1])
                    if torch.is_tensor(closs) and torch.is_nonzero(closs):
                        loss = mloss + self.beta * closs
                    else:
                        loss = mloss

            if not loss:
                continue

            _write_training_step_record(
                self, data, data_idx, output[1],
                mloss=mloss, closs=closs, total_loss=loss,
                iter_count=iter_count, epoch=self.epoch,
                training_mode=training_mode, device=self.device,
            )

            batch_pos = data_idx % batch_size
            do_update = (batch_pos == batch_size - 1) or (data_idx == num_data_iters - 1)

            # Accumulate gradients
            scaled_loss = loss / batch_size
            batch_loss += scaled_loss.item()
            scaled_loss.backward()

            if do_update:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # Primal step: update model params
                if self.opt is not None:
                    self.opt.step()
                    self.opt.zero_grad()
                
                iter_count += 1
                
                # Dual step: update lambda params (with reversed gradient for ascent)
                should_update_dual = (
                    self.copt is not None and
                    iter_count > c_warmup_iters and
                    iter_count - c_update_iter > c_update_freq
                )
                
                if should_update_dual:
                    # Reverse gradients for lambda (gradient ascent)
                    reverse_sign_grad(self.cmodel.parameters())
                    torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=10.0)
                    self.copt.step()
                    
                    c_update_iter = iter_count
                    c_update += 1
                    
                    # Update dual frequency
                    if c_freq_increase_freq > 0 and c_update % c_freq_increase_freq == 0:
                        c_update_freq += c_freq_increase
                    
                    # Update dual learning rate
                    self._update_dual_lr(c_lr_decay, c_lr_decay_param, c_update, c_lr)
                
                if self.copt is not None:
                    self.copt.zero_grad()
                
                batch_loss = 0.0
            
            yield (loss, metric, *output[:1])

        # Save session state
        c_session['iter'] = iter_count
        c_session['c_update_iter'] = c_update_iter
        c_session['c_update_freq'] = c_update_freq
        c_session['c_update'] = c_update

    def _update_dual_lr(self, c_lr_decay, c_lr_decay_param, c_update, c_lr):
        """Update learning rate for dual optimizer."""
        if c_lr_decay == 0:
            new_lr = lambda lr: c_lr * 1. / (1 + c_lr_decay_param * c_update)
        elif c_lr_decay == 1:
            new_lr = lambda lr: lr * np.sqrt(((c_update-1.) / c_lr_decay_param + 1.) / (c_update / c_lr_decay_param + 1.))
        elif c_lr_decay == 2:
            new_lr = lambda lr: lr * (((c_update-1.) / c_lr_decay_param + 1.) / (c_update / c_lr_decay_param + 1.))
        elif c_lr_decay == 3:
            assert c_lr_decay_param <= 1.
            new_lr = lambda lr: lr * c_lr_decay_param
        elif c_lr_decay == 4:
            new_lr = lambda lr: lr * np.sqrt((c_update+1) / (c_update+2))
        else:
            raise ValueError(f'c_lr_decay={c_lr_decay} not supported.')
        
        for param_group in self.copt.param_groups:
            param_group['lr'] = new_lr(param_group['lr'])
            
#=============================================================================
# Gumbel Primal-Dual Program
#=============================================================================

class GumbelPrimalDualProgram(GumbelTemperatureMixin, PrimalDualProgram):
    """Primal-Dual with Gumbel-Softmax for differentiable discrete sampling."""
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1,
                 use_gumbel=False, initial_temp=1.0, final_temp=0.1,
                 anneal_start_epoch=0, anneal_epochs=None, hard_gumbel=False,
                 **kwargs):
        super().__init__(graph, Model, beta=beta, **kwargs)
        self._init_gumbel(use_gumbel, initial_temp, final_temp,
                          anneal_start_epoch, anneal_epochs, hard_gumbel)
    
    def train(self, training_set, valid_set=None, test_set=None,
              num_epochs=None, **kwargs):
        self._auto_set_anneal_epochs(num_epochs)
        return super().train(training_set, valid_set=valid_set, test_set=test_set, **kwargs)

    def train_epoch(self, dataset, c_lr=0.05, c_warmup_iters=10,
                    c_freq_increase=5, c_freq_increase_freq=1,
                    c_lr_decay=4, c_lr_decay_param=1,
                    c_session={}, batch_size=1, dataset_size=None,
                    print_loss=True, training_mode='standard',
                    constraint_only=False, constraint_loss_scale=1.0,
                    **kwargs):
        """Primal-Dual epoch with Gumbel-Softmax."""
        self._update_temperature_for_epoch()

        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1')

        assert c_session
        
        self.model.mode(Mode.TRAIN)
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()

        iter_count = c_session['iter']
        c_update_iter = c_session['c_update_iter']
        c_update_freq = c_session['c_update_freq']
        c_update = c_session['c_update']

        num_data_iters = dataset_size
        if num_data_iters is None:
            if not hasattr(dataset, '__len__'):
                raise ValueError('dataset must have __len__ if dataset_size not provided')
            num_data_iters = len(dataset)

        batch_loss = 0.0
        
        for data_idx, data in enumerate(dataset):
            try:
                mloss, metric, *output = self.model(data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

            closs = None
            if training_mode == 'warmup':
                loss = mloss
            elif constraint_only:
                closs, *_ = self._call_cmodel_with_gumbel(output[1])
                if torch.is_tensor(closs) and torch.isfinite(closs) and abs(closs.item()) > 1e-8:
                    loss = mloss * 0.01 + closs * constraint_loss_scale
                else:
                    continue
            else:
                if iter_count < c_warmup_iters:
                    loss = mloss
                else:
                    closs, *_ = self._call_cmodel_with_gumbel(output[1])
                    if torch.is_tensor(closs) and torch.is_nonzero(closs):
                        loss = mloss + self.beta * closs
                    else:
                        loss = mloss

            if not loss:
                continue

            _write_training_step_record(
                self, data, data_idx, output[1],
                mloss=mloss, closs=closs, total_loss=loss,
                iter_count=iter_count, epoch=self.epoch,
                training_mode=training_mode, device=self.device,
            )

            batch_pos = data_idx % batch_size
            do_update = (batch_pos == batch_size - 1) or (data_idx == num_data_iters - 1)

            scaled_loss = loss / batch_size
            batch_loss += scaled_loss.item()
            scaled_loss.backward()

            if do_update:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                if self.opt is not None:
                    self.opt.step()
                    self.opt.zero_grad()
                
                iter_count += 1
                
                should_update_dual = (
                    self.copt is not None and
                    iter_count > c_warmup_iters and
                    iter_count - c_update_iter > c_update_freq
                )
                
                if should_update_dual:
                    reverse_sign_grad(self.cmodel.parameters())
                    torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=10.0)
                    self.copt.step()
                    
                    c_update_iter = iter_count
                    c_update += 1
                    
                    if c_freq_increase_freq > 0 and c_update % c_freq_increase_freq == 0:
                        c_update_freq += c_freq_increase
                    
                    self._update_dual_lr(c_lr_decay, c_lr_decay_param, c_update, c_lr)
                
                if self.copt is not None:
                    self.copt.zero_grad()
                
                batch_loss = 0.0
            
            yield (loss, metric, *output[:1])

        c_session['iter'] = iter_count
        c_session['c_update_iter'] = c_update_iter
        c_session['c_update_freq'] = c_update_freq
        c_session['c_update'] = c_update
        
        self._increment_epoch()

    def evaluate_condition(self, evaluate_data, device="cpu", threshold=0.5, return_dict=False):
        return _evaluate_condition_impl(self, evaluate_data, device=device, threshold=threshold, return_dict=return_dict)

#=============================================================================
# Inference Program
#=============================================================================

class InferenceProgram(LossProgram):
    """
    Program for training with program execution.

    During training, logical expressions either specified directly in the graph,
    or compiled from the dataset are executed using soft-logic. Parameters are
    then updated based on the soft-logic output and the provided ground-truth value
    of the logical expression.
    """
    DEFAULTCMODEL = InferenceModel
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        """
        Initializes an InferenceProgram instance.

        :param graph: The initialized graph either containing the logical expressions to be executed
            and/or called with `.compile_executable` to use the logical expressions in the dataset.
        :param Model: The class to use for the regular forward pass and
            supervised training (e.g., `SolverModel`).
        :param beta: The weight given to the CModel loss (in this case, the loss from the program
            execution output)
        """
        super().__init__(graph, Model, CModel=InferenceModel, beta=beta, **kwargs)

    def train(self, training_set, valid_set=None, test_set=None,
              batch_size=1, dataset_size=None, print_loss=True,
              warmup_epochs=0, constraint_epochs=0, **kwargs):
        """Setup optimizer and train."""
        if list(self.cmodel.parameters()):
            self.copt = torch.optim.Adam(self.cmodel.parameters(), lr=0.01)
        else:
            self.copt = None
        
        return super().train(
            training_set, valid_set=valid_set, test_set=test_set,
            batch_size=batch_size, dataset_size=dataset_size,
            print_loss=print_loss, warmup_epochs=warmup_epochs,
            constraint_epochs=constraint_epochs, **kwargs
        )

    def train_epoch(self, dataset, c_session={}, batch_size=1,
                    dataset_size=None, print_loss=True,
                    training_mode='standard', **kwargs):
        """Simple training: model loss + constraint loss."""

        import os as _os
        _mem_probe = _os.environ.get('DOMIKNOWS_MEM_PROBE') == '1'
        _mem_step = c_session.get('_mem_step', 0)

        self.model.mode(Mode.TRAIN)
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()

        iter_count = c_session.get('iter', 0)

        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()

            mloss, metric, *output = self.model(data)

            if training_mode == 'warmup':
                loss = mloss
            else:
                closs, *_ = self.cmodel(output[1])
                if torch.is_tensor(closs):
                    loss = mloss + self.beta * closs
                else:
                    loss = mloss

            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()

                # Gradient clipping to prevent explosion (e.g. constraint losses)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                if self.copt is not None:
                    torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=10.0)

                if self.opt is not None:
                    self.opt.step()
                if self.copt is not None:
                    self.copt.step()
                iter_count += 1

            if _mem_probe and torch.cuda.is_available():
                _mem_step += 1
                _alloc = torch.cuda.memory_allocated() / 1e9
                _res = torch.cuda.memory_reserved() / 1e9
                print(f"[mem_probe] step={_mem_step} alloc={_alloc:.2f}GB reserved={_res:.2f}GB", flush=True)

            yield (loss, metric, *output[:1])

        c_session['iter'] = iter_count
        c_session['_mem_step'] = _mem_step

    def evaluate_condition(self, evaluate_data, device="cpu", threshold=0.0, return_dict=False):
        return _evaluate_condition_impl(self, evaluate_data, device=device, threshold=threshold, return_dict=return_dict)

#=============================================================================
# Gumbel Inference Program 
#=============================================================================

class GumbelInferenceProgram(GumbelTemperatureMixin, InferenceProgram):
    """Inference Program with Gumbel-Softmax for differentiable discrete sampling."""
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1,
                 use_gumbel=False, initial_temp=1.0, final_temp=0.1,
                 anneal_start_epoch=0, anneal_epochs=None, hard_gumbel=False,
                 **kwargs):
        super().__init__(graph, Model, beta=beta, **kwargs)
        self._init_gumbel(use_gumbel, initial_temp, final_temp,
                          anneal_start_epoch, anneal_epochs, hard_gumbel)
    
    def train(self, training_set, valid_set=None, test_set=None,
              num_epochs=None, **kwargs):
        self._auto_set_anneal_epochs(num_epochs)
        return super().train(training_set, valid_set=valid_set, test_set=test_set, **kwargs)

    def train_epoch(self, dataset, c_session={}, batch_size=1,
                    dataset_size=None, print_loss=True,
                    training_mode='standard', **kwargs):
        """Inference training epoch with Gumbel-Softmax."""
        import os as _os
        _mem_probe = _os.environ.get('DOMIKNOWS_MEM_PROBE') == '1'
        _mem_step = c_session.get('_mem_step', 0)

        self._update_temperature_for_epoch()

        self.model.mode(Mode.TRAIN)
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()

        iter_count = c_session.get('iter', 0)

        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()

            mloss, metric, *output = self.model(data)

            if training_mode == 'warmup':
                loss = mloss
            else:
                closs, *_ = self._call_cmodel_with_gumbel(output[1])
                if torch.is_tensor(closs):
                    loss = mloss + self.beta * closs
                else:
                    loss = mloss

            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()

                # Gradient clipping to prevent explosion (e.g. constraint losses)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                if self.copt is not None:
                    torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=10.0)

                if self.opt is not None:
                    self.opt.step()
                if self.copt is not None:
                    self.copt.step()
                iter_count += 1

            if _mem_probe and torch.cuda.is_available():
                _mem_step += 1
                _alloc = torch.cuda.memory_allocated() / 1e9
                _res = torch.cuda.memory_reserved() / 1e9
                print(f"[mem_probe] step={_mem_step} alloc={_alloc:.2f}GB reserved={_res:.2f}GB", flush=True)

            yield (loss, metric, *output[:1])

        c_session['iter'] = iter_count
        c_session['_mem_step'] = _mem_step
        self._increment_epoch()

    def evaluate_condition(self, evaluate_data, device="cpu", threshold=0.5, return_dict=False):
        return _evaluate_condition_impl(self, evaluate_data, device=device, threshold=threshold, return_dict=return_dict)

#=============================================================================
# Sample Loss Program
#=============================================================================

class SampleLossProgram(LossProgram):
    """
    Sampling-based constraint training.
    """
    DEFAULTCMODEL = SampleLossModel
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=SampleLossModel, beta=beta, **kwargs)

    def train(self, training_set, valid_set=None, test_set=None,
              c_lr=0.05, c_warmup_iters=10, **kwargs):
        """Setup optimizer and train."""
        if list(self.cmodel.parameters()):
            self.copt = torch.optim.Adam(self.cmodel.parameters(), lr=c_lr)
        else:
            self.copt = None
        
        return super().train(
            training_set, valid_set=valid_set, test_set=test_set,
            c_warmup_iters=c_warmup_iters, **kwargs
        )

    def train_epoch(self, dataset, c_warmup_iters=0, c_session={}, **kwargs):
        """Sampling-based training epoch."""
        
        self.model.mode(Mode.TRAIN)
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()
        
        iter_count = c_session.get('iter', 0)
        
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            
            mloss, metric, *output = self.model(data)
            
            if iter_count < c_warmup_iters:
                loss = mloss
            else:
                closs, *_ = self.cmodel(output[1])
                if torch.is_tensor(closs):
                    loss = mloss + self.beta * closs
                else:
                    loss = mloss
                
                if loss != loss:  # NaN check
                    raise Exception("Calculated loss is nan")
            
            if self.opt is not None and torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()

                # Gradient clipping to prevent explosion (e.g. constraint losses)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                if self.copt is not None:
                    torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=10.0)

                self.opt.step()
                iter_count += 1
            
            if self.copt is not None and torch.is_tensor(loss) and loss.requires_grad:
                self.copt.step()
            
            yield (loss, metric, *output[:1])

        c_session['iter'] = iter_count

#=============================================================================
# Gumbel Sample Loss Program
#=============================================================================

class GumbelSampleLossProgram(GumbelTemperatureMixin, SampleLossProgram):
    """Sample Loss with Gumbel-Softmax support."""
    
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, beta=1,
                 use_gumbel=False, initial_temp=1.0, final_temp=0.1,
                 anneal_start_epoch=0, anneal_epochs=None, hard_gumbel=False,
                 **kwargs):
        super().__init__(graph, Model, beta=beta, **kwargs)
        self._init_gumbel(use_gumbel, initial_temp, final_temp,
                          anneal_start_epoch, anneal_epochs, hard_gumbel)
    
    def train(self, training_set, valid_set=None, test_set=None,
              num_epochs=None, **kwargs):
        self._auto_set_anneal_epochs(num_epochs)
        return super().train(training_set, valid_set=valid_set, test_set=test_set, **kwargs)
    
    def train_epoch(self, dataset, c_warmup_iters=0, c_session={}, **kwargs):
        """Sampling epoch with Gumbel-Softmax."""
        self._update_temperature_for_epoch()
        
        self.model.mode(Mode.TRAIN)
        self.model.train()
        self.model.reset()
        self.cmodel.train()
        self.cmodel.reset()
        
        iter_count = c_session.get('iter', 0)
        
        for data in dataset:
            if self.opt is not None:
                self.opt.zero_grad()
            if self.copt is not None:
                self.copt.zero_grad()
            
            mloss, metric, *output = self.model(data)
            
            if iter_count < c_warmup_iters:
                loss = mloss
            else:
                closs, *_ = self._call_cmodel_with_gumbel(output[1])
                
                if torch.is_tensor(closs):
                    loss = mloss + self.beta * closs
                else:
                    loss = mloss
                
                if loss != loss:
                    raise Exception("Calculated loss is nan")
            
            if self.opt is not None and torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()

                # Gradient clipping to prevent explosion (e.g. constraint losses)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                if self.copt is not None:
                    torch.nn.utils.clip_grad_norm_(self.cmodel.parameters(), max_norm=10.0)

                self.opt.step()
                iter_count += 1
            
            if self.copt is not None and loss:
                self.copt.step()
            
            yield (loss, metric, *output[:1])

        c_session['iter'] = iter_count
        self._increment_epoch()    
        
# =============================================================================
# GBI Program 
# =============================================================================

class GBIProgram(LossProgram):
    logger = logging.getLogger(__name__)

    def __init__(self, graph, Model, poi, beta=1, **kwargs):
        super().__init__(graph, Model, CModel=GBIModel, beta=beta, poi=poi, **kwargs)
        from domiknows.utils import setDnSkeletonMode
        setDnSkeletonMode(True)