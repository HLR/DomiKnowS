import logging
import warnings
from collections import OrderedDict

import numpy as np
import torch

from ...graph import DataNodeBuilder
from ..metric import MetricTracker, MacroAverageTracker
from domiknows import setup_logger, getProductionModeStatus
from domiknows.graph.logicalConstrain import queryL, sumL
from domiknows.solver.lossCalculator import multi_query_joint_nll

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

    #: Supported dual-optimization algorithms (R5 Phase A).
    DUAL_ALGORITHMS = ('ascent', 'augmented')
    #: Supported dual-variable granularities. 'constraint' = one dual per
    #: constraint template (Phase A); 'amortized' = per-grounding duals from a
    #: DualCritic network (Phase B).
    DUAL_GRANULARITIES = ('constraint', 'amortized')

    def __init__(self, graph,
                 tnorm='P',
                 counting_tnorm=None,
                 sample=False, sampleSize=0, sampleGlobalLoss=False, device='auto',
                 use_gumbel=False, temperature=1.0, hard_gumbel=False,
                 compile_lc=False,
                 dual_algorithm='ascent', dual_granularity='constraint',
                 al_rho_init=1.0, al_rho_growth=2.0, al_rho_max=100.0,
                 al_stagnation_tau=0.9,
                 critic_embed_dim=8, critic_hidden=32,
                 critic_fit_weight=1.0,
                 exclude_constraints=None):
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
        :param compile_lc: If True, evaluate the constraint loss with the compiled
            (batched-gather) evaluator instead of the per-datanode interpreter;
            unsupported constraint types fall back to the interpreter per
            constraint. Ignored when sample is True.
        :param dual_algorithm: 'ascent' (default) keeps the plain gradient-ascent
            Lagrangian dual updated by the program's constraint optimizer.
            'augmented' switches to an Augmented Lagrangian whose per-constraint
            multipliers are updated in closed form (no constraint optimizer) and
            adds a quadratic penalty ``(rho/2)*sum(v^2)``; see ``al_dual_update_``.
        :param dual_granularity: 'constraint' (default) — one dual per constraint
            template. 'amortized' — a DualCritic network predicts a per-grounding
            multiplier from detached features (R5 Phase B). With
            ``dual_algorithm='ascent'`` the critic is optimised by the program's
            constraint optimizer; with ``'augmented'`` (R5 Phase F) it is instead
            *regressed* onto the AL target ``lambda_c + rho_c * v_g``, since an
            augmented Lagrangian moves its multipliers in closed form and so has
            no ascent objective for a critic to maximise.
        :param al_rho_init: Initial per-constraint penalty coefficient (augmented).
        :param al_rho_growth: Multiplicative growth applied to a constraint's rho
            when its violation fails to shrink by ``al_stagnation_tau`` (augmented).
        :param al_rho_max: Upper bound on rho (augmented).
        :param al_stagnation_tau: A constraint is "stagnating" when its mean
            violation this dual window exceeds ``tau`` times the previous window's;
            rho then grows (augmented).
        :param critic_embed_dim: per-constraint embedding width for the DualCritic
            (amortized granularity).
        :param critic_hidden: hidden width of the DualCritic MLP (amortized).
        """
        super().__init__()
        self.graph = graph
        self.build = True

        self.tnorm = tnorm
        self.counting_tnorm = counting_tnorm
        self.compile_lc = compile_lc
        self.device = device

        if dual_algorithm not in self.DUAL_ALGORITHMS:
            raise ValueError(
                f"dual_algorithm must be one of {self.DUAL_ALGORITHMS}, got {dual_algorithm!r}")
        if dual_granularity not in self.DUAL_GRANULARITIES:
            raise ValueError(
                f"dual_granularity must be one of {self.DUAL_GRANULARITIES}, got {dual_granularity!r}")
        self.dual_algorithm = dual_algorithm
        self.dual_granularity = dual_granularity
        self.al_rho_growth = float(al_rho_growth)
        self.al_rho_max = float(al_rho_max)
        self.al_stagnation_tau = float(al_stagnation_tau)
        self.critic_embed_dim = int(critic_embed_dim)
        self.critic_hidden = int(critic_hidden)
        #: Weight of the critic's regression onto the AL target (Phase F only).
        self.critic_fit_weight = float(critic_fit_weight)

        self.sample = sample
        self.sampleSize = sampleSize
        self.sampleGlobalLoss = sampleGlobalLoss

        # Gumbel-Softmax parameters
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.hard_gumbel = hard_gumbel

        # Extract all logical constraints from the graph recursively.
        #
        # ``exclude_constraints`` drops constraints that something else already
        # guarantees — under an R3 factor-graph head a compiled hard constraint
        # has zero mass *by construction*, so its violation is identically zero.
        # Leaving it in would allocate a multiplier pinned at zero and feed
        # ``al_dual_update_`` an all-zero window, i.e. a dual that can only ever
        # learn nothing. Excluding it here keeps the dual system scoped to the
        # constraints that are actually still being fought for.
        self.exclude_constraints = set(exclude_constraints or ())
        #: Optional ``callable() -> set`` of constraint names to skip *this*
        #: step. Where ``exclude_constraints`` is a permanent construction-time
        #: decision, this is consulted per forward, so a constraint whose
        #: structural enforcement fell back at runtime gets its penalty back
        #: automatically instead of being left unconstrained.
        self.skip_provider = None
        self.constr = OrderedDict(
            (key, lc) for key, lc in graph.allLogicalConstrainsRecursive
            if key not in self.exclude_constraints
            and getattr(lc, 'lcName', key) not in self.exclude_constraints)
        nconstr = len(self.constr)
        if nconstr == 0:
            warnings.warn('No logical constraint detected in the graph. '
                          'PrimalDualModel will not generate any constraint loss.')

        # Lagrange multipliers, one per constraint.
        # - constraint x ascent: learnable Parameter, gradient ascent (copt).
        # - constraint x augmented: buffer, closed-form update (al_dual_update_);
        #   a buffer keeps it out of cmodel.parameters() so no copt is built and
        #   it still round-trips via state_dict.
        # - amortized x ascent: the multiplier is produced per grounding by a
        #   DualCritic (below); lmbd is kept as an unused buffer so shared code
        #   (reset_parameters/project_lmbd_/to) stays a harmless no-op.
        self.dual_critic = None
        if dual_granularity == 'amortized':
            from domiknows.program.model.dualCritic import DualCritic
            self.dual_critic = DualCritic(nconstr, embed_dim=self.critic_embed_dim,
                                          hidden=self.critic_hidden)
            self.register_buffer('lmbd', torch.ones(nconstr))
            if dual_algorithm == 'augmented':
                # R5 Phase F: the critic is regressed onto the AL target, so the
                # same penalty/statistics state the constraint-granular AL keeps
                # is needed here too.
                self.register_buffer('rho', torch.full((nconstr,), float(al_rho_init)))
                self.register_buffer('_al_viol_accum', torch.zeros(nconstr))
                self.register_buffer('_al_viol_count', torch.zeros(nconstr))
                self.register_buffer('_al_prev_mean_viol',
                                     torch.full((nconstr,), float('nan')))
        elif dual_algorithm == 'augmented':
            self.register_buffer('lmbd', torch.empty(nconstr))
            # Per-constraint quadratic-penalty coefficient and the running
            # violation statistics consumed by al_dual_update_.
            self.register_buffer('rho', torch.full((nconstr,), float(al_rho_init)))
            self.register_buffer('_al_viol_accum', torch.zeros(nconstr))
            self.register_buffer('_al_viol_count', torch.zeros(nconstr))
            self.register_buffer('_al_prev_mean_viol', torch.full((nconstr,), float('nan')))
        else:
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
        index = self.lmbd_index[key]
        return self.lmbd[index].clamp(min=0, max=self.lmbd_p[index])

    def project_lmbd_(self):
        """Project learnable Lagrange multipliers to their valid range."""
        if not hasattr(self, 'lmbd') or not hasattr(self, 'lmbd_p'):
            return
        with torch.no_grad():
            upper = self.lmbd_p.to(device=self.lmbd.device, dtype=self.lmbd.dtype)
            self.lmbd.clamp_(min=0)
            self.lmbd.copy_(torch.minimum(self.lmbd, upper))

    def _weighted_constraint_loss(self, key, lossTensor, groundingFeatures=None):
        """Weight one constraint's per-grounding violation vector into a scalar.

        constraint x ascent (default): reproduces the original behaviour exactly
        — ``lambda_c * sum(clamp(v, 0))`` over the NaN-filtered violations.

        constraint x augmented: Augmented-Lagrangian term ``lambda_c * S_c +
        (rho_c/2) * Q_c`` with ``S_c = sum(v)`` and ``Q_c = sum(v^2)``, and
        records ``S_c`` into the running statistics consumed by
        ``al_dual_update_`` (lambda/rho are buffers, so this adds no autograd
        path through the multipliers — the primal gradient stays
        ``lambda_c * dS_c/dtheta + rho_c * sum(v * dv/dtheta)``).

        amortized x ascent: ``sum_g lambda_g * v_g`` where ``lambda_g`` is the
        DualCritic's per-grounding multiplier (bounded to ``[0, lmbd_p_c]``).
        The critic reads *detached* features, so ``lambda_g`` carries gradient
        only to the critic (used by the ascent step) while ``v_g`` carries the
        primal gradient to the classifiers.
        """
        loss_value = lossTensor.clamp(min=0)
        finite_mask = (loss_value == loss_value)  # drop NaN groundings
        finite = loss_value[finite_mask]

        index = self.lmbd_index[key]

        if self.dual_granularity == 'amortized':
            if finite.numel() == 0:
                return loss_value.sum() * 0.0  # keeps a grad-connected zero
            feats = None
            if groundingFeatures is not None and torch.is_tensor(groundingFeatures):
                gf = groundingFeatures
                if gf.dim() == 1:
                    gf = gf.unsqueeze(-1)
                if gf.shape[0] == loss_value.shape[0]:
                    feats = gf[finite_mask]
            lam = self.dual_critic(index, finite.detach(), feats)  # [G'] in (0,1)
            lam = lam * self.lmbd_p[index]                          # scale to [0, lmbd_p]

            if self.dual_algorithm != 'augmented':
                return (lam * finite).sum()

            # R5 Phase F — amortized x augmented. Ascent maximises the critic's
            # own objective, which the AL has no equivalent of: its multipliers
            # move in closed form. So the critic is *regressed* onto the AL
            # target lambda_c + rho_c * v_g instead — the per-grounding value
            # the closed-form update would assign — while the primal keeps the
            # quadratic penalty. Both stay per-grounding, which is the point of
            # combining them.
            with torch.no_grad():
                self._al_viol_accum[index] += finite.sum().detach()
                self._al_viol_count[index] += 1
                target = (self.lmbd[index] + self.rho[index] * finite.detach()
                          ).clamp(min=0, max=float(self.lmbd_p[index]))

            critic_fit = torch.nn.functional.mse_loss(lam, target)
            quad = (finite * finite).sum()
            # The critic term carries no primal gradient (``finite`` is detached
            # inside the target and ``lam`` reaches only critic parameters), so
            # adding it here trains the critic through the existing constraint
            # optimizer without perturbing the learners' update.
            return (lam.detach() * finite).sum() + 0.5 * self.rho[index] * quad \
                + self.critic_fit_weight * critic_fit

        loss_nansum = finite.sum()

        if self.dual_algorithm != 'augmented':
            return self.get_lmbd(key) * loss_nansum

        with torch.no_grad():
            self._al_viol_accum[index] += loss_nansum.detach()
            self._al_viol_count[index] += 1

        lmbd = self.lmbd[index].clamp(min=0, max=self.lmbd_p[index])
        rho = self.rho[index]
        quad = (finite * finite).sum()
        return lmbd * loss_nansum + 0.5 * rho * quad

    def al_dual_update_(self):
        """Closed-form Augmented-Lagrangian multiplier update + penalty schedule.

        Invoked by the program at its dual-update points (in place of the
        gradient-ascent step). Consumes and resets the per-constraint violation
        statistics accumulated across the forward passes since the previous
        call. No-op unless ``dual_algorithm == 'augmented'``.
        """
        if self.dual_algorithm != 'augmented':
            return
        with torch.no_grad():
            has_data = self._al_viol_count > 0
            count = self._al_viol_count.clamp(min=1)
            mean_viol = self._al_viol_accum / count  # mean S_c over this window

            # Multiplier ascent, projected to [0, lmbd_p]; only touch constraints
            # that were actually evaluated this window.
            upper = self.lmbd_p.to(device=self.lmbd.device, dtype=self.lmbd.dtype)
            new_lmbd = torch.clamp(self.lmbd + self.rho * mean_viol, min=0)
            new_lmbd = torch.minimum(new_lmbd, upper)
            self.lmbd.copy_(torch.where(has_data, new_lmbd, self.lmbd))

            # Grow rho where the violation did not shrink by factor tau versus the
            # previous window (never on the first window — prev is NaN there).
            prev = self._al_prev_mean_viol
            stagnated = has_data & ~torch.isnan(prev) & (mean_viol > self.al_stagnation_tau * prev)
            grown = torch.minimum(self.rho * self.al_rho_growth,
                                  torch.full_like(self.rho, self.al_rho_max))
            self.rho.copy_(torch.where(stagnated, grown, self.rho))

            self._al_prev_mean_viol.copy_(torch.where(has_data, mean_viol, prev))
            self._al_viol_accum.zero_()
            self._al_viol_count.zero_()

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
            sampleSize=self.sampleSize,
            compiled=self.compile_lc and not self.sample
        )

        lmbd_loss = []
        if self.sampleGlobalLoss and constr_loss['globalLoss']:
            globalLoss = constr_loss['globalLoss']
            self.loss['globalLoss'](globalLoss)
            dtype = getattr(datanode, 'current_dtype', torch.float32)
            lmbd_loss = torch.tensor(globalLoss, dtype=dtype, requires_grad=True)
        else:
            skip = self.skip_provider() if self.skip_provider is not None else ()
            for key, loss in constr_loss.items():
                if key not in self.constr:
                    continue
                if key in skip:
                    # Enforced structurally this step: its violation is zero by
                    # construction, so a penalty term would only add noise.
                    continue

                if loss['lossTensor'] is not None:
                    # groundingFeatures (per-grounding literal probabilities) are
                    # present only on the compiled path; the DualCritic zero-fills
                    # when they are absent (interpreter path).
                    features = loss.get('groundingFeatures') if isinstance(loss, dict) else None
                    loss_ = self._weighted_constraint_loss(key, loss['lossTensor'], features)
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

    def __init__(self, graph, tnorm='P', counting_tnorm=None, device='auto', compile_lc=False,
                 dual_algorithm='ascent', dual_granularity='constraint',
                 al_rho_init=1.0, al_rho_growth=2.0, al_rho_max=100.0,
                 al_stagnation_tau=0.9,
                 critic_embed_dim=8, critic_hidden=32,
                 critic_fit_weight=1.0,
                 exclude_constraints=None):
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
        :param compile_lc: If True, evaluate the constraint loss with the compiled
        (batched-gather) evaluator; unsupported constraints fall back to the interpreter
        :param dual_algorithm: 'ascent' (default) or 'augmented' (Augmented Lagrangian);
            see :class:`LossModel`.
        :param dual_granularity: 'constraint' (default) or 'amortized' (R5 Phase B,
            per-grounding DualCritic; ascent only).
        :param al_rho_init: Initial augmented-Lagrangian penalty coefficient.
        :param al_rho_growth: rho growth factor on stagnation (augmented).
        :param al_rho_max: rho upper bound (augmented).
        :param al_stagnation_tau: stagnation threshold for rho growth (augmented).
        :param critic_embed_dim: DualCritic per-constraint embedding width (amortized).
        :param critic_hidden: DualCritic MLP hidden width (amortized).
        """
        super().__init__(graph, tnorm=tnorm, counting_tnorm=counting_tnorm, device=device,
                         compile_lc=compile_lc,
                         dual_algorithm=dual_algorithm, dual_granularity=dual_granularity,
                         al_rho_init=al_rho_init, al_rho_growth=al_rho_growth,
                         al_rho_max=al_rho_max, al_stagnation_tau=al_stagnation_tau,
                         critic_embed_dim=critic_embed_dim, critic_hidden=critic_hidden,
                         critic_fit_weight=critic_fit_weight,
                         exclude_constraints=exclude_constraints)
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


class SemanticLossModel(LossModel):
    """Constraint model using exact circuit weighted model counting.

    The default objective is the direct sum of ``-log(WMC)`` values. Set
    ``lambda_weighted=True`` to reuse :class:`LossModel`'s learned per-template
    multipliers.
    """

    def __init__(
        self,
        graph,
        *,
        lambda_weighted=False,
        circuit_backend=None,
        circuit_max_nodes=None,
        circuit_size_limit_action=None,
        circuit_aggregation=None,
        device="auto",
        dual_algorithm='ascent',
        dual_granularity='constraint',
        al_rho_init=1.0, al_rho_growth=2.0, al_rho_max=100.0,
        al_stagnation_tau=0.9,
        critic_embed_dim=8, critic_hidden=32,
    ):
        """
        :param lambda_weighted: weight each constraint's exact loss with the
            learned dual multipliers instead of summing raw ``-log(WMC)``. This
            is what composes semantic loss with the R5 dual mechanisms.
        :param circuit_aggregation: ``'joint'`` (default) or ``'per_grounding'``.
            Per-grounding is required for ``dual_granularity='amortized'``
            (R5B), which needs one violation entry per grounding.
        :param dual_algorithm / dual_granularity / al_* / critic_*: forwarded to
            :class:`LossModel`; they take effect only when ``lambda_weighted``.
        """
        if dual_granularity == 'amortized' and circuit_aggregation is None:
            # The amortized critic attributes per grounding; a joint scalar
            # would collapse it to a single row and defeat the mechanism.
            circuit_aggregation = 'per_grounding'

        super().__init__(
            graph=graph, device=device,
            dual_algorithm=dual_algorithm, dual_granularity=dual_granularity,
            al_rho_init=al_rho_init, al_rho_growth=al_rho_growth,
            al_rho_max=al_rho_max, al_stagnation_tau=al_stagnation_tau,
            critic_embed_dim=critic_embed_dim, critic_hidden=critic_hidden,
        )
        self.lambda_weighted = bool(lambda_weighted)
        self.circuit_backend = circuit_backend
        self.circuit_max_nodes = circuit_max_nodes
        self.circuit_size_limit_action = circuit_size_limit_action
        self.circuit_aggregation = circuit_aggregation
        self.constraints_seen = 0
        self.constraints_inexact = 0

    def forward(self, builder, build=None, **_):
        if build is None:
            build = self.build
        if not build and not isinstance(builder, DataNodeBuilder):
            raise ValueError(
                "SemanticLossModel must be invoked with `build` on or with "
                "a provided DataNode Builder."
            )

        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        datanode.inferLocal(keys=("softmax",))
        constraint_losses = datanode.calculateLcLoss(
            circuit=True,
            circuitBackend=self.circuit_backend,
            circuitMaxNodes=self.circuit_max_nodes,
            circuitSizeLimitAction=self.circuit_size_limit_action,
            circuitAggregation=self.circuit_aggregation,
        )
        # Fraction of constraints that had to abandon the exact circuit and fall
        # back to the Product t-norm (circuit budget exceeded). Surfaced so a
        # run can report how much of its "exact" loss really was exact.
        total_lc = 0
        inexact_lc = 0

        losses = []
        for key, loss_info in constraint_losses.items():
            constraint_key = key
            if constraint_key not in self.constr:
                constraint_key = getattr(loss_info.get("lc"), "lcName", key)
            if constraint_key not in self.constr or loss_info.get("lossTensor") is None:
                continue
            loss_tensor = loss_info["lossTensor"]
            total_lc += 1
            if loss_info.get("exact") is False:
                inexact_lc += 1
            if self.lambda_weighted:
                loss_value = self._weighted_constraint_loss(constraint_key, loss_tensor)
            else:
                loss_value = loss_tensor.mean()
            self.loss[constraint_key](loss_value)
            losses.append(loss_value)

        # Running exactness tally across the epoch (reset by LossModel.reset()).
        self.constraints_seen += total_lc
        self.constraints_inexact += inexact_lc

        if losses:
            total = torch.stack([loss.reshape(()) for loss in losses]).sum()
        else:
            dtype = getattr(datanode, "current_dtype", torch.float32)
            device = getattr(datanode, "current_device", self.device)
            total = torch.zeros((), device=device, dtype=dtype)
        return total, datanode, builder

    @property
    def exact_fraction(self):
        """Fraction of evaluated constraints that used the exact circuit.

        ``1.0`` means every constraint was compiled exactly; anything lower
        means the circuit budget was exceeded and those constraints silently
        degraded to the Product t-norm, so the reported loss is not fully exact.
        """
        if not self.constraints_seen:
            return float('nan')
        return 1.0 - (self.constraints_inexact / self.constraints_seen)

    def reset(self):
        super().reset()
        self.constraints_seen = 0
        self.constraints_inexact = 0

class InferenceModel(LossModel):
    """
    Class used to train from the program execution loss.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, graph,
                 tnorm='P',
                 loss=torch.nn.BCELoss,
                 query_loss=None,
                 counting_tnorm=None,
                 sample=False, sampleSize=0, sampleGlobalLoss=False, device='auto',
                 use_gumbel=False, temperature=1.0, hard_gumbel=False,
                 pos_weight=1.0,
                 include_global_constraint_loss=False,
                 global_constraint_loss_weight=1.0,
                 executable_constraint_loss_weight=1.0):
        """
        Initializes an instance of InferenceModel.

        :param graph: The initialized graph either containing the logical expressions to be executed
            and/or called with `.compile_executable` to use the logical expressions in the dataset.
        :param tnorm: Sets the method used to perform the soft-logic translation of the logical expressions.
            Defaults to 'P' (Product).
        :param loss: Loss function to use for binary program outputs.
        :param query_loss: Optional loss function for multiclass ``queryL``
            outputs. When omitted, ``loss`` is used for backward compatibility.
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
        :param include_global_constraint_loss: Include graph.logicalConstrains loss in addition to
            executable constraint BCE loss.
        :param global_constraint_loss_weight: Weight for graph-global constraint loss.
        :param executable_constraint_loss_weight: Weight for executable BCE loss.
        """
        self.graph = graph

        super().__init__(graph, tnorm=tnorm, counting_tnorm=counting_tnorm, 
                         sample=sample, sampleSize=sampleSize, 
                         sampleGlobalLoss=sampleGlobalLoss, device=device,
                         use_gumbel=use_gumbel, temperature=temperature, 
                         hard_gumbel=hard_gumbel)

        self.loss_func = loss()
        self.query_loss_func = query_loss() if query_loss is not None else self.loss_func
        self.include_global_constraint_loss = bool(include_global_constraint_loss)
        self.global_constraint_loss_weight = float(global_constraint_loss_weight)
        self.executable_constraint_loss_weight = float(executable_constraint_loss_weight)
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

    def _tensor_device(self):
        if self.device == 'auto' or self.device is None:
            return None
        return self.device

    def _zero_loss(self, datanode, requires_grad=True):
        dtype = getattr(datanode, 'current_dtype', torch.float32)
        return torch.tensor(
            0.0,
            dtype=dtype,
            device=self._tensor_device(),
            requires_grad=requires_grad,
        )

    def _calculate_global_constraint_loss(self, datanode):
        """Return graph-level constraint loss from graph.logicalConstrains only."""
        constr_loss = datanode.calculateLcLoss(
            tnorm=self.tnorm,
            counting_tnorm=self.counting_tnorm,
            sample=self.sample,
            sampleSize=self.sampleSize,
            # Keep this path per-constraint so executable constraints are not
            # folded into the graph-global component.
            sampleGlobalLoss=False,
        )

        losses = []
        for key, loss in constr_loss.items():
            if key not in self.constr or not isinstance(loss, dict):
                continue
            loss_tensor = loss.get('lossTensor')
            if loss_tensor is None:
                continue

            loss_value = loss_tensor.clamp(min=0)
            loss_sum = loss_value[loss_value == loss_value].sum()
            self.loss[key](loss_sum)
            losses.append(loss_sum)

        if losses:
            return sum(losses)
        return self._zero_loss(datanode)

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
        if len(read_labels) == 0 and not self.include_global_constraint_loss:
            raise ValueError('No active executable constraint labels found in datanode.')

        lc_context = None
        if read_labels:
            # Prepare shared context for executable loss calculation.
            lc_context = datanode._prepareLcLossContext(
                tnorm=self.tnorm,
                counting_tnorm=self.counting_tnorm,
            )

        executable_losses = []
        if read_labels:
            for lcName, lc in self.constr.items():
                if f'{lcName}/label' not in read_labels:
                    continue

                if not lc.active:
                    continue

                # Use datanode method to get the label.
                raw_lbl = datanode.getExecutableConstraintLabel(lcName)
                
                loss_dict = datanode.calculateSingleLcLoss(
                    lcName,
                    tnorm=self.tnorm,
                    counting_tnorm=self.counting_tnorm,
                    _context=lc_context
                )

                selection_distribution = loss_dict.get('selectionDistribution')
                if selection_distribution is not None:
                    predicted = selection_distribution.float().reshape(-1)
                    target = raw_lbl.float().to(predicted.device).reshape(-1)
                    if target.numel() != predicted.numel():
                        raise ValueError(
                            f"miotaL label for {lcName} has {target.numel()} values, "
                            f"but the constraint grounded {predicted.numel()} candidates"
                        )
                    if not torch.all((target == 0) | (target == 1)):
                        raise ValueError(
                            f"miotaL label for {lcName} must be a binary multi-hot vector"
                        )
                    if predicted.numel() == 0:
                        executable_losses.append(predicted.sum() * 0.0)
                        continue
                    eps = 1e-6
                    clamped = predicted.clamp(eps, 1.0 - eps)
                    predicted = predicted + (clamped - predicted).detach()
                    constraint_loss = self.loss_func(predicted, target)
                    executable_losses.append(constraint_loss)
                    continue

                if MONITORING_AVAILABLE and loss_dict.get('loss') is not None:
                    lcRepr = f'{lc.__class__.__name__} {lc.strEs()}'
                    log_single_lc(
                        constraint_name=lcName,
                        loss_dict=loss_dict,
                        label_tensor=raw_lbl,
                        lc_formulation=lcRepr
                    )

                query_distribution = loss_dict.get('queryDistribution')
                if query_distribution is not None:
                    inner_lc = getattr(lc, "innerLC", lc)
                    if isinstance(inner_lc, queryL) and inner_lc.is_multi_answer:
                        distribution = query_distribution.float()
                        _target, _chosen, _losses, constraint_loss = multi_query_joint_nll(
                            distribution,
                            raw_lbl,
                            inner_lc.num_subclasses,
                            label_name=f"multi-answer queryL {lcName}",
                        )
                        executable_losses.append(constraint_loss)
                        continue
                    try:
                        constraint_loss = self.query_loss_func(
                            query_distribution.float(), raw_lbl.long()
                        )
                    except Exception as exc:
                        raise TypeError(
                            "queryL executable constraints produce a multiclass distribution. "
                            "Use a multiclass loss such as domiknows.program.loss.NBCrossEntropyLoss."
                        ) from exc

                    if self._diag_step < self._diag_budget:
                        try:
                            qd = query_distribution.detach().float().flatten()
                            lb = raw_lbl.detach().long().flatten()
                            cl = constraint_loss.detach().float().flatten()
                            print(
                                f"[INFER_DIAG step={self._diag_step} lc={lcName}] "
                                f"queryDistribution={qd.tolist()} lbl={lb.tolist()} "
                                f"loss={cl.tolist()}",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"[INFER_DIAG error] {e}", flush=True)

                    executable_losses.append(constraint_loss)
                    continue

                if loss_dict.get('loss') is None:
                    continue
                    
                inner_lc = getattr(lc, "innerLC", lc)
                is_sumL = isinstance(inner_lc, sumL)
                constr_out = loss_dict['conversionSigmoid']
                if is_sumL:
                    # The numeric label has already been consumed while
                    # calculating the sum constraint.  conversionSigmoid is
                    # the resulting satisfaction probability, so its training
                    # target is true rather than the requested count itself.
                    lbl = torch.ones_like(
                        constr_out,
                        dtype=constr_out.dtype,
                        device=constr_out.device,
                    )
                else:
                    lbl = raw_lbl.float().to(device=constr_out.device)
                    if lbl.shape != constr_out.shape and lbl.numel() == 1:
                        lbl = torch.ones_like(
                            constr_out,
                            dtype=constr_out.dtype,
                            device=constr_out.device,
                        ) * lbl.reshape(-1)[0]
                #if torch.equal(constr_out, lbl):
                #    print(f"Constraint {lcName}: loss={constr_out}, label={lbl}" + (f", is_sumL={is_sumL}" if is_sumL else ""))
                # Avoid BCELoss saturation cliff using a STRAIGHT-THROUGH clamp:
                # forward sees a clamped value (no -inf log), but the gradient
                # flows back as if no clamp existed. A vanilla `tensor.clamp(...)`
                # would zero the gradient at saturation — which kills the recovery
                # gradient when convSig=0 with lbl=1 (the most informative case
                # for pushing atoms back up). Disabled if DOMIKNOWS_INFER_NO_CLAMP=1.
                import os as _os
                if not is_sumL and _os.environ.get('DOMIKNOWS_INFER_NO_CLAMP', '0') != '1':
                    _eps = 1e-6
                    _co_clamped = constr_out.clamp(_eps, 1.0 - _eps)
                    # Straight-through: forward = clamped, backward = identity.
                    constr_out = constr_out + (_co_clamped - constr_out).detach()
                constraint_loss = self.loss_func(constr_out.float(), lbl.float())

                if self._diag_step < self._diag_budget:
                    try:
                        co = constr_out.detach().float().flatten()
                        lb = lbl.detach().float().flatten()
                        cl = constraint_loss.detach().float().flatten()
                        # Dump a handful of atom probabilities that feed into this
                        # LC via _prepareLcLossContext, so we can see how close to
                        # saturation the atoms are on this step.
                        atom_summary = ""
                        try:
                            probs_ctx = None
                            for k in ('probs', 'predictions', 'softmax', 'localPredictions'):
                                if isinstance(lc_context, dict) and k in lc_context:
                                    probs_ctx = lc_context[k]
                                    break
                            if isinstance(probs_ctx, dict):
                                keys = list(probs_ctx.keys())[:3]
                                bits = []
                                for k in keys:
                                    v = probs_ctx[k]
                                    if hasattr(v, 'detach'):
                                        vv = v.detach().float().flatten()
                                        bits.append(f"{k}:{vv[:4].tolist()}")
                                if bits:
                                    atom_summary = " atoms=[" + "; ".join(bits) + "]"
                        except Exception:
                            pass
                        print(
                            f"[INFER_DIAG step={self._diag_step} lc={lcName}] "
                            f"convSig={co.tolist()} lbl={lb.tolist()} "
                            f"loss={cl.tolist()} is_sumL={is_sumL}{atom_summary}",
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

                executable_losses.append(constraint_loss)

        executable_loss = sum(executable_losses) if executable_losses else self._zero_loss(datanode)
        if self.include_global_constraint_loss:
            global_loss = self._calculate_global_constraint_loss(datanode)
        else:
            global_loss = self._zero_loss(datanode)
        loss = (
            self.executable_constraint_loss_weight * executable_loss
            + self.global_constraint_loss_weight * global_loss
        )
        self.last_executable_loss = executable_loss.detach() if torch.is_tensor(executable_loss) else executable_loss
        self.last_global_loss = global_loss.detach() if torch.is_tensor(global_loss) else global_loss
        self.last_total_constraint_loss = loss.detach() if torch.is_tensor(loss) else loss
            
        if MONITORING_AVAILABLE:
            log_memory() 
        
        self.inferenceLogger.info(f"Total loss: {loss.item()}")

        if self._diag_step < self._diag_budget:
            try:
                # Walk every datanode reachable from the root and print any
                # attribute whose key ends in '<local/softmax>'. Limit to the
                # first few matches so output stays readable.
                printed = 0
                import os as _os_diag
                limit = int(_os_diag.environ.get('DOMIKNOWS_INFER_DIAG_LIMIT', '6'))
                def _walk(dn, depth=0):
                    nonlocal printed
                    if printed >= limit:
                        return
                    attrs = getattr(dn, 'attributes', None) or {}
                    for key, val in attrs.items():
                        if printed >= limit:
                            return
                        if 'local/softmax' not in str(key):
                            continue
                        if not hasattr(val, 'detach'):
                            continue
                        vv = val.detach().float().flatten()
                        print(
                            f"[INFER_DIAG step={self._diag_step} dn={dn.getOntologyNode().name if dn.getOntologyNode() else '?'} "
                            f"key={key}] softmax={vv[:6].tolist()}",
                            flush=True,
                        )
                        printed += 1
                    for child in (getattr(dn, 'getChildDataNodes', lambda: [])() or []):
                        _walk(child, depth + 1)
                _walk(datanode)
                if printed == 0:
                    print(
                        f"[INFER_DIAG step={self._diag_step} concept] "
                        f"no 'local/softmax' attribute found on datanode tree",
                        flush=True,
                    )
            except Exception as e:
                print(f"[INFER_DIAG concept error] {type(e).__name__}: {e}", flush=True)
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
        explicit_temperature = temperature is not None
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
            if self.training and not explicit_temperature:
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
                    # Keep per-constraint successes when global aggregation is
                    # disabled.  Previously this replacement happened whenever
                    # a global-success sample existed, effectively making
                    # ``sampleGlobalLoss=False`` behave like global sampling.
                    if self.sampleGlobalLoss and constr_loss["globalSuccessCounter"] > 0:
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
