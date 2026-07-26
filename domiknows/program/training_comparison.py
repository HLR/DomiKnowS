"""Reusable harness for comparing constraint-training mechanisms.

Motivation: the "R" line of improvements to DomiKnowS constraint integration
(R1 compiled LC loss, R5 augmented-Lagrangian duals, and the planned R2–R4)
are all meant to improve *training* — either its speed (R1) or its constraint
satisfaction / convergence (R5+). This harness makes that measurable and
comparable on a fixed task, and is designed so each future R mechanism plugs
in as one more :class:`Variant`.

For each variant it trains a **freshly built** program from the same seed for
the same number of epochs, then reports:

* ``train_time_s``      — total wall-clock of ``program.train`` (R1 speed).
* ``closs_time_s``      — cumulative time spent in the constraint model's
                          forward pass, i.e. building the LC loss (R1's direct
                          target; measured via an instance timer on
                          ``cmodel.forward``).
* ``violation_before`` / ``violation_after`` — mean unweighted constraint
                          violation over the dataset before/after training
                          (R5's target: lower-after and a bigger drop mean the
                          constraints are better satisfied).
* task metrics          — whatever the caller's ``evaluate`` returns
                          (e.g. macro-F1 for conll04, exact-count for counting
                          tasks).

Usage sketch::

    def build_program(variant):
        # (re)declare the graph + sensors and return a PrimalDualProgram whose
        # construction merges variant.program_kwargs.
        ...
        return PrimalDualProgram(graph, Model, ..., **variant.program_kwargs)

    cmp = TrainingComparison(build_program, dataset, evaluate=my_eval, epochs=20)
    result = cmp.run()
    print(result.render())
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class Variant:
    """One training mechanism to compare.

    ``program_kwargs`` are merged into the caller's program construction.
    ``program_class`` lets a mechanism require a *different* Program class
    (R2's semantic loss needs ``SemanticLossProgram`` rather than
    ``PrimalDualProgram``); builders should honour it via
    ``variant.program_class or <their default>`` so future R mechanisms need no
    builder changes.
    """
    name: str
    program_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ''
    program_class: Optional[type] = None
    #: builder hints for mechanisms that change the model rather than the
    #: Program (e.g. R4's ``{'r4': 'synth'|'refine'}``); ignored by the harness.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve_program_class(self, default):
        """The Program class to build for this variant."""
        return self.program_class if self.program_class is not None else default


def _semantic_loss_program():
    """Imported lazily so this module stays free of a Program-layer import cycle."""
    from domiknows.program.lossprogram import SemanticLossProgram
    return SemanticLossProgram


#: Baseline vs. the mechanisms delivered so far. Extend this list as new R
#: mechanisms land (e.g. R3/R4).
DEFAULT_VARIANTS: List[Variant] = [
    Variant('baseline', {},
            'ascent duals + interpreter LC loss (pre-R1/R5)'),
    Variant('r1_compiled', {'compile_lc': True},
            'R1: compiled (batched-gather) LC loss'),
    Variant('r5a_augmented', {'dual_algorithm': 'augmented'},
            'R5A: augmented-Lagrangian duals'),
    Variant('r5b_amortized', {'dual_granularity': 'amortized'},
            'R5B: amortized per-grounding duals (DualCritic)'),
    Variant('r1_r5a', {'compile_lc': True, 'dual_algorithm': 'augmented'},
            'R1 + R5A combined'),
    Variant('r1_r5b', {'compile_lc': True, 'dual_granularity': 'amortized'},
            'R1 + R5B combined (critic reads compiled literal features)'),
    Variant('r2_semantic', {},
            'R2: exact semantic loss, -log P(constraint) via circuit WMC',
            program_class=_semantic_loss_program()),
    Variant('r2_r5a', {'lambda_weighted': True, 'dual_algorithm': 'augmented',
                       'training_style': 'primal_dual'},
            'R2 + R5A: exact semantic loss under augmented-Lagrangian duals',
            program_class=_semantic_loss_program()),
]


def _structured_program():
    """Imported lazily so this module stays free of a Program-layer import cycle."""
    from domiknows.program.lossprogram import StructuredProgram
    return StructuredProgram


#: R3/R4 change the model's *forward pass*, not just the Program's kwargs. Since
#: ``StructuredProgram`` builds a ``StructuredModel`` itself, these behave like
#: any other variant — the builder only has to honour
#: ``variant.resolve_program_class(...)``, exactly as it already does for R2.
#:
#: Kept out of :data:`DEFAULT_VARIANTS` because they change the architecture, so
#: a run should opt into comparing them explicitly.
R4_VARIANTS: List[Variant] = [
    Variant('r4_refine', {'refine': True, 'factor_graph': False},
            'R4B: constraint refinement layer (violation-gradient messages)',
            program_class=_structured_program(), metadata={'r4': 'refine'}),
    Variant('r4_refine_ablate', {'refine': True, 'factor_graph': False,
                                 'belief_flow': 'constraint_only'},
            'R4B ablation: refinement kept out of the supervised loss',
            program_class=_structured_program(), metadata={'r4': 'refine'}),
]


#: R3 replaces the head with exact inference in ``p(y | x, phi)`` and, with
#: ``partition='auto'``, drops the constraints it enforces from the loss/duals.
R3_VARIANTS: List[Variant] = [
    Variant('r3_factorgraph', {'refine': False, 'factor_graph': True},
            'R3: factor-graph head — constrained marginals as the forward pass',
            program_class=_structured_program(), metadata={'r3': 'factorgraph'}),
    Variant('r3_r4', {'refine': True, 'factor_graph': True},
            'R3 + R4B: refinement then exact constrained marginals',
            program_class=_structured_program(), metadata={'r3': 'factorgraph',
                                                           'r4': 'refine'}),
    Variant('r3_r1_r5a', {'refine': False, 'factor_graph': True,
                          'compile_lc': True, 'dual_algorithm': 'augmented'},
            'R3 + R1 + R5A: structure in the model, compiled AL-dual loss outside',
            program_class=_structured_program(), metadata={'r3': 'factorgraph'}),
]


@dataclass
class VariantResult:
    variant: Variant
    train_time_s: float
    closs_time_s: float
    violation_before: float
    violation_after: float
    metrics: Dict[str, float]
    error: Optional[str] = None

    @property
    def violation_drop(self) -> float:
        return self.violation_before - self.violation_after


@dataclass
class ComparisonResult:
    rows: List[VariantResult]

    def by_name(self, name: str) -> Optional[VariantResult]:
        for r in self.rows:
            if r.variant.name == name:
                return r
        return None

    def render(self) -> str:
        """A compact fixed-width comparison table."""
        metric_keys: List[str] = []
        for r in self.rows:
            for k in r.metrics:
                if k not in metric_keys:
                    metric_keys.append(k)

        headers = ['variant', 'train_s', 'closs_s', 'viol_before',
                   'viol_after', 'viol_drop'] + metric_keys
        widths = {h: len(h) for h in headers}

        def fmt(v):
            if isinstance(v, float):
                return f'{v:.4f}'
            return str(v)

        table = []
        for r in self.rows:
            if r.error is not None:
                row = {'variant': r.variant.name, 'train_s': 'ERROR'}
                row.update({h: '' for h in headers if h not in row})
                row['viol_before'] = r.error[:40]
            else:
                row = {
                    'variant': r.variant.name,
                    'train_s': fmt(r.train_time_s),
                    'closs_s': fmt(r.closs_time_s),
                    'viol_before': fmt(r.violation_before),
                    'viol_after': fmt(r.violation_after),
                    'viol_drop': fmt(r.violation_drop),
                }
                for k in metric_keys:
                    row[k] = fmt(r.metrics.get(k, float('nan')))
            for h in headers:
                widths[h] = max(widths[h], len(row.get(h, '')))
            table.append(row)

        def line(cells):
            return '  '.join(cells[h].ljust(widths[h]) for h in headers)

        out = [line({h: h for h in headers}),
               '  '.join('-' * widths[h] for h in headers)]
        out += [line(row) for row in table]

        legend = ['', 'variants:']
        for r in self.rows:
            legend.append(f'  {r.variant.name}: {r.variant.description}')
        return '\n'.join(out + legend)


def _install_closs_timer(program) -> Dict[str, float]:
    """Wrap ``program.cmodel.forward`` to accumulate its wall-clock time.

    ``nn.Module.__call__`` dispatches to ``forward``, so replacing the bound
    method transparently times every constraint-loss computation without
    touching the training loop.
    """
    timer = {'t': 0.0, 'calls': 0}
    cmodel = program.cmodel
    orig_forward = cmodel.forward

    def timed_forward(*args, **kwargs):
        start = perf_counter()
        try:
            return orig_forward(*args, **kwargs)
        finally:
            timer['t'] += perf_counter() - start
            timer['calls'] += 1

    cmodel.forward = timed_forward
    return timer


def _seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingComparison:
    """Compare constraint-training mechanisms on one task.

    :param build_program: ``callable(Variant) -> PrimalDualProgram``. Must
        build a *fresh* program (re-declaring sensors/learners) each call and
        merge ``variant.program_kwargs`` into the program construction.
    :param dataset: training data passed to ``program.train`` and reused for
        the violation measurement.
    :param evaluate: optional ``callable(program) -> dict`` returning task
        metrics (e.g. F1, accuracy) after training.
    :param variants: mechanisms to compare (defaults to :data:`DEFAULT_VARIANTS`).
    :param epochs / seed / device: training controls, applied identically to
        every variant.
    :param train_kwargs: extra kwargs forwarded to ``program.train`` (e.g.
        ``c_warmup_iters``, ``c_freq``, ``valid_set``); the same for all variants.
    :param optim: ``callable(params) -> torch.optim.Optimizer`` for the primal
        (model) step; defaults to SGD(lr=1e-3).
    :param violation_tnorm: t-norm for the (unweighted) violation metric; if
        None, uses each program's own ``cmodel.tnorm`` — pass a fixed value to
        measure all variants on identical footing.
    """

    def __init__(self, build_program: Callable[[Variant], Any], dataset,
                 evaluate: Optional[Callable[[Any], Dict[str, float]]] = None,
                 variants: Optional[List[Variant]] = None,
                 epochs: int = 10, seed: int = 0, device: str = 'cpu',
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 optim: Optional[Callable] = None,
                 violation_tnorm: Optional[str] = None,
                 violation_dataset=None,
                 warmup: bool = True,
                 continue_on_error: bool = True,
                 print_table: bool = True,
                 verbose: bool = True):
        self.build_program = build_program
        self.dataset = dataset
        # Set of items for the before/after violation metric. Defaults to the
        # training set; pass a small subset for expensive models (e.g. BERT)
        # where a full extra pass per variant is prohibitive.
        self.violation_dataset = violation_dataset if violation_dataset is not None else dataset
        self.evaluate = evaluate
        self.variants = variants if variants is not None else list(DEFAULT_VARIANTS)
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.train_kwargs = dict(train_kwargs) if train_kwargs else {}
        self.optim = optim if optim is not None else (
            lambda params: torch.optim.SGD(params, lr=1e-3))
        self.violation_tnorm = violation_tnorm
        # A throwaway build+train of the first variant before the measured loop
        # absorbs process-wide one-time costs (imports, lazy graph/datanode
        # setup) that would otherwise be charged entirely to whichever variant
        # runs first, making its train_time incomparable.
        self.warmup = warmup
        self.continue_on_error = continue_on_error
        self.print_table = print_table
        self.verbose = verbose

    def _log(self, msg):
        # print() (not logging) so results reach the screen even under
        # setProductionLogMode(), which filters the framework loggers.
        if self.verbose:
            print(msg, flush=True)

    # -- violation metric ---------------------------------------------------

    def mean_violation(self, program) -> float:
        """Mean unweighted constraint violation over the dataset.

        Runs the model forward per item, builds the datanode, and averages the
        NaN-filtered clamped LC loss tensors across all constraints — a direct,
        multiplier-free measure of how far the current predictions are from
        satisfying the graph constraints (0 = fully satisfied).
        """
        tnorm = self.violation_tnorm or getattr(program.cmodel, 'tnorm', 'P')
        counting_tnorm = getattr(program.cmodel, 'counting_tnorm', None)
        # Score executable-wrapped constraints too, so this metric covers the
        # same constraint population the exact-circuit path evaluates. Without
        # it, a graph using execute() would have its t-norm variants measured on
        # a strictly smaller constraint set than the semantic-loss variants.

        program.model.eval()
        totals: List[float] = []
        with torch.no_grad():
            for data in self.violation_dataset:
                out = program.model(data)
                builder = out[-1]
                builder.createBatchRootDN()
                datanode = builder.getDataNode(device=self.device)
                lc_losses = datanode.calculateLcLoss(
                    tnorm=tnorm, counting_tnorm=counting_tnorm,
                    includeExecutable=True)
                for res in lc_losses.values():
                    lt = res.get('lossTensor') if isinstance(res, dict) else None
                    if lt is None:
                        continue
                    lt = lt.clamp(min=0)
                    lt = lt[lt == lt]  # drop NaN
                    if lt.numel():
                        totals.append(float(lt.mean()))
        program.model.train()
        if not totals:
            return float('nan')
        return sum(totals) / len(totals)

    # -- run ----------------------------------------------------------------

    def _train(self, program):
        program.train(
            training_set=self.dataset,
            train_epoch_num=self.epochs,
            Optim=self.optim,
            device=self.device,
            **self.train_kwargs,
        )

    def _run_warmup(self):
        """One short throwaway run to absorb one-time process costs."""
        if not self.variants:
            return
        _seed_everything(self.seed)
        try:
            program = self.build_program(self.variants[0])
            saved_epochs = self.epochs
            self.epochs = 1
            try:
                self._train(program)
                self.mean_violation(program)
            finally:
                self.epochs = saved_epochs
        except Exception:  # noqa: BLE001 - warmup failures are non-fatal
            pass

    def run(self) -> ComparisonResult:
        self._log('=' * 72)
        self._log(f'Training-mechanism comparison — {len(self.variants)} variants, '
                  f'{self.epochs} epochs, seed {self.seed}, device {self.device}')
        self._log('=' * 72)

        if self.warmup:
            self._log('[warmup] absorbing one-time init cost ...')
            self._run_warmup()

        rows: List[VariantResult] = []
        for i, variant in enumerate(self.variants, 1):
            self._log(f'\n[{i}/{len(self.variants)}] {variant.name}: {variant.description}')
            _seed_everything(self.seed)
            try:
                program = self.build_program(variant)
                timer = _install_closs_timer(program)

                viol_before = self.mean_violation(program)
                self._log(f'    violation before training: {viol_before:.4f}')

                start = perf_counter()
                self._train(program)
                train_time = perf_counter() - start

                viol_after = self.mean_violation(program)
                metrics = self.evaluate(program) if self.evaluate else {}

                metric_str = '  '.join(f'{k}={v:.4f}' for k, v in metrics.items())
                self._log(f'    done in {train_time:.2f}s '
                          f'(constraint-loss {timer["t"]:.2f}s) | '
                          f'violation after {viol_after:.4f} '
                          f'(drop {viol_before - viol_after:+.4f})'
                          + (f' | {metric_str}' if metric_str else ''))

                rows.append(VariantResult(
                    variant=variant,
                    train_time_s=train_time,
                    closs_time_s=timer['t'],
                    violation_before=viol_before,
                    violation_after=viol_after,
                    metrics=metrics,
                ))
            except Exception as exc:  # noqa: BLE001 - surface but keep comparing
                if not self.continue_on_error:
                    raise
                import traceback
                traceback.print_exc()
                self._log(f'    ERROR: {type(exc).__name__}: {exc}')
                rows.append(VariantResult(
                    variant=variant, train_time_s=float('nan'),
                    closs_time_s=float('nan'), violation_before=float('nan'),
                    violation_after=float('nan'), metrics={},
                    error=f'{type(exc).__name__}: {exc}'))

        result = ComparisonResult(rows)
        if self.print_table:
            self._log('\n' + result.render() + '\n')
        return result
