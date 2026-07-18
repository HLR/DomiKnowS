"""Verifies the R-mechanism training-comparison harness on a real (light) task.

The harness (``domiknows.program.training_comparison``) trains the *same*
DomiKnowS task under the baseline mechanism and each R mechanism (R1 compiled
LC loss, R5A augmented-Lagrangian duals, and their combination) and reports a
comparison table. This test drives it on a small but genuine DomiKnowS
program — ``a`` contains ``b``, a 2-class ``answer`` on ``b``, and an
``exactL(answer.one, TARGET)`` counting constraint — with learnable,
constraint-consistent data, and checks the harness captures the right structure:

* R1 is numerics-preserving: ``r1_compiled`` must match ``baseline`` on the
  violation trajectory and task metrics exactly (only speed may differ).
* the augmented variants actually run the closed-form dual (no constraint
  optimizer) and still train.
* every variant produces a result with the constraint-loss time recorded.

The conll04 adapter (``Tasks/conll04/main-bert-compare.py``) uses the same
harness at real scale, where R1's speed win becomes visible.
"""

import numpy as np
import pytest
import torch

from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import exactL
from domiknows.sensor import Sensor
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.model.base import Mode
from domiknows.program.training_comparison import TrainingComparison, DEFAULT_VARIANTS

N, M, TARGET = 8, 6, 2


def _make_dataset(groups=3, seed=1):
    """Groups of M items; exactly TARGET are 'one', and feature[:,0] encodes
    the label so the classifier can actually learn it (constraint-consistent)."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(groups):
        feats = rng.randn(M, N)
        labels = [0] * M
        for j in rng.choice(M, TARGET, replace=False):
            labels[j] = 1
        for j in range(M):
            feats[j, 0] = 2.0 if labels[j] == 1 else -2.0
        data.append({'a': [0], 'b': feats.tolist(), 'label': labels})
    return data


def _build_program(variant, dataset):
    Sensor.clear()
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('cmp') as graph:
        a = Concept(name='a'); b = Concept(name='b')
        (acb,) = a.contains(b)
        answer = b(name='answer', ConceptClass=EnumConcept, values=['zero', 'one'])
        exactL(answer.__getattr__('one'), limit=TARGET)

    class Net(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.l = torch.nn.Sequential(torch.nn.Linear(n, n), torch.nn.ReLU(),
                                         torch.nn.Linear(n, 2))

        def forward(self, rel, x):
            return self.l(x)

    a['index'] = ReaderSensor(keyword='a')
    b['index'] = ReaderSensor(keyword='b')
    b['temp'] = ReaderSensor(keyword='label')
    b[acb] = EdgeSensor(b['index'], a['index'], relation=acb,
                        forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[answer] = ModuleLearner(acb, 'index', module=Net(N), device='cpu')
    b[answer] = FunctionalSensor(acb, 'temp', forward=lambda _, l: l, label=True)

    # Honour the variant's Program class (R2 needs SemanticLossProgram); any
    # future mechanism that needs its own Program works without a change here.
    program_cls = variant.resolve_program_class(PrimalDualProgram)
    program = program_cls(
        graph, SolverModel, poi=[a, b, answer], inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()), device='cpu', beta=1.0,
        **variant.program_kwargs)
    program._answer = answer
    return program


def _make_evaluate(dataset):
    def evaluate(program):
        answer = program._answer
        program.model.eval(); program.model.mode(Mode.TEST)
        errs = []
        with torch.no_grad():
            for dn in program.populate(dataset=dataset):
                ones = sum(
                    1 for ch in dn.getChildDataNodes()
                    if ch.getAttribute(answer, 'local/argmax').argmax().item() == 1)
                errs.append(abs(ones - TARGET))
        return {'count_err': sum(errs) / len(errs)}
    return evaluate


@pytest.fixture(scope='module')
def comparison_result():
    dataset = _make_dataset()
    cmp = TrainingComparison(
        build_program=lambda v: _build_program(v, dataset),
        dataset=dataset,
        evaluate=_make_evaluate(dataset),
        epochs=15, seed=0, device='cpu', violation_tnorm='P',
        train_kwargs=dict(c_warmup_iters=2, c_freq=1),
        optim=lambda p: torch.optim.Adam(p, lr=5e-3),
        print_table=True,
    )
    return cmp.run()


def test_all_variants_produced_results(comparison_result):
    names = {r.variant.name for r in comparison_result.rows}
    assert names == {v.name for v in DEFAULT_VARIANTS}
    for r in comparison_result.rows:
        assert r.error is None, f'{r.variant.name} errored: {r.error}'
        assert r.closs_time_s >= 0.0
        assert not np.isnan(r.violation_before)
        assert not np.isnan(r.violation_after)


def test_r1_is_numerics_identical_to_baseline(comparison_result):
    """R1 changes only how the LC loss is computed, not the result — so the
    compiled variant must match the baseline's violations and metrics exactly."""
    base = comparison_result.by_name('baseline')
    r1 = comparison_result.by_name('r1_compiled')
    assert r1.violation_before == pytest.approx(base.violation_before, abs=1e-6)
    assert r1.violation_after == pytest.approx(base.violation_after, abs=1e-6)
    assert r1.metrics['count_err'] == pytest.approx(base.metrics['count_err'], abs=1e-9)


def test_augmented_variants_use_closed_form_dual(comparison_result):
    """The augmented variants must have trained (violation recorded) — they use
    buffer multipliers with no constraint optimizer (verified structurally)."""
    for name in ('r5a_augmented', 'r1_r5a'):
        r = comparison_result.by_name(name)
        assert r.variant.program_kwargs.get('dual_algorithm') == 'augmented'
        assert r.error is None


def test_amortized_variants_present(comparison_result):
    """R5B (amortized DualCritic) variants ran without error."""
    for name in ('r5b_amortized', 'r1_r5b'):
        r = comparison_result.by_name(name)
        assert r is not None
        assert r.variant.program_kwargs.get('dual_granularity') == 'amortized'
        assert r.error is None
        assert not np.isnan(r.violation_after)


def test_semantic_variants_use_their_own_program_class(comparison_result):
    """R2 needs SemanticLossProgram, which the harness supplies via
    Variant.program_class without any builder-side special-casing."""
    from domiknows.program.lossprogram import SemanticLossProgram
    for name in ('r2_semantic', 'r2_r5a'):
        r = comparison_result.by_name(name)
        assert r is not None
        assert r.variant.program_class is SemanticLossProgram
        assert r.error is None
        assert not np.isnan(r.violation_after)


def test_training_reduces_violation(comparison_result):
    """With constraint-consistent, learnable data, training should not increase
    the mean constraint violation for any variant (drop >= a small tolerance)."""
    for r in comparison_result.rows:
        assert r.violation_drop >= -0.05, \
            f'{r.variant.name} increased violation by {-r.violation_drop:.4f}'


def test_construct_augmented_has_no_constraint_optimizer():
    """Structural guarantee behind the augmented variants: no copt is built."""
    dataset = _make_dataset(groups=1)
    from domiknows.program.training_comparison import Variant
    prog = _build_program(Variant('aug', {'dual_algorithm': 'augmented'}), dataset)
    assert prog._make_copt(0.05) is None


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
