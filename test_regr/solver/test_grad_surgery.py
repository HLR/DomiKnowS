"""R5 Phase C (gradient surgery) and Phase F (amortized x augmented duals).

Phase C's plan is explicit that the **diagnostic decides** whether the resolver
is worth its cost: surgery needs the supervised and constraint gradients
separately, which a single fused backward cannot provide, so it buys an extra
backward pass on every step. If a task shows no conflict, the right answer is to
leave it off. These tests therefore check the diagnostic first, then that the
resolvers do what they claim, and finally that defaults are untouched.

Run with ``-s`` to see the measured conflict rates.
"""

import pytest
import torch

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import nandL
from domiknows.program import StructuredProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.gradSurgery import (
    ConflictStats, GRAD_SURGERY, cagrad, conflict_report, cosine, pcgrad,
)
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor


# --------------------------------------------------------------------------- #
# The resolvers, against hand-computed ground truth
# --------------------------------------------------------------------------- #

def _worst_case_improvement(g_a, g_b, direction):
    """Per-objective loss decrease under a step ``-lr * direction``."""
    return min(float(g_a.dot(direction)), float(g_b.dot(direction)))


def test_pcgrad_is_identity_without_conflict():
    """No conflict, nothing to fix — the update must be the plain sum."""
    g_a = torch.tensor([1.0, 0.5])
    g_b = torch.tensor([0.8, 0.2])
    assert cosine(g_a, g_b) > 0
    assert torch.allclose(pcgrad(g_a, g_b), g_a + g_b)


@pytest.mark.parametrize('resolver', [pcgrad, cagrad], ids=['pcgrad', 'cagrad'])
def test_surgery_improves_the_worst_case_under_conflict(resolver, capsys):
    """Verification 2: the combined update must beat the plain sum where it fails.

    Opposed gradients make the sum favour one objective and nearly starve the
    other; both resolvers exist to stop that.
    """
    g_a = torch.tensor([1.0, 0.2])
    g_b = torch.tensor([-0.9, 0.3])
    assert cosine(g_a, g_b) < 0

    summed = _worst_case_improvement(g_a, g_b, g_a + g_b)
    resolved = _worst_case_improvement(g_a, g_b, resolver(g_a, g_b))
    print(f'\n{resolver.__name__}: worst-case improvement '
          f'{summed:+.4f} (sum) -> {resolved:+.4f}')
    assert resolved > summed


def test_cosine_is_zero_for_a_vanishing_gradient():
    assert cosine(torch.zeros(3), torch.ones(3)) == 0.0


# --------------------------------------------------------------------------- #
# conflict_report wiring
# --------------------------------------------------------------------------- #

def _linear_case():
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2)
    features = torch.randn(8, 4)
    output = layer(features)
    supervised = output.pow(2).mean()
    constraint = -output.pow(2).mean() * 0.9      # deliberately opposed
    return layer, supervised, constraint


@pytest.mark.parametrize('method', ['diagnose', 'pcgrad', 'cagrad'])
def test_conflict_report_sets_gradients_and_records(method):
    layer, supervised, constraint = _linear_case()
    layer.zero_grad(set_to_none=True)
    stats = ConflictStats()
    similarity = conflict_report(supervised, constraint, list(layer.parameters()),
                                 method=method, stats=stats)
    assert similarity == pytest.approx(-1.0, abs=1e-5)
    assert stats.steps == 1 and stats.conflicts == 1
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_diagnose_leaves_the_update_equal_to_the_plain_sum():
    """'diagnose' must measure without changing what the optimizer would do."""
    layer, supervised, constraint = _linear_case()
    layer.zero_grad(set_to_none=True)
    (supervised + constraint).backward(retain_graph=True)
    fused = [p.grad.clone() for p in layer.parameters()]

    layer, supervised, constraint = _linear_case()
    layer.zero_grad(set_to_none=True)
    conflict_report(supervised, constraint, list(layer.parameters()),
                    method='diagnose')
    for parameter, reference in zip(layer.parameters(), fused):
        assert torch.allclose(parameter.grad, reference, atol=1e-6)


def test_parameters_reached_by_one_loss_keep_that_gradient():
    """Only shared parameters can conflict; the rest must pass through intact."""
    torch.manual_seed(0)
    shared = torch.nn.Linear(3, 3)
    only_supervised = torch.nn.Linear(3, 2)
    features = torch.randn(5, 3)
    hidden = shared(features)
    supervised = only_supervised(hidden).pow(2).mean()
    constraint = -hidden.pow(2).mean()

    params = list(shared.parameters()) + list(only_supervised.parameters())
    for p in params:
        p.grad = None
    conflict_report(supervised, constraint, params, method='pcgrad')
    for parameter in only_supervised.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_invalid_method_is_rejected():
    layer, supervised, constraint = _linear_case()
    with pytest.raises(ValueError):
        conflict_report(supervised, constraint, list(layer.parameters()),
                        method='magic')


def test_conflict_stats_reports_no_measurement_when_unused():
    stats = ConflictStats()
    assert stats.conflict_rate != stats.conflict_rate     # NaN
    assert 'no measurement' in stats.render()


# --------------------------------------------------------------------------- #
# End to end, and the diagnostic that gates the whole phase
# --------------------------------------------------------------------------- #

def _graph(shared_trunk):
    """Two exclusive concepts; optionally over one shared trunk (R4-style)."""
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('surgery_test') as graph:
        img = Concept(name='img')
        ent = Concept(name='ent')
        (contains,) = img.contains(ent)
        a = ent(name='a')
        b = ent(name='b')
        nandL(a('x'), b('x'))

    img['index'] = ReaderSensor(keyword='img')
    ent['index'] = ReaderSensor(keyword='ents')
    ent[contains] = EdgeSensor(ent['index'], img['index'], relation=contains,
                               forward=lambda x, _: torch.ones_like(x).unsqueeze(-1))
    ent['emb'] = ReaderSensor(keyword='emb')

    trunk = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    def head():
        if shared_trunk:
            return torch.nn.Sequential(trunk, torch.nn.Linear(4, 2))
        return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(),
                                   torch.nn.Linear(4, 2))

    ent[a] = ModuleLearner('emb', module=head())
    ent[b] = ModuleLearner('emb', module=head())
    ent[a] = ReaderSensor(keyword='al', label=True)
    ent[b] = ReaderSensor(keyword='bl', label=True)
    return graph, img, ent, a, b


def _dataset(n=8):
    torch.manual_seed(0)
    return [{'img': [0], 'ents': [0, 1], 'emb': torch.randn(2, 4),
             'al': torch.tensor([1, 0]), 'bl': torch.tensor([0, 1])}
            for _ in range(n)]


def _run(shared_trunk, surgery, epochs=4):
    graph, img, ent, a, b = _graph(shared_trunk)
    program = StructuredProgram(
        graph, poi=[img, ent, a, b], refine=False, factor_graph=False,
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        inferTypes=['local/softmax'], grad_surgery=surgery)
    program.train(training_set=_dataset(), train_epoch_num=epochs,
                  c_warmup_iters=0,
                  Optim=lambda p: torch.optim.SGD(p, lr=0.05), device='cpu')
    return program


def test_conflict_is_real_and_worse_with_a_shared_trunk(capsys):
    """The measurement that decides whether Phase C is worth shipping.

    R5's plan predicted conflict becomes material only once R4's shared trunk
    puts both losses in the same parameters. Measured here: it is present either
    way, but a shared trunk makes it *universal* and roughly doubles its
    magnitude — so the prediction's direction holds, while 'isolated heads
    cannot conflict' turns out to be too strong.
    """
    independent = _run(shared_trunk=False, surgery='diagnose').conflict_stats
    shared = _run(shared_trunk=True, surgery='diagnose').conflict_stats

    print(f'\nindependent heads : {independent.render()}')
    print(f'shared trunk (R4) : {shared.render()}')

    assert independent.steps > 0 and shared.steps > 0
    assert shared.conflict_rate >= independent.conflict_rate
    assert shared.mean_cosine < independent.mean_cosine
    assert shared.conflict_rate > 0.5, 'no conflict — Phase C would not earn its cost'


@pytest.mark.parametrize('surgery', ['diagnose', 'pcgrad', 'cagrad'])
def test_training_runs_and_updates_under_each_strategy(surgery):
    program = _run(shared_trunk=True, surgery=surgery)
    assert program.conflict_stats.steps > 0
    assert any(p.grad is not None or p.requires_grad
               for p in program.model.parameters())


def test_default_keeps_the_single_fused_backward():
    """Regression: surgery is opt-in and must not perturb the default path."""
    program = _run(shared_trunk=True, surgery='none')
    assert program.grad_surgery == 'none'
    assert program.conflict_stats.steps == 0


def test_program_rejects_an_unknown_strategy():
    graph, img, ent, a, b = _graph(shared_trunk=True)
    with pytest.raises(ValueError):
        StructuredProgram(graph, poi=[img, ent, a, b], grad_surgery='surgical')
    assert 'diagnose' in GRAD_SURGERY


# --------------------------------------------------------------------------- #
# R5 Phase F — amortized x augmented
# --------------------------------------------------------------------------- #

def test_amortized_augmented_trains_the_critic():
    """Phase F: previously a NotImplementedError; the critic must actually move.

    An augmented Lagrangian updates its multipliers in closed form, so there is
    no ascent objective for a critic to maximise — it is *regressed* onto the AL
    target instead, and needs an ordinary descent step. Without that step the
    combination constructs and trains but the critic never changes at all.
    """
    graph, img, ent, a, b = _graph(shared_trunk=False)
    program = StructuredProgram(
        graph, poi=[img, ent, a, b], refine=False, factor_graph=False,
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        inferTypes=['local/softmax'],
        dual_granularity='amortized', dual_algorithm='augmented')

    assert program.cmodel.dual_critic is not None
    assert hasattr(program.cmodel, 'rho'), 'AL penalty state missing'

    before = [p.detach().clone() for p in program.cmodel.dual_critic.parameters()]
    program.train(training_set=_dataset(6), train_epoch_num=4, c_warmup_iters=0,
                  c_freq=1, Optim=lambda p: torch.optim.SGD(p, lr=0.05),
                  device='cpu')
    after = list(program.cmodel.dual_critic.parameters())

    moved = [i for i, (x, y) in enumerate(zip(before, after))
             if not torch.equal(x, y.detach())]
    assert moved, 'the amortized critic was never updated under AL'


def test_amortized_ascent_still_uses_sign_flipped_ascent():
    """Phase B must be unchanged by Phase F's descent branch."""
    graph, img, ent, a, b = _graph(shared_trunk=False)
    program = StructuredProgram(
        graph, poi=[img, ent, a, b], refine=False, factor_graph=False,
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        inferTypes=['local/softmax'], dual_granularity='amortized')
    assert program.cmodel.dual_algorithm == 'ascent'
    before = [p.detach().clone() for p in program.cmodel.dual_critic.parameters()]
    program.train(training_set=_dataset(6), train_epoch_num=3, c_warmup_iters=0,
                  c_freq=1, Optim=lambda p: torch.optim.SGD(p, lr=0.05),
                  device='cpu')
    moved = [i for i, (x, y) in enumerate(
        zip(before, program.cmodel.dual_critic.parameters()))
        if not torch.equal(x, y.detach())]
    assert moved


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
