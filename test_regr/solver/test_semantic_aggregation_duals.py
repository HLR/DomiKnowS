"""R2 follow-through: grounding aggregation and composition with the R5 duals.

Covers the two structural gaps that remained after the exact semantic-loss
engine landed:

* ``circuit_aggregation='per_grounding'`` — one ``-log P`` per grounding instead
  of a single joint scalar. Keeps the loss scale independent of how many
  groundings a data item has, and is what per-grounding dual mechanisms need.
* ``SemanticLossProgram(training_style='primal_dual')`` — running the exact loss
  under the primal-dual epoch so R5A (augmented Lagrangian) and R5B (amortized
  DualCritic) actually apply to it.
"""

import numpy as np
import pytest
import torch

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import nandL
from domiknows.sensor import Sensor
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import SemanticLossProgram
from domiknows.program.model.pytorch import SolverModel

N, M = 6, 4


def _build(**cmodel_kwargs):
    """`a` contains `b`; nandL(p, q) grounds once per b instance, so G == M."""
    np.random.seed(0); torch.manual_seed(0)
    Sensor.clear()
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r2_agg') as graph:
        a = Concept(name='a'); b = Concept(name='b')
        (acb,) = a.contains(b)
        p_ = b(name='p'); q_ = b(name='q')
        nandL(p_, q_)

    class Net(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.l = torch.nn.Sequential(torch.nn.Linear(n, n), torch.nn.ReLU(),
                                         torch.nn.Linear(n, 2))

        def forward(self, rel, x):
            return self.l(x)

    a['index'] = ReaderSensor(keyword='a')
    b['index'] = ReaderSensor(keyword='b')
    b['lp'] = ReaderSensor(keyword='lp')
    b['lq'] = ReaderSensor(keyword='lq')
    b[acb] = EdgeSensor(b['index'], a['index'], relation=acb,
                        forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[p_] = ModuleLearner(acb, 'index', module=Net(N), device='cpu')
    b[p_] = FunctionalSensor(acb, 'lp', forward=lambda _, l: l, label=True)
    b[q_] = ModuleLearner(acb, 'index', module=Net(N), device='cpu')
    b[q_] = FunctionalSensor(acb, 'lq', forward=lambda _, l: l, label=True)

    dataset = [{'a': [0],
                'b': [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
                'lp': [1, 0, 1, 0], 'lq': [0, 1, 0, 1]}]
    program = SemanticLossProgram(
        graph, SolverModel, poi=[a, b, p_, q_], inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()), device='cpu',
        **cmodel_kwargs)
    return program, dataset


def _losses(program, dataset, aggregation):
    _l, _m, _d, builder = program.model(dataset[0])
    builder.createBatchRootDN()
    dn = builder.getDataNode(device='cpu')
    dn.inferLocal(keys=("softmax",))
    return dn.calculateLcLoss(circuit=True, circuitAggregation=aggregation)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_joint_is_scalar_per_grounding_is_a_vector():
    program, dataset = _build()
    joint = _losses(program, dataset, 'joint')
    per = _losses(program, dataset, 'per_grounding')

    for name, res in joint.items():
        assert res['lossTensor'].shape == (1,)
        assert res['aggregation'] == 'joint'
        assert res['groundingCount'] == M
    for name, res in per.items():
        assert res['lossTensor'].shape == (M,)
        assert res['aggregation'] == 'per_grounding'


def test_joint_equals_sum_of_independent_groundings():
    """These groundings share no variable, so the joint -log P must equal the
    sum of the per-grounding ones — a check that both paths agree where they
    provably should."""
    program, dataset = _build()
    joint = _losses(program, dataset, 'joint')
    per = _losses(program, dataset, 'per_grounding')

    for name in joint:
        joint_loss = joint[name]['lossTensor'].sum().item()
        summed = per[name]['lossTensor'].sum().item()
        assert joint_loss == pytest.approx(summed, rel=1e-5)


def test_invalid_aggregation_rejected():
    program, dataset = _build()
    with pytest.raises(ValueError):
        _losses(program, dataset, 'nope')


def test_per_grounding_loss_is_differentiable():
    program, dataset = _build()
    per = _losses(program, dataset, 'per_grounding')
    total = sum(res['lossTensor'].sum() for res in per.values())
    params = [p for p in program.model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads)


# ---------------------------------------------------------------------------
# Composition with the R5 duals
# ---------------------------------------------------------------------------

def test_semantic_loss_under_augmented_duals():
    program, dataset = _build(lambda_weighted=True, dual_algorithm='augmented',
                              training_style='primal_dual')
    assert program._make_copt(0.05) is None  # closed-form duals, no copt
    lmbd0 = program.cmodel.lmbd.clone()

    program.train(training_set=dataset, train_epoch_num=8,
                  Optim=lambda p: torch.optim.SGD(p, lr=0.05),
                  c_warmup_iters=1, c_freq=0, device='cpu')

    assert not torch.allclose(program.cmodel.lmbd, lmbd0)
    assert program.cmodel.exact_fraction == pytest.approx(1.0)


def test_amortized_duals_force_per_grounding_aggregation():
    """The critic attributes per grounding, so a joint scalar would defeat it —
    the model selects per-grounding aggregation automatically."""
    program, dataset = _build(lambda_weighted=True, dual_granularity='amortized',
                              training_style='primal_dual')
    assert program.cmodel.circuit_aggregation == 'per_grounding'

    emb0 = program.cmodel.dual_critic.constraint_embedding.weight.detach().clone()
    program.train(training_set=dataset, train_epoch_num=8,
                  Optim=lambda p: torch.optim.SGD(p, lr=0.05),
                  c_warmup_iters=1, c_freq=0, device='cpu')
    emb1 = program.cmodel.dual_critic.constraint_embedding.weight.detach()
    assert not torch.allclose(emb1, emb0)


def test_fixed_style_still_default_and_trains():
    program, dataset = _build()
    assert program.training_style == 'fixed'
    program.train(training_set=dataset, train_epoch_num=3,
                  Optim=lambda p: torch.optim.SGD(p, lr=0.05),
                  c_warmup_iters=0, device='cpu')


def test_invalid_training_style_rejected():
    with pytest.raises(ValueError):
        _build(training_style='nope')


if __name__ == '__main__':
    pytest.main([__file__])
