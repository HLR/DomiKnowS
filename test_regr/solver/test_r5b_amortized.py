"""R5 Phase B tests — amortized per-grounding duals (DualCritic).

Covers:
3. Critic unit behaviour: per-grounding multiplier bounded to ``[0, lmbd_p]``,
   the primal gradient reaches the classifier with NO gradient leaking through
   the (detached) critic input, an ascent step increases ``sum_g lambda_g*v_g``,
   and it works both without literal features (interpreter path) and with them
   (compiled path).
5. End-to-end: an amortized program trains and the critic is optimised, on the
   interpreter and the compiled (``compile_lc=True``) LC paths.
"""

import numpy as np
import pytest
import torch

from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import nandL, exactL
from domiknows.sensor import Sensor
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.model.lossModel import PrimalDualModel
from domiknows.program.model.dualCritic import DualCritic, N_LITERAL_FEATURES


@pytest.fixture
def one_constraint_graph():
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r5b_one') as graph:
        w = Concept(name='w')
        a = w(name='a'); b = w(name='b')
        nandL(a, b)
    key = next(iter(graph.allLogicalConstrainsRecursive))[0]
    return graph, key


# ---------------------------------------------------------------------------
# 3. Critic unit behaviour
# ---------------------------------------------------------------------------

def test_amortized_state_shape(one_constraint_graph):
    graph, _ = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_granularity='amortized')
    assert m.dual_critic is not None
    assert not isinstance(m.lmbd, torch.nn.Parameter)         # lmbd is an unused buffer
    param_names = [n for n, _ in m.named_parameters()]
    assert param_names and all(n.startswith('dual_critic.') for n in param_names)


def test_multiplier_within_bounds():
    critic = DualCritic(nconstr=3, embed_dim=4, hidden=8)
    v = torch.rand(20)
    lam = critic(1, v)  # in (0, 1) before scaling
    assert lam.shape == (20,)
    assert (lam > 0).all() and (lam < 1).all()


def test_primal_gradient_no_leak_through_critic(one_constraint_graph):
    """lambda_g reads a detached violation, so the gradient to the classifier
    is lambda_g * dv/dtheta and nothing flows to theta through the multiplier."""
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_granularity='amortized')

    v = torch.tensor([0.2, 0.7, 0.4], requires_grad=True)
    loss = m._weighted_constraint_loss(key, v)
    g, = torch.autograd.grad(loss, v, retain_graph=True)

    # Recompute the multipliers the critic produced for these violations and
    # check the gradient equals exactly lambda_g (the detached coefficient).
    with torch.no_grad():
        lam = m.dual_critic(m.lmbd_index[key], v.detach()) * m.lmbd_p[m.lmbd_index[key]]
    assert torch.allclose(g, lam, atol=1e-6)
    assert (g >= 0).all() and (g <= float(m.lmbd_p[m.lmbd_index[key]]) + 1e-4).all()


def test_ascent_step_increases_weighted_violation(one_constraint_graph):
    """A gradient-ascent step on the critic must increase sum_g lambda_g*v_g."""
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_granularity='amortized')
    v = torch.tensor([0.3, 0.6, 0.5])

    before = float(m._weighted_constraint_loss(key, v).detach())
    opt = torch.optim.SGD(m.dual_critic.parameters(), lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        obj = m._weighted_constraint_loss(key, v)
        (-obj).backward()   # ascent = descend the negative
        opt.step()
    after = float(m._weighted_constraint_loss(key, v).detach())
    assert after > before


def test_literal_features_used_when_present(one_constraint_graph):
    """Providing aligned [G, L] literal features changes the multipliers
    (the critic consumes them), and misalignment is never silently ignored."""
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_granularity='amortized')
    v = torch.tensor([0.2, 0.7, 0.4])

    no_feat = float(m._weighted_constraint_loss(key, v).detach())
    aligned = float(m._weighted_constraint_loss(key, v, torch.rand(3, 2)).detach())
    assert no_feat != aligned            # features influenced the multiplier
    with pytest.raises(ValueError, match="5 rows for 3 violations"):
        m._weighted_constraint_loss(key, v, torch.rand(5, 2))


def test_critic_literal_summary_width():
    critic = DualCritic(nconstr=2)
    summ = critic.literal_summary(torch.rand(6, 4), torch.device('cpu'), torch.float32)
    assert summ.shape == (6, N_LITERAL_FEATURES)


def test_critic_literal_summary_ignores_ragged_padding():
    critic = DualCritic(nconstr=1)
    features = torch.tensor([
        [0.2, 0.8, float('nan')],
        [0.4, float('nan'), float('nan')],
    ])
    summary = critic.literal_summary(
        features, torch.device('cpu'), torch.float32)
    assert torch.allclose(summary, torch.tensor([
        [0.5, 0.2, 0.8],
        [0.4, 0.4, 0.4],
    ]))


def test_compiled_amortized_mode_requires_features(one_constraint_graph):
    graph, key = one_constraint_graph
    model = PrimalDualModel(
        graph, device='cpu', dual_granularity='amortized', compile_lc=True)
    with pytest.raises(RuntimeError, match="requires groundingFeatures"):
        model._weighted_constraint_loss(key, torch.tensor([0.3, 0.6]))


# ---------------------------------------------------------------------------
# 5. End-to-end amortized training (interpreter + compiled)
# ---------------------------------------------------------------------------

def _build_amortized_program(compile_lc, N=6, M=4, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    Sensor.clear()
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r5b_e2e') as graph:
        a = Concept(name='a'); b = Concept(name='b')
        (acb,) = a.contains(b)
        answer = b(name='answer', ConceptClass=EnumConcept, values=['zero', 'one'])
        exactL(answer.__getattr__('one'), limit=2)

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

    dataset = [{'a': [0],
                'b': [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
                'label': [1, 1, 0, 0]}]
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, answer], inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()), device='cpu',
        dual_granularity='amortized', compile_lc=compile_lc)
    return program, dataset


@pytest.mark.parametrize('compile_lc', [False, True])
def test_end_to_end_amortized_training(compile_lc):
    program, dataset = _build_amortized_program(compile_lc)

    # The critic's parameters are the constraint optimizer's parameters.
    assert program._make_copt(0.01) is not None
    emb_before = program.cmodel.dual_critic.constraint_embedding.weight.detach().clone()

    program.train(training_set=dataset, train_epoch_num=5,
                  Optim=lambda p: torch.optim.SGD(p, lr=0.05),
                  c_warmup_iters=1, c_freq=0, device='cpu')

    # The ascent step actually moved the critic.
    emb_after = program.cmodel.dual_critic.constraint_embedding.weight.detach()
    assert not torch.allclose(emb_after, emb_before)


if __name__ == '__main__':
    pytest.main([__file__])
