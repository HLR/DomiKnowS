"""R5 Phase A tests — finer/augmented-Lagrangian duals on the constraint loss.

Covers:
1. Default (ascent) path is bit-identical to the pre-R5 weighting.
2. Augmented-Lagrangian unit behaviour: closed-form multiplier update,
   projection to ``[0, lmbd_p]``, rho growth on stagnation, and the
   ``lambda + rho*v`` primal gradient.
4. Checkpoint bundle: augmented dual state round-trips through save/load;
   ascent stays flat; old flat checkpoints still load.
6. Behaviour probe: the AL mechanism drives a violation lower than a fixed
   multiplier at an equal step budget.
"""

import pytest
import torch

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import nandL
from domiknows.program.model.lossModel import PrimalDualModel


@pytest.fixture
def one_constraint_graph():
    """A minimal graph carrying a single nandL constraint."""
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r5_one') as graph:
        w = Concept(name='w')
        a = w(name='a'); b = w(name='b')
        nandL(a, b)
    key = next(iter(graph.allLogicalConstrainsRecursive))[0]
    return graph, key


@pytest.fixture
def two_constraint_graph():
    """A graph with two independent nandL constraints (indices 0 and 1)."""
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r5_two') as graph:
        w = Concept(name='w')
        a = w(name='a'); b = w(name='b'); c = w(name='c')
        nandL(a, b)
        nandL(a, c)
    keys = [k for k, _ in graph.allLogicalConstrainsRecursive]
    return graph, keys


# ---------------------------------------------------------------------------
# 1. Default (ascent) path unchanged
# ---------------------------------------------------------------------------

def test_ascent_weighting_bit_identical(one_constraint_graph):
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu')  # ascent default

    v = torch.tensor([0.2, 0.5, float('nan'), 0.1], requires_grad=True)
    got = m._weighted_constraint_loss(key, v)

    # Exact pre-R5 formula.
    ref_value = v.clamp(min=0)
    ref = m.get_lmbd(key) * ref_value[ref_value == ref_value].sum()

    assert torch.equal(got, ref)
    g_got, = torch.autograd.grad(got, v, retain_graph=True)
    g_ref, = torch.autograd.grad(ref, v)
    assert torch.equal(g_got, g_ref)


def test_ascent_state_shape(one_constraint_graph):
    graph, _ = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu')
    assert isinstance(m.lmbd, torch.nn.Parameter)
    assert [n for n, _ in m.named_parameters()] == ['lmbd']
    assert not hasattr(m, 'rho')


def test_augmented_state_shape(one_constraint_graph):
    graph, _ = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_algorithm='augmented')
    # lambda/rho/stats are buffers → no trainable params → no copt is built.
    assert not isinstance(m.lmbd, torch.nn.Parameter)
    assert list(m.named_parameters()) == []
    buf_names = {n for n, _ in m.named_buffers()}
    assert {'lmbd', 'rho', '_al_viol_accum', '_al_viol_count', '_al_prev_mean_viol'} <= buf_names


def test_invalid_dual_options(one_constraint_graph):
    graph, _ = one_constraint_graph
    with pytest.raises(ValueError):
        PrimalDualModel(graph, device='cpu', dual_algorithm='nope')
    with pytest.raises(ValueError):
        PrimalDualModel(graph, device='cpu', dual_granularity='nope')
    # amortized (R5B) is valid on its own but not yet with augmented duals.
    with pytest.raises(NotImplementedError):
        PrimalDualModel(graph, device='cpu', dual_granularity='amortized',
                        dual_algorithm='augmented')


# ---------------------------------------------------------------------------
# 2. Augmented-Lagrangian unit behaviour
# ---------------------------------------------------------------------------

def test_al_quadratic_gradient(one_constraint_graph):
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_algorithm='augmented', al_rho_init=3.0)
    # Put lambda at a known value.
    with torch.no_grad():
        m.lmbd.fill_(2.0)

    v = torch.tensor([0.2, 0.5, 0.1], requires_grad=True)
    loss = m._weighted_constraint_loss(key, v)
    g, = torch.autograd.grad(loss, v)
    # d/dv [ lambda*sum(v) + 0.5*rho*sum(v^2) ] = lambda + rho*v
    assert torch.allclose(g, 2.0 + 3.0 * v.detach())


def test_al_accumulation_and_multiplier_update(one_constraint_graph):
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_algorithm='augmented', al_rho_init=1.0)

    # Two forward passes accumulate S_c = sum(v).
    m._weighted_constraint_loss(key, torch.tensor([0.2, 0.6]))   # S=0.8
    m._weighted_constraint_loss(key, torch.tensor([0.4]))        # S=0.4
    assert torch.allclose(m._al_viol_count, torch.tensor([2.0]))
    assert torch.allclose(m._al_viol_accum, torch.tensor([1.2]))

    m.al_dual_update_()
    # lambda <- clamp(1 + rho * mean(S)) = 1 + 1*0.6 = 1.6
    assert torch.allclose(m.lmbd, torch.tensor([1.6]))
    assert torch.allclose(m._al_prev_mean_viol, torch.tensor([0.6]))
    # accumulators reset
    assert torch.allclose(m._al_viol_count, torch.tensor([0.0]))
    assert torch.allclose(m._al_viol_accum, torch.tensor([0.0]))


def test_al_multiplier_projection(one_constraint_graph):
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_algorithm='augmented', al_rho_init=1000.0)
    upper = float(m.lmbd_p[m.lmbd_index[key]])
    # A large violation would blow lambda past its cap; it must clamp to lmbd_p.
    m._weighted_constraint_loss(key, torch.tensor([1.0, 1.0, 1.0]))
    m.al_dual_update_()
    assert float(m.lmbd[m.lmbd_index[key]]) == pytest.approx(upper)


def test_al_rho_growth_on_stagnation(one_constraint_graph):
    graph, key = one_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_algorithm='augmented',
                        al_rho_init=1.0, al_rho_growth=2.0, al_stagnation_tau=0.9)

    # Window 1: establishes the baseline; rho never grows on the first window.
    m._weighted_constraint_loss(key, torch.tensor([0.5]))
    m.al_dual_update_()
    assert torch.allclose(m.rho, torch.tensor([1.0]))

    # Window 2: violation basically unchanged (0.5 > 0.9*0.5) → stagnation → rho grows.
    m._weighted_constraint_loss(key, torch.tensor([0.5]))
    m.al_dual_update_()
    assert torch.allclose(m.rho, torch.tensor([2.0]))

    # Window 3: violation shrinks a lot (0.1 < 0.9*0.5) → no growth.
    m._weighted_constraint_loss(key, torch.tensor([0.1]))
    m.al_dual_update_()
    assert torch.allclose(m.rho, torch.tensor([2.0]))


def test_al_only_touches_evaluated_constraints(two_constraint_graph):
    graph, keys = two_constraint_graph
    m = PrimalDualModel(graph, device='cpu', dual_algorithm='augmented', al_rho_init=1.0)
    i0, i1 = m.lmbd_index[keys[0]], m.lmbd_index[keys[1]]

    # Only evaluate constraint 0 this window.
    m._weighted_constraint_loss(keys[0], torch.tensor([0.5]))
    m.al_dual_update_()

    assert m.lmbd[i0] != 1.0                      # updated
    assert torch.allclose(m.lmbd[i1], torch.tensor(1.0))  # untouched
    assert torch.isnan(m._al_prev_mean_viol[i1])          # never seen


# ---------------------------------------------------------------------------
# 4. Checkpoint bundle round-trip
# ---------------------------------------------------------------------------

def _make_program(graph, dual_algorithm='ascent'):
    from domiknows.program.lossprogram import PrimalDualProgram
    from domiknows.program.model.pytorch import SolverModel
    from domiknows.program.metric import MacroAverageTracker
    from domiknows.program.loss import NBCrossEntropyLoss

    return PrimalDualProgram(
        graph, SolverModel,
        poi=[c for c in graph.concepts.values()],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        device='cpu',
        dual_algorithm=dual_algorithm,
    )


def test_checkpoint_bundle_roundtrip_augmented(one_constraint_graph, tmp_path):
    graph, key = one_constraint_graph
    prog = _make_program(graph, dual_algorithm='augmented')
    assert prog.copt is None or list(prog.cmodel.parameters()) == []

    # Mutate the dual state so a reset would be detectable.
    with torch.no_grad():
        prog.cmodel.lmbd.fill_(7.5)
        prog.cmodel.rho.fill_(4.0)
        prog.cmodel._al_prev_mean_viol.fill_(0.3)

    path = str(tmp_path / 'ckpt_al.pt')
    prog.save(path)  # auto-bundles in augmented mode

    saved = torch.load(path, weights_only=True)
    assert isinstance(saved, dict) and 'model' in saved and 'cmodel' in saved

    fresh = _make_program(graph, dual_algorithm='augmented')
    # Confirm it starts from defaults, then load restores the saved state.
    assert not torch.allclose(fresh.cmodel.lmbd, torch.tensor([7.5]))
    fresh.load(path)
    assert torch.allclose(fresh.cmodel.lmbd, torch.tensor([7.5]))
    assert torch.allclose(fresh.cmodel.rho, torch.tensor([4.0]))
    assert torch.allclose(fresh.cmodel._al_prev_mean_viol, torch.tensor([0.3]))


def test_checkpoint_flat_for_ascent_and_backcompat(one_constraint_graph, tmp_path):
    graph, key = one_constraint_graph
    prog = _make_program(graph, dual_algorithm='ascent')

    path = str(tmp_path / 'ckpt_flat.pt')
    prog.save(path)  # ascent → historical flat model-only checkpoint

    saved = torch.load(path, weights_only=True)
    assert not (isinstance(saved, dict) and 'model' in saved and 'cmodel' in saved)

    # A flat checkpoint still loads through the bundle-aware loader.
    fresh = _make_program(graph, dual_algorithm='ascent')
    fresh.load(path)


# ---------------------------------------------------------------------------
# 6. Behaviour probe — AL drives a violation lower than a fixed multiplier
# ---------------------------------------------------------------------------

def test_al_reduces_violation_vs_fixed_multiplier(one_constraint_graph):
    graph, key = one_constraint_graph

    def run(augmented):
        torch.manual_seed(0)
        if augmented:
            m = PrimalDualModel(graph, device='cpu', dual_algorithm='augmented',
                                al_rho_init=1.0, al_rho_growth=2.0, al_stagnation_tau=0.9)
        else:
            m = PrimalDualModel(graph, device='cpu')  # fixed lambda=1, no penalty/update

        theta = torch.zeros(1, requires_grad=True)  # sigmoid(0)=0.5 initial violation
        opt = torch.optim.SGD([theta], lr=0.5)
        for step in range(60):
            opt.zero_grad()
            v = torch.sigmoid(theta)          # scalar violation in (0,1), target 0
            loss = m._weighted_constraint_loss(key, v)
            loss.backward()
            opt.step()
            if augmented and (step + 1) % 5 == 0:
                m.al_dual_update_()
        return torch.sigmoid(theta).item()

    v_al = run(augmented=True)
    v_base = run(augmented=False)

    assert v_al < v_base, f'AL={v_al} should beat fixed-multiplier baseline={v_base}'
    assert v_al < 0.1  # AL drives the violation substantially toward satisfaction


# ---------------------------------------------------------------------------
# End-to-end: program.train drives the augmented dual branch of train_epoch
# ---------------------------------------------------------------------------

def _build_trainable_al_program(N=6, M=4, seed=0):
    """A minimal trainable graph + augmented PrimalDualProgram (self-contained)."""
    import numpy as np
    from domiknows.graph import EnumConcept
    from domiknows.graph.logicalConstrain import exactL
    from domiknows.sensor import Sensor
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, FunctionalSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.program.metric import MacroAverageTracker
    from domiknows.program.loss import NBCrossEntropyLoss
    from domiknows.program.lossprogram import PrimalDualProgram
    from domiknows.program.model.pytorch import SolverModel

    np.random.seed(seed); torch.manual_seed(seed)
    Sensor.clear()
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('r5_e2e') as graph:
        a = Concept(name='a'); b = Concept(name='b')
        (a_contain_b,) = a.contains(b)
        b_answer = b(name='answer', ConceptClass=EnumConcept, values=['zero', 'one'])
        exactL(b_answer.__getattr__('one'), 2)

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
    b[a_contain_b] = EdgeSensor(b['index'], a['index'], relation=a_contain_b,
                                forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[b_answer] = ModuleLearner(a_contain_b, 'index', module=Net(N), device='cpu')
    b[b_answer] = FunctionalSensor(a_contain_b, 'temp', forward=lambda _, l: l, label=True)

    dataset = [{'a': [0],
                'b': [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
                'label': [1] * M}]

    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer], inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()), device='cpu',
        dual_algorithm='augmented', al_rho_init=1.0)
    return program, dataset


def test_end_to_end_augmented_training(tmp_path):
    program, dataset = _build_trainable_al_program()

    # Pure Augmented Lagrangian: no constraint optimizer is built.
    program.copt = program._make_copt(0.05)
    assert program.copt is None

    lmbd0 = program.cmodel.lmbd.clone()
    program.train(training_set=dataset, train_epoch_num=8,
                  Optim=lambda p: torch.optim.SGD(p, lr=0.05),
                  c_warmup_iters=1, c_freq=0, device='cpu')

    # The augmented dual branch of train_epoch must have fired: multipliers
    # moved and the per-window violation baseline was recorded.
    assert not torch.allclose(program.cmodel.lmbd, lmbd0)
    assert not torch.isnan(program.cmodel._al_prev_mean_viol).all()

    # Trained dual state round-trips through the checkpoint bundle.
    path = str(tmp_path / 'trained_al.pt')
    program.save(path)
    reloaded, _ = _build_trainable_al_program()
    reloaded.load(path)
    assert torch.allclose(reloaded.cmodel.lmbd, program.cmodel.lmbd)
    assert torch.allclose(reloaded.cmodel.rho, program.cmodel.rho)


if __name__ == '__main__':
    pytest.main([__file__])
