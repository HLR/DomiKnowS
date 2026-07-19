"""Compiled-path (R1) coverage for constraint kinds that used to fall back.

Each test builds its own graph and asserts the compiled evaluator produces
*identical* loss tensors and gradients to the interpreter, and that it did not
silently fall back. Falling back is always correct but forfeits R1's speedup,
so "no fallback" is as much a part of the contract as the numbers.

Each test constructs its graph fresh (``Sensor.clear()`` + ``Graph.clear()``)
because property-sensor assignments stack across a pytest process.
"""

import numpy as np
import pytest
import torch

from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import nandL, ifL, andL, exactL, eqL, fixedL
from domiknows.sensor import Sensor
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel

N, M = 6, 4
TNORMS = ('L', 'G', 'P')


class _Net(torch.nn.Module):
    def __init__(self, n, out=2):
        super().__init__()
        self.l = torch.nn.Sequential(torch.nn.Linear(n, n), torch.nn.ReLU(),
                                     torch.nn.Linear(n, out))

    def forward(self, rel, x):
        return self.l(x)


def _features(seed=0):
    rng = np.random.RandomState(seed)
    return [((rng.rand(N) - rng.rand(N))).tolist() for _ in range(M)]


def _datanode(program, data):
    _l, _m, _d, builder = program.model(data)
    builder.createBatchRootDN()
    dn = builder.getDataNode(device='cpu')
    dn.inferLocal(keys=("softmax",))
    return dn


def _assert_parity(dn, tnorm='P'):
    """Interpreter vs compiled: same constraints, same values, same gradients."""
    ref = dn.calculateLcLoss(tnorm=tnorm)
    cmp = dn.calculateLcLoss(tnorm=tnorm, compiled=True)

    assert set(ref.keys()) == set(cmp.keys()), \
        f'constraint sets differ: {set(ref)} vs {set(cmp)}'

    for name in ref:
        rt, ct = ref[name]['lossTensor'], cmp[name]['lossTensor']
        if rt is None or ct is None:
            assert rt is None and ct is None, f'{name}: one path produced None'
            continue
        assert rt.shape == ct.shape, f'{name}: shape {rt.shape} vs {ct.shape}'
        rnan, cnan = torch.isnan(rt), torch.isnan(ct)
        assert torch.equal(rnan, cnan), f'{name}: NaN masks differ'
        assert torch.allclose(rt[~rnan], ct[~cnan], atol=1e-6), \
            f'{name}: values differ\ninterpreter={rt}\ncompiled={ct}'
    return ref, cmp


def _count_fallbacks(dn, monkeypatch, tnorm='P'):
    """Run the compiled path; report fallbacks and compiled head evaluations.

    Only fallbacks that actually *produce a loss* are counted. Constraints both
    paths decline to evaluate (a non-head ``eqL`` from inside a path, or
    ``fixedL`` itself, which ``LossCalculator`` returns ``None`` for) would
    otherwise be miscounted as work that failed to compile.
    """
    from domiknows.solver.compiled import formula as formula_mod

    calls = {'fallback': 0, 'compiled_head': 0}

    orig = formula_mod.LossCalculator.calculate_single_lc_loss

    def spy(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        if isinstance(self, formula_mod.CompiledLossCalculator) and result is not None:
            calls['fallback'] += 1
        return result

    orig_eval = formula_mod.CompiledConstraintEvaluator.constructCompiled

    def spy_eval(self, *args, **kwargs):
        if kwargs.get('headLC'):
            calls['compiled_head'] += 1
        return orig_eval(self, *args, **kwargs)

    monkeypatch.setattr(formula_mod.LossCalculator, 'calculate_single_lc_loss', spy)
    monkeypatch.setattr(formula_mod.CompiledConstraintEvaluator, 'constructCompiled', spy_eval)
    result = dn.calculateLcLoss(tnorm=tnorm, compiled=True)
    return result, calls


# ---------------------------------------------------------------------------
# fixedL — previously disabled compilation for the ENTIRE graph
# ---------------------------------------------------------------------------

def _build_fixed_program(gate_values):
    np.random.seed(0); torch.manual_seed(0)
    Sensor.clear()
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('fx') as graph:
        a = Concept(name='a'); b = Concept(name='b')
        (acb,) = a.contains(b)
        p_ = b(name='p'); q_ = b(name='q')
        nandL(p_, q_)
        # Pin `p` wherever b's `gate` attribute is True.
        fixedL(p_('x', eqL(b, 'gate', {True})), active=True)

    a['index'] = ReaderSensor(keyword='a')
    b['index'] = ReaderSensor(keyword='b')
    b['gate'] = ReaderSensor(keyword='gate')
    b['lp'] = ReaderSensor(keyword='lp')
    b['lq'] = ReaderSensor(keyword='lq')
    b[acb] = EdgeSensor(b['index'], a['index'], relation=acb,
                        forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[p_] = ModuleLearner(acb, 'index', module=_Net(N), device='cpu')
    b[p_] = FunctionalSensor(acb, 'lp', forward=lambda _, l: l, label=True)
    b[q_] = ModuleLearner(acb, 'index', module=_Net(N), device='cpu')
    b[q_] = FunctionalSensor(acb, 'lq', forward=lambda _, l: l, label=True)

    data = {'a': [0], 'b': _features(), 'gate': gate_values,
            'lp': [1, 0, 1, 0], 'lq': [0, 1, 0, 1]}
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, p_, q_], inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()), device='cpu')
    return program, data, p_


@pytest.mark.parametrize('gate_values', [
    [True, True, True, True],     # every row pinned
    [True, False, True, False],   # mixed
    [False, False, False, False], # gate never matches -> nothing pinned
])
def test_fixedL_graph_matches_interpreter(gate_values):
    program, data, _ = _build_fixed_program(gate_values)
    dn = _datanode(program, data)
    for tnorm in TNORMS:
        _assert_parity(dn, tnorm)


def test_fixedL_graph_no_longer_falls_back(monkeypatch):
    """The whole-graph fixedL disable is gone: constraints compile normally."""
    program, data, _ = _build_fixed_program([True, False, True, False])
    dn = _datanode(program, data)
    result, calls = _count_fallbacks(dn, monkeypatch)
    assert result, 'no constraints evaluated'
    assert calls['fallback'] == 0, \
        f"{calls['fallback']} constraints fell back despite fixedL support"
    assert calls['compiled_head'] >= len(result), \
        f"compiled evaluator ran {calls['compiled_head']} times for {len(result)} results"


def test_fixed_substitution_actually_applied():
    """A pinned row's value must be the hard label-derived constant, not the
    model probability — otherwise the parity tests could pass vacuously."""
    program, data, p_ = _build_fixed_program([True, True, True, True])
    dn = _datanode(program, data)

    from domiknows.solver.compiled.grounding import ProbabilityStore
    store = ProbabilityStore(dn, "/local/softmax", graphs=[program.graph])
    entry = store._entry('p')
    assert entry['fixed_gate'] is not None, 'fixedL spec was not detected'
    assert bool(entry['fixed_gate'].all()), 'all rows should be gated'

    # The interpreter compares the label against e[2] — which is 0 for a binary
    # concept even though the probability is read at index 1 — so label==0 pins
    # to 1.0 and label==1 pins to 0.0. Replicating that inversion is the point.
    gathered = store.gather_variable([[d] for d in entry['dns']], (p_, 'p', None, 1))[0][0]
    expected = (entry['fixed_label'] == 0).to(gathered.dtype)
    assert torch.allclose(gathered, expected), \
        f'fixed values not applied: got {gathered}, expected {expected}'

    # And they must differ from the raw model probabilities, else nothing happened.
    raw = entry['matrix'][:, 1]
    assert not torch.allclose(gathered, raw), 'fixed substitution had no effect'


# ---------------------------------------------------------------------------
# EnumConcept — indexed classes (fast path) and bare references (enum_all)
# ---------------------------------------------------------------------------

def _build_enum_program(bare):
    """`bare=True` uses a whole EnumConcept reference (e[2] is None → enum_all);
    `bare=False` uses explicit class members."""
    np.random.seed(0); torch.manual_seed(0)
    Sensor.clear()
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('en') as graph:
        a = Concept(name='a'); b = Concept(name='b')
        (acb,) = a.contains(b)
        label = b(name='lab', ConceptClass=EnumConcept, values=['zero', 'one', 'two'])
        if bare:
            # A bare EnumConcept reference inside a counting constraint.
            exactL(label('x'), limit=2)
        else:
            exactL(label.__getattr__('one'), limit=2)

    a['index'] = ReaderSensor(keyword='a')
    b['index'] = ReaderSensor(keyword='b')
    b['temp'] = ReaderSensor(keyword='label')
    b[acb] = EdgeSensor(b['index'], a['index'], relation=acb,
                        forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[label] = ModuleLearner(acb, 'index', module=_Net(N, out=3), device='cpu')
    b[label] = FunctionalSensor(acb, 'temp', forward=lambda _, l: l, label=True)

    data = {'a': [0], 'b': _features(), 'label': [1, 1, 0, 2]}
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, label], inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()), device='cpu')
    return program, data, label


@pytest.mark.parametrize('bare', [False, True])
def test_enum_concept_matches_interpreter(bare):
    program, data, _ = _build_enum_program(bare)
    dn = _datanode(program, data)
    for tnorm in TNORMS:
        _assert_parity(dn, tnorm)


@pytest.mark.parametrize('bare', [False, True])
def test_enum_concept_no_fallback(bare, monkeypatch):
    program, data, _ = _build_enum_program(bare)
    dn = _datanode(program, data)
    result, calls = _count_fallbacks(dn, monkeypatch)
    assert result
    assert calls['fallback'] == 0, f"{calls['fallback']} constraints fell back"


def test_enum_all_uses_the_batched_path():
    """A bare EnumConcept reference takes the batched gather: one [G] tensor per
    class, all inside a single group — not the per-datanode slow path."""
    program, data, label = _build_enum_program(bare=True)
    dn = _datanode(program, data)

    from domiknows.solver.compiled.grounding import ProbabilityStore
    store = ProbabilityStore(dn, "/local/softmax", graphs=[program.graph])
    entry = store._entry('lab')
    gathered = store.gather_variable([[d] for d in entry['dns']], (label, 'lab', None, 3))

    assert len(gathered) == 1, 'expected a single batched group'
    assert len(gathered[0]) == 3, 'expected one column per enum class'
    for column in gathered[0]:
        assert column.shape == (len(entry['dns']),)


def test_enum_all_keeps_every_class_column():
    """Every one of the K class probabilities must survive the gather.

    This previously collapsed to class 0, which silently reduced ``sameL`` to
    "is the first class" (see
    test_regr/examples/same_different/test_sameL_loss_uses_every_enum_class).
    The columns must also stay in declared class order, since consumers index
    them positionally — ``sameVar`` reads ``group[j]`` for subclass ``j``.
    """
    program, data, label = _build_enum_program(bare=True)
    dn = _datanode(program, data)

    from domiknows.solver.compiled.grounding import ProbabilityStore
    store = ProbabilityStore(dn, "/local/softmax", graphs=[program.graph])
    entry = store._entry('lab')
    columns = store.gather_variable([[d] for d in entry['dns']], (label, 'lab', None, 3))[0]

    matrix = entry['matrix']
    assert matrix.shape[1] == 3, 'expected 3 enum classes'
    # Distinguishable classes, else the ordering assertion proves nothing.
    assert not torch.allclose(matrix[:, 0], matrix[:, 1])

    for j, column in enumerate(columns):
        assert torch.allclose(column, matrix[:, j]), \
            f'column {j} does not match class {j} of the probability matrix'


# ---------------------------------------------------------------------------
# sumL — needs the runtime label threaded into the evaluator
# ---------------------------------------------------------------------------

def _calculators(dn):
    """Interpreter and compiled calculators sharing one solver, both primed."""
    from domiknows.solver.compiled import CompiledLossCalculator
    from domiknows.solver.lossCalculator import LossCalculator

    solver, _ = dn.getILPSolver(conceptsRelations=dn.collectConceptsAndRelations())
    solver.current_device = dn.current_device
    dn.inferLocal()

    compiled = CompiledLossCalculator(solver)
    compiled.calculateLoss(dn, 'P')  # primes the ProbabilityStore
    return LossCalculator(solver), compiled


def _build_sum_program():
    np.random.seed(0); torch.manual_seed(0)
    Sensor.clear()
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph('sm') as graph:
        a = Concept(name='a'); b = Concept(name='b')
        (acb,) = a.contains(b)
        p_ = b(name='p'); q_ = b(name='q')
        from domiknows.graph.logicalConstrain import sumL
        sumL(p_('x'), q_('y'), active=True)

    a['index'] = ReaderSensor(keyword='a')
    b['index'] = ReaderSensor(keyword='b')
    b['lp'] = ReaderSensor(keyword='lp')
    b['lq'] = ReaderSensor(keyword='lq')
    b[acb] = EdgeSensor(b['index'], a['index'], relation=acb,
                        forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[p_] = ModuleLearner(acb, 'index', module=_Net(N), device='cpu')
    b[p_] = FunctionalSensor(acb, 'lp', forward=lambda _, l: l, label=True)
    b[q_] = ModuleLearner(acb, 'index', module=_Net(N), device='cpu')
    b[q_] = FunctionalSensor(acb, 'lq', forward=lambda _, l: l, label=True)

    data = {'a': [0], 'b': _features(), 'lp': [1, 0, 1, 0], 'lq': [0, 1, 0, 1]}
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, p_, q_], inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()), device='cpu')
    return program, data, graph


@pytest.mark.parametrize('target', [1, 2, 3])
def test_sumL_with_label_matches_interpreter(target):
    """The compiled evaluator must thread `label` into sumL exactly like the
    interpreter — without it, summationVar silently returns None (a vanishing
    loss), which is why this used to be a hard fallback."""
    program, data, graph = _build_sum_program()
    dn = _datanode(program, data)
    interp, compiled = _calculators(dn)

    lc = next(lc for lc in graph.logicalConstrains.values() if lc.headLC)

    ref = interp.calculate_single_lc_loss(dn=dn, lc=lc, key='/local/softmax',
                                          tnorm='P', label=target)
    cmp = compiled.calculate_single_lc_loss(dn=dn, lc=lc, key='/local/softmax',
                                            tnorm='P', label=target)

    assert ref is not None and cmp is not None
    rt, ct = ref['lossTensor'], cmp['lossTensor']
    assert rt is not None and ct is not None, 'sumL produced no loss (label not threaded?)'
    assert torch.allclose(rt, ct, atol=1e-6), \
        f'sumL loss differs: interpreter={rt} compiled={ct}'
    assert 'expectedCount' in cmp, 'compiled path must report expectedCount for sumL'


def test_sumL_compiled_does_not_fall_back(monkeypatch):
    program, data, graph = _build_sum_program()
    dn = _datanode(program, data)
    interp, compiled = _calculators(dn)
    lc = next(lc for lc in graph.logicalConstrains.values() if lc.headLC)

    from domiknows.solver.compiled import formula as formula_mod
    calls = {'fallback': 0}
    orig = formula_mod.LossCalculator.calculate_single_lc_loss

    def spy(self, *args, **kwargs):
        r = orig(self, *args, **kwargs)
        if isinstance(self, formula_mod.CompiledLossCalculator) and r is not None:
            calls['fallback'] += 1
        return r

    monkeypatch.setattr(formula_mod.LossCalculator, 'calculate_single_lc_loss', spy)
    compiled.calculate_single_lc_loss(dn=dn, lc=lc, key='/local/softmax',
                                      tnorm='P', label=2)
    assert calls['fallback'] == 0, 'sumL still falls back to the interpreter'


# ---------------------------------------------------------------------------
# Executable LCs — opt-in, because the default exclusion is load-bearing
# ---------------------------------------------------------------------------

def test_include_executable_defaults_off_and_is_threaded():
    """`execute()` moves a constraint out of `logicalConstrains`, so the loss
    path cannot see it by default.

    That default protects `InferenceProgram`, which scores executable
    constraints itself and adds a separately weighted graph-global term — if
    this path also scored them they would be counted twice. The opt-in exists
    so a caller that wants one number over *every* constraint (e.g. comparing
    against the exact-circuit path, which always iterates both populations) can
    ask for it explicitly.
    """
    import inspect
    from domiknows.solver.lossCalculator import LossCalculator
    from domiknows.solver.compiled import CompiledLossCalculator
    from domiknows.graph.dataNode import DataNode

    for fn in (LossCalculator.calculateLoss, CompiledLossCalculator.calculateLoss):
        sig = inspect.signature(fn)
        assert 'include_executable' in sig.parameters
        assert sig.parameters['include_executable'].default is False

    dn_sig = inspect.signature(DataNode.calculateLcLoss)
    assert 'includeExecutable' in dn_sig.parameters
    assert dn_sig.parameters['includeExecutable'].default is False


def test_graph_without_executables_is_unaffected_by_the_flag():
    """Turning the flag on must be a no-op when nothing used execute()."""
    program, data, _ = _build_fixed_program([True, False, True, False])
    dn = _datanode(program, data)

    off = dn.calculateLcLoss(tnorm='P', compiled=True)
    on = dn.calculateLcLoss(tnorm='P', compiled=True, includeExecutable=True)

    assert set(off.keys()) == set(on.keys())
    for name in off:
        a, b = off[name]['lossTensor'], on[name]['lossTensor']
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert torch.allclose(a, b, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])
