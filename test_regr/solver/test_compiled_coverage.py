"""Compiled-path (R1) coverage for constraint kinds that used to fall back.

Each test builds its own graph and asserts the compiled evaluator produces
*identical* loss tensors and gradients to the interpreter, and that it did not
silently fall back. Falling back is always correct but forfeits R1's speedup,
so "no fallback" is as much a part of the contract as the numbers.

Each test constructs its graph fresh (``Sensor.clear()`` + ``Graph.clear()``)
because property-sensor assignments stack across a pytest process.
"""

from collections import OrderedDict

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
    assert 'includeGlobal' in dn_sig.parameters
    assert dn_sig.parameters['includeGlobal'].default is True


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


def test_executable_only_compiled_scope_matches_interpreter():
    """InferenceProgram can compile labels without enabling global loss."""
    from test_regr.tiny_dynamic_graph.example_dynamic_graph import (
        build_reusable_dynamic_graph,
    )

    context = build_reusable_dynamic_graph(device='cpu')
    context.graph.set_active_concepts(None)
    row = next(iter(context.datasets.values()))[0]
    _loss, _metric, _datanode, builder = context.program.model(row)
    builder.createBatchRootDN()
    dn = builder.getDataNode(device='cpu')
    dn.inferLocal(keys=('softmax',))

    reference = dn.calculateLcLoss(
        tnorm='P', compiled=False,
        includeExecutable=True, includeGlobal=False)
    compiled = dn.calculateLcLoss(
        tnorm='P', compiled=True,
        includeExecutable=True, includeGlobal=False)

    assert reference
    assert set(reference) == set(compiled)
    for name in reference:
        expected = reference[name]
        actual = compiled[name]
        assert actual.get('compiled') is True
        assert actual.get('executableName') == name
        for key in ('lossTensor', 'conversionSigmoid'):
            expected_tensor = expected.get(key)
            actual_tensor = actual.get(key)
            if expected_tensor is None or actual_tensor is None:
                assert expected_tensor is None and actual_tensor is None
            else:
                assert torch.allclose(expected_tensor, actual_tensor, atol=1e-6)

    solver, _ = dn.getILPSolver(
        conceptsRelations=dn.collectConceptsAndRelations())
    calculator = solver._compiled_loss_calculator
    bindings_before = calculator.cache_info()['data_bindings']
    inference_loss, *_ = context.program.cmodel(builder)
    bindings_after = calculator.cache_info()['data_bindings']
    assert torch.isfinite(inference_loss)
    assert context.program.cmodel.include_global_constraint_loss is False
    assert bindings_after == bindings_before + 1


@pytest.mark.parametrize('tnorm', TNORMS)
def test_custom_formula_subclass_uses_compiled_protocol_without_registration(
        tnorm, monkeypatch):
    """Third-party formulas no longer fall back because of an allowlist."""
    from domiknows.graph.logicalConstrain import LogicalConstrain
    from domiknows.solver.compiled import formula as formula_module
    from test_regr.tiny_dynamic_graph.example_dynamic_global_constraints import (
        build_dynamic_constraint_example,
    )

    class customIfL(LogicalConstrain):
        def __call__(self, model, processor, variables,
                     headConstrain=False, integrate=False):
            return self.createLogicalConstrains(
                'If', processor.ifVar, model, variables, headConstrain)

    context = build_dynamic_constraint_example(device='cpu')
    with context.graph:
        custom = customIfL(
            context.concepts['red']('z'),
            context.concepts['animal'](path='z'),
            name='custom_red_implies_animal',
        )
    context.graph.set_active_concepts(None)
    row = context.entries[0][1]
    _loss, _metric, _datanode, builder = context.program.model(row)
    builder.createBatchRootDN()
    dn = builder.getDataNode(device='cpu')
    dn.inferLocal(keys=('softmax',))

    reference = dn.calculateLcLoss(tnorm=tnorm, compiled=False)
    calls = {'fallback': 0}
    interpreter_single = formula_module.LossCalculator.calculate_single_lc_loss

    def record_fallback(self, *args, **kwargs):
        calls['fallback'] += 1
        return interpreter_single(self, *args, **kwargs)

    monkeypatch.setattr(
        formula_module.LossCalculator,
        'calculate_single_lc_loss',
        record_fallback,
    )
    compiled = dn.calculateLcLoss(tnorm=tnorm, compiled=True)

    assert custom.lcName in reference
    assert custom.lcName in compiled
    assert compiled[custom.lcName].get('compiled') is True
    assert calls['fallback'] == 0
    assert torch.allclose(
        reference[custom.lcName]['lossTensor'],
        compiled[custom.lcName]['lossTensor'],
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Phase 2 — persistent formula plans and tensorized candidate binding
# ---------------------------------------------------------------------------

def test_compiled_plans_persist_across_data_items():
    """A solver keeps one immutable formula plan while DataNodes change."""
    program, data, _ = _build_fixed_program([True, False, True, False])

    first_dn = _datanode(program, data)
    first_dn.calculateLcLoss(tnorm='P', compiled=True)
    solver, _ = first_dn.getILPSolver(
        conceptsRelations=first_dn.collectConceptsAndRelations())
    calculator = solver._compiled_loss_calculator
    evaluator_id = id(calculator._evaluator)
    probability_store_id = id(calculator._prob_store)
    first_info = calculator.cache_info()

    second_dn = _datanode(program, data)
    assert second_dn is not first_dn
    second_dn.calculateLcLoss(tnorm='P', compiled=True)
    second_solver, _ = second_dn.getILPSolver(
        conceptsRelations=second_dn.collectConceptsAndRelations())
    second_info = calculator.cache_info()

    assert second_solver is solver
    assert second_solver._compiled_loss_calculator is calculator
    assert id(calculator._evaluator) == evaluator_id
    assert id(calculator._prob_store) == probability_store_id
    assert second_info['misses'] == first_info['misses']
    assert second_info['hits'] > first_info['hits']
    assert second_info['data_bindings'] == first_info['data_bindings'] + 1
    assert second_info['tensorized_candidate_calls'] > \
        first_info['tensorized_candidate_calls']
    assert second_info['candidate_fallback_calls'] == 0


def test_compiled_plan_cache_invalidates_after_formula_mutation():
    """Structural edits cannot leave an obsolete execution plan cached."""
    program, data, _ = _build_fixed_program([True, False, True, False])
    dn = _datanode(program, data)
    dn.calculateLcLoss(tnorm='P', compiled=True)
    solver, _ = dn.getILPSolver(
        conceptsRelations=dn.collectConceptsAndRelations())
    cache = solver._compiled_loss_calculator.plan_cache
    lc = next(lc for lc in program.graph.logicalConstrains.values() if lc.headLC)

    original_plan = cache.get(lc)
    original_invalidations = cache.invalidations
    lc.e.append(1)
    try:
        replacement_plan = cache.get(lc)
    finally:
        lc.e.pop()

    assert replacement_plan is not original_plan
    assert cache.invalidations == original_invalidations + 1


def test_compiled_plan_cache_hit_does_not_rewalk_formula(monkeypatch):
    """Revision checks, not recursive signatures, guard the hot cache path."""
    program, data, _ = _build_fixed_program([True, False, True, False])
    dn = _datanode(program, data)
    dn.calculateLcLoss(tnorm='P', compiled=True)
    solver, _ = dn.getILPSolver(
        conceptsRelations=dn.collectConceptsAndRelations())
    cache = solver._compiled_loss_calculator.plan_cache
    lc = next(lc for lc in program.graph.logicalConstrains.values() if lc.headLC)
    plan = cache.get(lc)

    from domiknows.solver.compiled import plan as plan_module

    def unexpected_formula_walk(_lc):
        raise AssertionError('formula signature recomputed on a cache hit')

    monkeypatch.setattr(plan_module, 'constraint_signature', unexpected_formula_walk)
    assert cache.get(lc) is plan


# ---------------------------------------------------------------------------
# Phase 3 — graph-wide batching of identity-grounded unary implications
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tnorm', TNORMS)
def test_batched_unary_implications_match_interpreter_and_gradients(tnorm):
    """Several KB rules share one [rules, objects] Torch invocation."""
    from test_regr.tiny_dynamic_graph.example_dynamic_global_constraints import (
        build_dynamic_constraint_example,
    )

    context = build_dynamic_constraint_example(device='cpu')
    context.graph.set_active_concepts(None)
    row = context.entries[0][1]
    _loss, _metric, _datanode, builder = context.program.model(row)
    builder.createBatchRootDN()
    dn = builder.getDataNode(device='cpu')
    dn.inferLocal(keys=('softmax',))

    reference = dn.calculateLcLoss(tnorm=tnorm)
    compiled = dn.calculateLcLoss(tnorm=tnorm, compiled=True)
    assert set(reference) == set(compiled)

    for name in reference:
        assert torch.allclose(
            reference[name]['lossTensor'], compiled[name]['lossTensor'], atol=1e-6)
        assert compiled[name].get('batchedFormula') is True

    parameters = tuple(context.shared_model.parameters())
    reference_total = sum(
        result['lossTensor'].sum() for result in reference.values())
    compiled_total = sum(
        result['lossTensor'].sum() for result in compiled.values())
    reference_grads = torch.autograd.grad(
        reference_total, parameters, retain_graph=True, allow_unused=True)
    compiled_grads = torch.autograd.grad(
        compiled_total, parameters, retain_graph=True, allow_unused=True)
    for expected, actual in zip(reference_grads, compiled_grads):
        if expected is None or actual is None:
            assert expected is None and actual is None
        else:
            assert torch.allclose(expected, actual, atol=1e-6)

    solver, _ = dn.getILPSolver(
        conceptsRelations=dn.collectConceptsAndRelations())
    info = solver._compiled_loss_calculator.cache_info()
    assert info['batched_formula_groups'] == 1
    assert info['batched_formula_constraints'] == len(reference)
    assert info['batched_formula_fallbacks'] == 0


@pytest.mark.parametrize('tnorm', TNORMS)
def test_batched_implication_primitive_preserves_row_semantics(tnorm):
    """Godel's row-level zero branch and Product zeros remain exact."""
    from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods

    processor = lcLossBooleanMethods()
    processor.current_device = 'cpu'
    processor.current_dtype = torch.float32
    processor.setTNorm(tnorm)
    antecedent = torch.tensor([
        [0.0, 0.3, 0.7, 0.5],
        [0.2, 0.5, 0.8, 0.4],
    ], requires_grad=True)
    consequent = torch.tensor([
        [0.0, 0.3, 0.2, 0.9],
        [0.1, 0.5, 0.9, 0.4],
    ], requires_grad=True)

    expected = torch.stack([
        processor.ifVar(
            None, antecedent[row], consequent[row], onlyConstrains=True)
        for row in range(antecedent.shape[0])
    ])
    actual = processor.ifVarBatched(
        None, antecedent, consequent, onlyConstrains=True)
    assert torch.allclose(expected, actual, atol=1e-7)

    expected_grads = torch.autograd.grad(
        expected.sum(), (antecedent, consequent), retain_graph=True,
        allow_unused=True)
    actual_grads = torch.autograd.grad(
        actual.sum(), (antecedent, consequent), allow_unused=True)
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        if expected_grad is None or actual_grad is None:
            assert expected_grad is None and actual_grad is None
        else:
            assert torch.allclose(expected_grad, actual_grad, atol=1e-7)


def test_batched_unary_implications_respect_dynamic_concept_activation():
    """The adjacency index selects only rules whose two concepts are active."""
    from test_regr.tiny_dynamic_graph.example_dynamic_global_constraints import (
        active_rule_names,
        build_dynamic_constraint_example,
    )

    context = build_dynamic_constraint_example(device='cpu')
    active, row = context.entries[0]
    expected_rule_names = {
        context.rules[name].lcName
        for name in active_rule_names(context, context.examples[0])
    }
    context.graph.set_active_concepts(active)
    _loss, _metric, _datanode, builder = context.program.model(row)
    builder.createBatchRootDN()
    dn = builder.getDataNode(device='cpu')
    dn.inferLocal(keys=('softmax',))

    result = dn.calculateLcLoss(tnorm='P', compiled=True)
    assert set(result) == expected_rule_names
    assert all(item.get('batchedFormula') is True for item in result.values())

    solver, _ = dn.getILPSolver(
        conceptsRelations=dn.collectConceptsAndRelations())
    info = solver._compiled_loss_calculator.cache_info()
    assert info['batched_formula_constraints'] == len(expected_rule_names)
    assert info['batched_formula_fallbacks'] == 0


def test_batched_formula_index_rebuilds_after_rule_mutation():
    """A persistent graph batch cannot retain a stale target literal."""
    from test_regr.tiny_dynamic_graph.example_dynamic_global_constraints import (
        build_dynamic_constraint_example,
    )

    context = build_dynamic_constraint_example(device='cpu')
    context.graph.set_active_concepts(None)
    row = context.entries[0][1]
    _loss, _metric, _datanode, builder = context.program.model(row)
    builder.createBatchRootDN()
    dn = builder.getDataNode(device='cpu')
    dn.inferLocal(keys=('softmax',))

    dn.calculateLcLoss(tnorm='P', compiled=True)
    solver, _ = dn.getILPSolver(
        conceptsRelations=dn.collectConceptsAndRelations())
    calculator = solver._compiled_loss_calculator
    before = calculator.cache_info()

    rule = context.rules['red_implies_colored']
    original_target = rule.e[2]
    rule.e[2] = context.rules['dog_implies_animal'].e[2]
    try:
        reference = dn.calculateLcLoss(tnorm='P', compiled=False)
        compiled = dn.calculateLcLoss(tnorm='P', compiled=True)
        after = calculator.cache_info()

        assert torch.allclose(
            reference[rule.lcName]['lossTensor'],
            compiled[rule.lcName]['lossTensor'],
            atol=1e-6,
        )
        assert compiled[rule.lcName].get('batchedFormula') is True
        assert after['batch_index_rebuilds'] == \
            before['batch_index_rebuilds'] + 1
        assert after['invalidations'] == before['invalidations'] + 1
    finally:
        rule.e[2] = original_target


if __name__ == '__main__':
    pytest.main([__file__])


# ---------------------------------------------------------------------------
# Multi-relation conjunctions: operands enumerated over different tuples
# ---------------------------------------------------------------------------

def test_shared_variable_conjunction_keeps_operands_independent():
    """`andL(A('z','x'), B('z','y'))` must not conflate x with y.

    Both relations are enumerated as 4x4 = 16 rows with `z` on the outer axis,
    so multiplying them row-by-row pairs A's (z,x) with B's (z,y) at the *same*
    inner index — silently imposing x == y. The correct reading quantifies the
    unshared variables away independently:

        phi(z) = (exists x. A(z,x)) and (exists y. B(z,y))

    This fixture makes the difference observable: A only ever holds at x == 0
    and B only at y == 1, so for every z there IS an x and a y satisfying both,
    but never with x == y. The conflating evaluation therefore reports the
    conjunction as unsatisfiable everywhere, while the correct one does not.
    """
    from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor
    from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods

    processor = lcLossBooleanMethods()
    processor.current_device = torch.device('cpu')
    processor.setTNorm('P')

    n = 4
    # A(z,x) true only when x == 0 ; B(z,y) true only when y == 1
    a = torch.tensor([1.0 if (r % n) == 0 else 0.0 for r in range(n * n)])
    b = torch.tensor([1.0 if (r % n) == 1 else 0.0 for r in range(n * n)])

    variables = OrderedDict([('A', [[a]]), ('B', [[b]])])
    bindings = {
        'A': (('z', 'x'), [(r // n, r % n) for r in range(n * n)]),
        'B': (('z', 'y'), [(r // n, r % n) for r in range(n * n)]),
    }

    reduced = LogicalConstraintConstructor.reduceToCommonGrounding(
        variables, bindings, processor)

    assert reduced is not variables, 'reduction should engage on differing tuples'
    for name in ('A', 'B'):
        column = reduced[name][0][0]
        assert column.shape == (n,), f'{name} should reduce onto z, got {tuple(column.shape)}'
        assert torch.allclose(column, torch.ones(n)), \
            f'{name}: exists-quantification should hold for every z, got {column}'

    # The conjunction is now satisfiable for every z ...
    joined = processor.andVar(None, reduced['A'][0][0], reduced['B'][0][0],
                              onlyConstrains=False)
    assert torch.allclose(joined, torch.ones(n))

    # ... whereas the old row-by-row product forced x == y and vanished.
    conflated = processor.andVar(None, a, b, onlyConstrains=False)
    assert torch.allclose(conflated, torch.zeros(n * n))


def test_cogrounded_conjunction_is_untouched():
    """Operands sharing one variable tuple keep their exact current behaviour."""
    from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor
    from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods

    processor = lcLossBooleanMethods()
    processor.current_device = torch.device('cpu')
    processor.setTNorm('P')

    keys = [(r,) for r in range(4)]
    variables = OrderedDict([
        ('A', [[torch.rand(4)]]),
        ('B', [[torch.rand(4)]]),
    ])
    bindings = {'A': (('z',), keys), 'B': (('z',), keys)}

    reduced = LogicalConstraintConstructor.reduceToCommonGrounding(
        variables, bindings, processor)
    assert reduced is variables, 'co-grounded operands must not be rewritten'


# ---------------------------------------------------------------------------
# queryVar: the answer must track the model under every t-norm
# ---------------------------------------------------------------------------

def _query_answer(tnorm, class_probs, selection, temperature=1.0):
    """Run queryVar directly with a known selection and class matrix."""
    from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods

    processor = lcLossBooleanMethods()
    processor.current_device = torch.device('cpu')
    processor.setTNorm(tnorm)

    subclasses = [(None, 'a', 0), (None, 'b', 1)]
    subclass_data = [[torch.tensor(p) for p in row] for row in class_probs]
    return processor.queryVar(
        None, None, subclasses, [torch.tensor(selection)],
        subclass_data=subclass_data, onlyConstrains=False,
        temperature=temperature, logicMethodName='QUERY')


@pytest.mark.parametrize('tnorm', ['P', 'L'])
def test_query_answer_is_not_squashed_toward_uniform(tnorm):
    """A confident model must yield a confident answer.

    ``subclass_scores = sum_i sel_i * c_ij`` is already the marginal
    probability of each class, because ``sel`` is a distribution over entities
    and each ``c_i`` a distribution over classes. Running softmax over it treats
    a probability as a logit: for two classes the answer could then never leave
    ``[0.27, 0.73]``, no matter how certain the model was.
    """
    # Selection is certain about entity 0, whose class is certain too.
    answer = _query_answer(tnorm, class_probs=[[1.0, 0.0], [0.0, 1.0]],
                           selection=[1.0, 0.0])

    assert answer.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert answer[0].item() > 0.9, \
        f'{tnorm}: confident model gave a squashed answer {answer}'


@pytest.mark.parametrize('tnorm', ['P', 'L'])
def test_query_answer_tracks_the_model(tnorm):
    """Flipping the model's class prediction must flip the answer."""
    favours_a = _query_answer(tnorm, [[0.9, 0.1], [0.5, 0.5]], [1.0, 0.0])
    favours_b = _query_answer(tnorm, [[0.1, 0.9], [0.5, 0.5]], [1.0, 0.0])

    assert favours_a[0].item() > favours_a[1].item()
    assert favours_b[1].item() > favours_b[0].item()
    assert favours_a[0].item() == pytest.approx(favours_b[1].item(), abs=1e-5)


@pytest.mark.parametrize('tnorm', ['P', 'L'])
def test_query_answer_follows_the_selection(tnorm):
    """Selecting a different entity must return that entity's class."""
    class_probs = [[1.0, 0.0], [0.0, 1.0]]
    first = _query_answer(tnorm, class_probs, [1.0, 0.0])
    second = _query_answer(tnorm, class_probs, [0.0, 1.0])

    assert first[0].item() > 0.9
    assert second[1].item() > 0.9


def test_query_temperature_sharpens():
    """Temperature still sharpens, now in probability space."""
    soft = _query_answer('P', [[0.6, 0.4]], [1.0], temperature=1.0)
    sharp = _query_answer('P', [[0.6, 0.4]], [1.0], temperature=0.25)

    assert sharp[0].item() > soft[0].item()
    assert sharp.sum().item() == pytest.approx(1.0, abs=1e-5)
