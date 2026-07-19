"""Equivalence tests for the compiled logical-constraint loss path.

Runs ``DataNode.calculateLcLoss`` with the per-datanode interpreter and with
``compiled=True`` on the nested_relations example (real datanode graph with
words/phrases/pairs) and asserts the per-constraint ``lossTensor`` values and
the gradients reaching the prediction tensors are identical.

Constraint coverage of the example graph plus the extra constraints registered
here: ``nandL``, ``ifL`` + ``andL`` with forward ``path=`` hops, nested
``orL(andL, andL)`` over relation paths, ``atLeastL`` over a nested ``ifL``,
``existsL`` over a **reversed** path (``rel.reversed``, one-to-many expansion),
and conll04-style ``ifL(phrase('x'), exactL(...('x')))`` variable re-binding.
"""

import pytest
import torch

# Importing the example's fixture function registers the `case` fixture for
# this module as well; model_declaration wires the TestSensors onto the graph.
from test_regr.examples.nested_relations.test_main import model_declaration, test_case  # noqa: F401


# Shapes of the cases built so far in this process. Sensors stack on the
# shared example graph and re-execute on every later build, so two builds with
# differently-shaped data in one process would fail the TestSensor
# expected-input assertions (rebuilds with the seeded example case survive
# only because its tensors are regenerated bit-identically).
_BUILT_PHRASE_COUNTS = set()


def _build_datanode(case):
    from test_regr.examples.nested_relations.config import CONFIG

    _BUILT_PHRASE_COUNTS.add(len(case.phrase.raw))
    lbp = model_declaration(CONFIG.Model, case)
    _loss, _metric, datanode, _builder = lbp.model({})
    return datanode


@pytest.fixture(scope='module')
def extra_constraints():
    """Add reversed-path and exactL constraints; remove them on teardown."""
    from test_regr.examples.nested_relations.graph import (
        app_graph, phrase, people, organization, location, other, o,
        work_for, rel_pair_phrase1)
    from domiknows.graph.logicalConstrain import ifL, exactL, existsL

    before = set(app_graph.logicalConstrains.keys())

    with app_graph:
        # Reversed path: a people phrase must head at least one work_for pair.
        # Exercises impactLinks traversal and one-to-many expansion.
        ifL(people('x'),
            existsL(work_for(path=('x', rel_pair_phrase1.reversed.name))),
            active=True)

        # conll04-style label exclusivity via variable re-binding.
        ifL(phrase('x'),
            exactL(people('x'), organization('x'), location('x'), other('x'), o('x')),
            active=True)

    added = set(app_graph.logicalConstrains.keys()) - before
    yield added

    for name in added:
        del app_graph.logicalConstrains[name]


def _assert_loss_tensors_equal(lcName, ref, cmp, tnorm):
    rt = ref['lossTensor']
    ct = cmp['lossTensor']

    if rt is None or ct is None:
        assert rt is None and ct is None, \
            f'{lcName} ({tnorm}): one path produced a lossTensor, the other None'
        return

    assert rt.shape == ct.shape, \
        f'{lcName} ({tnorm}): shapes differ {rt.shape} vs {ct.shape}'

    rnan = torch.isnan(rt)
    cnan = torch.isnan(ct)
    assert torch.equal(rnan, cnan), f'{lcName} ({tnorm}): NaN masks differ'

    assert torch.allclose(rt[~rnan], ct[~cnan], atol=1e-6), \
        f'{lcName} ({tnorm}): values differ\ninterpreter={rt}\ncompiled={ct}'


def _total_loss(lcResult):
    parts = []
    for result in lcResult.values():
        t = result['lossTensor']
        if t is not None:
            parts.append(t[~torch.isnan(t)].sum())
    assert parts, 'no constraint produced a loss'
    return torch.stack(parts).sum()


@pytest.mark.parametrize('tnorm', ['L', 'G', 'P'])
def test_compiled_matches_interpreter(case, extra_constraints, tnorm):
    grad_leaves = [
        case.phrase.people, case.phrase.organization, case.phrase.location,
        case.phrase.other, case.phrase.O,
        case.pair.work_for,
        case.word.Iword, case.word.Bword, case.word.Eword,
    ]
    for t in grad_leaves:
        t.requires_grad_(True)

    datanode = _build_datanode(case)

    ref = datanode.calculateLcLoss(tnorm=tnorm)
    cmp = datanode.calculateLcLoss(tnorm=tnorm, compiled=True)

    assert ref, 'interpreter produced no constraint results'
    assert set(ref.keys()) == set(cmp.keys()), \
        f'constraint sets differ: {set(ref)} vs {set(cmp)}'

    for lcName in ref:
        _assert_loss_tensors_equal(lcName, ref[lcName], cmp[lcName], tnorm)

    # Gradient equivalence: the constraint loss must push the same corrections
    # into every prediction tensor on both paths.
    g_ref = torch.autograd.grad(_total_loss(ref), grad_leaves,
                                retain_graph=True, allow_unused=True)
    g_cmp = torch.autograd.grad(_total_loss(cmp), grad_leaves,
                                retain_graph=True, allow_unused=True)

    for leaf_idx, (a, b) in enumerate(zip(g_ref, g_cmp)):
        if a is None and b is None:
            continue
        # A path may register a leaf as "used with zero gradient" while the
        # other never touches it — numerically both mean zero gradient.
        if a is None:
            assert torch.count_nonzero(b) == 0, f'leaf {leaf_idx}: extra gradient in compiled path'
            continue
        if b is None:
            assert torch.count_nonzero(a) == 0, f'leaf {leaf_idx}: missing gradient in compiled path'
            continue
        assert torch.allclose(a, b, atol=1e-6), \
            f'leaf {leaf_idx} ({tnorm}): gradients differ\ninterpreter={a}\ncompiled={b}'


def test_compiled_path_actually_used(case, extra_constraints, monkeypatch):
    """Guard against vacuous equivalence: the compiled evaluator must handle
    every supported head constraint itself, with zero interpreter fallbacks."""
    from domiknows.solver.compiled import formula as formula_mod

    calls = {'compiled_head': 0, 'fallback': 0}

    orig_eval = formula_mod.CompiledConstraintEvaluator.constructCompiled

    def spy_eval(self, *args, **kwargs):
        if kwargs.get('headLC'):
            calls['compiled_head'] += 1
        return orig_eval(self, *args, **kwargs)

    monkeypatch.setattr(formula_mod.CompiledConstraintEvaluator, 'constructCompiled', spy_eval)

    orig_single = formula_mod.LossCalculator.calculate_single_lc_loss

    def spy_single(self, *args, **kwargs):
        if isinstance(self, formula_mod.CompiledLossCalculator):
            calls['fallback'] += 1
        return orig_single(self, *args, **kwargs)

    monkeypatch.setattr(formula_mod.LossCalculator, 'calculate_single_lc_loss', spy_single)

    datanode = _build_datanode(case)
    result = datanode.calculateLcLoss(tnorm='P', compiled=True)

    assert result
    assert calls['compiled_head'] >= len(result), \
        f"compiled evaluator ran for {calls['compiled_head']} constraints, expected >= {len(result)}"
    assert calls['fallback'] == 0, \
        f"{calls['fallback']} constraints fell back to the interpreter"


def test_compiled_calculator_reports_supported_types():
    """The Phase-1 supported set must cover the conll04-class constraints."""
    from domiknows.solver.compiled import lc_tree_supported, SUPPORTED_LC_TYPES
    from domiknows.graph.logicalConstrain import (
        notL, andL, orL, nandL, ifL, norL, xorL, iffL, forAllL,
        atMostL, atLeastL, exactL, existsL,
        atMostAL, atLeastAL, exactAL, existsAL,
        greaterL, equalCountsL,
    )

    for lcType in (notL, andL, orL, nandL, ifL, norL, xorL, forAllL,
                   atMostL, atLeastL, exactL, existsL,
                   atMostAL, atLeastAL, exactAL, existsAL,
                   greaterL, equalCountsL):
        assert issubclass(lcType, SUPPORTED_LC_TYPES), lcType.__name__
    assert issubclass(iffL, SUPPORTED_LC_TYPES)


def test_compiled_calculator_excludes_unverified_types():
    """The negative half of the contract.

    A type only belongs in SUPPORTED_LC_TYPES once a parity case proves the
    compiled result matches the interpreter. These are excluded on purpose:

    * ``eqL`` as a direct LC element — its ``__call__`` takes no
      ``headConstrain``/``integrate``, so the evaluator's call would TypeError.
      (``eqL`` inside a ``path=`` is fine; it is resolved structurally.)
    * ``queryL`` / ``iotaL`` — the interpreter's own t-norm loss for these
      raises, so there is no working reference to compare against.

    Without this assertion, widening the tuple by accident would be invisible.
    """
    from domiknows.graph.logicalConstrain import eqL, queryL, iotaL
    from domiknows.solver.compiled import SUPPORTED_LC_TYPES as SUPPORTED

    for lcType in (eqL, queryL, iotaL):
        assert not issubclass(lcType, SUPPORTED), \
            f'{lcType.__name__} is in SUPPORTED_LC_TYPES without a parity case'


def test_compile_lc_flag_reaches_cmodel():
    """PrimalDualProgram(..., compile_lc=True) must forward the flag into the
    constraint model via LossProgram's signature matching."""
    from inspect import signature
    from domiknows.program.model.lossModel import LossModel, PrimalDualModel
    from test_regr.examples.nested_relations.graph import graph

    assert 'compile_lc' in signature(PrimalDualModel.__init__).parameters
    assert 'compile_lc' in signature(LossModel.__init__).parameters

    cmodel = PrimalDualModel(graph, tnorm='P', device='cpu', compile_lc=True)
    assert cmodel.compile_lc is True

    cmodel_default = PrimalDualModel(graph, tnorm='P', device='cpu')
    assert cmodel_default.compile_lc is False


def _make_scaled_case(num_phrases):
    """Generate a nested_relations-shaped case with many phrases/pairs.

    Same structure as the example fixture but with ``num_phrases`` phrases and
    ``num_phrases**2`` pairs, to measure how both loss paths scale with the
    number of constraint groundings.
    """
    from domiknows.utils import Namespace

    torch.manual_seed(7)
    device = torch.device('cpu')
    n = num_phrases

    word_emb = torch.randn(4, 2048, device=device)

    pw1 = torch.zeros(n, 4, device=device)
    pw2 = torch.zeros(n, 4, device=device)
    pcw = torch.zeros(n, 4, device=device)
    for i in range(n):
        pw1[i, i % 4] = 1.0
        pw2[i, (i + 1) % 4] = 1.0
        pcw[i, i % 4] = 1.0

    phrase_emb = pw1 @ word_emb + pw2 @ word_emb

    pair_indices = [(i, j) for i in range(n) for j in range(n)]
    pa1 = torch.zeros(len(pair_indices), n, device=device)
    pa2 = torch.zeros(len(pair_indices), n, device=device)
    for r, (i, j) in enumerate(pair_indices):
        pa1[r, i] = 1.0
        pa2[r, j] = 1.0
    pair_emb = torch.cat((pa1 @ phrase_emb, pa2 @ phrase_emb), dim=1)

    def probs(rows):
        p = torch.rand(rows, 1, device=device)
        return torch.cat((1 - p, p), dim=1)

    case = {
        'sentence': {'raw': 'scaled benchmark sentence'},
        'word': {
            'scw': torch.ones(4, 1, device=device),
            'raw': ['w0', 'w1', 'w2', 'w3'],
            'emb': word_emb,
            'Eword': probs(4), 'Iword': probs(4), 'Bword': probs(4), 'Oword': probs(4),
        },
        'phrase': {
            'pcw_backward': pcw,
            'scp': torch.ones(n, 1, device=device),
            'emb': phrase_emb,
            'people': probs(n), 'organization': probs(n), 'location': probs(n),
            'other': probs(n), 'O': probs(n),
            'pw1_backward': pw1, 'pw2_backward': pw2,
            'raw': [f'p{i}' for i in range(n)],
        },
        'pair': {
            'pa1_backward': pa1, 'pa2_backward': pa2, 'emb': pair_emb,
            'work_for': probs(n * n), 'live_in': probs(n * n),
            'located_in': probs(n * n), 'orgbase_on': probs(n * n), 'kill': probs(n * n),
        },
    }
    return Namespace(case)


@pytest.mark.benchmark
def test_compiled_benchmark(case, extra_constraints):
    """Report interpreter vs compiled wall-clock for the constraint loss."""
    from time import perf_counter

    datanode = _build_datanode(case)

    # Warm both paths (softmax caching, lazy imports).
    datanode.calculateLcLoss(tnorm='P')
    datanode.calculateLcLoss(tnorm='P', compiled=True)

    n = 20

    # Alternate the two paths so cache warm-up and background drift do not
    # bias either measurement.
    interpreter_s = 0.0
    compiled_s = 0.0
    for _ in range(n):
        start = perf_counter()
        datanode.calculateLcLoss(tnorm='P')
        interpreter_s += perf_counter() - start

        start = perf_counter()
        datanode.calculateLcLoss(tnorm='P', compiled=True)
        compiled_s += perf_counter() - start
    interpreter_s /= n
    compiled_s /= n

    print(f'\nconstraint loss per item: interpreter={interpreter_s*1000:.1f}ms '
          f'compiled={compiled_s*1000:.1f}ms speedup={interpreter_s/compiled_s:.2f}x')

    # Sanity bound only — timing assertions are kept loose to avoid flakiness.
    assert compiled_s <= interpreter_s * 1.5


@pytest.mark.benchmark
def test_compiled_benchmark_scaled(extra_constraints):
    """Interpreter vs compiled at a realistic grounding count.

    Needs a process where no differently-shaped case was built yet (see
    _BUILT_PHRASE_COUNTS); run it alone: pytest -k benchmark_scaled.
    """
    from time import perf_counter

    if _BUILT_PHRASE_COUNTS - {25}:
        pytest.skip('scaled benchmark needs a fresh process; run it alone with -k benchmark_scaled')

    case = _make_scaled_case(25)  # 25 phrases -> 625 pairs
    datanode = _build_datanode(case)

    datanode.calculateLcLoss(tnorm='P')
    datanode.calculateLcLoss(tnorm='P', compiled=True)

    n = 3
    interpreter_s = 0.0
    compiled_s = 0.0
    for _ in range(n):
        start = perf_counter()
        datanode.calculateLcLoss(tnorm='P')
        interpreter_s += perf_counter() - start

        start = perf_counter()
        datanode.calculateLcLoss(tnorm='P', compiled=True)
        compiled_s += perf_counter() - start
    interpreter_s /= n
    compiled_s /= n

    print(f'\n[25 phrases / 625 pairs] constraint loss per item: '
          f'interpreter={interpreter_s*1000:.0f}ms compiled={compiled_s*1000:.0f}ms '
          f'speedup={interpreter_s/compiled_s:.2f}x')

    assert compiled_s <= interpreter_s * 1.5


if __name__ == '__main__':
    pytest.main([__file__])
