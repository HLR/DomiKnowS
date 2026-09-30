"""Assertions specify desired behavior. Failures are intentional repro evidence.

Test doubles below replace an external solver, never DomiKnowS implementation.
No test uses gold labels to construct an ILP answer.
"""
import __main__
import itertools
import logging
import types
from collections import OrderedDict

import pytest
import torch

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.dataNode import DataNode
from domiknows.graph.logicalConstrain import LogicalConstrain, andL, orL, nandL, execute, miotaL
from domiknows.solver.answerModule import AnswerSolver
from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor as L
from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods


def binary_scene(selector=False):
    with Graph('repro') as graph:
        scene = Concept(name='scene'); item = Concept(name='item')
        scene.contains(item)
        flag = item(name='flag')
        if selector:
            execute(miotaL(flag('x'), threshold=0.5, hard=False))
    root = DataNode(instanceID=0, ontologyNode=scene)
    child = DataNode(instanceID=0, ontologyNode=item)
    root.addChildDataNode(child)
    child.attributes['<flag>/local/softmax'] = torch.tensor([0.01, 0.99])
    return graph, root, child, flag


@pytest.mark.parametrize('populate', [True, False])
def test_D01_direct_decode_uses_winner_and_restores_snapshot(populate):
    """Actual AnswerSolver orchestration, deterministic solver/decoder spies.

    This isolates decode ordering and key selection without a Gurobi license.
    It is not an end-to-end solver test (see test_native.py).
    """
    graph, root, child, flag = binary_scene(selector=True)
    child.attributes['<flag>/ILP'] = torch.tensor([1.0])  # previous world
    c = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    c.attributes['ELC0/label'] = torch.tensor(1.)
    root.addChildDataNode(c)

    class SolvedWorld:
        def _calculateILPSelection(self, *args, **kwargs):
            return {'objective': 0.0, 'values': {}}
        def populateILPSelection(self, dn, concepts, values):
            child.attributes['<flag>/ILP'] = torch.tensor([0.0])

    solver = AnswerSolver(graph, solver=SolvedWorld())
    observed = []
    def decoder(self, lc, dn, key):
        observed.append((key, child.attributes['<flag>/ILP'].item()))
        return []
    solver._decode_miota = types.MethodType(decoder, solver)
    solver.solve_active_constraints(root, ['ELC0'], ((flag, flag.name, None, 1),), populate=populate)
    expected = [(('ILP',), 0.0)]
    assert observed == expected, f'Decode must see solved zero using ILP key; observed {observed}'
    assert child.attributes['<flag>/ILP'].item() == (0.0 if populate else 1.0)


@pytest.mark.parametrize('truth', [0.0, 1.0])
def test_D02_binary_ilp_leaf_returns_scalar(truth):
    _, _, child, _ = binary_scene()
    child.attributes['<flag>/ILP'] = torch.tensor([truth])
    constructor = L(logging.getLogger('repro'))
    constructor.current_device = torch.device('cpu')
    actual = constructor.getMLResult(child, '<flag>/ILP', ('flag', 1, 0), 0, loss=True)
    assert torch.is_tensor(actual), f'Expected ILP truth {truth}, got {actual!r}'
    assert actual.numel() == 1 and actual.item() == truth


def test_D05_grounding_mismatch_must_not_silently_drop_constraint():
    lc = object.__new__(LogicalConstrain)
    calls = []
    def builder(model, *args, onlyConstrains=False):
        calls.append(args)
        return 1
    with pytest.raises((ValueError, RuntimeError)):
        result = lc.createLogicalConstrains('AND', builder, object(),
            OrderedDict(a=[[1], [0], [0], [1]], b=[[1], [0]]), True)
        assert not calls, 'Fixture expected mismatch before builder invocation'
        # The correct contract is an explicit error, not an empty constraint.


@pytest.mark.parametrize('n',[3,11])
def test_D06_hard_witness_survives_joint_expansion(n):
    """Default dimensions reproduce the historical 11^6 witness deletion.

    Native miota decoding uses the loss interpreter, which calls this method
    without an exact-hard flag. Preserve current defaults; do not lower limits.
    """
    names = list('abcdef')
    excluded = (0 if n<=4 else next(i for i in range(n) if i not in torch.topk(torch.ones(n), 4).indices.tolist()))
    operands = OrderedDict((v, [[torch.ones(n)]]) for v in names)
    bindings = {v: ((v,), [(i,) for i in range(n)]) for v in names}
    pairs = list(itertools.product(range(n), repeat=2))
    operands['edge'] = [[torch.tensor([float(i == j == excluded) for i, j in pairs])]]
    bindings['edge'] = (('b', 'c'), pairs)
    # Independent witness: every variable can equal excluded.
    assert operands['edge'][0][0][excluded*n+excluded].item() == 1
    expanded, joined, _ = L.expandToJointGrounding(
        operands, bindings, {v: list(range(n)) for v in names}, protect=('a',))
    assert joined
    conjunction = torch.stack([value[0][0] for value in expanded.values()]).prod(dim=0)
    assert bool((conjunction > .5).any()), 'Valid Boolean witness disappeared during decoding expansion'


def test_D07_same_script_workers_do_not_share_solution_path(tmp_path, monkeypatch):
    """Path collision repro, not a probabilistic write-race test."""
    from domiknows.utils import _default_log_dir
    script = tmp_path / 'runner.py'
    script.touch()
    monkeypatch.setattr(__main__, '__file__', str(script), raising=False)
    paths = []
    for name in ('worker1', 'worker2'):
        cwd = tmp_path / name
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        paths.append(_default_log_dir() + '/GurobiSolution.sol')
    assert paths[0] != paths[1], f'Distinct workers collide on {paths[0]}; use per-run output namespace'


def test_D08_clear_cache_visits_linked_nodes_and_preserves_predictions():
    """Checks supported cleanup helper, not unsupported automatic cache reuse."""
    _, root, child, _ = binary_scene()
    child.attributes['<flag>/ILP/x'] = object()
    child.attributes['<flag>/ILP'] = torch.tensor([1.0])
    AnswerSolver._clear_ilp_cache(root)
    assert '<flag>/ILP/x' not in child.attributes
    assert '<flag>/local/softmax' in child.attributes
    assert '<flag>/ILP' in child.attributes


def test_D09_collect_populated_binary_child_results():
    _, root, child, flag = binary_scene()
    child.attributes['<flag>/ILP'] = torch.tensor([1.0])
    actual = root.collectInferredResults(flag, 'ILP')
    assert actual.numel() == 1 and actual.item() == 1.0, f'Populated child returned {actual}'


@pytest.mark.parametrize('operator,method', [(andL, 'andVar'), (orL, 'orVar'), (nandL, 'nandVar')])
def test_T02_three_operand_head_is_violation_not_truth(operator, method):
    """Training-path historical defect; keep separate from ILP conclusions."""
    lc = object.__new__(LogicalConstrain)
    processor = lcLossBooleanMethods()
    processor.current_device = torch.device('cpu')
    processor.tnorm = 'P'
    for bits in itertools.product((0., 1.), repeat=3):
        values = OrderedDict((str(i), [[torch.tensor(x)]]) for i, x in enumerate(bits))
        actual = lc.createLogicalConstrains(method, getattr(processor, method), None, values, True)[0][0]
        truth = (all(bits) if operator is andL else any(bits) if operator is orL else not all(bits))
        assert torch.as_tensor(actual).item() == 1.0 - float(truth)


def test_T01_product_implication_has_finite_gradient_for_tiny_satisfied_antecedent():
    processor = lcLossBooleanMethods()
    processor.current_device = torch.device('cpu')
    processor.tnorm = 'P'
    a = torch.tensor([1e-30, .7], requires_grad=True)
    b = torch.tensor([.2, .3], requires_grad=True)
    loss = processor.ifVar(None, a, b, onlyConstrains=True)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(a.grad).all(), f'Finite loss but antecedent gradient = {a.grad}'
    assert torch.isfinite(b.grad).all()


def test_hypothesis_search_does_not_compute_iis_for_infeasible_hypotheses():
    """An infeasible hypothesis is an expected outcome of the search."""
    graph, root, child, flag = binary_scene(selector=False)
    with graph:
        from domiknows.graph.logicalConstrain import existsL
        execute(existsL(flag('x')))
    c = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    c.attributes['ELC0/label'] = torch.tensor(1.)
    root.addChildDataNode(c)
    seen = []

    class Solver:
        def _calculateILPSelection(self, *args, **kwargs):
            seen.append(kwargs.get('computeIIS', 'missing'))
            return None  # every hypothesis infeasible
        def populateILPSelection(self, *args):
            pass

    solver = AnswerSolver(graph, solver=Solver())
    solver.solve_active_constraints(root, ['ELC0'], ((flag, flag.name, None, 1),),
                                    populate=False, raise_on_infeasible=False)
    assert seen and all(value is False for value in seen), seen


def test_D05_operand_without_groundings_skips_the_constraint():
    """No groundings at all is an empty constraint, not a grounding mismatch.

    A nested constraint that found nothing to ground yields an empty operand
    (satisfaction reports hit this); only operands that all have groundings but
    in different numbers are a misalignment (see the D05 test above).
    """
    lc = object.__new__(LogicalConstrain)
    calls = []
    def builder(model, *args, onlyConstrains=False):
        calls.append(args)
        return 1
    result = lc.createLogicalConstrains('IF', builder, object(),
        OrderedDict(a=[[1]], b=[]), False)
    assert result == [] and not calls

