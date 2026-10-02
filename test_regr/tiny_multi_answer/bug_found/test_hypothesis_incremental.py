"""The hypothesis search solves one shared model (hypothesis part added, solved and
removed) instead of rebuilding the model for every hypothesis.  That must not
change any answer: every scenario below runs through both paths and compares the
winning hypotheses, the objective and every solved ILP variable.

Native tests (real Gurobi); they need DOMIKNOWS_REPRO_NATIVE=1.
"""
import random

import pytest
import torch

from domiknows.graph import Graph, Concept, Relation, execute, existsL, iotaL, queryL, sumL, atLeastL
from domiknows.graph.dataNode import DataNode
from domiknows.solver import ilpOntSolverFactory
from domiknows.solver.answerModule import AnswerSolver


def fresh_state():
    Graph.clear(); Concept.clear(); Relation.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None


def finish(graph, root):
    node = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    for name in graph.executableLCs:
        node.attributes[f'{name}/label'] = torch.tensor(1.)
    root.addChildDataNode(node)
    root.inferLocal(keys=('softmax',))
    root.setActiveExecutableLCs()
    return root


def items(root, item, n, rng, binary=(), classes=(), target=None):
    for i in range(n):
        child = DataNode(instanceID=i, ontologyNode=item)
        for concept in binary:
            child.attributes[f'<{concept.name}>'] = torch.tensor([rng.uniform(-2, 2), rng.uniform(-2, 2)])
        for concept in classes:
            child.attributes[f'<{concept.name}>'] = torch.tensor([rng.uniform(-2, 2), rng.uniform(-2, 2)])
        if target is not None:
            child.attributes[f'<{target.name}>'] = torch.tensor([0., 4.] if i == 1 else [4., 0.])
        root.addChildDataNode(child)


def scenario_count(seed):
    """sumL hypotheses: one per possible count, built from indicator constraints."""
    rng = random.Random(seed)
    with Graph('count') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        flag = item(name='flag')
        execute(sumL(flag('x')))
    root = DataNode(instanceID=0, ontologyNode=scene)
    items(root, item, 6, rng, binary=(flag,))
    return graph, finish(graph, root), (flag,)


def scenario_joint_boolean(seed):
    rng = random.Random(seed)
    with Graph('joint') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        red = item(name='red'); blue = item(name='blue')
        execute(existsL(red('x')))
        execute(existsL(blue('x')))
    root = DataNode(instanceID=0, ontologyNode=scene)
    items(root, item, 4, rng, binary=(red, blue))
    return graph, finish(graph, root), (red, blue)


def scenario_count_and_boolean(seed):
    """Joint search: count hypotheses x boolean hypotheses (product of two specs)."""
    rng = random.Random(seed)
    with Graph('mixed') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        flag = item(name='flag'); other = item(name='other')
        execute(sumL(flag('x')))
        execute(existsL(other('x')))
    root = DataNode(instanceID=0, ontologyNode=scene)
    items(root, item, 4, rng, binary=(flag, other))
    return graph, finish(graph, root), (flag, other)


def scenario_class_query(seed):
    rng = random.Random(seed)
    with Graph('query') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        target = item(name='target')
        material = item(name='material')
        subs = [item(name=f'sub{k}') for k in range(4)]
        for sub in subs:
            sub.is_a(material)
        execute(queryL(material, iotaL(target('x'))))
    root = DataNode(instanceID=0, ontologyNode=scene)
    items(root, item, 5, rng, binary=tuple(subs), target=target)
    return graph, finish(graph, root), (target, *subs)


SCENARIOS = [scenario_count, scenario_joint_boolean, scenario_count_and_boolean, scenario_class_query]


def run(builder, seed, incremental, monkeypatch):
    """Solve one scenario; returns (result, normalized ILP values, which path ran)."""
    fresh_state()
    graph, root, concepts = builder(seed)
    solver, concept_tuples = root.getILPSolver(root.collectConceptsAndRelations())
    if not incremental:
        monkeypatch.setattr(solver, '_hypothesisBatchSupported', lambda *a, **k: False)
    calls = {'iter': 0, 'rebuild': 0}
    real_iter, real_rebuild = solver._iterHypothesisSelections, solver._calculateILPSelection

    def spy_iter(*a, **k):
        calls['iter'] += 1
        return real_iter(*a, **k)

    def spy_rebuild(*a, **k):
        calls['rebuild'] += 1
        return real_rebuild(*a, **k)

    monkeypatch.setattr(solver, '_iterHypothesisSelections', spy_iter)
    monkeypatch.setattr(solver, '_calculateILPSelection', spy_rebuild)
    result = AnswerSolver(graph, solver=solver, compiled=True).solve_active_constraints(
        root, list(graph.executableLCs), concept_tuples, key=('local', 'softmax'),
        fun=torch.log, populate=True, raise_on_infeasible=True)
    values = {(getattr(k[0], 'name', str(k[0])), k[1], k[2], k[3]): v for k, v in result['values'].items()}
    return result, values, calls


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('builder', SCENARIOS, ids=lambda b: b.__name__.replace('scenario_', ''))
def test_incremental_hypothesis_search_matches_the_rebuild_per_hypothesis(native_license, builder, seed, monkeypatch):
    new, new_values, new_calls = run(builder, seed, incremental=True, monkeypatch=monkeypatch)
    old, old_values, old_calls = run(builder, seed, incremental=False, monkeypatch=monkeypatch)
    assert new_calls['iter'] == 1 and new_calls['rebuild'] == 0, f'incremental path not taken: {new_calls}'
    assert old_calls['iter'] == 0 and old_calls['rebuild'] >= 1, f'fallback path not taken: {old_calls}'
    assert dict(new['hypotheses']) == dict(old['hypotheses'])
    assert new['objective'] == pytest.approx(old['objective'], abs=1e-9)
    assert new_values == old_values, 'the solved ILP variables differ between the two searches'
