"""Does the queryL hypothesis actually bind the answer to the selected entity?

Items 0 and 1 have fixed colors (0 red, 1 blue). The selector picks item `s`.
The answer for ``queryL(color, iotaL(...))`` must be the color of the selected
item: 'red' for s == 0, 'blue' for s == 1.  If the hypothesis constraint is
silently dropped, the class hypotheses tie and the first one ('red') always
wins, so the s == 1 case fails.

Native (real Gurobi) tests; they need DOMIKNOWS_REPRO_NATIVE=1.
"""
import itertools

import pytest
import torch

from domiknows.graph import Graph, Concept, andL, execute, iotaL, queryL
from domiknows.graph.concept import EnumConcept
from domiknows.graph.dataNode import DataNode
from domiknows.solver.answerModule import AnswerSolver

STRONG = 4.0


def logits(truth):
    return torch.tensor([0.0, STRONG] if truth else [STRONG, 0.0])


def activate(root, graph):
    node = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    node.attributes['ELC0/label'] = torch.tensor(1.)
    root.addChildDataNode(node)
    root.inferLocal(keys=('softmax',))
    root.setActiveExecutableLCs()


def solve(root, graph):
    solver, concepts = root.getILPSolver(root.collectConceptsAndRelations())
    return AnswerSolver(graph, solver=solver, compiled=True).solve_active_constraints(
        root, ['ELC0'], concepts, key=('local', 'softmax'), fun=torch.log,
        populate=True, raise_on_infeasible=True)


def color_logits(i):
    # item 0 red, item 1 blue, item 2 red
    return torch.tensor([STRONG, 0.0] if i != 1 else [0.0, STRONG])


@pytest.mark.parametrize('selected', [0, 1])
def test_entity_selector_answer_follows_selected_item(native_license, selected):
    with Graph('entity_query') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        target = item(name='target')
        color = item(name='color', ConceptClass=EnumConcept, values=['red', 'blue'])
        execute(queryL(color, iotaL(target('x'))))
    root = DataNode(instanceID=0, ontologyNode=scene)
    for i in range(3):
        child = DataNode(instanceID=i, ontologyNode=item)
        child.attributes['<target>'] = logits(i == selected)
        child.attributes['<color>'] = color_logits(i)
        root.addChildDataNode(child)
    activate(root, graph)
    result = solve(root, graph)
    assert result['hypotheses']['ELC0'] == ('red' if selected != 1 else 'blue')


@pytest.mark.parametrize('selected', [0, 1])
def test_relational_selector_answer_follows_selected_item(native_license, selected):
    """Relational selector: a class variable expanded over pair rows (n*n)."""
    with Graph('relational_query') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        pair = Concept(name='pair'); arg1, arg2 = pair.has_a(arg1=item, arg2=item)
        target = item(name='target'); mark = item(name='mark')
        rel = pair(name='rel')
        color = item(name='color', ConceptClass=EnumConcept, values=['red', 'blue'])
        execute(queryL(color, iotaL(andL(target('a'), rel('a', 'b'), mark('b')))))
    root = DataNode(instanceID=0, ontologyNode=scene)
    nodes = []
    for i in range(3):
        child = DataNode(instanceID=i, ontologyNode=item)
        child.attributes['<target>'] = logits(i == selected)
        child.attributes['<mark>'] = logits(i == 2)
        child.attributes['<color>'] = color_logits(i)
        root.addChildDataNode(child)
        nodes.append(child)
    for i, j in itertools.product(range(3), repeat=2):
        row = DataNode(instanceID=3 * i + j, ontologyNode=pair)
        row.addRelationLink(arg1.name, nodes[i]); row.addRelationLink(arg2.name, nodes[j])
        row.attributes['<rel>'] = logits(j == 2 and i in (0, 1))
        root.addChildDataNode(row)
    activate(root, graph)
    result = solve(root, graph)
    assert result['hypotheses']['ELC0'] == ('red' if selected != 1 else 'blue')


@pytest.mark.parametrize('compute_iis',[True,False])
def test_hypothesis_search_writes_no_iis_file_for_infeasible_hypothesis(
        native_license, tmp_path, monkeypatch, compute_iis):
    """Real Gurobi: one class hypothesis is infeasible because the class is
    forbidden.  The search must not spend time on (or write) an IIS for it,
    whatever the process-wide flag says."""
    from domiknows.graph import notL
    from domiknows.utils import setComputeIIS
    monkeypatch.setenv('DOMIKNOWS_LOG_DIR', str(tmp_path))
    with Graph('infeasible_hypothesis') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        target = item(name='target')
        material = item(name='material')
        metal = item(name='metal'); rubber = item(name='rubber')
        metal.is_a(material); rubber.is_a(material)
        notL(rubber('x'))                      # 'rubber' can never hold
        execute(queryL(material, iotaL(target('x'))))
    root = DataNode(instanceID=0, ontologyNode=scene)
    child = DataNode(instanceID=0, ontologyNode=item)
    child.attributes['<target>'] = logits(True)
    child.attributes['<metal>'] = logits(False)
    child.attributes['<rubber>'] = logits(True)   # prefers the forbidden class
    root.addChildDataNode(child)
    activate(root, graph)
    setComputeIIS(compute_iis)
    try:
        result = solve(root, graph)
    finally:
        setComputeIIS(True)
    assert result['hypotheses']['ELC0'] == 'metal'
    assert not (tmp_path / 'GurobiInfeasible.ilp').exists()
