"""Hand-checkable acquisitions and pool/session isolation."""
import copy
import json
import math

import pytest
import torch

from domiknows.graph import Graph, Concept, DataNode, Relation, ifL, atMostL, EnumConcept, existsL, nandL
from domiknows.solver import ilpOntSolverFactory
from domiknows.solver.active_learning import ConstraintActiveLearner


@pytest.fixture(autouse=True)
def clean():
    from domiknows import utils
    flags = utils.dnSkeletonMode, utils.dnSkeletonModeFull
    utils.dnSkeletonMode = utils.dnSkeletonModeFull = False
    Graph.clear(); Concept.clear(); Relation.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None
    yield
    utils.dnSkeletonMode, utils.dnSkeletonModeFull = flags
    Graph.clear(); Concept.clear(); Relation.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None


def pool():
    with Graph('acquisition') as graph:
        world = Concept('world')
        item = Concept('item')
        world.contains(item)
        flag, other = item(name='flag'), item(name='other')
        rule = ifL(flag('x'), other('x'))
    roots, children = [], []
    for index, (a, b) in enumerate(((.8, .1), (.5, .8), (.1, .9))):
        root = DataNode(instanceID=index, ontologyNode=world)
        node = DataNode(instanceID=0, ontologyNode=item)
        for concept, p in ((flag, a), (other, b)):
            values = torch.tensor([1-p, p])
            node.attributes[f'<{concept.name}>'] = torch.log(values)
            node.attributes[f'<{concept.name}>/local/softmax'] = values
        root.addChildDataNode(node)
        roots.append(root); children.append(node)
    return graph, roots, children, flag, other, rule


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
@pytest.mark.parametrize('compiled', [True, False])
def test_conflicting_example_ranked_and_preserves_outputs(mode, compiled):
    _, roots, children, flag, other, _ = pool()
    marker = torch.tensor([42.])
    children[0].attributes['<flag>/ILP'] = marker
    attributes = [(node, dict(node.attributes)) for node in children]
    learner = ConstraintActiveLearner(mode=mode, strategy='constraint_conflict', compiled=compiled)
    report = learner.select(roots, budget=1)
    payload = report.to_dict()
    assert payload['selected'][0]['sample_id'] == 0, payload['excluded']
    target = next(t for t in payload['selected'][0]['targets'] if t['path'].endswith('/flag'))
    expected = {'ilp': 1., 'circuit': .72, 'tnorm': .875}[mode]
    assert target['components']['constraint_conflict'] == pytest.approx(expected)
    assert target['effective_weights'] == {'constraint_conflict': 1.}
    assert target['related_rules'] == ['acquisition/LC0']
    assert target['evidence_ids']
    assert set(target['evidence_ids']) <= {record['id'] for record in payload['selected'][0]['evidence']}
    assert json.loads(report.to_json()) == payload
    assert 'Suggestions are model predictions' in report.render_text()
    for node, attrs in attributes:
        assert node.attributes.keys() == attrs.keys()
        assert all(node.attributes[k] is v for k, v in attrs.items())


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_truncation_does_not_change_scores_or_selection(mode):
    _, roots, _, _, _, _ = pool()
    full = ConstraintActiveLearner(mode=mode).select(roots, budget=2).to_dict()
    empty = ConstraintActiveLearner(mode=mode, evidence_limit=0).select(roots, budget=2).to_dict()
    assert [e['sample_id'] for e in full['selected']] == [e['sample_id'] for e in empty['selected']]
    for a, b in zip(full['ranked'], empty['ranked']):
        assert a['score'] == pytest.approx(b['score'])
        assert [t['components'] for t in a['targets']] == [t['components'] for t in b['targets']]
        assert b['evidence'] == []
        assert b['truncation']['omitted'] > 0
        assert all(not t['evidence_ids'] for t in b['targets'])


def test_uncertainty_masks_ties_and_diversity():
    _, roots, _, _, _, _ = pool()
    learner = ConstraintActiveLearner(mode='tnorm', strategy='uncertainty', top_targets=1)
    first = learner.select(roots, budget=3).to_dict()
    assert first['selected'][0]['sample_id'] == 1
    assert first['selected'][0]['targets'][0]['components']['uncertainty'] == pytest.approx(1.)
    ids = [t['id'] for e in first['ranked'] if e['sample_id'] == 1 for t in e['targets']]
    next_round = learner.select(roots, budget=3, annotations={1: ids}).to_dict()
    assert 1 not in [e['sample_id'] for e in next_round['selected']]
    assert next_round['excluded'][0]['reason'] == 'fully_annotated_or_no_targets'
    partial = learner.select(roots, budget=3, annotations={1: ids[:1]}).to_dict()
    assert len(next(e for e in partial['ranked'] if e['sample_id'] == 1)['targets']) == 1
    with pytest.raises(ValueError, match='unknown targets'):
        learner.select(roots, budget=1, annotations={0: ['invalid']})
    with pytest.raises(ValueError, match='unknown sample'):
        learner.select(roots, budget=1, annotations={99: []})
    with pytest.raises(ValueError, match='Duplicate'):
        learner.select([roots[0], roots[0]], budget=1)
    diverse = ConstraintActiveLearner(mode='tnorm', strategy='uncertainty', diversity_weight=1.).select(
        roots, budget=2, embeddings={0: [1., 0.], 1: [1., 0.], 2: [-1., 0.]}).to_dict()
    assert [e['sample_id'] for e in diverse['selected']] == [1, 2]
    assert diverse['selected'][1]['diversity_score'] == pytest.approx(1.)


def test_unavailable_disagreement_excludes_rather_than_substitutes():
    _, roots, _, _, _, _ = pool()
    report = ConstraintActiveLearner(mode='circuit', strategy='inference_disagreement').select(roots, budget=2).to_dict()
    assert not report['selected']
    assert all(e['reason'] == 'no_available_weighted_signal' for e in report['excluded'])
    weighted = ConstraintActiveLearner(mode='tnorm').select(roots, budget=1).to_dict()['selected'][0]
    assert weighted['unsupported']
    assert all('inference_disagreement' not in t['effective_weights'] for t in weighted['targets'])


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_session_reuses_evidence_and_reacts_to_rules_and_probabilities(mode):
    _, roots, children, _, _, rule = pool()
    learner = ConstraintActiveLearner(mode=mode)
    with learner.session() as session:
        first = session.select(roots, budget=2)
        saved = first.to_dict()
        repeat = session.select(roots, budget=2).to_dict()
        assert repeat['stats']['explanations_reused'] == 3
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        changed = session.select(roots, budget=2).to_dict()
        assert changed['stats']['explanations_reused'] == 2
        fresh = learner.select(roots, budget=2).to_dict()
        assert changed['selected'] == fresh['selected']
        assert first.to_dict() == saved
        rule.active = False
        changed = session.select(roots, budget=2).to_dict()
        assert changed['stats']['explanations_reused'] == 0
        assert all(t['components']['constraint_conflict'] is None for e in changed['ranked'] for t in e['targets'])
        session.select(roots[:1], budget=1)
        assert len(session._entries) == 1
    session.close()
    with pytest.raises(RuntimeError, match='closed'):
        session.select(roots, budget=1)


def test_infeasibility_failure_and_cache_recovery(monkeypatch):
    graph, roots, children, flag, _, _ = pool()
    with graph:
        existsL(flag('x'))
        impossible = atMostL(flag('x'), 0)
    learner = ConstraintActiveLearner(mode='ilp')
    result = learner.select(roots, budget=1).to_dict()
    assert not result['selected']
    assert all(e['reason'] == 'infeasible' and e['review_required'] for e in result['excluded'])
    impossible.active = False
    with learner.session() as session:
        session.select(roots, budget=1)
        old = session._entries['0']['payload']
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.9, .1]))
        with monkeypatch.context() as patch:
            patch.setattr(roots[0], 'explainInference', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('failure')))
            failed = session.select(roots, budget=1).to_dict()
        assert failed['excluded'][0]['reason'] == 'evaluation_failed'
        assert session._entries['0']['payload'] is old
        recovered = session.select(roots, budget=1).to_dict()
        assert not recovered['excluded']


def test_enum_entropy_and_graph_size_aggregation():
    with Graph('enum_pool') as graph:
        world, item = Concept('world'), Concept('item')
        world.contains(item)
        color = item(name='color', ConceptClass=EnumConcept, values=['red', 'blue', 'green'])
    roots = []
    for index, count in enumerate((1, 8)):
        root = DataNode(instanceID=index, ontologyNode=world)
        for i in range(count):
            child = DataNode(instanceID=i, ontologyNode=item)
            values = torch.tensor([1/3, 1/3, 1/3])
            child.attributes['<color>'] = values.log()
            child.attributes['<color>/local/softmax'] = values
            root.addChildDataNode(child)
        roots.append(root)
    report = ConstraintActiveLearner(color, mode='tnorm', strategy='uncertainty').select(roots, budget=2).to_dict()
    assert [e['sample_id'] for e in report['selected']] == [0, 1]
    assert all(e['score'] == pytest.approx(1.) for e in report['selected'])
    assert report['selected'][0]['targets'][0]['class_names'] == ['red', 'blue', 'green']


@pytest.mark.parametrize('options', [{'weights': {'bad': 1}}, {'weights': {'uncertainty': -1}},
    {'top_targets': 0}, {'evidence_limit': -1}, {'diversity_weight': 2}, {'mode': 'bad'}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        ConstraintActiveLearner(**options)


def test_random_is_seeded_and_report_is_detached():
    _, roots, _, _, _, _ = pool()
    learner = ConstraintActiveLearner(mode='tnorm', strategy='random', seed=14)
    first = learner.select(roots, budget=3)
    second = learner.select(list(reversed(roots)), budget=3)
    assert first.to_dict()['selected'] == second.to_dict()['selected']
    payload = first.to_dict()
    payload['selected'].clear()
    assert len(first.to_dict()['selected']) == 3
    assert not learner.select(roots, budget=0).to_dict()['selected']
    with pytest.raises(ValueError):
        learner.select(roots, budget=1, embeddings={0: [0., 0.]})


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_relation_targets_and_rewiring(mode):
    graph, roots, children, flag, _, rule = pool()
    rule.active = False
    with graph:
        pair = Concept('pair')
        pair.has_a(arg1=children[0].ontologyNode, arg2=children[0].ontologyNode)
        roots[0].ontologyNode.contains(pair)
        related = pair(name='related')
        ifL(related('x', 'y'), flag('x'))
    relation = DataNode(instanceID=0, ontologyNode=pair)
    values = torch.tensor([.01, .99])
    relation.attributes['<related>'] = values.log()
    relation.attributes['<related>/local/softmax'] = values
    relation.addRelationLink('arg1', children[0])
    relation.addRelationLink('arg2', children[0])
    roots[0].addChildDataNode(relation)
    children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
    learner = ConstraintActiveLearner(mode=mode)
    result = learner.select(roots[:1], budget=1).to_dict()
    assert not result['excluded'], result['excluded']
    target = next(t for t in result['selected'][0]['targets'] if t['path'].endswith('/related'))
    assert target['entity_path'] == 'acquisition/pair'
    assert target['related_rules']
    assert target['components']['constraint_conflict'] is not None


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_disjointness_and_counting_conflict(mode):
    graph, roots, children, flag, other, rule = pool()
    rule.active = False
    with graph:
        nandL(flag('x'), other('x'))
        atMostL(flag('x'), 0)
    for concept in (flag, other):
        children[0].attributes[f'<{concept.name}>/local/softmax'].copy_(torch.tensor([.1, .9]))
    result = ConstraintActiveLearner(mode=mode, strategy='constraint_conflict').select(roots[:1], budget=1).to_dict()
    assert not result['excluded'], result['excluded']
    targets = result['selected'][0]['targets']
    assert all(t['components']['constraint_conflict'] > 0 for t in targets)
    assert 'acquisition/LC2' in next(t for t in targets if t['path'].endswith('/flag'))['related_rules']
    expected = 1. if mode == 'ilp' else .855
    assert next(t for t in targets if t['path'].endswith('/flag'))['components']['constraint_conflict'] == pytest.approx(expected)


def test_masks_only_rescore_and_inference_settings_invalidate_session():
    _, roots, _, _, _, _ = pool()
    learner = ConstraintActiveLearner(mode='tnorm')
    with learner.session() as session:
        first = session.select(roots, budget=1).to_dict()
        chosen = first['selected'][0]
        mask = {chosen['sample_id']: [t['id'] for t in chosen['targets']]}
        changed = session.select(roots, budget=1, annotations=mask).to_dict()
        assert changed['stats']['explanations_reused'] == 3
        learner.options['threshold'] = .7
        changed = session.select(roots, budget=1).to_dict()
        assert changed['stats']['explanations_reused'] == 0
        assert not changed['excluded']


def test_failed_configuration_update_preserves_previous_session(monkeypatch):
    _, roots, _, _, _, _ = pool()
    learner = ConstraintActiveLearner(mode='tnorm')
    with learner.session() as session:
        session.select(roots[:1], budget=1)
        previous = session._entries['0']
        learner.options['threshold'] = .7
        with monkeypatch.context() as patch:
            patch.setattr(roots[0], 'explainInference', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('failure')))
            result = session.select(roots[:1], budget=1).to_dict()
        assert result['excluded'][0]['reason'] == 'evaluation_failed'
        assert session._entries['0'] is previous
        assert not previous['engine'].closed
        del learner.options['threshold']
        result = session.select(roots[:1], budget=1).to_dict()
        assert result['stats']['explanations_reused'] == 1
