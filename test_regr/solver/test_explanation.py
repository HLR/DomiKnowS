"""Independent, small-world checks for explanation semantics and isolation."""
import json

import pytest
import torch

from domiknows.graph import (Graph, Concept, DataNode, EnumConcept, Relation, execute,
                            existsL, andL, notL, ifL, sumL, atMostL, fixedL, eqL, nandL)
from domiknows.solver import ilpOntSolverFactory


@pytest.fixture(autouse=True)
def reset():
    from domiknows import utils
    skeleton_flags = (utils.dnSkeletonMode, utils.dnSkeletonModeFull)
    utils.dnSkeletonMode = utils.dnSkeletonModeFull = False
    ilpOntSolverFactory.clear()
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.collectedConceptsAndRelations = None
    yield
    utils.dnSkeletonMode, utils.dnSkeletonModeFull = skeleton_flags
    ilpOntSolverFactory.clear()
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.collectedConceptsAndRelations = None


def scene(query=True, implication=False):
    with Graph('explanation') as graph:
        world = Concept('world')
        item = Concept('item')
        world.contains(item)
        flag = item(name='flag')
        other = item(name='other')
        if implication:
            ifL(flag('x'), other('x'))
        if query:
            execute(existsL(flag('x')))
    root = DataNode(instanceID=0, ontologyNode=world)
    root.current_device = 'cpu'
    children = []
    for index, probability in enumerate((0.8, 0.3)):
        child = DataNode(instanceID=index, ontologyNode=item)
        child.current_device = 'cpu'
        for concept, p in ((flag, probability), (other, 0.2)):
            child.attributes[f'<{concept.name}>'] = torch.log(torch.tensor([1-p, p]))
            child.attributes[f'<{concept.name}>/local/softmax'] = torch.tensor([1-p, p])
        root.addChildDataNode(child)
        children.append(child)
    if query:
        constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
        constraint.current_device = 'cpu'
        constraint.attributes['ELC0/label'] = torch.tensor(0)
        root.addChildDataNode(constraint)
    return graph, root, flag, other, children


@pytest.mark.parametrize('compiled', [True, False])
@pytest.mark.parametrize('mode', ['tnorm', 'circuit', 'ilp'])
def test_report_and_counterfactual_isolation(mode, compiled):
    graph, root, flag, other, children = scene()
    original = [(dict(c.attributes), {k: v.clone() for k, v in c.attributes.items()}) for c in children]
    original_graph = list(graph.executableLCs)
    report = root.explainInference(flag, mode=mode, compiled=compiled, tnorm='P')
    payload = report.to_dict()
    assert payload['status'] in ('complete', 'optimal'), payload
    assert json.loads(report.to_json()) == payload
    assert payload['evidence']
    assert any(e['kind'] == 'evaluation' for e in payload['evidence'])
    identifier = payload['predictions'][0]['id']
    alternative = report.counterfactual({identifier: False}).to_dict()
    assert alternative['status'] in ('complete', 'optimal'), alternative
    if mode == 'ilp':
        assert alternative['predictions'][0]['inferred_label'] is False
        assert 'objective_difference' in alternative['comparison']
    else:
        field = 'score' if mode == 'tnorm' else 'probability'
        assert payload['answers']['ELC0'][field] == pytest.approx(0.86)
        assert alternative['answers']['ELC0'][field] == pytest.approx(0.7)  # selected false answer: 1 - .3
        assert alternative['answers']['ELC0']['distribution'] == pytest.approx([0.7, 0.3])
    for child, (attrs, tensors) in zip(children, original):
        assert child.attributes.keys() == attrs.keys()
        assert all(child.attributes[k] is attrs[k] and torch.equal(child.attributes[k], tensors[k]) for k in attrs)
        assert child.gurobiModel is None
    assert list(graph.executableLCs) == original_graph


@pytest.mark.parametrize('mode', ['tnorm', 'circuit', 'ilp'])
def test_adhoc_and_truncation(mode):
    graph, root, flag, other, _ = scene(query=False)
    before = (list(graph.logicalConstrains), list(graph.executableLCs), list(graph.concepts))
    report = root.explainInference(flag, mode=mode, queries={'present': "existsL(flag('x'))"},
                                   queryNamespace={'flag': flag}, evidence_limit=1)
    payload = report.to_dict()
    assert 'present' in payload['answers'], payload
    assert len(payload['evidence']) == 1
    assert payload['truncation']['omitted'] > 0
    assert before == (list(graph.logicalConstrains), list(graph.executableLCs), list(graph.concepts))


def test_ilp_grounded_implication_and_infeasible_counterfactual():
    _, root, flag, other, _ = scene(query=False, implication=True)
    report = root.explainInference(flag, other)
    payload = report.to_dict()
    assert payload['status'] == 'optimal', payload
    assert payload['local_verification']['explanation/LC0']['satisfied'] == pytest.approx(50.)
    assert payload['ilp_verification']['explanation/LC0']['satisfied'] == pytest.approx(100.)
    assert any(p.get('related_evidence_ids') for p in payload['predictions'] if p.get('changed'))
    predictions = payload['predictions']
    a = next(p for p in predictions if p['path'].endswith('/flag') and p['instance_id'] == 0)
    b = next(p for p in predictions if p['path'].endswith('/other') and p['instance_id'] == 0)
    alternative = report.counterfactual({a['id']: True, b['id']: False}).to_dict()
    assert alternative['status'] == 'infeasible', alternative
    assert alternative['conflicts']
    rows = alternative['conflicts'][0]['rows']
    assert any('rule' in r for r in rows)
    assert any(r.get('kind') == 'counterfactual' for r in rows)
    assert 'objective_difference' not in alternative['comparison']


def test_summary_validates_references_and_keeps_report():
    _, root, flag, _, _ = scene()
    report = root.explainInference(flag, mode='tnorm')
    before = report.to_dict()
    evidence_id = before['evidence'][0]['id']
    assert report.summarize(lambda p: {'summary': 'Explanation', 'evidence_ids': [evidence_id]})['status'] == 'complete'
    assert report.summarize(lambda p: {'summary': 'Invented', 'evidence_ids': ['unknown']})['status'] == 'failed'
    def failing(p):
        p.clear()
        raise RuntimeError('provider failed')
    assert report.summarize(failing)['status'] == 'failed'
    def invented(p):
        p['evidence'].append({'id': 'invented'})
        return {'summary': 'Invalid', 'evidence_ids': ['invented']}
    assert report.summarize(invented)['status'] == 'failed'
    assert report.to_dict() == before


def test_invalid_assignments_and_options():
    _, root, flag, _, _ = scene()
    report = root.explainInference(flag, mode='tnorm')
    identifier = report.to_dict()['predictions'][0]['id']
    with pytest.raises(ValueError):
        report.counterfactual({identifier: 1})
    with pytest.raises(ValueError):
        report.counterfactual({'missing': True})
    with pytest.raises(TypeError):
        root.explainInference(flag, typo=True)
    with pytest.raises(ValueError):
        root.explainInference(flag, evidence_limit=-1)


@pytest.mark.parametrize('compiled', [True, False])
def test_shared_leaf_circuit_is_not_fuzzy_product(compiled):
    _, root, flag, _, _ = scene(query=False)
    options = {'queries': {'both': "existsL(andL(flag('x'), flag('x')))"},
               'queryNamespace': {'flag': flag}, 'compiled': compiled}
    exact = root.explainInference(flag, mode='circuit', **options).to_dict()
    fuzzy = root.explainInference(flag, mode='tnorm', tnorm='P', **options).to_dict()
    # Two worlds: P(any flag)=1-(1-.8)(1-.3). Duplicate literals are identical.
    assert exact['answers']['both']['truth_probability'] == pytest.approx(.86)
    # Product fuzzy conjunction repeats its operand numerically.
    assert fuzzy['answers']['both']['truth_score'] == pytest.approx(1-(1-.8**2)*(1-.3**2))
    assert any(e['kind'] == 'leaf' and e['entity']['instance_id'] == 0 for e in exact['evidence'])


def test_circuit_zero_probability_evidence_is_undefined():
    _, root, flag, _, children = scene()
    children[0].attributes[f'<{flag.name}>'] = torch.tensor([0., float('-inf')])
    children[0].attributes[f'<{flag.name}>/local/softmax'] = torch.tensor([1., 0.])
    report = root.explainInference(flag, mode='circuit')
    identifier = report.to_dict()['predictions'][0]['id']
    result = report.counterfactual({identifier: True}).to_dict()
    assert result['status'] == 'undefined_evidence'
    assert result['answers'] == {}


def enum_scene():
    with Graph('enum_explanation') as graph:
        world = Concept('world')
        item = Concept('item')
        world.contains(item)
        color = item(name='color', ConceptClass=EnumConcept, values=['red', 'blue'])
        execute(existsL(color.red('x')))
    root = DataNode(instanceID=0, ontologyNode=world)
    root.current_device = 'cpu'
    child = DataNode(instanceID=0, ontologyNode=item)
    child.current_device = 'cpu'
    child.attributes['<color>'] = torch.log(torch.tensor([.7, .3]))
    child.attributes['<color>/local/softmax'] = torch.tensor([.7, .3])
    root.addChildDataNode(child)
    constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    constraint.attributes['ELC0/label'] = torch.tensor(0)
    root.addChildDataNode(constraint)
    return root, color


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
@pytest.mark.parametrize('compiled', [True, False])
def test_enum_counterfactual_uses_class_names(mode, compiled):
    root, color = enum_scene()
    report = root.explainInference(color, mode=mode, compiled=compiled)
    baseline = report.to_dict()
    assert baseline['status'] in ('complete', 'optimal'), baseline
    prediction = baseline['predictions'][0]
    assert prediction['class_names'] == ['red', 'blue']
    result = report.counterfactual({prediction['id']: 'blue'}).to_dict()
    assert result['status'] in ('complete', 'optimal'), result
    if mode == 'ilp':
        assert result['predictions'][0]['inferred_label'] == 'blue'
        assert result['answers']['ELC0'] is False
    else:
        assert result['answers']['ELC0']['distribution'] == pytest.approx([1., 0.])
    with pytest.raises(ValueError):
        report.counterfactual({prediction['id']: 1})


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_count_query_reselects_answer(mode):
    graph, root, flag, _, _ = scene(query=False)
    with graph:
        execute(sumL(flag('x')))
    constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    constraint.attributes['ELC0/label'] = torch.tensor(0)
    root.addChildDataNode(constraint)
    report = root.explainInference(flag, mode=mode)
    baseline = report.to_dict()
    predictions = baseline['predictions']
    altered = report.counterfactual({p['id']: True for p in predictions}).to_dict()
    if mode == 'ilp':
        assert baseline['answers']['ELC0'] == 1
        assert altered['answers']['ELC0'] == 2
    else:
        assert baseline['answers']['ELC0']['distribution'] == pytest.approx([.14, .62, .24])
        assert altered['answers']['ELC0']['distribution'] == pytest.approx([0., 0., 1.])


@pytest.mark.parametrize('mode', ['circuit', 'ilp'])
@pytest.mark.parametrize('compiled', [True, False])
def test_fixed_label_conflicts_with_evidence(mode, compiled):
    graph, root, flag, _, children = scene()
    with graph:
        fixedL(flag('x', eqL(children[0].ontologyNode, 'pinned', {True})))
    for child in children:
        child.attributes['pinned'] = torch.tensor(True)
        child.attributes['<flag>/label'] = torch.tensor(1)
    report = root.explainInference(flag, mode=mode, compiled=compiled)
    result = report.counterfactual({report.to_dict()['predictions'][0]['id']: False}).to_dict()
    assert result['status'] == ('impossible_evidence' if mode == 'circuit' else 'infeasible'), result


def test_circuit_size_limit_has_no_fallback():
    _, root, flag, _, _ = scene()
    report = root.explainInference(flag, mode='circuit', circuit_backend='bdd', circuit_max_nodes=2).to_dict()
    assert report['status'] == 'size_limit', report
    assert report['requested_backend'] == 'circuit'
    assert report['answers'] == {}


def test_graph_changes_invalidate_counterfactual():
    graph, root, flag, other, _ = scene()
    report = root.explainInference(flag, mode='tnorm')
    with graph:
        ifL(flag('x'), other('x'))
    with pytest.raises(ValueError, match='Graph changed'):
        report.counterfactual({})


def test_batch_reports_keep_sample_ids_and_order():
    graph, first, flag, _, _ = scene()
    from domiknows.solver.explanation.session import _clone
    second = _clone(first)
    second.instanceID = 1
    with graph:
        batch = Concept('batch')
        batch.contains(first.ontologyNode)
    graph.batch = batch
    root = DataNode(instanceID=0, ontologyNode=batch)
    root.current_device = 'cpu'
    root.addChildDataNode(first)
    root.relationLinks['contains'].append(second)
    reports = root.explainInference(flag, mode='circuit')
    assert [r.to_dict()['sample'] for r in reports] == [0, 1]
    ids = [r.to_dict()['predictions'][0]['id'] for r in reports]
    assert ids[0] != ids[1]
    assert reports[1].counterfactual({ids[1]: False}).to_dict()['sample'] == 1


def test_failure_restores_graph_and_caller_cache():
    graph, root, flag, _, children = scene(query=False)
    before = (list(graph.logicalConstrains), graph.constraint_revision, graph.logicalConstrains)
    with pytest.raises(NameError):
        root.explainInference(flag, mode='tnorm', queries={'invalid': 'existsL(missing("x"))'})
    assert list(graph.logicalConstrains) == before[0]
    assert graph.constraint_revision == before[1]
    assert graph.logicalConstrains is before[2]
    assert all(not any('/ILP' in k for k in c.attributes) for c in children)


@pytest.mark.parametrize('minimize', [True, False])
def test_objective_configuration_and_custom_transform(minimize):
    _, root, flag, _, _ = scene(query=False)
    report = root.explainInference(flag, minimizeObjective=minimize, fun=lambda p: torch.log(p))
    baseline = report.to_dict()
    assert baseline['objective']['mode'] == 'custom'
    assert baseline['objective']['direction'] == ('minimize' if minimize else 'maximize')
    assert baseline['predictions'][0]['inferred_label'] is (not minimize)
    alternate = report.counterfactual({baseline['predictions'][0]['id']: minimize}).to_dict()
    assert alternate['status'] == 'optimal'
    assert 'objective_difference' in alternate['comparison']


@pytest.mark.parametrize('tnorm,expected', [('P', .86), ('L', 1.), ('G', .8)])
@pytest.mark.parametrize('compiled', [True, False])
def test_tnorms_have_manual_expected_scores(tnorm, expected, compiled):
    _, root, flag, _, _ = scene()
    report = root.explainInference(flag, mode='tnorm', tnorm=tnorm, compiled=compiled).to_dict()
    assert report['answers']['ELC0']['truth_score'] == pytest.approx(expected)
    leaves = [e for e in report['evidence'] if e['kind'] == 'leaf']
    assert any(e['entity']['instance_id'] == 0 and e['value'] == pytest.approx(.8) for e in leaves)


@pytest.mark.parametrize('backend', ['bdd', 'pysdd'])
def test_circuit_conditioning_backend_and_evidence_mass(backend):
    _, root, flag, _, _ = scene()
    report = root.explainInference(flag, mode='circuit', circuit_backend=backend)
    baseline = report.to_dict()
    altered = report.counterfactual({baseline['predictions'][0]['id']: False}).to_dict()
    assert altered['actual_backend'] == backend
    assert altered['evidence_probability'] == pytest.approx(.2)
    assert altered['answers']['ELC0']['truth_probability'] == pytest.approx(.3)


def test_priority_runs_and_inactive_rules():
    graph, root, flag, _, _ = scene(query=False)
    with graph:
        existsL(flag('x'), p=100)
        atMostL(flag('x'), 0, p=50)
        inactive = existsL(flag('x'), active=False)
    report = root.explainInference(flag).to_dict()
    assert report['status'] == 'optimal', report
    assert [r['priority'] for r in report['runs']] == [100, 50]
    assert [r['status'] for r in report['runs']] == ['optimal', 'infeasible']
    assert not any(e.get('rule') == inactive.lcName for e in report['evidence'])


def test_interrupted_solver_is_not_infeasible(monkeypatch):
    import gurobipy
    _, root, flag, _, _ = scene(query=False)
    optimize = gurobipy.Model.optimize
    def interrupted(model, *args, **kwargs):
        model.Params.TimeLimit = 0
        return optimize(model, *args, **kwargs)
    def unexpected_iis(model):
        pytest.fail('IIS must not run for interrupted solves')
    monkeypatch.setattr(gurobipy.Model, 'optimize', interrupted)
    monkeypatch.setattr(gurobipy.Model, 'computeIIS', unexpected_iis)
    report = root.explainInference(flag).to_dict()
    assert report['status'] == 'interrupted_or_ambiguous', report
    assert report['conflicts'] == []
    assert report['objective'] is None


def test_retained_solver_model_is_untouched():
    _, root, flag, _, children = scene(query=False)
    root.inferILPResults(flag)
    existing = [c.attributes['<flag>/ILP'] for c in children]
    tensors = [value.clone() for value in existing]
    concepts = root.collectConceptsAndRelations((flag,))
    solver, _ = root.getILPSolver(concepts)
    models = list(solver.model)
    solver_state = dict(solver.__dict__)
    original_model = root.gurobiModel
    report = root.explainInference(flag)
    report.counterfactual({report.to_dict()['predictions'][0]['id']: False})
    assert root.gurobiModel is original_model
    assert list(solver.model) == models
    assert solver.__dict__.keys() == solver_state.keys()
    assert all(solver.__dict__[k] is v for k, v in solver_state.items())
    assert all(c.attributes['<flag>/ILP'] is original and torch.equal(original, tensor)
               for c, original, tensor in zip(children, existing, tensors))


def test_counterfactual_chaining_and_detached_payload():
    _, root, flag, _, _ = scene()
    report = root.explainInference(flag, mode='circuit')
    ids = [p['id'] for p in report.to_dict()['predictions']]
    alternative = report.counterfactual({ids[0]: False})
    chained = alternative.counterfactual({ids[1]: False}).to_dict()
    assert chained['answers']['ELC0']['truth_probability'] == pytest.approx(0.)
    payload = report.to_dict()
    payload['predictions'].clear()
    assert report.to_dict()['predictions']
    assert len(report.to_dict()['counterfactuals']) == 1


@pytest.mark.parametrize('compiled', [True, False])
def test_relation_rule_grounding_and_conflict(compiled):
    graph, root, flag, _, children = scene(query=False)
    with graph:
        pair = Concept('pair')
        pair.has_a(arg1=children[0].ontologyNode, arg2=children[0].ontologyNode)
        root.ontologyNode.contains(pair)
        related = pair(name='related')
        ifL(related('x', 'y'), flag('x'))
    relation = DataNode(instanceID=17, ontologyNode=pair)
    relation.current_device = 'cpu'
    relation.attributes['<related>'] = torch.log(torch.tensor([.1, .9]))
    relation.attributes['<related>/local/softmax'] = torch.tensor([.1, .9])
    relation.addRelationLink('arg1', children[0])
    relation.addRelationLink('arg2', children[1])
    root.addChildDataNode(relation)
    report = root.explainInference(flag, related, compiled=compiled)
    baseline = report.to_dict()
    assert baseline['status'] == 'optimal', baseline
    a = next(p for p in baseline['predictions'] if p['path'].endswith('/flag') and p['instance_id'] == 0)
    r = next(p for p in baseline['predictions'] if p['path'].endswith('/related'))
    assert a['inferred_label'] is True
    assert r['inferred_label'] is True
    assert any(e['kind'] == 'leaf' and e['entity']['instance_id'] == 17 for e in baseline['evidence'])
    alternate = report.counterfactual({a['id']: False, r['id']: True}).to_dict()
    assert alternate['status'] == 'infeasible', alternate
    assert any(source.get('variables') for conflict in alternate['conflicts'] for source in conflict['rows'])


def test_disjointness_cannot_force_both_labels():
    graph, root, flag, other, _ = scene(query=False)
    with graph:
        nandL(flag('x'), other('x'))
    report = root.explainInference(flag, other)
    predictions = report.to_dict()['predictions']
    ids = [p['id'] for p in predictions if p['instance_id'] == 0]
    assert report.counterfactual({identifier: True for identifier in ids}).to_dict()['status'] == 'infeasible'


def test_fixed_labels_work_in_ordinary_ilp():
    graph, root, flag, _, children = scene(query=False)
    with graph:
        fixedL(flag('x', eqL(children[0].ontologyNode, 'pinned', {True})))
    for child in children:
        child.attributes['pinned'] = torch.tensor(True)
        child.attributes['<flag>/label'] = torch.tensor(1)
    root.inferILPResults(flag)
    assert all(c.attributes['<flag>/ILP'].item() == 1 for c in children)


@pytest.mark.parametrize('mode', ['tnorm', 'circuit', 'ilp'])
@pytest.mark.parametrize('full', [False, True])
def test_sensor_built_samples_and_skeleton_are_isolated(mode, full):
    from domiknows import setDnSkeletonMode
    from test_regr.solver.test_compiled_coverage import _build_fixed_program, _datanode
    from domiknows.solver.explanation.session import _nodes
    setDnSkeletonMode(full, full=full)
    try:
        program, data, _ = _build_fixed_program([False] * 4)
        dn = _datanode(program, data)
        builder = dn.myBuilder
        assert builder is not None
        builder_keys = list(builder)
        full_flag = builder.skeletonDataNodeFull
        stored = [(node, dict(node.attributes)) for node in _nodes(dn)]
        result = dn.explainInference(mode=mode)
        reports = result if isinstance(result, list) else [result]
        assert reports
        for report in reports:
            payload = report.to_dict()
            assert payload['status'] in ('complete', 'optimal'), payload
            assert payload['predictions']
            identifier = payload['predictions'][0]['id']
            report.counterfactual({identifier: False})
        assert list(builder) == builder_keys
        assert builder.skeletonDataNodeFull == full_flag
        for node, attributes in stored:
            assert node.attributes.keys() == attributes.keys()
            assert all(node.attributes[k] is v for k, v in attributes.items())
    finally:
        setDnSkeletonMode(False)


def test_hypothesis_ties_preserve_order_and_counterfactual_cost():
    _, root, flag, _, children = scene()
    for child in children:
        child.attributes['<flag>'] = torch.log(torch.tensor([.5, .5]))
        child.attributes['<flag>/local/softmax'] = torch.tensor([.5, .5])
    report = root.explainInference(flag)
    baseline = report.to_dict()
    assert baseline['answers']['ELC0'] is True
    assert sum(run['selected'] for run in baseline['runs']) == 1
    assert next(run for run in baseline['runs'] if run['selected'])['hypotheses'] == {'ELC0': True}
    alternate = report.counterfactual({p['id']: False for p in baseline['predictions']}).to_dict()
    assert alternate['answers']['ELC0'] is False
    assert alternate['comparison']['objective_difference'] == pytest.approx(0.)
    assert all('hypotheses' in conflict for conflict in alternate['conflicts'])


def test_contradictory_rules_report_proven_infeasibility():
    graph, root, flag, _, _ = scene(query=False)
    with graph:
        existsL(flag('x'))
        atMostL(flag('x'), 0)
    report = root.explainInference(flag).to_dict()
    assert report['status'] == 'infeasible', report
    assert report['conflicts'][0]['irreducible'] is True
    assert report['conflicts'][0]['minimum_certified'] is False
    assert len({r['rule'] for r in report['conflicts'][0]['rows'] if 'rule' in r}) == 2


def test_unsupported_executable_is_explicit():
    graph, root, flag, _, _ = scene(query=False)
    with graph:
        execute(andL(flag('x'), flag('x')))
    constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    constraint.attributes['ELC0/label'] = torch.tensor(0)
    root.addChildDataNode(constraint)
    report = root.explainInference(flag).to_dict()
    assert report['status'] == 'unsupported'
    assert 'andL' in report['unsupported'][0]
    assert root.gurobiModel is None


def test_existing_circuit_option_spellings_and_query_filter():
    _, root, flag, _, _ = scene()
    payload = root.explainInference(flag, mode='circuit', circuitBackend='bdd',
                                    circuitAggregation='joint', constraints='ELC0').to_dict()
    assert payload['actual_backend'] == 'bdd'
    assert payload['answers']['ELC0']['truth_probability'] == pytest.approx(.86)
    assert root.explainInference(flag, mode='tnorm', constraints=[]).to_dict()['answers'] == {}
    with pytest.raises(ValueError):
        root.explainInference(flag, constraints=[])
    with pytest.raises(ValueError):
        root.explainInference(flag, mode='circuit', circuitSizeLimitAction='warn')
    with pytest.raises(TypeError):
        root.explainInference(flag, mode='circuit', circuitBackend='bdd', circuit_backend='bdd')


def test_false_executable_answer_is_not_a_violated_gold_goal():
    _, root, flag, _, _ = scene()
    report = root.explainInference(flag)
    baseline = report.to_dict()
    alternate = report.counterfactual({p['id']: False for p in baseline['predictions']}).to_dict()
    assert alternate['answers']['ELC0'] is False
    assert alternate['status'] == 'optimal'
    assert alternate['local_verification'] == alternate['ilp_verification'] == {}


def test_summary_never_receives_raw_attributes_or_namespace_values():
    _, root, flag, _, children = scene(query=False)
    root.instanceValue = 'PRIVATE_SAMPLE_TEXT'
    children[0].attributes['note'] = 'PRIVATE_ATTRIBUTE'
    report = root.explainInference(flag, mode='tnorm', queries={'present': "existsL(flag('x'))"},
        queryNamespace={'flag': flag, 'note': 'PRIVATE_NAMESPACE'})
    serialized = report.to_json()
    assert 'PRIVATE_' not in serialized
    def adapter(payload):
        assert 'PRIVATE_' not in repr(payload)
        return {'summary': 'Grounded evidence', 'evidence_ids': [payload['evidence'][0]['id']]}
    assert report.summarize(adapter)['status'] == 'complete'
