"""Incremental/fresh semantic counterexamples and actual reuse checks."""
import copy

import pytest
import torch

from domiknows.graph import (Graph, Concept, DataNode, Relation, EnumConcept, execute,
                            existsL, ifL, atMostL, sumL, fixedL, eqL)
from domiknows.solver import ilpOntSolverFactory


@pytest.fixture(autouse=True)
def reset():
    from domiknows import utils
    flags = utils.dnSkeletonMode, utils.dnSkeletonModeFull
    utils.dnSkeletonMode = utils.dnSkeletonModeFull = False
    Graph.clear(); Concept.clear(); Relation.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None
    yield
    utils.dnSkeletonMode, utils.dnSkeletonModeFull = flags
    Graph.clear(); Concept.clear(); Relation.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None


def scene(query=True):
    with Graph('incremental') as graph:
        world = Concept('world')
        item = Concept('item')
        world.contains(item)
        flag = item(name='flag')
        other = item(name='other')
        rule = ifL(flag('x'), other('x'))
        if query:
            execute(existsL(flag('x')))
    root = DataNode(instanceID=0, ontologyNode=world)
    root.current_device = 'cpu'
    children = []
    for index, probability in enumerate((.8, .3)):
        children.append(add(root, item, flag, other, index, probability))
    if query:
        constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
        constraint.attributes['ELC0/label'] = torch.tensor(0)
        root.addChildDataNode(constraint)
    return graph, root, item, flag, other, rule, children


def add(root, item, flag, other, index, probability):
    child = DataNode(instanceID=index, ontologyNode=item)
    child.current_device = 'cpu'
    for concept, p in ((flag, probability), (other, .7)):
        child.attributes[f'<{concept.name}>'] = torch.log(torch.tensor([1-p, p]))
        child.attributes[f'<{concept.name}>/local/softmax'] = torch.tensor([1-p, p])
    root.addChildDataNode(child)
    return child


def fresh(root, mode):
    with root.incrementalInference(mode=mode) as session:
        return session.infer(populate=False)


def parity(root, mode, result):
    reference = fresh(root, mode)
    assert reference.status == result.status, (reference.error, result.error)
    if mode == 'ilp':
        assert result.objective == pytest.approx(reference.objective)
        assert result.answers == reference.answers
        # Non-tied examples here have a unique selected assignment.
        assert result.assignments == reference.assignments
    else:
        assert result.answers.keys() == reference.answers.keys()
        for name, value in result.answers.items():
            assert value['distribution'] == pytest.approx(reference.answers[name]['distribution'])
            assert value['answer'] == reference.answers[name]['answer']


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_updates_match_fresh_and_no_change_reuses(mode):
    _, root, _, flag, _, _, children = scene()
    with root.incrementalInference(mode=mode) as session:
        first = session.infer(populate=False)
        assert first.status in ('complete', 'optimal'), first.error
        parity(root, mode, first)
        repeat = session.infer(populate=False)
        assert repeat.stats['evaluation_ms'] == 0
        assert repeat.stats['constraints_reused'] > 0
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.9, .1]))
        changed = session.infer(populate=False)
        assert changed.status in ('complete', 'optimal'), changed.error
        parity(root, mode, changed)
        if mode != 'ilp':
            assert changed.answers['ELC0']['distribution'][1] == pytest.approx(.37)
        if mode == 'ilp':
            assert changed.stats['ilp_models_reused'] > 0
            assert changed.stats['ilp_models_created'] == 0
        elif mode == 'circuit':
            assert changed.stats['circuit_structures_reused'] > 0


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_entity_add_remove_and_rule_changes(mode):
    graph, root, item, flag, other, rule, children = scene()
    with root.incrementalInference(mode=mode) as session:
        assert session.infer(populate=False).status in ('optimal', 'complete')
        third = add(root, item, flag, other, 2, .95)
        result = session.infer(populate=False)
        assert result.status in ('optimal', 'complete'), result.error
        parity(root, mode, result)
        if mode == 'ilp':
            assert result.stats['variables_added'] > 0
        root.removeChildDataNode(third)
        result = session.infer(populate=False)
        assert result.status in ('optimal', 'complete'), result.error
        parity(root, mode, result)
        rule.active = False
        result = session.infer(populate=False)
        assert result.status in ('optimal', 'complete'), result.error
        parity(root, mode, result)
        with graph:
            added = atMostL(flag('x'), 1)
        result = session.infer(populate=False)
        assert result.status in ('optimal', 'complete'), result.error
        parity(root, mode, result)
        added.e[-1] = 0
        result = session.infer(populate=False)
        assert result.status in ('optimal', 'complete'), result.error
        parity(root, mode, result)
        graph.logicalConstrains.pop(added.lcName)
        result = session.infer(populate=False)
        assert result.status in ('optimal', 'complete'), result.error
        parity(root, mode, result)


def test_populate_return_only_and_closure():
    _, root, _, _, _, _, children = scene(query=False)
    session = root.incrementalInference()
    result = session.infer(populate=False)
    assert result.status == 'optimal', result.error
    assert not any('/ILP' in key for child in children for key in child.attributes)
    session.infer()
    assert all('<flag>/ILP' in child.attributes for child in children)
    session.close(); session.close()
    with pytest.raises(RuntimeError, match='closed'):
        session.infer()


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_adhoc_queries_and_hints(mode):
    graph, root, _, flag, _, _, children = scene(query=False)
    before = list(graph.executableLCs)
    with root.incrementalInference(mode=mode, queries={'present': "existsL(flag('x'))"},
                                   queryNamespace={'flag': flag}) as session:
        first = session.infer(populate=False)
        assert first.status in ('optimal', 'complete'), first.error
        assert 'present' in first.answers
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        altered = session.infer(changes={'concepts': []}, populate=False)
        assert altered.status in ('optimal', 'complete'), altered.error
        assert 'flag' in altered.changes['concepts']
    assert list(graph.executableLCs) == before


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
@pytest.mark.parametrize('compiled', [True, False])
def test_relation_rewiring_with_identical_counts(mode, compiled):
    graph, root, item, flag, other, rule, children = scene(query=False)
    with graph:
        pair = Concept('pair')
        pair.has_a(arg1=item, arg2=item)
        root.ontologyNode.contains(pair)
        related = pair(name='related')
        ifL(related('x', 'y'), flag('x'))
    relation = DataNode(instanceID=0, ontologyNode=pair)
    relation.attributes['<related>'] = torch.log(torch.tensor([.01, .99]))
    relation.attributes['<related>/local/softmax'] = torch.tensor([.01, .99])
    relation.addRelationLink('arg1', children[0])
    relation.addRelationLink('arg2', children[1])
    root.addChildDataNode(relation)
    with root.incrementalInference(mode=mode, compiled=compiled) as session:
        first = session.infer(populate=False)
        assert first.status in ('complete', 'optimal'), first.error
        relation.removeRelationLink('arg1', children[0])
        relation.addRelationLink('arg1', children[1])
        changed = session.infer(populate=False)
        assert changed.status in ('complete', 'optimal'), changed.error
        parity(root, mode, changed)
        assert 'arg1' in changed.changes['relations']
        if mode == 'ilp':
            assert changed.stats['ilp_models_patched'] > 0
            selected = next(a for a in changed.assignments
                if a['concept'].endswith('/flag') and a['instance_id'] == 1)
            assert selected['value'] == 1


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_count_domain_and_enum_updates(mode):
    graph, root, item, flag, other, _, children = scene(query=False)
    with graph:
        execute(sumL(flag('x')))
        color = item(name='color', ConceptClass=EnumConcept, values=['red', 'blue'])
        execute(existsL(color.red('x')))
    for child in children:
        child.attributes['<color>'] = torch.log(torch.tensor([.8, .2]))
        child.attributes['<color>/local/softmax'] = torch.tensor([.8, .2])
    constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    constraint.attributes.update({'ELC0/label': torch.tensor(0), 'ELC1/label': torch.tensor(0)})
    root.addChildDataNode(constraint)
    with root.incrementalInference(mode=mode) as session:
        first = session.infer(populate=False)
        assert first.status in ('complete', 'optimal'), first.error
        third = add(root, item, flag, other, 2, .99)
        third.attributes['<color>'] = torch.log(torch.tensor([.2, .8]))
        third.attributes['<color>/local/softmax'] = torch.tensor([.2, .8])
        changed = session.infer(populate=False)
        assert changed.status in ('complete', 'optimal'), changed.error
        parity(root, mode, changed)
        if mode == 'ilp':
            assert changed.answers['ELC0'] == 2
        else:
            assert len(changed.answers['ELC0']['distribution']) == 4
        children[0].attributes['<color>/local/softmax'].copy_(torch.tensor([.01, .99]))
        changed = session.infer(populate=False)
        assert changed.status in ('complete', 'optimal'), changed.error
        parity(root, mode, changed)


def test_rule_priority_and_infeasible_rollback():
    graph, root, _, flag, _, _, children = scene(query=False)
    with graph:
        positive = existsL(flag('x'))
        negative = atMostL(flag('x'), 0, active=False)
    with root.incrementalInference() as session:
        first = session.infer()
        assert first.status == 'optimal', first.error
        previous = session.last_result.to_dict()
        negative.active = True
        failed = session.infer()
        assert failed.status == 'infeasible', failed.error
        assert session.last_result.to_dict() == previous
        assert all('<flag>/ILP' not in c.attributes for c in children)
        negative.p = 50
        changed = session.infer()
        assert changed.status == 'optimal', changed.error
        parity(root, 'ilp', changed)
        assert changed.stats['full_rebuild_reasons']
        positive.active = False
        changed = session.infer(populate=False)
        assert changed.status == 'optimal', changed.error
        parity(root, 'ilp', changed)


@pytest.mark.parametrize('mode', ['ilp', 'circuit', 'tnorm'])
def test_fixed_label_attributes_and_schema_rebuild(mode):
    graph, root, item, flag, other, _, children = scene()
    with graph:
        fixedL(flag('x', eqL(item, 'pinned', {True})))
    for child in children:
        child.attributes['pinned'] = torch.tensor(True)
        child.attributes['<flag>/label'] = torch.tensor(1)
    with root.incrementalInference(mode=mode) as session:
        first = session.infer(populate=False)
        assert first.status in ('complete', 'optimal'), first.error
        children[0].attributes['<flag>/label'] = torch.tensor(0)
        changed = session.infer(populate=False)
        assert changed.status in ('complete', 'optimal'), changed.error
        parity(root, mode, changed)
        with graph:
            new = item(name='new_flag')
        for child in children:
            child.attributes['<new_flag>'] = torch.log(torch.tensor([.9, .1]))
            child.attributes['<new_flag>/local/softmax'] = torch.tensor([.9, .1])
        changed = session.infer(populate=False)
        assert changed.status in ('complete', 'optimal'), changed.error
        assert 'schema changed' in changed.stats['full_rebuild_reasons']
        parity(root, mode, changed)


def test_unrelated_soft_rules_and_result_independence():
    graph, root, _, flag, other, _, children = scene(query=False)
    with graph:
        standalone = existsL(other('x'))
    with root.incrementalInference(mode='tnorm') as session:
        first = session.infer(populate=False)
        original = first.to_dict()
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        changed = session.infer(populate=False)
        assert changed.status == 'complete', changed.error
        assert changed.stats['constraints_reused'] == 1
        assert first.to_dict() == original
        changed.answers.clear()
        assert session.infer(populate=False).answers


def test_batch_membership_and_sample_isolation():
    graph, root, item, flag, other, _, _ = scene(query=False)
    with graph:
        batch = Concept('batch')
        batch.contains(root.ontologyNode)
    graph.batch = batch
    wrapper = DataNode(instanceID=0, ontologyNode=batch)
    second = DataNode(instanceID=1, ontologyNode=root.ontologyNode)
    add(second, item, flag, other, 0, .1)
    wrapper.addChildDataNode(root); wrapper.addChildDataNode(second)
    with wrapper.incrementalInference(mode='tnorm') as session:
        results = session.infer(populate=False)
        assert len(results) == 2
        assert all(r.status == 'complete' for r in results)
        wrapper.removeChildDataNode(second)
        results = session.infer(populate=False)
        assert len(results) == 1
        assert len(session._samples) == 1


def test_legacy_model_cache_cannot_reuse_equal_counts_with_changed_rules():
    graph, root, _, flag, _, _, _ = scene(query=False)
    concepts = root.collectConceptsAndRelations((flag,))
    solver, _ = root.getILPSolver(concepts)
    solver.reuse_model = True
    root.inferILPResults(flag)
    with graph:
        atMostL(flag('x'), 0)
    root.inferILPResults(flag)
    values = [c.getAttribute(flag, 'ILP').item()
              for c in root.findDatanodes(select=root.findRootConceptOrRelation(flag))]
    assert values == [0., 0.]


def test_failed_evaluation_rolls_back_and_recovers(monkeypatch):
    _, root, _, _, _, _, children = scene()
    with root.incrementalInference(mode='circuit') as session:
        first = session.infer()
        original = session.last_result.to_dict()
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        from domiknows.solver.executableInference import ExecutableInference
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise RuntimeError('injected failure')
            patch.setattr(ExecutableInference, '_infer_one', fail)
            failed = session.infer()
        assert failed.status == 'error'
        assert 'injected failure' in failed.error
        assert session.last_result.to_dict() == original
        recovered = session.infer(populate=False)
        assert recovered.status == 'complete', recovered.error
        parity(root, 'circuit', recovered)
        assert first.answers


def test_circuit_limit_fallback_matches_cold_after_changes():
    _, root, _, _, _, _, children = scene()
    options = dict(mode='circuit', circuit_backend='bdd', circuit_max_nodes=3,
                   circuit_size_limit_action='raise')
    with root.incrementalInference(**options) as session:
        assert session.infer(populate=False).status == 'complete'
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        changed = session.infer(populate=False)
        with root.incrementalInference(**options) as cold:
            reference = cold.infer(populate=False)
        assert changed.status == reference.status == 'complete', changed.error
        assert changed.answers == reference.answers
        assert changed.stats.get('backend_fallbacks')


@pytest.mark.parametrize('mode', ['tnorm', 'circuit', 'ilp'])
@pytest.mark.parametrize('full', [False, True])
def test_sensor_samples_and_skeleton_return_only(mode, full):
    from domiknows import setDnSkeletonMode
    from test_regr.solver.test_compiled_coverage import _build_fixed_program, _datanode
    from domiknows.solver.explanation.session import _nodes
    setDnSkeletonMode(full, full=full)
    try:
        program, data, _ = _build_fixed_program([False] * 4)
        root = _datanode(program, data)
        builder_keys = list(root.myBuilder)
        stored = [(n, dict(n.attributes)) for n in _nodes(root)]
        with root.incrementalInference(mode=mode) as session:
            results = session.infer(populate=False)
            results = results if isinstance(results, list) else [results]
            assert all(r.status in ('complete', 'optimal') for r in results), results
            again = session.infer(populate=False)
            again = again if isinstance(again, list) else [again]
            assert all(r.stats['evaluation_ms'] == 0 for r in again)
        assert list(root.myBuilder) == builder_keys
        for node, attrs in stored:
            assert node.attributes.keys() == attrs.keys()
            assert all(node.attributes[k] is v for k, v in attrs.items())
    finally:
        setDnSkeletonMode(False)


def test_interrupted_solve_does_not_commit_or_leave_stale_outputs(monkeypatch):
    from gurobipy import GRB
    _, root, _, _, _, _, children = scene()
    with root.incrementalInference(mode='ilp') as session:
        assert session.infer().status == 'optimal'
        previous = session.last_result.to_dict()
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        def interrupted(solver, *args, **kwargs):
            solver.statuses = [GRB.INTERRUPTED]
            return None
        with monkeypatch.context() as patch:
            patch.setattr(type(session._solver), '_calculateILPSelection', interrupted)
            result = session.infer()
        assert result.status == 'interrupted_or_ambiguous'
        assert session.last_result.to_dict() == previous
        assert all('<flag>/ILP' not in n.attributes for n in children)
        assert session.infer().status == 'optimal'


def test_mutation_during_evaluation_rejected_and_recoverable(monkeypatch):
    from domiknows.solver.executableInference import ExecutableInference
    _, root, _, _, _, _, children = scene()
    original = ExecutableInference._infer_one
    def mutating(*args, **kwargs):
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        return original(*args, **kwargs)
    with root.incrementalInference(mode='tnorm') as session:
        with monkeypatch.context() as patch:
            patch.setattr(ExecutableInference, '_infer_one', mutating)
            result = session.infer()
        assert result.status == 'error'
        assert 'Concurrent graph mutation' in result.error
        assert session.last_result is None
        result = session.infer(populate=False)
        assert result.status == 'complete', result.error
        assert result.answers['ELC0']['distribution'][1] == pytest.approx(.335)


def test_minimization_update_and_prediction_layout_rebuild():
    _, root, _, _, _, _, children = scene(query=False)
    options = dict(mode='ilp', minimizeObjective=True)
    with root.incrementalInference(**options) as session:
        assert session.infer(populate=False).status == 'optimal'
        children[0].attributes['<flag>/local/softmax'].copy_(torch.tensor([.95, .05]))
        updated = session.infer(populate=False)
        with root.incrementalInference(**options) as cold:
            expected = cold.infer(populate=False)
        assert updated.objective == pytest.approx(expected.objective)
        assert updated.assignments == expected.assignments
        children[0].attributes['<flag>/local/softmax'] = torch.tensor([.95, .05], dtype=torch.float64)
        updated = session.infer(populate=False)
        assert updated.status == 'optimal', updated.error
        assert updated.changes['input_layout']
        assert updated.stats['ilp_models_created'] > 0
