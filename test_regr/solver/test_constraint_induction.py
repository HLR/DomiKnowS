"""Independent tri-valued mining checks and human review/registry isolation."""
import copy
import json

import pytest
import torch
from domiknows.graph import Graph, Concept, DataNode, Relation, EnumConcept
from domiknows.solver import ilpOntSolverFactory
from domiknows.solver.constraint_induction import ConstraintInducer


@pytest.fixture(autouse=True)
def clear():
    from domiknows import utils
    flags = utils.dnSkeletonMode, utils.dnSkeletonModeFull
    utils.dnSkeletonMode = utils.dnSkeletonModeFull = False
    Graph.clear(); Concept.clear(); Relation.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None
    yield
    utils.dnSkeletonMode, utils.dnSkeletonModeFull = flags
    Graph.clear(); Concept.clear(); Relation.clear(); ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None


def scene():
    with Graph('induction') as graph:
        world, item = Concept('world'), Concept('item')
        world.contains(item)
        person, employee = item(name='person'), item(name='employee')
    roots, children = [], []
    for i in range(4):
        root = DataNode(instanceID=i, ontologyNode=world)
        node = DataNode(instanceID=0, ontologyNode=item)
        root.addChildDataNode(node)
        for concept, p in ((employee, .9), (person, .1)):
            node.attributes[f'<{concept.name}>/label'] = torch.tensor(1)
            node.attributes[f'<{concept.name}>/local/softmax'] = torch.tensor([1-p, p])
            node.attributes[f'<{concept.name}>'] = torch.tensor([1-p, p]).log()
        roots.append(root); children.append(node)
    inducer = ConstraintInducer(person, employee)
    book = inducer.propose(roots[:2], validation_examples=roots[2:], guidance=['employee implies person'])
    candidate = next(c['id'] for c in book.to_dict()['candidates'] if any(o['method']=='guidance' for o in c['origins']))
    return graph, roots, children, person, employee, inducer, book, candidate


def candidate(book, identifier):
    return next(c for c in book.to_dict()['candidates'] if c['id']==identifier)


def test_annotations_predictions_and_deterministic_mining_are_separate():
    graph, roots, children, person, employee, inducer, book, cid = scene()
    payload = candidate(book, cid)
    assert payload['status']=='proposed'
    assert payload['validation_status']=='passed'
    annotation = payload['metrics']['validation']['annotation']
    prediction = payload['metrics']['validation']['prediction']
    assert annotation['confidence']==1
    assert annotation['covered_examples']==2
    assert prediction['confidence']==0
    assert prediction['violations']==2
    assert all(e['kind']=='contradiction' for e in prediction['evidence'])
    assert not graph.logicalConstrains
    again = inducer.propose(roots[:2], validation_examples=roots[2:], guidance=['employee implies person'])
    assert book.to_dict()==again.to_dict()
    assert json.loads(book.to_json())==book.to_dict()
    assert 'human review required' in book.render_text()
    detached=book.to_dict();detached['candidates'].clear()
    assert book.to_dict()['candidates']


def test_unknown_labels_and_vacuous_implication_not_false_evidence():
    _, roots, children, _, _, inducer, _, _ = scene()
    children[2].attributes.pop('<person>/label')
    children[2].attributes['<person>/label']=torch.tensor(float('nan'))
    children[3].attributes['<employee>/label']=torch.tensor(0)
    children[3].attributes.pop('<person>/label')
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['employee implies person'])
    cid=next(c['id'] for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance')
    metrics=candidate(book,cid)['metrics']['validation']['annotation']
    assert metrics['unknown_groundings']==1
    assert metrics['vacuous_groundings']==1
    assert metrics['confidence'] is None
    assert metrics['violations']==0
    with pytest.raises(ValueError,match='override'):
        book.approve(cid,reviewer='human',rationale='reviewed')


def test_reviews_edits_revalidation_activation_and_rollback():
    graph, roots, _, _, _, _, book, cid=scene()
    decision=book.approve(cid,reviewer='alice',rationale='Domain definition verified')
    activation=book.activate(decision,graph)
    head=graph.logicalConstrains[activation['rule_path'].rsplit('/',1)[-1]]
    assert head.induction_provenance['decision']==decision
    with pytest.raises(ValueError):book.activate(decision,graph)
    with pytest.raises(ValueError):book.edit(cid,candidate(book,cid)['specification'],reviewer='alice',rationale='edit')
    event=book.rollback(activation,reviewer='alice',rationale='Pause for a narrower scope')
    assert event['action']=='rollback' and not head.active
    spec=candidate(book,cid)['specification'];spec['priority']=90
    edited=book.edit(cid,spec,reviewer='alice',rationale='Use a lower priority')
    assert edited['version']==2 and edited['metrics'] is None
    assert edited['versions'][0]['version']==1
    with pytest.raises(ValueError):book.activate(decision,graph)
    with pytest.raises(ValueError):book.approve(cid,reviewer='alice',rationale='not yet revalidated')
    book.revalidate()
    decision=book.approve(cid,reviewer='alice',rationale='Rechecked new version')
    activation=book.activate(decision,graph)
    assert activation['version']==2
    assert graph.logicalConstrains[activation['rule_path'].rsplit('/',1)[-1]].p==90


@pytest.mark.parametrize('mode',['ilp','circuit','tnorm'])
def test_preview_comparisons_restore_graph_outputs_and_models(mode):
    graph, roots, children, person, employee, _, book, cid=scene()
    marker=torch.tensor([42.]);children[0].attributes['<employee>/ILP']=marker
    original=dict(children[0].attributes)
    registry=list(graph.logicalConstrains.items())
    preview=book.preview(cid,roots[:1],modes=[mode])
    payload=preview.to_dict()['results'][0]
    assert payload['baseline']['status'] in ('optimal','complete')
    assert payload['proposed']['status'] in ('optimal','complete')
    assert list(graph.logicalConstrains.items())==registry
    assert children[0].attributes.keys()==original.keys()
    assert all(children[0].attributes[k] is v for k,v in original.items())
    if mode=='ilp':
        assert payload['changed_labels']
        assert payload['objective_difference']==pytest.approx(-.8)
    else:
        assert payload['objective_difference'] is None
        assert payload['proposed']['answers']
    json.loads(preview.to_json())


def test_group_preview_infeasibility_and_registry_restoration_after_exception(monkeypatch):
    graph, roots, _, person, employee, _, _, _=scene()
    inducer=ConstraintInducer(employee,templates=['counting'])
    book=inducer.propose(roots[:2],validation_examples=roots[2:],
        guidance=['At most 0 employee per example','At least 1 employee per example'],
        complete_domains={i:['employee'] for i in range(4)})
    ids=[c['id'] for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance']
    result=book.preview(ids,roots[:1],modes=['ilp']).to_dict()['results'][0]
    assert result['proposed']['status']=='infeasible'
    assert result['proposed']['conflicts']
    assert result['objective_difference'] is None
    assert not graph.logicalConstrains
    original=roots[0].explainInference
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1)
        if len(calls)>1:raise RuntimeError('injected preview failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(roots[0],'explainInference',fail)
    with pytest.raises(RuntimeError):book.preview(ids,roots[:1],modes=['tnorm'])
    assert not graph.logicalConstrains


def test_guidance_adapter_is_structured_and_rejects_unknown_symbols():
    graph, roots, _, person, employee, inducer, _, _=scene()
    def adapter(request):
        assert request['text']=='Some domain guidance'
        assert request['vocabulary']
        assert not any('attributes' in entry for entry in request['vocabulary'])
        return {'candidates':[dict(template='implication',antecedent='employee',consequent='person')]}
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['Some domain guidance'],guidance_adapter=adapter)
    assert any(c['origins'][0].get('adapter') for c in book.to_dict()['candidates'])
    bad=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['unsafe'],
        guidance_adapter=lambda request:{'candidates':[dict(template='implication',antecedent='employee',consequent='__import__("os")')]})
    unresolved=next(c for c in bad.to_dict()['candidates'] if c['specification'] is None)
    assert unresolved['validation_status']=='unresolved'
    with pytest.raises(ValueError):bad.approve(unresolved['id'],reviewer='bob',rationale='cannot approve',override=True)
    bad.edit(unresolved['id'],dict(template='implication',antecedent='employee',consequent='person'),reviewer='bob',rationale='Resolved the intended concepts')
    bad.revalidate()
    assert candidate(bad,unresolved['id'])['validation_status']=='passed'
    assert not graph.logicalConstrains


def test_clustered_confidence_and_evidence_truncation():
    graph, roots, children, person, employee, _, _, _=scene()
    children[2].attributes['<person>/label']=torch.tensor(0)
    for i in range(1,100):
        n=DataNode(instanceID=i,ontologyNode=children[3].ontologyNode)
        n.attributes['<person>/label']=torch.tensor(1);n.attributes['<employee>/label']=torch.tensor(1)
        roots[3].addChildDataNode(n)
    inducer=ConstraintInducer(person,employee,evidence_limit=0)
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['employee implies person'])
    c=next(c for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance')
    metrics=c['metrics']['validation']['annotation']
    assert metrics['confidence']==.5
    assert metrics['micro_confidence']==pytest.approx(100/101)
    assert metrics['evidence']==[] and metrics['truncation']['omitted']==101
    full=ConstraintInducer(person,employee).propose(roots[:2],validation_examples=roots[2:],guidance=['employee implies person'])
    other=candidate(full,c['id'])['metrics']['validation']['annotation']
    assert metrics['confidence']==other['confidence']
    assert metrics['stability']==other['stability']


def test_prediction_only_default_and_explicit_policy():
    _, roots, children, person, employee, _, _, _=scene()
    for node in children:
        node.attributes.pop('<person>/label');node.attributes.pop('<employee>/label')
        node.attributes['<person>/local/softmax']=torch.tensor([.1,.9])
    for permitted in (False,True):
        book=ConstraintInducer(person,employee,allow_prediction_only=permitted).propose(
            roots[:2],validation_examples=roots[2:],guidance=['employee implies person'])
        c=next(c for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance')
        assert c['validation_status']==('passed' if permitted else 'needs_review')
        assert c['validation_source']==('prediction' if permitted else 'annotation')


def test_leakage_stale_data_and_forged_approvals_rejected():
    graph, roots, children, person, employee, inducer, book, cid=scene()
    with pytest.raises(ValueError,match='disjoint'):
        inducer.propose(roots[:2],validation_examples=roots[1:])
    decision=book.approve(cid,reviewer='alice',rationale='reviewed')
    forged={**decision,'reviewer':'mallory'}
    with pytest.raises(ValueError):book.activate(forged,graph)
    children[2].attributes['<person>/label']=torch.tensor(0)
    with pytest.raises(ValueError,match='revalidate'):book.activate(decision,graph)
    book.revalidate()
    with pytest.raises(ValueError):book.activate(decision,graph)
    override=book.approve(cid,reviewer='alice',rationale='Domain requirement despite annotation contradiction',override=True)
    assert override['override'] and override['validation_status']=='needs_review'
    with Graph('wrong') as other:
        Concept('unrelated')
    with pytest.raises(ValueError):book.activate(override,other)
    assert not graph.logicalConstrains


@pytest.mark.parametrize('mode',['ilp','circuit','tnorm'])
def test_enum_restrictions_and_disjointness_compile(mode):
    graph, roots, children, person, employee, _, _, _=scene()
    with graph:
        color=children[0].ontologyNode(name='color',ConceptClass=EnumConcept,values=['red','blue'])
    for child in children:
        child.attributes['<color>/label']=torch.tensor(0)
        child.attributes['<color>/local/softmax']=torch.tensor([.9,.1])
        child.attributes['<color>']=torch.tensor([.9,.1]).log()
    book=ConstraintInducer(color,person).propose(roots[:2],validation_examples=roots[2:],
        guidance=['color=red implies person','color=blue and person are disjoint'])
    ids=[c['id'] for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance']
    assert len(ids)==2
    report=book.preview(ids,roots[:1],modes=[mode]).to_dict()['results'][0]
    assert report['proposed']['status'] in ('complete','optimal'),report['proposed']['unsupported']
    assert not graph.logicalConstrains


def relational_scene():
    graph,roots,children,person,employee,_,_,_=scene()
    with graph:
        pair=Concept('pair')
        pair.has_a(arg1=children[0].ontologyNode,arg2=children[0].ontologyNode)
        roots[0].ontologyNode.contains(pair)
        employment=pair(name='employment')
    for root,child in zip(roots,children):
        rel=DataNode(instanceID=0,ontologyNode=pair)
        rel.addRelationLink('arg1',child);rel.addRelationLink('arg2',child)
        rel.attributes['<employment>/label']=torch.tensor(1)
        rel.attributes['<employment>/local/softmax']=torch.tensor([.1,.9])
        rel.attributes['<employment>']=torch.tensor([.1,.9]).log()
        root.addChildDataNode(rel)
    return graph,roots,children,person,employee,employment


@pytest.mark.parametrize('mode',['ilp','circuit','tnorm'])
def test_relation_typing_and_per_owner_counting(mode):
    graph,roots,children,person,employee,employment=relational_scene()
    specs=[dict(template='relation_typing',relation='employment',roles={'arg1':'employee','arg2':'person'}),
           dict(template='counting',predicate='employment',operator='at_most',limit=1,group_by='arg1')]
    book=ConstraintInducer(person,employee,employment).propose(roots[:2],validation_examples=roots[2:],
        guidance=['Employment roles and counts'],guidance_adapter=lambda request:{'candidates':specs},
        complete_domains={i:['employment'] for i in range(4)})
    candidates=[c for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance']
    assert len(candidates)==2
    assert all(c['metrics']['validation']['annotation']['confidence']==1 for c in candidates)
    result=book.preview([c['id'] for c in candidates],roots[:1],modes=[mode]).to_dict()['results'][0]
    assert result['proposed']['status'] in ('complete','optimal'),result['proposed']['unsupported']
    assert not graph.logicalConstrains


def test_counting_requires_complete_annotated_domains_and_missing_role_is_unknown():
    _,roots,children,person,employee,employment=relational_scene()
    inducer=ConstraintInducer(employment,templates=['counting'])
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['At most 1 employment per arg1.'])
    c=book.to_dict()['candidates'][0]
    assert c['metrics']['validation']['annotation']['unknown_groundings']==2
    assert c['metrics']['validation']['annotation']['confidence'] is None
    rel=next(node for node in roots[2].relationLinks['contains'] if node.ontologyNode.name=='pair')
    rel.removeRelationLink('arg1',children[2])
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['At most 1 employment per arg1'],
        complete_domains={i:['employment'] for i in range(4)})
    c=next(c for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance')
    assert c['metrics']['validation']['annotation']['unknown_groundings']==1


def test_activation_invalidates_incremental_session():
    graph,roots,_,person,employee,_,book,cid=scene()
    with roots[0].incrementalInference(person,employee,mode='ilp') as session:
        before=session.infer(populate=False)
        decision=book.approve(cid,reviewer='reviewer',rationale='Verified implication')
        activation=book.activate(decision,graph)
        after=session.infer(populate=False)
        assert after.status=='optimal'
        assert after.changes['rules']
        assert after.objective < before.objective
        book.deactivate(activation,reviewer='reviewer',rationale='Temporarily pause')
        again=session.infer(populate=False)
        assert again.objective==pytest.approx(before.objective)


def test_no_holdout_cannot_be_approved_and_adapter_failure_stays_reviewable():
    _,roots,_,person,employee,inducer,_,_=scene()
    book=inducer.propose(roots[:2],guidance=['employee implies person'])
    cid=next(c['id'] for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance')
    with pytest.raises(ValueError,match='Held-out'):book.approve(cid,reviewer='x',rationale='override',override=True)
    def failed(request):raise RuntimeError('adapter failed')
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['Domain text'],guidance_adapter=failed)
    unresolved=next(c for c in book.to_dict()['candidates'] if c['specification'] is None)
    assert 'adapter failed' in unresolved['unresolved']
    assert any(c['specification'] is not None for c in book.to_dict()['candidates'])
    book.reject(unresolved['id'],reviewer='x',rationale='Needs better guidance')
    assert candidate(book,unresolved['id'])['status']=='rejected'


def test_reactivation_requires_new_decision_and_old_handle_cannot_undo_it():
    graph,roots,_,_,_,_,book,cid=scene()
    decision=book.approve(cid,reviewer='x',rationale='checked')
    first=book.activate(decision,graph)
    book.deactivate(first,reviewer='x',rationale='paused')
    preview=book.preview(cid,roots[:1],modes=['ilp']).to_dict()['results'][0]
    assert preview['changed_labels']
    assert not graph.logicalConstrains[first['rule_path'].rsplit('/',1)[-1]].active
    with pytest.raises(ValueError):book.activate(decision,graph)
    newer=book.approve(cid,reviewer='x',rationale='Resume the unchanged reviewed version')
    second=book.activate(newer,graph)
    with pytest.raises(ValueError):book.rollback(first,reviewer='x',rationale='old handle')
    book.rollback(second,reviewer='x',rationale='new handle')


def test_sparse_registry_names_and_external_rule_edits_are_guarded():
    from domiknows.graph import ifL
    graph,roots,_,person,employee,_,book,cid=scene()
    with graph:
        existing=ifL(person('x'),employee('x'))
    graph.logicalConstrains['LC9']=graph.logicalConstrains.pop(existing.lcName)
    existing.lcName='LC9'
    book.revalidate()
    decision=book.approve(cid,reviewer='x',rationale='checked')
    activation=book.activate(decision,graph)
    assert graph.logicalConstrains['LC9'] is existing
    head=graph.logicalConstrains[activation['rule_path'].rsplit('/',1)[-1]]
    book.deactivate(activation,reviewer='x',rationale='pause')
    head.p=50
    book.revalidate()
    newer=book.approve(cid,reviewer='x',rationale='review unchanged specification')
    with pytest.raises(ValueError,match='externally'):book.activate(newer,graph)


def test_candidate_bound_and_settings_change_require_revalidation():
    graph,roots,_,person,employee,_,_,_=scene()
    inducer=ConstraintInducer(person,employee,max_candidates=1)
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['employee implies person','person implies employee'])
    payload=book.to_dict()
    assert len(payload['candidates'])==1 and payload['truncation']['omitted_at_least']>0
    cid=payload['candidates'][0]['id']
    decision=book.approve(cid,reviewer='x',rationale='checked')
    inducer.policy['min_support_examples']=3
    with pytest.raises(ValueError):book.activate(decision,graph)
    book.revalidate()
    assert candidate(book,cid)['validation_status']=='needs_review'


def test_batch_siblings_are_not_mixed_in_extraction():
    graph,roots,_,person,employee,inducer,_,_=scene()
    with graph:
        batch=Concept('batch')
        batch.contains(roots[0].ontologyNode)
    graph.batch=batch
    wrapper=DataNode(instanceID=99,ontologyNode=batch)
    for root in roots:wrapper.addChildDataNode(root)
    book=inducer.propose(roots[:2],validation_examples=roots[2:],guidance=['employee implies person'])
    c=next(c for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance')
    assert c['metrics']['validation']['annotation']['applicable_groundings']==2


@pytest.mark.parametrize('full',[False,True])
def test_sensor_built_annotations_and_skeleton_are_isolated(full):
    from domiknows import setDnSkeletonMode
    from test_regr.solver.test_compiled_coverage import _build_fixed_program,_datanode
    from domiknows.solver.constraint_induction.data import sample_nodes
    setDnSkeletonMode(full,full=full)
    try:
        program,data,p=_build_fixed_program([False]*4)
        roots=[]
        for index in range(3):
            root=_datanode(program,copy.deepcopy(data))
            root.instanceID=index
            roots.append(root)
        stored=[(node,dict(node.attributes)) for root in roots for node in sample_nodes(root)]
        q=p.getOntologyGraph().findConcept('q')
        book=ConstraintInducer(p,q).propose(roots[:1],validation_examples=roots[1:],guidance=['p and q are disjoint'])
        c=next(c for c in book.to_dict()['candidates'] if c['origins'][0]['method']=='guidance')
        metrics=c['metrics']['validation']['annotation']
        assert metrics['applicable_groundings']==8
        assert metrics['confidence']==1
        for node,attributes in stored:
            assert node.attributes.keys()==attributes.keys()
            assert all(node.attributes[key] is value for key,value in attributes.items())
    finally:
        setDnSkeletonMode(False)
