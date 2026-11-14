from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
import torch
from .. import agent
from .grounding import phrases,rank,region
from .contract import ExecutorContract


def test_cli_extends_existing_parser_without_argument_collisions():
    from .evaluate import build_parser
    args=build_parser().parse_args(['--checkpoint','snapshot.pt','--configuration','config.json',
        '--condition','raw','--experiment-output','output'])
    assert args.condition=='raw' and str(args.experiment_output)=='output'


def test_changed_manifest_or_cached_inputs_are_rejected(tmp_path):
    from ...protocol import sha
    from .provenance import validate_inputs
    config={}
    for name in ('observations','features','libero_manifest','archive_predictions'):
        path=tmp_path/name;path.write_text('original');config[name]=str(path);config[name+'_sha256']=sha(path)
    validate_inputs(config)
    (tmp_path/'libero_manifest').write_text('changed')
    with pytest.raises(ValueError,match='libero_manifest'):validate_inputs(config)


def test_raw_query_excludes_relations_without_changing_planner_input(monkeypatch):
    seen=[]
    monkeypatch.setattr(agent,'instruction_query',lambda p,s:seen.append(s) or torch.zeros(3))
    head=lambda *a:torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]])
    raw='Put the red box in the basket';bound={'instruction':raw+'\nInitial scene relations: arbitrary',
        'entity_table':('entity_000','entity_001'),'images':('raw','numbered')}
    effective,predicted,_=agent.predicted_context(None,head,bound,torch.zeros(2,3),torch.ones(2,dtype=torch.bool),grounding_instruction=raw)
    assert seen==[raw] and effective['instruction'].startswith(bound['instruction'])
    assert effective['images']==bound['images'] and predicted=={'source':0,'destination':1}
    agent.predicted_context(None,head,bound,torch.zeros(2,3),torch.ones(2,dtype=torch.bool))
    assert seen[-1]==bound['instruction'] # archived mode remains reproducible
    with pytest.raises(ValueError,match='supervision'):
        agent.predicted_context(None,head,{**bound,'goals':[1]},torch.zeros(2,3),torch.ones(2,dtype=torch.bool),grounding_instruction=raw)


def test_external_default_uses_raw_public_query_and_legacy_override_is_explicit(monkeypatch):
    from ...protocol import context
    seen=[];monkeypatch.setattr(agent,'instruction_query',lambda p,s:seen.append(s) or torch.zeros(3))
    public={'instruction':'Put the unfamiliar package in the basket','images':('raw','numbered'),
        'entities':['entity_000','entity_001'],'movable':['entity_000'],'entity_types':{},'initial_relations':[]}
    bound=context(public,'libero');head=lambda *a:torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]])
    feature=torch.zeros(2,3);visible=torch.ones(2,dtype=torch.bool)
    effective,_,_=agent.predicted_context(None,head,bound,feature,visible)
    assert seen==[public['instruction']] and effective['instruction'].startswith(bound['instruction'])
    agent.predicted_context(None,head,bound,feature,visible,grounding_instruction=bound['instruction'])
    assert seen[-1]==bound['instruction']


def test_rlbench_public_context_carries_query_without_copying_private_fields(monkeypatch):
    from ...protocol import context
    from ....generalization import rlbench_protocol
    monkeypatch.setattr(rlbench_protocol,'planning_context',lambda p:{'instruction':p['instruction']+'\nScene relations'})
    public={'instruction':'Move the package','goals':['hidden'],'reference_plan':['hidden']}
    bound=context(public,'rlbench')
    assert bound=={'instruction':'Move the package\nScene relations','grounding_instruction':'Move the package'}


@pytest.mark.parametrize('text,source,dest',[
    ('Pick up the black bowl between the plate and the cup and place it on the plate','black bowl between the plate and the cup','plate'),
    ('Pick the orange juice and place it in the basket','orange juice','basket'),
    ('Please move an unfamiliar package onto a tray.','unfamiliar package','tray')])
def test_instruction_only_role_phrases(text,source,dest):
    assert phrases(text)=={'source':source,'destination':dest}


def test_unknown_language_does_not_invent_grounding():
    assert phrases('Open the drawer')=={'source':None,'destination':None}


def test_retrieval_discriminates_competitors_and_preserves_permuted_ids():
    q=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    images=torch.tensor([[[.9,.1,0.],[.9,.1,0.]],[[1.,0.,0.],[1.,0.,0.]],[[0.,1.,0.],[0.,1.,0.]]])
    visible=torch.tensor([True,True,True]);p,d=rank(q,images,visible)
    assert p=={'source':1,'destination':2} and d['source']['nearest_competitor_ids'][0]==0
    assert d['source']['margin']>0
    order=torch.tensor([2,0,1]);permuted,_=rank(q,images[order],visible[order])
    assert {k:int(order[v]) for k,v in permuted.items()}==p
    p,_=rank(q,images,torch.tensor([True,False,True]));assert p['source']==0
    assert rank(q,images,torch.zeros(3,dtype=torch.bool))[0]=={'source':None,'destination':None}


def test_rgb_region_is_reproducible_and_does_not_mutate_observation():
    pytest.importorskip('cv2')
    pixels=np.full((480,480,3),170,np.uint8);pixels[190:280,220:260]=[220,30,30]
    image=Image.fromarray(pixels);a,pa=region(image,(240,235));b,pb=region(image,(240,235))
    assert pa==pb and np.array_equal(a,b) and np.array_equal(image,pixels)
    x0,y0,x1,y1=pa['box'];assert x0<=240<x1 and y0<=235<y1 and a.size[0]<192


@pytest.fixture(scope='module')
def native():
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    from ....generalization.shared_constraints import bind_shared
    vocab=PlanVocabulary.from_world(build_vlabench_world_graph('object_fixes_contract_test'))
    public={'entities':['entity_000','entity_001','entity_002'],'movable':['entity_000','entity_001']}
    bound={'entity_table':public['entities'],'generation_static_fixture_ids':[2]}
    base=bind_shared(vocab,12,bound)
    return ExecutorContract(base,vocab,public,{'pick':'target_entity_name','place':'target_container_name'}),vocab


def test_native_contract_enforces_executor_not_goal(native):
    from domiknows.generation.dfa.predicate_monitor import require_bound_predicate_dfa
    d,v=native;label=v.label_for_token;s=d.start_state
    assert require_bound_predicate_dfa(d) is d
    assert label('skill:pick') in d.allowed_tokens(s)
    for token in ('skill:pour','skill:pull','skill:place'):
        assert label(token) not in d.allowed_tokens(s)
    assert d.step(s,v.eos_label) is None
    s=d.step(s,label('skill:pick'));assert d.step(s,label('arg:target_container_name')) is None
    s=d.step(s,label('arg:target_entity_name'));assert d.step(s,label('obj:2')) is None
    assert d.step(s,label('obj:3')) is None
    s=d.step(s,label('obj:1'));assert d.step(s,label('skill:pick')) is None and d.step(s,v.eos_label) is None
    s=d.step(s,label('skill:place'));s=d.step(s,label('arg:target_container_name'))
    assert d.step(s,label('obj:1')) is None
    s=d.step(s,label('obj:0')) # legal WRONG target must remain available, no gold guidance
    s=d.step(s,v.eos_label);assert d.is_accepting(s)


def test_native_contract_budget_can_finish_full_pair(native):
    d,v=native;s=d.start_state
    labels=[v.label_for_token(t) for t in ('skill:pick','arg:target_entity_name','obj:0','skill:place','arg:target_container_name','obj:2')]+[v.eos_label]
    assert not d.allowed_tokens(s,remaining_steps=5)
    for i,label in enumerate(labels):
        assert label in d.allowed_tokens(s,remaining_steps=len(labels)-i-1)
        s=d.step(s,label);assert s is not None
    assert d.is_accepting(s)
    # A final EOS is allowed with zero steps left, avoiding a budget off-by-one.
    assert not d.allowed_tokens(s,remaining_steps=0)
