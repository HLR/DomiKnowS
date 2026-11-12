from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
import torch
from torch import nn

from .markers import templates,recognize,render
from .grounder import RoleGrounder,instruction_query,grounding_loss,predict_roles,role_targets,known_pointer_steps
from .agent import predicted_context


def test_isolated_launcher_preserves_corpus_identity_and_guard(tmp_path,monkeypatch):
    import importlib.util
    import json
    from .campaign import runtime_environment
    from test_regr.VLABenchAgentInterface import dataset_protocol
    root=tmp_path/'official';episode=root/'task/example0'
    (episode/'env_config').mkdir(parents=True);(episode/'output').mkdir()
    (episode/'env_config/scene.json').write_text('{"task":{"components":[]}}')
    (episode/'output/plan.json').write_text('[]')
    (root/'dataset_role.json').write_text(json.dumps({'role':'evaluation',
        'provenance':{'repo_id':'VLABench/vlm_evaluation_v1.0'}}))
    env=runtime_environment(tmp_path/'isolated-source',root)
    assert env['GIT_CONFIG_KEY_0']=='safe.directory'
    assert env['GIT_CONFIG_VALUE_0']==str(tmp_path/'isolated-source')
    monkeypatch.setenv('VLABENCH_OFFICIAL_EVALUATION_ROOT',env['VLABENCH_OFFICIAL_EVALUATION_ROOT'])
    spec=importlib.util.spec_from_file_location('isolated_dataset_guard',dataset_protocol.__file__)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    report=guard.guard_training_dataset(root,allow_internal_evaluation_training=True)
    assert report['official_evaluation_training_seen'] is True and report['episodes']==1
    with pytest.raises(ValueError,match='cannot be used for training'):
        guard.guard_training_dataset(root)
    with pytest.raises(ValueError,match='canonical official evaluation root'):
        guard.guard_training_dataset(tmp_path,allow_internal_evaluation_training=True)


def test_two_entity_training_covers_identity_and_swap_equally():
    from .augmentation import identity_epoch
    assert sum(identity_epoch('case',2,epoch) for epoch in range(4))==2
    assert sum(identity_epoch('case',8,epoch) for epoch in range(4))==1
    assert [identity_epoch('case',8,epoch) for epoch in range(4)]==[identity_epoch('case',8,epoch) for epoch in range(4)]


def test_recursive_deployment_paths_are_portable_to_linux(tmp_path):
    from .deploy import export_files
    package=tmp_path/'joint';folder=package/'transfer/anonymous_grounding';folder.mkdir(parents=True)
    (folder/'module.py').write_text('x=1')
    (folder/'atlas.json').write_text('{}')
    (package/'checkpoint.py').write_text('guard=True')
    (package/'main.py').write_text('auditor_hook=True')
    result=export_files(package)
    assert set(result)=={'transfer/anonymous_grounding/module.py','transfer/anonymous_grounding/atlas.json','checkpoint.py','main.py'}
    assert all('\\' not in name for name in result)


def pictures(tmp_path,value=11):
    raw=np.full((960,960,3),[70,100,130],dtype=np.uint8)
    stamp=next(s for v,s,o in templates(64) if v==value)
    mask=raw.copy();mask[100:127,100:100+stamp.shape[1]]=stamp[...,None]
    rp=tmp_path/'raw.png';mp=tmp_path/'mask.png';Image.fromarray(raw).save(rp);Image.fromarray(mask).save(mp)
    return rp,mp,raw,mask,stamp


def test_complete_multidigit_is_not_its_surviving_single_digit(tmp_path):
    raw,mask,_,pixels,stamp=pictures(tmp_path)
    proof=recognize(raw,mask,20)
    assert proof['visible_ids']==[11]
    # Occlude the first '1'; the remaining complete '1' is NOT object ID 1.
    pixels[100:127,100:117]=[30,60,90];Image.fromarray(pixels).save(mask)
    assert 1 not in recognize(raw,mask,20)['visible_ids']


def test_reannotation_starts_from_clean_rgb_not_old_marker_or_palette(tmp_path):
    raw,mask,rgb,pixels,stamp=pictures(tmp_path,value=2)
    pixels[300:350,300:350]=[0,255,0];Image.fromarray(pixels).save(mask)
    proof=recognize(raw,mask,3)
    output=tmp_path/'permuted.png';render(raw,proof['markers'],[2,0,1],output)
    changed=np.array(Image.open(output))
    assert np.array_equal(changed[300:350,300:350],rgb[300:350,300:350])
    assert np.array_equal(changed[100:102,100:102],rgb[100:102,100:102])
    assert proof['visible_ids']==[2]


def test_physical_text_and_ambiguous_duplicate_tags_are_not_bindings(tmp_path):
    raw,mask,rgb,pixels,stamp=pictures(tmp_path,value=2)
    Image.fromarray(pixels).save(raw)
    assert recognize(raw,mask,3)['visible_ids']==[]
    Image.fromarray(rgb).save(raw)
    pixels[200:227,200:200+stamp.shape[1]]=stamp[...,None];Image.fromarray(pixels).save(mask)
    proof=recognize(raw,mask,3)
    assert proof['visible_ids']==[] and proof['ambiguous_omitted']


def test_colored_occlusion_cannot_turn_part_of_another_number_into_an_id(tmp_path):
    raw,mask,_,pixels,stamp=pictures(tmp_path,value=4)
    pixels[100:127,100]=[80,60,40];Image.fromarray(pixels).save(mask)
    assert recognize(raw,mask,9)['visible_ids']==[4]
    # With possible IDs 14/24/34, the visible digit needs full word evidence.
    assert 4 not in recognize(raw,mask,40)['visible_ids']


def test_grounder_is_equivariant_and_supports_variable_entity_counts():
    torch.manual_seed(2);model=RoleGrounder(4,6,8).eval()
    query=torch.randn(4);features=torch.randn(5,6);visible=torch.tensor([1,1,0,1,1],dtype=torch.bool)
    order=torch.tensor([2,4,0,3,1]);before=model(query,features,visible);after=model(query,features[order],visible[order])
    torch.testing.assert_close(after[:-1],before[:-1][order],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(after[-1],before[-1],atol=1e-6,rtol=1e-5)
    assert model(query,features[:2],visible[:2]).shape==(3,2)


class TextModel(nn.Module):
    def __init__(self):super().__init__();self.emb=nn.Embedding(5,4)
    def forward(self,input_ids,**kwargs):return SimpleNamespace(hidden_states=(self.emb(input_ids),))


def planner():
    model=TextModel();seen=[]
    def prepare(domain,context,tokens):
        seen.append(context)
        assert context['entity_table']==() and context['images']==()
        return {'input_ids':torch.tensor([[1,2]])},[1]
    return SimpleNamespace(model=model,_prepare_inputs_and_boundaries=prepare,_inputs_to_model_device=lambda x:x),seen


def test_grounding_gradient_reaches_shared_query_backbone():
    plan,_=planner();head=RoleGrounder(4,6,8)
    query=instruction_query(plan,'put the apple in the bowl');visible=torch.tensor([True,True,True])
    logits=head(query,torch.randn(3,6),visible)
    loss,_=grounding_loss(logits,visible,{'source':{1},'destination':{2}});loss.backward()
    assert plan.model.emb.weight.grad is not None and torch.any(plan.model.emb.weight.grad!=0)
    assert head.score[-1].weight.grad is not None and torch.any(head.score[-1].weight.grad!=0)


def test_wrong_predicted_roles_are_never_replaced_by_gold():
    plan,seen=planner()
    class WrongHead(nn.Module):
        def forward(self,q,features,visible):return torch.tensor([[5.,4.],[0.,1.],[-1.,-1.]],requires_grad=True)
    context={'instruction':'put the apple in the bowl','entity_table':('a','b'),'images':('raw','numbered')}
    effective,predictions,logits=predicted_context(plan,WrongHead(),context,torch.ones(2,6),torch.ones(2,dtype=torch.bool))
    assert predictions=={'source':0,'destination':0}
    assert effective['entity_table']==('numbered object 0','numbered object 1')
    loss,_=grounding_loss(logits,torch.ones(2,dtype=torch.bool),{'source':{1},'destination':{1}})
    assert loss>0 and 'source ID 0' in effective['instruction'] and 'destination ID 0' in effective['instruction']
    assert 'Predicted' not in seen[0]['instruction']
    with pytest.raises(ValueError,match='supervision'):
        predicted_context(plan,WrongHead(),{**context,'reference_plan':[1]},torch.ones(2,6),torch.ones(2,dtype=torch.bool))


def test_unknown_positive_roles_are_not_negative_or_gold_substitutions():
    logits=torch.tensor([[1.,2.],[-float('inf'),-float('inf')],[0.,0.]],requires_grad=True)
    visible=torch.tensor([True,False]);targets={'source':{1},'destination':{1}}
    loss,coverage=grounding_loss(logits,visible,targets)
    assert loss==0 and torch.isfinite(loss)
    assert coverage['source']=={'targets':1,'observable_targets':0}
    loss.backward();assert torch.all(logits.grad==0)
    empty=RoleGrounder(4,6,8)(torch.randn(4),torch.randn(2,6),torch.zeros(2,dtype=torch.bool))
    assert predict_roles(empty,torch.zeros(2,dtype=torch.bool))=={'source':None,'destination':None}


def test_pointer_supervision_keeps_syntax_and_only_observed_pointers():
    vocabulary=SimpleNamespace(token_for_label=lambda label:['skill:pick','obj:0','obj:1','<eos>'][label])
    labels=torch.tensor([0,2,1,3]);keep=known_pointer_steps(labels,vocabulary,torch.tensor([True,False]))
    assert keep.tolist()==[True,False,True,True]
    targets=role_targets([{'name':'pick','params':{'target_entity_name':'apple','duration':2}},
                         {'name':'place','params':{'target_container_name':1}}],('apple','bowl'))
    assert targets=={'source':{0},'destination':{1}}


def test_standard_loader_rejects_snapshot_that_requires_grounding(tmp_path):
    from ...checkpoint import load_joint_checkpoint
    path=tmp_path/'requires_head.pt';torch.save({'metrics':{'grounding_required':True}},path)
    with pytest.raises(ValueError,match='grounding loader|anonymous_grounding loader'):
        load_joint_checkpoint(path,runtime=None,planner=SimpleNamespace(),controller=None,map_location='cpu')


def test_unknown_pointer_has_zero_planning_gradient_and_hints_do_not_bind_dfa(monkeypatch):
    from . import objective
    logits=nn.Parameter(torch.zeros(1,4,5));captured=[]
    vocabulary=SimpleNamespace(eos_label=4,label_count=5,
        token_for_label=lambda i:['skill:pick','arg:target_entity_name','obj:0','obj:1','<eos>'][i])
    runtime=SimpleNamespace(vlabench_vocabulary=vocabulary,max_vlabench_operations=1,
        dfa_for=lambda domain,context:captured.append(context) or object())
    fake=SimpleNamespace(device='cpu',sequence_logits=lambda *a:logits,shift_right=lambda *a:None)
    monkeypatch.setattr(objective,'allowed_label_sets',lambda *a,**k:{i:set(range(5)) for i in range(4)})
    original={'instruction':'move the object'};effective={'instruction':'move the object; predicted ID 0'}
    loss,coverage=objective.planning_loss(fake,runtime,original,effective,torch.tensor([0,1,3,4]),torch.tensor([True,False]))
    loss.backward()
    assert captured==[original] and coverage['unknown_pointer_tokens']==1
    assert torch.all(logits.grad[0,2]==0) and torch.any(logits.grad[0,0]!=0)
