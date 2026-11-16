from types import SimpleNamespace
import torch
import pytest
from ..object_fixes.test_fixes import native
from .verification_analysis import shadow_trace, accepts, plan_labels
from .binding import BindingHead
from .agent import load_binding
from ...protocol import sha
from ...train import tensor_digest


def test_counterfactual_stops_at_first_unsupported_token(native):
    policy,vocabulary=native
    tokens=['skill:pick','arg:target_entity_name','obj:0','skill:pull']
    rows=[{'prefix':tokens[:i],'raw_argmax':token,'chosen':token} for i,token in enumerate(tokens)]
    rows.append({'prefix':['INVALID PREFIX MUST NOT BE REPLAYED']})
    result=shadow_trace(policy,rows,vocabulary,61)
    assert result['first_divergence']==3 and result['stopped_at_rejected_prefix']
    assert len(result['steps'])==4 and result['steps'][-1]['reason']=='executor_unsupported_skill'


def test_real_contract_keeps_wrong_object_and_repeated_relocation(native):
    policy,vocabulary=native
    plan=[{'name':'pick','params':{'target_entity_name':1}},
          {'name':'place','params':{'target_container_name':0}}]
    assert accepts(policy,plan_labels(plan,vocabulary),61)
    assert accepts(policy,plan_labels(plan+plan,vocabulary),61)


def test_loader_uses_checkpoint_specific_saved_heads_and_rejects_head_swap(tmp_path,monkeypatch):
    from .... import checkpoint as checkpoints
    controller=SimpleNamespace(image_encoder=torch.nn.Linear(2,2));encoder=tensor_digest(controller.image_encoder)
    payloads={}
    for name,gain in [('initialization',0.),('epoch1',1.)]:
        checkpoint=tmp_path/(name+'.pt');checkpoint.write_bytes(name.encode())
        head=BindingHead(4,6,8);head.gain.data.fill_(gain);path=tmp_path/(name+'_grounding.pt')
        torch.save({'format':'operation-binding-v1','dimensions':{'query_dim':4,'visual_dim':6,'hidden_dim':8,'max_gain':4.},'state_dict':head.state_dict()},path)
        payloads[str(checkpoint)]={'metrics':{'initialization_before_updates':name=='initialization',
            'grounding':{'head_file':path.name,'sha256':sha(path),'visual_encoder_sha256':encoder}}}
    monkeypatch.setattr(checkpoints,'load_joint_checkpoint',lambda path,**kwargs:payloads[str(path)])
    for name,gain in [('initialization',0.),('epoch1',1.)]:
        head,_=load_binding(tmp_path/(name+'.pt'),runtime=None,planner=SimpleNamespace(),controller=controller,device='cpu')
        assert float(head.gain.detach())==gain
    payloads[str(tmp_path/'initialization.pt')]['metrics']['grounding']['head_file']='epoch1_grounding.pt'
    with pytest.raises(ValueError,match='sidecar changed'):
        load_binding(tmp_path/'initialization.pt',runtime=None,planner=SimpleNamespace(),controller=controller,device='cpu')


def test_manifest_metadata_qualification_preserves_read_only_native_artifacts(tmp_path):
    import json
    from .verification_campaign import qualify_cohort
    native=tmp_path/'data/validation';native.mkdir(parents=True)
    bridge=tmp_path/'runtime/evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'
    bridge.parent.mkdir(parents=True);bridge.write_text('qualified bridge');executor=bridge.parent/'fixed_executor.py';executor.write_text('qualified executor')
    episodes=[{'episode_id':str(i),'split':'evaluation','state_index':30+i%5,'state':'unchanged-state','scoring_private':'unchanged-goal'} for i in range(100)]
    episodes += [{'episode_id':'cal'+str(i),'split':'calibration'} for i in range(100)]
    path=native/'manifest.json';path.write_text(json.dumps({'episodes':episodes,'fresh_states':[40,41,42,43,44]}));before=sha(path)
    (native/'calibration_reuse.json').write_text(json.dumps({'manifest_sha256':before,'success_rate':.96,'bridge_source_sha256':sha(bridge),'executor_source_sha256':sha(executor)}))
    qualify_cohort(tmp_path,'validation',{'libero_base':str(tmp_path/'runtime')})
    qualified=json.loads((tmp_path/'cohorts/validation/manifest_qualified.json').read_text())
    assert qualified['fresh_states']==[30,31,32,33,34] and qualified['episodes']==episodes
    assert sha(path)==before and not (native/'manifest_qualified.json').exists()
