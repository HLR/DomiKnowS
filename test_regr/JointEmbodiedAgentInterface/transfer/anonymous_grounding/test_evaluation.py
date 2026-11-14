from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
import torch
from .eval_observations import libero_markers,observed_markers


def test_rlbench_native_tag_style_preserves_multidigit_ids(tmp_path):
    raw=Image.fromarray(np.full((384,384,3),100,dtype=np.uint8));numbered=raw.copy();draw=ImageDraw.Draw(numbered)
    for pointer,x,y in [(0,30,40),(10,150,100)]:
        draw.rectangle((x-2,y-2,x+16,y+14),fill='white',outline='black');draw.text((x,y),str(pointer),fill='black')
    r=tmp_path/'raw.png';n=tmp_path/'numbered.png';raw.save(r);numbered.save(n)
    proof=observed_markers(r,n,12,'rlbench-v33')
    assert proof['visible_ids']==[0,10]
    assert all(m['view']==0 for m in proof['markers'])


def test_external_grounding_uses_only_unique_public_stamps(tmp_path):
    raw=Image.fromarray(np.full((256,256,3),100,dtype=np.uint8));numbered=raw.copy();draw=ImageDraw.Draw(numbered)
    for pointer,x,y in [(0,20,30),(1,80,100)]:
        draw.rectangle((x-3,y-3,x+14,y+13),fill='white',outline='black');draw.text((x,y),str(pointer),fill='black')
    r=tmp_path/'raw.png';n=tmp_path/'numbered.png';raw.save(r);numbered.save(n)
    proof=libero_markers(r,n,3)
    assert proof['visible_ids']==[0,1] # unseen object 2 is never filled in from a goal
    draw.rectangle((140-3,140-3,140+14,140+13),fill='white',outline='black');draw.text((140,140),'1',fill='black');numbered.save(n)
    assert libero_markers(r,n,3)['visible_ids']==[0] # ambiguous duplicate cannot assign an object


def test_generation_binds_original_observations_and_uses_predicted_roles(monkeypatch):
    from . import evaluate
    from ... import eval_generalization
    from ...generalization import shared_constraints
    from test_regr.VLABenchAgentInterface import plan_codec
    import contextlib
    original={'instruction':'pick apple','entity_table':('apple','bowl'),'images':()}
    effective={**original,'entity_table':('numbered object 0','numbered object 1'),
               'instruction':'pick apple; predicted source 1'}
    monkeypatch.setattr(evaluate,'predicted_context',lambda *a:(effective,{'source':1,'destination':0},None))
    monkeypatch.setattr(eval_generalization,'input_record',lambda p,c:{'instruction':c['instruction'],'entities':c['entity_table']})
    monkeypatch.setattr(plan_codec,'labels_to_plan',lambda *a,**k:[{'name':'pick','params':{'target_entity_name':1}}])
    monkeypatch.setattr(shared_constraints,'rule_trace',lambda *a:{'triggered_rejections':{}})
    captured=[]
    def sample(domain,context,dfa,**kwargs):
        assert context==effective
        planner._prepare_inputs_and_boundaries(domain,context)
        return [0],None
    planner=SimpleNamespace(device='cpu',sample_labels=sample,_prepare_inputs_and_boundaries=lambda *a:None)
    runtime=SimpleNamespace(vlabench_vocabulary=object(),max_vlabench_operations=12,
        world=SimpleNamespace(vlabench=object()),domain_scope=lambda *a:contextlib.nullcontext(),
        dfa_for=lambda domain,context:captured.append(context) or object())
    feature={'features':torch.zeros(2,1548),'visible':torch.ones(2,dtype=torch.bool)}
    prediction,error,_,_,_,roles=evaluate.generation(planner,object(),runtime,original,feature,'full-metadata')
    assert captured==[original] and error is None
    assert prediction[0]['params']['target_entity_name']==roles['source']==1
