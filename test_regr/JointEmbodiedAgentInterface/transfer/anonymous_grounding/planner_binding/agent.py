"""Required loader and predicted-only inference for operation-binding snapshots."""
from pathlib import Path
import torch
from ...protocol import sha
from ..agent import load_head
from ..grounder import condition
from ..object_fixes.grounding import rank
from .binding import BindingHead,FORBIDDEN,operation_bindings


def primary(context,text_cache,features,visible):
    if any(k in context for k in FORBIDDEN):raise ValueError('primary grounding received private supervision')
    instruction=context.get('grounding_instruction',context['instruction']);query=text_cache[instruction]
    predicted,_=rank(query['features'].to(features.device),features[:,:1536].reshape(len(features),2,768),visible)
    for role in predicted:
        if query['phrases'][role] is None:predicted[role]=None
    return condition(context,predicted),predicted


def load_binding(checkpoint,*,runtime,planner,controller,device):
    from ....checkpoint import load_joint_checkpoint
    planner._grounding_extension_active=True
    payload=load_joint_checkpoint(checkpoint,runtime=runtime,planner=planner,controller=controller,
        map_location=device,allow_knowledge_ablation=True)
    spec=payload['metrics']['grounding'];path=Path(checkpoint).parent/spec['head_file']
    if path.name!=spec['head_file'] or sha(path)!=spec['sha256']:raise ValueError('operation head sidecar changed')
    saved=torch.load(path,map_location=device,weights_only=True)
    if saved.get('format')!='operation-binding-v1':raise ValueError('operation-binding checkpoint required')
    head=BindingHead(**saved['dimensions']).to(device);head.load_state_dict(saved['state_dict'],strict=True)
    head.visual_encoder_sha256=spec['visual_encoder_sha256']
    from ...train import tensor_digest
    if tensor_digest(controller.image_encoder)!=head.visual_encoder_sha256:raise ValueError('binding feature encoder changed')
    return head.eval(),payload


def predict(planner,head,runtime,context,features,visible,primary_roles,policy,public=None):
    from ....models import FreeDecoding
    from ....generalization.shared_constraints import bind_shared
    from ..object_fixes.contract import ExecutorContract
    from ..object_fixes.evaluate import CAPABILITIES
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
    from ....eval_generalization import input_record
    effective=condition(context,primary_roles);planner.free_decoding=policy=='none'
    if policy=='none':dfa=FreeDecoding(runtime.vlabench_vocabulary.label_count,runtime.vlabench_vocabulary.eos_label)
    elif policy=='executor':
        dfa=ExecutorContract(bind_shared(runtime.vlabench_vocabulary,runtime.max_vlabench_operations,context),
            runtime.vlabench_vocabulary,public,CAPABILITIES,runtime.max_vlabench_operations)
    else:raise ValueError('unknown binding constraint policy')
    rendered=input_record(planner,effective);original=planner._prepare_inputs_and_boundaries;verified=False
    def checked(domain,actual,active_tokens=()):
        nonlocal verified
        if not verified:
            if domain!='vlabench' or input_record(planner,actual)!=rendered:raise AssertionError('constraint binding changed actual neural inputs')
            verified=True
        return original(domain,actual,active_tokens)
    planner._prepare_inputs_and_boundaries=checked
    try:
        with torch.no_grad(),operation_bindings(planner,head,features,visible) as events,runtime.domain_scope('vlabench'):
            labels,_=planner.sample_labels('vlabench',effective,dfa,max_steps=runtime.max_vlabench_operations*5+1,deterministic=True)
    finally:planner._prepare_inputs_and_boundaries=original
    return labels_to_plan(labels,runtime.vlabench_vocabulary,world=runtime.world.vlabench),effective,[
        {'step':e['step'],'role':e['role'],'probabilities':torch.softmax(e['logits'][:,e['column']],0).cpu().tolist()}
        for e in events],dfa.trace(labels) if policy=='executor' else {}
