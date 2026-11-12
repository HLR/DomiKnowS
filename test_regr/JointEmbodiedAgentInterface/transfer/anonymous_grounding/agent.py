"""Required loader and predicted-only inference for grounding-extended snapshots."""
from pathlib import Path
import torch
from .grounder import RoleGrounder,instruction_query,predict_roles,condition
from ..protocol import sha


def load_head(payload,checkpoint,device):
    spec=payload['metrics']['grounding']
    if Path(spec['head_file']).name!=spec['head_file']:raise ValueError('grounding head must accompany its checkpoint')
    path=Path(checkpoint).parent/spec['head_file']
    if sha(path)!=spec['sha256']:raise ValueError('grounding sidecar hash differs')
    saved=torch.load(path,map_location=device,weights_only=True)
    model=RoleGrounder(**saved['dimensions']).to(device)
    model.load_state_dict(saved['state_dict'],strict=True)
    model.visual_encoder_sha256=spec['visual_encoder_sha256']
    return model


def load_snapshot(checkpoint,*,runtime,planner,controller,device):
    from ...checkpoint import load_joint_checkpoint
    planner._grounding_extension_active=True
    payload=load_joint_checkpoint(checkpoint,runtime=runtime,planner=planner,controller=controller,
        map_location=device,allow_knowledge_ablation=True)
    head=load_head(payload,checkpoint,device).eval()
    from ..train import tensor_digest
    if tensor_digest(controller.image_encoder)!=head.visual_encoder_sha256:
        raise ValueError('snapshot visual encoder differs from grounding feature provenance')
    return head,payload


def predicted_context(planner,head,context,features,visible):
    if any(key in context for key in ('reference','reference_plan','operation_sequence','target_plan_labels',
            'native_success','hidden_goal','goal','goals','obj_of_interest','scoring_private','reference_labels')):
        raise ValueError('generation context contains supervision or evaluator state')
    logits=head(instruction_query(planner,context['instruction']),features,visible)
    predictions=predict_roles(logits,visible)
    return condition(context,predictions),predictions,logits


def generate(planner,head,runtime,context,features,visible,policy='none'):
    from ...models import FreeDecoding
    from ...generalization.shared_constraints import bind_shared,rule_trace
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
    planner.eval();head.eval()
    with torch.no_grad():
        effective,predictions,_=predicted_context(planner,head,context,features,visible)
        planner.free_decoding=policy=='none'
        if policy=='none':dfa=FreeDecoding(runtime.vlabench_vocabulary.label_count,runtime.vlabench_vocabulary.eos_label)
        elif policy=='shared-verified':dfa=bind_shared(runtime.vlabench_vocabulary,runtime.max_vlabench_operations,context)
        elif policy=='full-metadata':dfa=runtime.dfa_for('vlabench',context)
        else:raise ValueError('unsupported constraint policy')
        with runtime.domain_scope('vlabench'):
            labels,_=planner.sample_labels('vlabench',effective,dfa,max_steps=runtime.max_vlabench_operations*5+1,deterministic=True)
        plan=labels_to_plan(labels,runtime.vlabench_vocabulary,world=runtime.world.vlabench)
    return {'prediction':plan,'predicted_grounding':predictions,'effective_context':effective,
            'rule_trace':rule_trace(dfa,labels) if policy!='none' else {}}
