"""Shared anonymous observations and explicitly typed, checkpoint-owned weights."""
from pathlib import Path
import random
import json
import torch
import numpy as np
from ..protocol import read,freeze,sha,digest
from ..matched_baseline.protocol import prompt_core,common_prompt,normalize_ids
from .protocol import VERSION,checkpoint_kind,anonymous_entities


class DecodingFailure(ValueError):
    def __init__(self,message,labels=(),raw_text=''):
        super().__init__(message);self.labels=list(labels);self.raw_text=raw_text


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)


def source_context(planner,row):
    from ...utils.planner_context import planner_context
    context=planner_context('vlabench',row['item'])
    # The common prefix deliberately excludes task IDs, oracle targets, role
    # predictions and gold-derived facts. Canonical bindings remain loss-only.
    aliases=anonymous_entities(len(context['entity_table']))
    public={'entities':aliases,'movable':aliases,
            'instruction':context['instruction'],'initial_relations':[], 'images':list(context['images'])}
    context['_matched_core']=prompt_core(public,planner.vocabulary('vlabench'))
    context['_matched_roles']=None
    context['_public_entity_aliases']=aliases
    return context


def build_runtime(c,output,device='cuda:0',*,training_args=None):
    from ... import main as joint
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    args=training_args
    if args is None:
        args=joint.build_parser().parse_args(read(c['original_configuration'])['arguments'])
        args.output=str(output);args.initialize_from=None;args.resume=None;args.device=device
        args.save_initialization=False;args.stage1_eai_warmup_steps=0
        args.ce_only=True;args.ce_dfa_mass_only=False;args.ce_dfa_mass_weight=0.
        args.no_dfa_official=False;args.allow_internal_evaluation_training=True
        args.scene_verified_training_references=True
    apply_research_arguments(args);apply_knowledge_argument(args)
    Path(output).mkdir(parents=True,exist_ok=True)
    def audit(runtime,prepared,out):
        # The frozen reference hashes are checked below against the qualified
        # correction archive. Verify its completed audit, without rerunning an
        # oracle DFA or relying on process-dependent pickle/set ordering.
        rows=[(split,item) for split,items in prepared.items() for item in items]
        folder=Path(c['reference_audit'])
        if read(folder/'complete.json')['count']!=len(rows):raise ValueError('reference audit coverage changed')
        for i,row in enumerate(rows):
            proof=read(folder/f'{i:05d}.json')
            if proof['result_sha256']!=digest(proof['result']) or not proof['result']['passed']:
                raise ValueError('reference audit certificate failed: '+str(i))
        freeze(Path(out)/'training_constraint_audit.json',{'examples':len(rows),'conflicts':[],
            'reused_proof_sha256':sha(folder/'complete.json'),'generation_constraints_from_targets':False})
    values=joint._prepare(args,torch.device(device),path_auditor=audit)
    runtime,eai_train,eai_valid,splits,_=values
    if (len(eai_train),len(eai_valid))!=(350,88):raise ValueError('canonical EAI identities changed')
    from ...utils.planner_context import planner_context
    archived=read(c['canonical_source_inputs'])['inputs']
    for split,rows in [('train',eai_train),('validation',eai_valid)]:
        expected=[r['context'] for r in archived if r['domain']=='eai' and r['split']==split]
        actual=[json.loads(json.dumps(planner_context('eai',item),default=str)) for item in rows]
        identity=lambda value:(value.get('instruction'),value.get('goal'))
        if [identity(v) for v in actual]!=[identity(v) for v in expected]:
            raise ValueError('canonical archived EAI instructions/goals or identity order changed')
    if {k:[e.episode_id for e in v] for k,v in splits.items()}!=read(c['split_manifest'])['splits']:
        raise ValueError('historical source split changed')
    for name in ('training_reference_policy.json','training_reference_hashes.json'):
        if sha(Path(output)/name)!=sha(Path(c['reference_root'])/name):raise ValueError('corrected reference policy changed')
    freeze(Path(output)/'eai_identities.json',{split:[digest({k:(v.tolist() if torch.is_tensor(v) else v) for k,v in item.items()
        if k in ('instruction','instruction_text','causal_prompt_text','tl_goal','target_action_labels')})
        for item in rows] for split,rows in [('train',eai_train),('validation',eai_valid)]})
    return args,values


def fresh_planner(c,runtime,arm):
    from ...models import JointQwenVLPlanner
    from .precision import default_policy
    quant_policy=default_policy()
    seed()
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,model_id=c['model'],load_in_4bit=True,
        decoder_hidden_size=c['decoder_hidden_size'])
    if arm=='native':
        for p in planner.label_heads['vlabench'].parameters():p.requires_grad_(False)
    # Domain scopes may restore baseline flags; update that baseline too.
    planner._base_parameter_requires_grad={p:p.requires_grad for p in planner.parameters()}
    planner.free_decoding=True
    planner.quant_matmul_policy=quant_policy
    return planner


def save_checkpoint(path,planner,runtime,arm,epoch,settings_sha,optimizer=None,scheduler=None,*,provenance=None):
    path=Path(path)
    if path.exists():raise ValueError('preserve existing checkpoint')
    names={n for n,p in planner.named_parameters() if p.requires_grad or n.startswith('label_heads.')}
    state={n:v.detach().cpu() for n,v in planner.state_dict().items() if n in names}
    payload={'format':VERSION,'arm':arm,'epoch':epoch,'initialization_before_updates':epoch==0,
        'planner':state,'settings_sha256':settings_sha,
        'vocabularies':{d:list(planner.vocabulary(d).tokens) for d in ('eai','vlabench')},
        'optimizer':optimizer.state_dict() if optimizer else None,'scheduler':scheduler.state_dict() if scheduler else None,
        'grounding_residual':False,'primary_role_hints':False,'fresh_pretrained_initialization':provenance is None}
    if provenance is not None:payload['provenance']=provenance
    payload['quant_matmul_policy']=planner.quant_matmul_policy
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
    freeze(path.with_suffix('.json'),{k:v for k,v in payload.items() if k not in ('planner','optimizer','scheduler')})


def load_checkpoint(path,planner,arm,settings_sha=None):
    payload=torch.load(path,map_location='cpu',weights_only=True);checkpoint_kind(payload,arm)
    if payload.get('quant_matmul_policy')!=planner.quant_matmul_policy:raise ValueError('checkpoint quantized compute policy differs')
    if settings_sha and payload['settings_sha256']!=settings_sha:raise ValueError('resume settings changed')
    if payload['vocabularies']!={d:list(planner.vocabulary(d).tokens) for d in ('eai','vlabench')}:
        raise ValueError('checkpoint vocabulary differs')
    expected={n for n,p in planner.named_parameters() if p.requires_grad or n.startswith('label_heads.')}
    if set(payload['planner'])!=expected:raise ValueError('checkpoint components differ')
    result=planner.load_state_dict(payload['planner'],strict=False)
    if result.unexpected_keys:raise ValueError('unexpected checkpoint components')
    return payload


def native_decode(planner,context,policy=None,budget=768,observer=None):
    from transformers import LogitsProcessorList
    from test_regr.VLABenchAgentInterface.native_planning import NativeConstraintProcessor
    from ..native_baseline import parse_plan
    inputs,_=planner._prepare_inputs_and_boundaries('vlabench',context,())
    inputs=planner._inputs_to_model_device(inputs);length=inputs['input_ids'].shape[1]
    processors=[]
    if policy:processors.append(NativeConstraintProcessor(policy,length,budget))
    if observer:processors.insert(0,observer)
    with torch.inference_mode():
        ids=planner.model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=budget,
            logits_processor=LogitsProcessorList(processors),eos_token_id=planner.processor.tokenizer.eos_token_id)[0,length:].tolist()
    raw=planner.processor.tokenizer.decode(ids,skip_special_tokens=True)
    if not ids or ids[-1]!=planner.processor.tokenizer.eos_token_id:raise DecodingFailure('native token budget exhausted',ids,raw)
    if policy and not policy.accepts(ids):raise DecodingFailure('nonaccepting native termination',ids,raw)
    try:plan=parse_plan(raw)
    except ValueError as exc:raise DecodingFailure(str(exc),ids,raw) from exc
    return plan,ids,raw


def graph_decode(planner,runtime,context,policy=None,budget=61):
    from ...models import FreeDecoding
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan,plan_to_tokens
    vocab=runtime.vlabench_vocabulary
    policy=policy or FreeDecoding(vocab.label_count,vocab.eos_label)
    planner.free_decoding=type(policy).__name__=='FreeDecoding'
    from ..anonymous_grounding.planner_binding.verification_protocol import exact_policy_search
    with torch.inference_mode(),runtime.domain_scope('vlabench'),exact_policy_search(policy,[]):
        labels,_=planner.sample_labels('vlabench',context,policy,max_steps=budget,deterministic=True)
    if not labels or labels[-1]!=vocab.eos_label:raise DecodingFailure('graph token budget exhausted',labels)
    plan=labels_to_plan(labels,vocab,world=runtime.world.vlabench)
    try:canonical=plan_to_tokens(plan,context['entity_table'],world=runtime.world.vlabench)
    except ValueError as exc:raise DecodingFailure(str(exc),labels) from exc
    if [vocab.token_for_label(int(label)) for label in labels]!=canonical:
        raise DecodingFailure('noncanonical graph-label syntax; parsing would discard generated labels',labels)
    return plan,labels,''
