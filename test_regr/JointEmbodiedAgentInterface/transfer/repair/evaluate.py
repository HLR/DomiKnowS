"""Fixed confirmation inference; private goals are opened only after generation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from ..protocol import read,freeze,sha,digest,episodes,public_input
from ..matched_baseline.protocol import common_prompt,raw_format_valid
from ..matched_baseline.evaluate import input_record
from ...generalization.shared_constraints import rule_trace
from ...generalization.executor_policy import (CAPABILITIES, EXECUTOR, POLICIES, bind_plan_policy,
    compiled_rule_names, pick_place_profile)
from ...generalization.libero_protocol import abstract_score
from .common import build_runtime,fresh_planner,load_checkpoint,native_decode,graph_decode
from .protocol import verify,normalize_anonymous_prediction,anonymous_entities
from .surface import surface_decoder,SURFACE_VERSION
from .preference import VISUAL_SOURCE, FrozenVisualSource, bind_visual_evidence
from .input_protocol import (INPUT_PROTOCOLS, IMAGE_INSTRUCTION, generation_context,
    constraint_context, constraint_public, contract_record, validate_resume_contract)

SYSTEMS={'base':'native','native-trained':'native','joint-initialization':'joint','joint-trained':'joint'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--arm',choices=SYSTEMS,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',choices=POLICIES,required=True)
    p.add_argument('--input-protocol',choices=INPUT_PROTOCOLS,default=IMAGE_INSTRUCTION)
    p.add_argument('--preference', choices=['none', VISUAL_SOURCE], default='none')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();c=read(a.config);verify(c)
    if a.preference != 'none' and (SYSTEMS[a.arm] != 'joint' or a.policy != EXECUTOR
                                  or a.input_protocol != IMAGE_INSTRUCTION):
        p.error('visual-source requires Joint labels, strict image/instruction inputs, and shared-verified-executor')
    if a.seed!=42:raise ValueError('seed differs')
    if c.get('evaluation_input_protocol',a.input_protocol)!=a.input_protocol:
        raise ValueError('input protocol differs from frozen evaluation campaign')
    gate=read(Path(c['output'])/'frozen_selection.json')
    spec=gate['systems'][a.arm]
    if str(a.checkpoint)!=spec['path'] or sha(a.checkpoint)!=spec['sha256']:raise ValueError('source-selected checkpoint differs')
    frozen=Path(c['output'])/'confirmation/manifest.json'
    if sha(a.manifest)!=sha(frozen):raise ValueError('confirmation manifest differs')
    audit=read(c['confirmation_frozen_input_audit'])
    if not audit['passed'] or audit['confirmation_manifest_sha256']!=sha(a.manifest):
        raise ValueError('confirmation settled-scene / decoded-image audit failed or changed')
    output=a.output;output.mkdir(parents=True,exist_ok=True)
    if (output/'settings.json').exists():
        validate_resume_contract(read(output/'settings.json'),a.input_protocol)
    _,(runtime,_,_,_,_)=build_runtime(c,output/'runtime')
    arm=SYSTEMS[a.arm];planner=fresh_planner(c,runtime,arm)
    payload=load_checkpoint(a.checkpoint,planner,arm);planner.eval();vocab=runtime.vlabench_vocabulary
    executor = pick_place_profile() if a.policy in (CAPABILITIES, EXECUTOR) else None
    rules = compiled_rule_names(a.policy, executor, vocab)
    unsupported_labels = ({'skill:' + skill for skill in vocab.skills
                           if skill not in {action.name for action in executor.actions}} if executor else set())
    settings={'config_sha256':sha(a.config),'source_manifest_sha256':sha(c['source_manifest']),'checkpoint_sha256':sha(a.checkpoint),'arm':a.arm,
        'manifest_sha256':sha(a.manifest),'source_selection_sha256':sha(Path(c['output'])/'frozen_selection.json'),
        'policy':a.policy,'surface':SURFACE_VERSION,
        'shared_rules':compiled_rule_names('shared-verified'),
        'native_budget':768,'graph_budget':61,'horizon':1000,'seed':42,'precision':'NF4/BF16',
        **contract_record(a.input_protocol),'representation':arm,
        'quant_matmul_policy':planner.quant_matmul_policy,
        'source_qualified':spec['qualified']}
    if executor:
        settings.update(executor_profile=json.loads(json.dumps(asdict(executor))), compiled_rules=rules)
    visual = FrozenVisualSource(c['vision_model'], planner.device) if a.preference == VISUAL_SOURCE else None
    if visual:
        settings['preference'] = visual.settings
    freeze(output/'settings.json',settings)
    if (output/'report.json').exists():return
    if (output/'predictions.jsonl').exists():raise ValueError('partial condition preserved; explicit new output required')
    from test_regr.VLABenchAgentInterface.native_planning import byte_token_map,NativeConstraintProcessor
    token_bytes=byte_token_map(planner.processor.tokenizer) if arm=='native' else None
    records=[];decoder_cache={}
    with common_prompt(planner),(output/'predictions.jsonl').open('x') as stream,(output/'decisions.jsonl').open('x') as decisions:
        for i,e in enumerate(episodes(read(a.manifest))):
            identity=e['episode_id'];public=public_input(e)
            bound=generation_context(public,vocab,a.input_protocol);core=bound['_matched_core']
            dfa_bound=constraint_context(public) if a.policy!='none' else None
            if dfa_bound is not None:
                freeze(output/'constraint_inputs'/f'{i:04d}.json',{'episode_id':identity,
                    'compiled_rules':rules,'bindings':dfa_bound,
                    'goals_or_references_used':False,'model_receives_bindings':a.input_protocol!=IMAGE_INSTRUCTION})
            raw_inputs,_=planner._prepare_inputs_and_boundaries('vlabench',bound,())
            rendered=input_record(planner.processor,core,public,None,raw_inputs)
            freeze(output/'inputs'/f'{i:04d}.json',{'episode_id':identity,**rendered})
            preference = None
            if visual:
                evidence = visual.observe(public)
                preference = bind_visual_evidence(evidence, len(public['entities']))
                freeze(output/'preference_inputs'/f'{i:04d}.json', {
                    'episode_id': identity, 'image_hashes': rendered['image_hashes'],
                    'model_input_hash': digest(rendered), 'observation': evidence,
                    'bound_evidence': preference.evidence,
                })
            prediction=[];error=None;labels=[];raw='';graph_decisions=[];policy=None
            try:
                if arm=='native':
                    if a.policy!='none':
                        observed=constraint_public(public)
                        slot=(tuple(observed['entities']),tuple(observed['movable']))
                        if slot not in decoder_cache:
                            decoder_cache[slot]=surface_decoder(token_bytes,planner.processor.tokenizer.eos_token_id,
                                observed,vocab,12,policy_name=a.policy,executor_profile=executor,
                                constraint_bindings=dfa_bound)
                        policy=decoder_cache[slot]
                    length=raw_inputs['input_ids'].shape[1]
                    mask=NativeConstraintProcessor(policy,length,768) if policy else None
                    class Observer:
                        def __call__(self,ids,scores):
                            raw_top=int(scores[0].argmax());changed=mask(ids,scores) if mask else scores
                            allowed=torch.isfinite(changed[0]);probabilities=torch.softmax(scores[0].float(),0)
                            top=scores[0].float().topk(2).values
                            decisions.write(json.dumps({'episode_id':identity,'step':ids.shape[1]-length,
                                'raw_top':raw_top,'selected_top':int(changed[0].argmax()),
                                'raw_top_rejected':not bool(allowed[raw_top]),
                                'allowed':allowed.nonzero().flatten().tolist(),
                                'allowed_mass':float(probabilities[allowed].sum()),'raw_logit_margin':float(top[0]-top[1])})+'\n')
                            return changed
                    prediction,labels,raw=native_decode(planner,bound,observer=Observer())
                    if policy and not policy.accepts(labels):raise ValueError('nonaccepting constrained JSON')
                else:
                    policy=bind_plan_policy(a.policy,vocab,12,dfa_bound,executor)
                    if preference is not None:
                        policy.generation_preference = preference
                    def observe(row):
                        rejection = row.get('rejection')
                        if executor and rejection and rejection['kind'] == 'grammar':
                            if row['raw_argmax'] in unsupported_labels:
                                rejection['violated_rules'] = [{'rule': 'executor_unsupported_skill', 'phase': 'skill'}]
                        graph_decisions.append(row)
                        decisions.write(json.dumps({'episode_id':identity,**row},default=str)+'\n')
                    planner.decision_observer=observe
                    prediction,labels,raw=graph_decode(planner,runtime,bound,policy)
            except (ValueError,RuntimeError,KeyError) as exc:
                if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                error={'type':type(exc).__name__,'message':str(exc)}
                labels=getattr(exc,'labels',labels);raw=getattr(exc,'raw_text',raw)
                if arm=='joint' and not labels:
                    labels=[vocab.label_for_token(r['chosen']) for r in graph_decisions]
            original=prediction;changes=[]
            if public['entities']!=anonymous_entities(len(public['entities'])):raise ValueError('public anonymous IDs changed')
            try:prediction,changes=normalize_anonymous_prediction(original,len(public['entities']),vocab.argument_keys)
            except ValueError as exc:
                prediction=[];error=error or {'type':type(exc).__name__,'message':str(exc)}
            if sha(e['scoring_private'])!=e['files'][e['scoring_private']]:raise ValueError('scoring reference changed')
            private=read(e['scoring_private']);score=abstract_score(prediction,private)
            goal_sources={private['entities'].index(g[1]) for g in private['goals']}
            goal_destinations={private['entities'].index(g[2]) for g in private['goals']}
            selected_sources={op['params']['target_entity_name'] for op in prediction if type(op.get('params',{}).get('target_entity_name')) is int}
            selected_destinations={op['params']['target_container_name'] for op in prediction if type(op.get('params',{}).get('target_container_name')) is int}
            grounding={'source_correct':selected_sources==goal_sources,'destination_correct':selected_destinations==goal_destinations}
            trace={}
            if a.policy!='none' and prediction:
                from test_regr.VLABenchAgentInterface.plan_codec import plan_to_tokens
                try:
                    canonical=[vocab.label_for_token(t) for t in plan_to_tokens(prediction,public['entities'],world=runtime.world.vlabench)]
                    canonical_policy = policy if arm == 'joint' else bind_plan_policy(a.policy,vocab,12,dfa_bound,executor)
                    trace=rule_trace(canonical_policy,canonical)
                except (ValueError,KeyError) as exc:trace={'unavailable':str(exc)}
            decisions.write(json.dumps({'episode_id':identity,'compiled_rule_trace':trace})+'\n')
            record={'episode_id':identity,'prediction':prediction,'raw_prediction':original,
                'normalizations':changes,'raw_format_valid':raw_format_valid(original,vocab,len(public['entities'])) and error is None,
                'generation_error':error,'labels':labels,'raw_text':raw,'input_hash':digest(rendered),
                'core_hash':rendered['core_sha256'],'image_hashes':rendered['image_hashes'],
                'grounding':grounding,'compiled_rule_trace':trace,
                'truncated':bool(error and ('budget' in error['message'] or 'terminate' in error['message'])),**score}
            records.append(record);stream.write(json.dumps(record)+'\n');stream.flush();decisions.flush()
            print(f'[repair-eval] {a.arm} {a.policy} {i+1}/100 error={error}',flush=True)
    planner.decision_observer=None
    freeze(output/'report.json',{'examples':100,**contract_record(a.input_protocol),
        'abstract_percent':sum(r['goal_success'] and not r['generation_error'] for r in records),
        'valid_percent':sum(r['valid_plan'] and not r['generation_error'] for r in records),
        'raw_format_percent':sum(r['raw_format_valid'] for r in records),
        'normalized_episodes':sum(bool(r['normalizations']) for r in records),
        'generation_errors':sum(bool(r['generation_error']) for r in records),
        'core_hashes':{r['episode_id']:r['core_hash'] for r in records},
        'input_hashes':{r['episode_id']:r['input_hash'] for r in records}})


if __name__=='__main__':main()
