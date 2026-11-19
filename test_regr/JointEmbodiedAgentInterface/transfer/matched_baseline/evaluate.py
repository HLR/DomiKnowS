"""One frozen model/constraint condition, with identical public input construction."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

from ..protocol import read,freeze,sha,digest,episodes,public_input,context
from .protocol import MODELS,POLICIES,VERSION,prompt_core,role_prompt,chat,common_prompt,normalize_ids,raw_format_valid


def input_record(processor,core,public,roles,inputs):
    import torch
    tensors={}
    for name,value in inputs.items():
        if torch.is_tensor(value):
            import hashlib
            tensor=value.detach().cpu().contiguous()
            tensors[name]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                          'sha256':hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
    return {'core_sha256':digest(core),'rendered_prompt':chat(processor,role_prompt(core,roles),len(public['images'])),
            'image_hashes':{p:sha(p) for p in public['images']},'tensors':tensors}


def load_joint(c,key,device):
    import torch
    from ... import eval_planning
    from ...main import _controller
    from ...models import JointQwenVLPlanner
    from ..anonymous_grounding.agent import load_snapshot
    from ..anonymous_grounding.planner_binding.agent import load_binding
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    spec=c['snapshots'][key]
    args=eval_planning.build_eval_parser().parse_args(['--checkpoint',spec['path'],'--planning-dir',c['planning_dir'],
        '--planner-model',c['model'],'--vision-model',c['vision_model'],'--image-size','384','384',
        '--knowledge','metadata','--observation-mode','metadata-assisted','--device',device,'--load-in-4bit'])
    args.vlabench_planning_dir=c['planning_dir'];args.split='test'
    apply_research_arguments(args);apply_knowledge_argument(args)
    runtime,_=eval_planning._prepare_eval(args,torch.device(device))
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,model_id=c['model'],load_in_4bit=True,
        decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller=_controller(args,torch.device(device))
    loader=load_snapshot if key=='A' else load_binding
    head,payload=loader(spec['path'],runtime=runtime,planner=planner,controller=controller,device=device)
    if sha(Path(spec['path']).parent/payload['metrics']['grounding']['head_file'])!=spec['head_sha256']:
        raise ValueError('checkpoint-specific grounding head differs')
    if key=='initialization' and not payload['metrics'].get('initialization_before_updates'):
        raise ValueError('control is not pre-continuation initialization')
    planner.eval();head.eval();_ANONYMOUS_OBJECTS.set(True)
    details={'stage':payload['stage'],'epoch':payload['epoch'],'own_head_loaded':True,
             'visual_encoder_sha256':head.visual_encoder_sha256}
    if hasattr(head,'gain'):details['residual_gain']=float((head.max_gain*torch.tanh(head.gain)).detach())
    del controller,payload
    return planner,head,runtime,details


def verify(c):
    for path,expected in c['input_hashes'].items():
        if sha(path)!=expected:raise ValueError('frozen input differs: '+path)
    for name,expected in read(c['source_manifest'])['files'].items():
        if sha(Path(c['source'])/name)!=expected:raise ValueError('frozen source differs: '+name)


def main():
    import torch
    from PIL import Image
    from transformers import LogitsProcessorList
    from test_regr.VLABenchAgentInterface.native_planning import FrozenNativePlanner,NativeConstraintProcessor
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary,labels_to_plan
    from ..native_baseline import compiled_decoder,parse_plan
    from ...models import FreeDecoding
    from ...generalization.shared_constraints import bind_shared,SHARED_RULES
    from ...generalization.libero_protocol import abstract_score
    from ..anonymous_grounding.agent import predicted_context
    from ..anonymous_grounding.grounder import condition
    from ..anonymous_grounding.planner_binding.binding import operation_bindings
    from ..anonymous_grounding.planner_binding.verification_protocol import exact_policy_search
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--model-key',choices=MODELS,required=True);p.add_argument('--policy',choices=POLICIES,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();c=read(a.config);verify(c);key=a.model_key;root=Path(c['output'])/'evaluation'/key/a.policy
    root.mkdir(parents=True,exist_ok=True);random.seed(42);torch.manual_seed(42)
    selected=episodes(read(c['manifest']));features={r['episode_id']:r for r in read(c['features'])['episodes']}
    prior=read(c['priors']);priors={r['episode_id']:r for r in prior['episodes']}
    if key=='base':
        planner=FrozenNativePlanner.load(c['model'],load_in_4bit=True);head=runtime=None
        # Match Joint's non-quantized parameter casts as well as NF4 compute.
        # This prepares precision only; it creates no adapter or learned head.
        from test_regr.VLABenchAgentInterface.models import prepare_kbit_model
        planner.model=prepare_kbit_model(planner.model,gradient_checkpointing=True).eval()
        planner.model.requires_grad_(False)
        vocabulary=PlanVocabulary.default(max_entities=c['max_entities']);model_details={'native_lm_head':True}
    else:
        planner,head,runtime,model_details=load_joint(c,key,'cuda:0');vocabulary=runtime.vlabench_vocabulary
        if head.visual_encoder_sha256!=read(c['features'])['encoder_sha256']:raise ValueError('visual encoder differs')
    if vocabulary.max_entities!=c['max_entities']:raise ValueError('vocabulary entity limit differs')
    if a.preflight:
        freeze(Path(c['output'])/('preflight_'+key+'.json'),{'passed':True,'model':key,
            'vocabulary_checksum':vocabulary.checksum,'vocabulary_tokens':list(vocabulary.tokens),
            'model_details':model_details,'config_sha256':sha(a.config),'scored_episodes':0})
        print('[matched-libero] preflight passed '+key,flush=True);return
    settings={'config_sha256':sha(a.config),'model':key,'policy':a.policy,'protocol':VERSION,
        'dtype':'4-bit NF4 / BF16 compute','decoding':'greedy','seed':42,'model_details':model_details,
        'checkpoint':c['snapshots'].get(key),'shared_rules':sorted(SHARED_RULES),'vocabulary_checksum':vocabulary.checksum,
        'native_token_budget':768,'graph_label_budget':61,'max_operations':12,
        'normalization':'exact public entity alias or canonical decimal ID only; applies to every model',
        'method_hints':'own learned primary roles' if key=='A' else 'frozen paired SigLIP roles' if key!='base' else 'none'}
    freeze(root/'settings.json',settings)
    if (root/'report.json').exists():return
    if (root/'predictions.jsonl').exists():raise ValueError('incomplete condition preserved; no implicit resume')
    decoders={};rows=[]
    with (root/'predictions.jsonl').open('x') as out,(root/'decisions.jsonl').open('x') as decisions:
        for i,e in enumerate(selected):
            identity=e['episode_id'];public=public_input(e);core=prompt_core(public,vocabulary)
            roles=None;error=None;prediction=[];raw_text='';labels=[];record=None
            if key=='base':
                images=[Image.open(f).convert('RGB') for f in public['images']]
                inputs=planner.processor(text=[chat(planner.processor,core,len(images))],images=images,padding=True,return_tensors='pt')
                record=input_record(planner.processor,core,public,None,inputs)
                inputs=inputs.to(next(planner.model.parameters()).device);length=inputs.input_ids.shape[1]
                decoder=None;processor=None
                if a.policy!='none':
                    slot=(tuple(public['entities']),tuple(public['movable']))
                    if slot not in decoders:decoders[slot]=compiled_decoder(planner.token_bytes,planner.tokenizer.eos_token_id,public,vocabulary,12)
                    decoder=decoders[slot];processor=NativeConstraintProcessor(decoder,length,768)
                class Observer:
                    def __call__(self,ids,scores):
                        raw=int(scores[0].argmax());changed=processor(ids,scores) if processor else scores
                        decisions.write(json.dumps({'episode_id':identity,'step':ids.shape[1]-length,'raw_top':raw,
                            'selected_top':int(changed[0].argmax()),'raw_top_rejected':not bool(torch.isfinite(changed[0,raw])),
                            'allowed_count':int(torch.isfinite(changed[0]).sum())})+'\n');return changed
                try:
                    with torch.inference_mode():
                        labels=planner.model.generate(**inputs,max_new_tokens=768,do_sample=False,use_cache=True,
                            logits_processor=LogitsProcessorList([Observer()]),eos_token_id=planner.tokenizer.eos_token_id)[0,length:].tolist()
                    raw_text=planner.tokenizer.decode(labels,skip_special_tokens=True)
                    if decoder is not None and not decoder.accepts(labels):raise ValueError('native decoding did not terminate acceptingly')
                    prediction=parse_plan(raw_text)
                except (ValueError,RuntimeError,KeyError) as exc:
                    if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                    error={'type':type(exc).__name__,'message':str(exc)}
                finally:
                    for image in images:image.close()
            else:
                bound=context(public,'libero');bound['_matched_core']=core
                f=features[identity]
                if sha(f['path'])!=f['sha256']:raise ValueError('visual feature changed')
                values=torch.load(f['path'],map_location=planner.device,weights_only=True)
                visual,visible=values['features'].float(),values['visible']
                with torch.no_grad(),common_prompt(planner):
                    if key=='A':effective,roles,_=predicted_context(planner,head,bound,visual,visible)
                    else:roles=priors[identity]['predictions'];effective=condition(bound,roles)
                    effective['_matched_roles']=roles
                    raw_inputs,_=planner._prepare_inputs_and_boundaries('vlabench',effective,())
                    record=input_record(planner.processor,core,public,roles,raw_inputs)
                    dfa=(FreeDecoding(vocabulary.label_count,vocabulary.eos_label) if a.policy=='none' else bind_shared(vocabulary,12,bound))
                    planner.free_decoding=a.policy=='none'
                    planner.decision_observer=lambda row:decisions.write(json.dumps({'episode_id':identity,**row},default=str)+'\n')
                    binding=operation_bindings(planner,head,visual,visible) if key!='A' else nullcontext()
                    try:
                        with exact_policy_search(dfa,[]),binding,runtime.domain_scope('vlabench'):
                            labels,_=planner.sample_labels('vlabench',effective,dfa,max_steps=61,deterministic=True)
                        prediction=labels_to_plan(labels,vocabulary,world=runtime.world.vlabench)
                    except (ValueError,RuntimeError,KeyError) as exc:
                        if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                        error={'type':type(exc).__name__,'message':str(exc)}
                planner.decision_observer=None
            freeze(root/'inputs'/f'{i:04d}.json',{'episode_id':identity,'core':core,'predicted_roles':roles,**record})
            raw_prediction=prediction
            prediction,changes=normalize_ids(raw_prediction,public['entities'])
            # Scoring-private data is first opened after neural generation.
            if sha(e['scoring_private'])!=e['files'][e['scoring_private']]:raise ValueError('scoring input changed')
            score=abstract_score(prediction,read(e['scoring_private']))
            row={'episode_id':identity,'prediction':prediction,'raw_prediction':raw_prediction,'normalizations':changes,
                'raw_format_valid':raw_format_valid(raw_prediction,vocabulary,len(public['entities'])) and error is None,
                'generation_error':error,'raw_text':raw_text,'labels':labels,'predicted_roles':roles,
                'core_hash':record['core_sha256'],'image_hashes':record['image_hashes'],'input_hash':digest(record),**score}
            out.write(json.dumps(row)+'\n');out.flush();decisions.flush();rows.append(row)
            print(f'[matched-libero] {key} {a.policy} {i+1}/100 raw={row["raw_format_valid"]} valid={score["valid_plan"]} goal={score["goal_success"]} error={error}',flush=True)
    report={'examples':100,'model':key,'policy':a.policy,'raw_format_percent':sum(r['raw_format_valid'] for r in rows),
        'normalized_episodes':sum(bool(r['normalizations']) for r in rows),
        'valid_percent':sum(r['valid_plan'] and not r['generation_error'] for r in rows),
        'abstract_percent':sum(r['goal_success'] and not r['generation_error'] for r in rows),
        'generation_errors':sum(bool(r['generation_error']) for r in rows),
        'input_hashes':{r['episode_id']:r['input_hash'] for r in rows},'core_hashes':{r['episode_id']:r['core_hash'] for r in rows}}
    freeze(root/'report.json',report);print('[matched-libero] COMPLETE '+json.dumps({k:v for k,v in report.items() if not k.endswith('hashes')}),flush=True)


if __name__=='__main__':main()
