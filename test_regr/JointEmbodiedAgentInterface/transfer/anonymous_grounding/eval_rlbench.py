"""Observation-only RLBench grounding and fixed-snapshot planning evaluation."""
import argparse
from pathlib import Path
from collections import Counter
import json,random
import torch
from PIL import Image
from ..protocol import read,freeze,sha,digest,episodes,public_input,context as external_context
from ..campaign import save,utc
from .eval_observations import observed_markers,features,verify_configuration
from .evaluate import generation
from .agent import load_snapshot


def prepare(config,output):
    verify_configuration(config)
    freeze(output/'settings.json',{'config_sha256':sha(config['configuration_path']),
        'manifest_sha256':sha(config['rlbench_manifest']),'source_sha256':sha(__file__),
        'input':'exact public RLBench stamps and original RGB; one camera normalized to 480; no private files read'})
    rows=[]
    for episode in episodes(read(config['rlbench_manifest'])):
        public=public_input(episode);raw,numbered=public['images'];proof=observed_markers(raw,numbered,len(public['entities']),'rlbench-v33')
        folder=output/'episodes'/digest(episode['episode_id'])[:24];path=folder/'crop_rgb.png'
        with Image.open(raw) as image:
            w,h=image.size
            if w!=h:raise ValueError('RLBench single-camera adapter requires square images')
            folder.mkdir(parents=True,exist_ok=True)
            if path.exists():raise ValueError('preserve previous preparation')
            image.convert('RGB').resize((480,480),Image.Resampling.BILINEAR).save(path)
        markers=[{**m,'box':[round(x*480/w),round(y*480/h),round(mw*480/w),round(mh*480/h)]}
                 for m in proof['markers'] for x,y,mw,mh in [m['box']]]
        rows.append({'episode_id':episode['episode_id'],'split':'evaluation','canonical_entities':public['entities'],
            'raw_image':str(path),'binding':{**proof,'markers':markers},
            'files':{raw:sha(raw),numbered:sha(numbered),str(path):sha(path)},
            'public_input':episode['generation_input'],'public_sha256':sha(episode['generation_input'])})
    if not any(r['binding']['markers'] for r in rows):raise ValueError('public marker recognition failed')
    freeze(output/'manifest.json',{'rlbench':rows,'complete':True,'references_read':False,'target_training_used':False,
        'coverage':{'episodes':len(rows),'entities':sum(len(r['canonical_entities']) for r in rows),
            'visible_entities':sum(len(r['binding']['visible_ids']) for r in rows),
            'zero_binding':sum(not r['binding']['markers'] for r in rows)},
        'adapter':'same normalized single-camera crop method as LIBERO; native RLBench annotation style'})


def evaluate(config,args):
    from ... import eval_planning
    from ...main import _controller
    from ...models import JointQwenVLPlanner
    from ...generalization.rlbench_protocol import plan_validity,validate_calibration
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    validate_calibration(config['rlbench_manifest'],config['rlbench_calibration'],config['executor_source'])
    base=eval_planning.build_eval_parser().parse_args(['--checkpoint',config['checkpoint'],'--planning-dir',config['planning_dir'],
        '--planner-model','/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct','--vision-model',config['vision_model'],
        '--image-size','384','384','--device','cuda:0','--seed','42','--load-in-4bit'])
    base.knowledge='metadata';base.observation_mode='metadata-assisted';base.split='test';base.vlabench_planning_dir=str(base.planning_dir)
    apply_research_arguments(base);apply_knowledge_argument(base);random.seed(42);torch.manual_seed(42)
    runtime,_=eval_planning._prepare_eval(base,torch.device(base.device))
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=base.planner_model,load_in_4bit=True,decoder_hidden_size=base.planner_decoder_hidden_dim)
    controller=_controller(base,torch.device(base.device))
    head,payload=load_snapshot(config['checkpoint'],runtime=runtime,planner=planner,controller=controller,device=base.device)
    planner.eval();head.eval();_ANONYMOUS_OBJECTS.set(True)
    proof=read(args.features)
    if proof['checkpoint_sha256']!=config['checkpoint_sha256'] or proof['encoder_sha256']!=head.visual_encoder_sha256:
        raise ValueError('feature checkpoint or encoder differs')
    if proof['observations_sha256']!=sha(args.observations):raise ValueError('frozen observations differ')
    cache={}
    for record in proof['episodes']:
        if record['dataset']!='rlbench' or sha(record['path'])!=record['sha256']:raise ValueError('frozen RLBench features differ')
        cache[record['episode_id']]=torch.load(record['path'],map_location='cpu',weights_only=True)
    selected=episodes(read(config['rlbench_manifest']))
    if set(cache)!={e['episode_id'] for e in selected}:raise ValueError('RLBench feature coverage differs')
    root=args.output/args.policy;root.mkdir(parents=True,exist_ok=True)
    settings={'checkpoint_sha256':config['checkpoint_sha256'],'head_sha256':config['grounding_sha256'],
        'manifest_sha256':sha(config['rlbench_manifest']),'observations_sha256':sha(args.observations),
        'features_sha256':sha(args.features),'source_manifest_sha256':config['evaluation_source_manifest_sha256'],
        'configuration_sha256':sha(args.config),'policy':args.policy,'seed':42,'decoding':'greedy',
        'protocol':'source-only anonymous grounding; original RLBench raw/numbered images; predicted roles only'}
    freeze(root/'settings.json',settings)
    if (root/'predictions.jsonl').exists():raise ValueError('preserve prior predictions')
    rows=[];inputs={};violations=Counter();triggers=Counter()
    with (root/'predictions.jsonl').open('x') as stream,(root/'decisions.jsonl').open('x') as decisions:
        for index,episode in enumerate(selected):
            public=public_input(episode);context=external_context(public,'rlbench');identity=episode['episode_id']
            planner.decision_observer=lambda row:decisions.write(json.dumps({'episode_id':identity,**row},default=str)+'\n')
            prediction,error,trace,_,rendered,roles=generation(planner,head,runtime,context,cache[identity],args.policy)
            score=plan_validity(prediction,public);inputs[identity]=digest(rendered);violations.update(v['reason'] for v in score['violations'])
            triggers.update(trace['triggered_rejections'])
            row={'episode_id':identity,'prediction':prediction,'generation_error':error,'input_hash':inputs[identity],
                'predicted_roles':roles,'rule_trace':trace,**score}
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();decisions.flush()
            freeze(root/'rendered_inputs'/f'{index:03d}.json',{'episode_id':identity,'input':rendered,'predicted_roles':roles})
            save(root/'status.json',{'stage':'planning','completed':index+1,'total':100,'updated_utc':utc()})
            if (index+1)%10==0:print(f'[grounding-RLBench] {args.policy} {index+1}/100',flush=True)
    planner.decision_observer=None
    freeze(root/'report.json',{'episodes':100,'valid_plan_rate':sum(r['valid_plan'] and not r['generation_error'] for r in rows)/100,
        'generation_errors':sum(r['generation_error'] is not None for r in rows),'input_hashes':inputs,
        'violations':dict(violations),'triggered_rejections':dict(triggers),'settings':settings})
    save(root/'status.json',{'stage':'planning-complete','updated_utc':utc()})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','features','evaluate']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--observations',type=Path);p.add_argument('--features',type=Path)
    p.add_argument('--policy',choices=['none','shared-verified']);a=p.parse_args();c=read(a.config);c['configuration_path']=str(a.config)
    verify_configuration(c)
    if a.mode=='prepare':prepare(c,a.output)
    elif a.mode=='features':features(c,a.observations,a.output,torch.device('cuda:0'))
    else:evaluate(c,a)


if __name__=='__main__':main()
