"""Frozen fresh-state evaluation; held-out goals opened only after generation."""
from pathlib import Path
import random
import json
import torch
from ...protocol import read,freeze,sha,digest,episodes,public_input,context
from ...campaign import save,utc
from .agent import load_binding,predict
from ..grounder import condition


def main():
    from .... import eval_planning
    from ....main import _controller,_evaluate_eai
    from ....models import JointQwenVLPlanner
    from ....eval_generalization import input_record
    from ....generalization.libero_protocol import abstract_score
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    p=eval_planning.build_eval_parser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--constraint-policy',choices=['none','executor'],required=True)
    p.add_argument('--experiment-output',type=Path,required=True);p.add_argument('--evaluate-eai',action='store_true')
    a=p.parse_args();c=read(a.config);root=Path(c['output']);out=a.experiment_output
    for name in ('external_manifest','features','observations'):
        if sha(c[name])!=c[name+'_sha256']:raise ValueError('frozen campaign input changed')
    fresh=read(root/'inputs_complete.json')
    for name,path in [('fresh_manifest',Path(c['external_manifest'])),('fresh_features',root/'fresh_features/manifest.json'),
        ('fresh_priors',root/'fresh_priors.json'),('fresh_observations',root/'fresh_observations/manifest.json')]:
        if sha(path)!=fresh[name+'_sha256']:raise ValueError('fresh evaluation inputs changed')
    a.knowledge='metadata';a.observation_mode='metadata-assisted';a.vlabench_planning_dir=str(a.planning_dir)
    apply_research_arguments(a);apply_knowledge_argument(a);random.seed(42);torch.manual_seed(42)
    runtime,_=eval_planning._prepare_eval(a,torch.device(a.device))
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=a.planner_model,load_in_4bit=True,decoder_hidden_size=a.planner_decoder_hidden_dim)
    controller=_controller(a,torch.device(a.device));head,payload=load_binding(a.checkpoint,runtime=runtime,planner=planner,controller=controller,device=a.device)
    planner.eval();_ANONYMOUS_OBJECTS.set(True)
    proof=read(root/'fresh_features/manifest.json')
    if proof['encoder_sha256']!=head.visual_encoder_sha256:raise ValueError('trained controller/feature encoder mismatch')
    features={r['episode_id']:r for r in proof['episodes']};priors={r['episode_id']:r for r in read(root/'fresh_priors.json')['episodes']}
    settings={'checkpoint_sha256':sha(a.checkpoint),'head_sha256':payload['metrics']['grounding']['sha256'],
        'manifest_sha256':sha(c['external_manifest']),'inputs_sha256':sha(root/'inputs_complete.json'),
        'source_sha256':c['source_manifest_sha256'],'constraint_policy':a.constraint_policy,'seed':42,
        'selection':'source validation only','fresh_states':[40,41,42,43,44],'role_references_in_generation':False}
    freeze(out/'settings.json',settings)
    if (out/'predictions.jsonl').exists():raise ValueError('preserve incomplete evaluation; use a fresh output')
    rows=[]
    with (out/'predictions.jsonl').open('x') as stream,(out/'decisions.jsonl').open('x') as decisions:
        for i,e in enumerate(episodes(read(c['external_manifest']))):
            identity=e['episode_id'];public=public_input(e);bound=context(public,'libero');record=features[identity]
            if sha(record['path'])!=record['sha256']:raise ValueError('fresh feature changed')
            values=torch.load(record['path'],map_location=a.device,weights_only=True);error=None;events=[];trace={}
            effective=condition(bound,priors[identity]['predictions'])
            planner.decision_observer=lambda row:decisions.write(json.dumps({'episode_id':identity,**row},default=str)+'\n')
            try:
                prediction,effective,events,trace=predict(planner,head,runtime,bound,values['features'].float(),values['visible'],
                    priors[identity]['predictions'],a.constraint_policy,public)
            except (ValueError,RuntimeError,KeyError) as exc:prediction=[];error={'type':type(exc).__name__,'message':str(exc)}
            rendered=input_record(planner,effective);freeze(out/'rendered'/f'{i:04d}.json',{'episode_id':identity,'input':rendered})
            # Evaluator-only targets; never supplied to the binding hook or planner.
            if sha(e['scoring_private'])!=e['files'][e['scoring_private']]:raise ValueError('held-out target changed')
            score=abstract_score(prediction,read(e['scoring_private']))
            row={'episode_id':identity,'suite':e['suite'],'prediction':prediction,'generation_error':error,
                'input_hash':digest(rendered),'observation_hash':digest(public),'operation_grounding':events,'rule_trace':trace,**score}
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();decisions.flush()
            save(out/'status.json',{'completed':i+1,'total':100,'updated_utc':utc()})
            if (i+1)%10==0:print(f'[binding-eval] {a.constraint_policy} {i+1}/100',flush=True)
    freeze(out/'summary.json',{'episodes':100,'abstract_successes':sum(r['goal_success'] for r in rows),
        'valid_plans':sum(r['valid_plan'] and not r['generation_error'] for r in rows),
        'generation_errors':sum(r['generation_error'] is not None for r in rows),'settings':settings})
    if a.evaluate_eai:
        from test_regr.EmbodiedAgentInterface.dataset import load_eai_dataset,split_train_dev
        from test_regr.EmbodiedAgentInterface.reward import make_eai_reward_function
        data=load_eai_dataset(a.eai_dataset,split=a.eai_split,data_path=a.eai_data_path,device='cpu',max_steps=a.eai_max_steps)
        _,valid=split_train_dev(data,dev_fraction=.2)
        if len(valid)!=88:raise ValueError('canonical EAI validation identity count changed')
        for item in valid:item['reward_function']=make_eai_reward_function(item,vocabulary=runtime.eai_vocabulary,mode='dense',world_bundle=runtime.world.eai)
        planner.decision_observer=None;planner.free_decoding=False
        freeze(out/'eai.json',_evaluate_eai(planner,runtime,valid,limit=None))
    save(out/'status.json',{'stage':'complete','completed':100,'updated_utc':utc()})


if __name__=='__main__':main()
