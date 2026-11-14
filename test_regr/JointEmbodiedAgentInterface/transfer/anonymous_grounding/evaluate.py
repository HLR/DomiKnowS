"""Selected grounding snapshot: matched EAI, VLABench and LIBERO protocols."""
import json
import random
from collections import Counter
from pathlib import Path
import torch
from ..protocol import read,freeze,sha,digest,episodes,public_input,context as external_context
from ..campaign import save,utc
from .agent import load_snapshot,predicted_context


def generation(planner,head,runtime,context,feature,policy):
    from ...models import FreeDecoding
    from ...eval_generalization import input_record
    from ...generalization.shared_constraints import bind_shared,rule_trace
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
    planner.free_decoding=policy=='none';labels=[];error=None
    with torch.no_grad():
        effective,predictions,_=predicted_context(planner,head,context,feature['features'].to(planner.device).float(),
                                                feature['visible'].to(planner.device))
        rendered=input_record(planner,effective)
        if policy=='none':dfa=FreeDecoding(runtime.vlabench_vocabulary.label_count,runtime.vlabench_vocabulary.eos_label)
        elif policy=='shared-verified':dfa=bind_shared(runtime.vlabench_vocabulary,runtime.max_vlabench_operations,context)
        elif policy=='full-metadata':dfa=runtime.dfa_for('vlabench',context)
        else:raise ValueError('unsupported constraint policy')
        original=planner._prepare_inputs_and_boundaries;verified=False
        def checked(domain,actual,active_tokens=()):
            nonlocal verified
            if not verified:
                if domain!='vlabench' or input_record(planner,actual)!=rendered:
                    raise AssertionError('constraints changed actual generation inputs')
                verified=True
            return original(domain,actual,active_tokens)
        planner._prepare_inputs_and_boundaries=checked
        try:
            with runtime.domain_scope('vlabench'):
                labels,_=planner.sample_labels('vlabench',effective,dfa,
                    max_steps=runtime.max_vlabench_operations*5+1,deterministic=True)
            prediction=labels_to_plan(labels,runtime.vlabench_vocabulary,world=runtime.world.vlabench)
        except (ValueError,RuntimeError,KeyError) as exc:
            prediction=[];error={'type':type(exc).__name__,'message':str(exc)}
        finally:planner._prepare_inputs_and_boundaries=original
    trace=rule_trace(dfa,labels) if policy!='none' else {'declared':[],'applicable_events':{},'triggered_rejections':{},'events':[]}
    return prediction,error,trace,dfa,rendered,predictions


def evaluate(args,config,planner,head,runtime,splits,settings):
    from ...utils.planner_context import planner_context
    from ...utils.paired_references import PairedReferenceAudit
    from ...generalization.libero_protocol import abstract_score
    from test_regr.VLABenchAgentInterface.reward import score_vlabench_plan
    obs=read(args.observations);features=read(args.features)
    bindings={r['episode_id']:r for r in obs['vlabench']}
    cached={}
    for row in features['episodes']:
        if sha(row['path'])!=row['sha256']:raise ValueError('frozen visual feature changed')
        cached[row['episode_id']]=torch.load(row['path'],map_location='cpu',weights_only=True)
    if features['encoder_sha256']!=head.visual_encoder_sha256:raise ValueError('grounder encoder differs')
    if features['observations_sha256']!=sha(args.observations) or features['checkpoint_sha256']!=sha(args.checkpoint):
        raise ValueError('feature checkpoint or observation provenance differs')
    reports={}
    datasets=['vlabench'] if args.constraint_policy=='full-metadata' else ['vlabench','libero']
    for dataset in datasets:
        folder=args.experiment_output/dataset/args.constraint_policy;folder.mkdir(parents=True,exist_ok=True)
        freeze(folder/'settings.json',settings)
        if (folder/'report.json').exists():
            reports[dataset]=read(folder/'report.json');continue
        if (folder/'predictions.jsonl').exists():raise ValueError('incomplete evaluation preserved; use a fresh output')
        selected=splits['test'] if dataset=='vlabench' else episodes(read(config['libero_manifest']))
        expected=450 if dataset=='vlabench' else 100
        if len(selected)!=expected:raise ValueError('evaluation episode coverage differs')
        valid=errors=0;inputs={};triggers=Counter();rows=[]
        with (folder/'predictions.jsonl').open('x') as stream,(folder/'decisions.jsonl').open('x') as decisions:
            paired=PairedReferenceAudit(stream) if dataset=='vlabench' else None
            for index,example in enumerate(selected):
                if dataset=='vlabench':
                    identity=example.episode_id;binding=bindings[identity]
                    if tuple(binding['canonical_entities'])!=tuple(example.entities):raise ValueError('canonical entity order changed')
                    if any(sha(p)!=h for p,h in binding['files'].items()):raise ValueError('test observation changed')
                    bound=planner_context('vlabench',example)
                    bound['images']=(binding['raw_image'],binding['canonical_image'])
                else:
                    identity=example['episode_id'];public=public_input(example);bound=external_context(public,'libero')
                planner.decision_observer=lambda row:decisions.write(json.dumps({'episode_id':identity,**row},default=str)+'\n')
                prediction,error,trace,dfa,rendered,roles=generation(planner,head,runtime,bound,cached[identity],args.constraint_policy)
                inputs[identity]=digest(rendered);errors+=int(error is not None);triggers.update(trace['triggered_rejections'])
                freeze(args.experiment_output/'rendered_inputs'/args.constraint_policy/dataset/f'{index:04d}.json',
                    {'episode_id':identity,'input':rendered,'predicted_roles':roles})
                decisions.write(json.dumps({'episode_id':identity,'predicted_roles':roles,'rule_trace':trace})+'\n');decisions.flush()
                if dataset=='vlabench':
                    paired.observe(example,prediction,generation_error=error)
                    score=score_vlabench_plan(prediction,example.operation_sequence,example.dependency,
                        entity_table=example.entities,world_bundle=runtime.world.vlabench,
                        task_name=example.task,task_context=bound,policy=dfa)
                    valid+=int(score.valid and error is None)
                else:
                    # Hidden BDDL goals are first opened after generation.
                    score=abstract_score(prediction,read(example['scoring_private']))
                    row={'episode_id':identity,'prediction':prediction,'generation_error':error,
                         'input_hash':inputs[identity],'predicted_roles':roles,'rule_trace':trace,**score}
                    rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    valid+=int(score['valid_plan'] and error is None)
                save(args.experiment_output/('status_'+args.constraint_policy+'.json'),
                    {'stage':'evaluating','dataset':dataset,'policy':args.constraint_policy,
                     'completed':index+1,'total':expected,'updated_utc':utc()})
                if (index+1)%10==0:print(f'[grounding-eval] {dataset} {args.constraint_policy} {index+1}/{expected}',flush=True)
            if paired:
                report=paired.report();metric=report['corrected']['planning_metrics']
                dims=metric['official_mapping']['legacy_total_score_by_dimension']
                report.update(dimensions=dims,Macro=sum(dims.values())/len(dims),
                              progress_score=metric['overall']['score'],exact_match=report['corrected']['strict_plan_match'])
            else:report={'abstract_goal_success':sum(r['goal_success'] and not r['generation_error'] for r in rows)/expected}
        report.update(examples=expected,valid_plan_rate=valid/expected,generation_errors=errors,
                      input_hashes=inputs,triggered_rejections=dict(triggers),settings=settings)
        freeze(folder/'report.json',report);reports[dataset]=report
    planner.decision_observer=None
    return reports


def main():
    from ... import eval_planning
    from ...main import _controller,_evaluate_eai
    from ...models import JointQwenVLPlanner
    from ...utils.planner_context import planner_context
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    parser=eval_planning.build_eval_parser()
    parser.add_argument('--configuration',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True);parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--constraint-policy',choices=['none','shared-verified','full-metadata'],required=True)
    parser.add_argument('--experiment-output',type=Path,required=True);parser.add_argument('--evaluate-eai',action='store_true')
    args=parser.parse_args();config=read(args.configuration)
    from .eval_observations import verify_configuration
    verify_configuration(config)
    if sha(args.checkpoint)!=config['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
    args.knowledge='metadata';args.observation_mode='metadata-assisted';args.split='test';args.vlabench_planning_dir=str(args.planning_dir)
    apply_research_arguments(args);apply_knowledge_argument(args)
    random.seed(42);torch.manual_seed(42)
    runtime,splits=eval_planning._prepare_eval(args,torch.device(args.device))
    if [e.episode_id for e in splits['test']]!=read(args.checkpoint.parent/'split_identities.json')['splits']['test']:
        raise ValueError('frozen test identities differ')
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,model_id=args.planner_model,
        load_in_4bit=args.load_in_4bit,decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller=_controller(args,torch.device(args.device))
    head,payload=load_snapshot(args.checkpoint,runtime=runtime,planner=planner,controller=controller,device=args.device)
    planner.eval();head.eval();_ANONYMOUS_OBJECTS.set(True)
    settings={'checkpoint_sha256':sha(args.checkpoint),'head_sha256':payload['metrics']['grounding']['sha256'],
        'observations_sha256':sha(args.observations),'features_sha256':sha(args.features),
        'source_manifest_sha256':config['evaluation_source_manifest_sha256'],
        'configuration_sha256':sha(args.configuration),'constraint_policy':args.constraint_policy,
        'seed':42,'decoding':'greedy','selection':'source validation only, continuation epoch 2',
        'protocol':'anonymous source-only grounding; VLA reannotated RGB; original LIBERO views; predicted roles only'}
    freeze(args.experiment_output/('settings_'+args.constraint_policy+'.json'),settings)
    reports=evaluate(args,config,planner,head,runtime,splits,settings)
    if args.evaluate_eai:
        from test_regr.EmbodiedAgentInterface.dataset import load_eai_dataset,split_train_dev
        eai=load_eai_dataset(args.eai_dataset,split=args.eai_split,data_path=args.eai_data_path,device='cpu',max_steps=args.eai_max_steps)
        _,validation=split_train_dev(eai,dev_fraction=.2)
        if len(validation)!=88:raise ValueError('canonical EAI validation must contain 88 examples')
        from test_regr.EmbodiedAgentInterface.reward import make_eai_reward_function
        for item in validation:
            item['reward_function']=make_eai_reward_function(item,vocabulary=runtime.eai_vocabulary,
                mode='dense',world_bundle=runtime.world.eai)
        planner.free_decoding=False
        with (args.experiment_output/'eai_predictions.jsonl').open('x') as stream:
            def observer(item,labels,result):
                stream.write(json.dumps({'instruction':planner_context('eai',item)['instruction'],
                                        'labels':labels,'goal':result},default=str)+'\n');stream.flush()
            report=_evaluate_eai(planner,runtime,validation,limit=None,prediction_observer=observer)
        freeze(args.experiment_output/'eai.json',report)
    freeze(args.experiment_output/('summary_'+args.constraint_policy+'.json'),
        {'settings':settings,'reports':{k:{n:v for n,v in r.items() if n not in ('original','corrected','input_hashes')}
                                      for k,r in reports.items()}})
    save(args.experiment_output/('status_'+args.constraint_policy+'.json'),{'stage':'complete','updated_utc':utc()})


if __name__=='__main__':main()
