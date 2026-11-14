"""Matched, frozen-snapshot LIBERO grounding/action-contract diagnostics."""
import argparse
from collections import Counter
from pathlib import Path
import random
import torch
from ...protocol import read,sha,digest,freeze,episodes,public_input,context,jsonl,compare_baseline
from ...campaign import save,utc
from ..agent import load_snapshot,predicted_context
from ..grounder import condition as add_roles
from ..eval_observations import verify_configuration
from .contract import ExecutorContract
from .provenance import validate_inputs

CONDITIONS=('baseline','raw','contract','raw-contract','aligned','aligned-contract')
CAPABILITIES={'pick':'target_entity_name','place':'target_container_name'}


def generate(planner,head,runtime,bound,feature,public,variant,aligned=None):
    from ....models import FreeDecoding
    from ....eval_generalization import input_record
    from ....generalization.shared_constraints import bind_shared
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
    use_contract=variant.endswith('contract');planner.free_decoding=not use_contract
    labels=[];error=None
    with torch.no_grad():
        if variant.startswith('aligned'):
            roles=aligned['predictions'];effective=add_roles(bound,roles)
        else:
            # Legacy controls are explicit, including with the corrected public-context API.
            options={'grounding_instruction':public['instruction'] if variant.startswith('raw') else bound['instruction']}
            effective,roles,_=predicted_context(planner,head,bound,feature['features'].to(planner.device).float(),
                                               feature['visible'].to(planner.device),**options)
        rendered=input_record(planner,effective)
        if use_contract:
            base=bind_shared(runtime.vlabench_vocabulary,runtime.max_vlabench_operations,bound)
            dfa=ExecutorContract(base,runtime.vlabench_vocabulary,public,CAPABILITIES,runtime.max_vlabench_operations)
        else:dfa=FreeDecoding(runtime.vlabench_vocabulary.label_count,runtime.vlabench_vocabulary.eos_label)
        original=planner._prepare_inputs_and_boundaries;verified=False
        def checked(domain,actual,active_tokens=()):
            nonlocal verified
            if not verified:
                if domain!='vlabench' or input_record(planner,actual)!=rendered:
                    raise AssertionError('actual planner inputs changed during binding')
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
    return prediction,error,dfa.trace(labels) if use_contract else {},rendered,roles,labels


def role_score(prediction,roles,spec):
    # Evaluator-only: never called before model generation has returned.
    gold={role:{spec['entities'].index(g[column]) for g in spec['goals']}
          for role,column in [('source',1),('destination',2)]}
    result={}
    for role,skill,key in [('source','pick','target_entity_name'),('destination','place','target_container_name')]:
        pointer=next((o.get('params',{}).get(key) for o in prediction if o.get('name')==skill),None)
        result[role]={'grounder_correct':roles[role] in gold[role], 'planner_correct':pointer in gold[role],
                      'gold_ids':sorted(gold[role]),'planner_id':pointer}
    return result


def metrics(rows):
    n=len(rows)
    return {'episodes':n,'valid_plans':sum(r['valid_plan'] and not r['generation_error'] for r in rows),
            'abstract_successes':sum(r['goal_success'] and not r['generation_error'] for r in rows),
            'generation_errors':sum(r['generation_error'] is not None for r in rows),
            'unsupported_skill_plans':sum(any(v['reason']=='unsupported_skill' for v in r['violations']) for r in rows),
            **{f'{which}_{role}_correct':sum(r['role_scores'][role][which+'_correct'] for r in rows)
               for which in ('grounder','planner') for role in ('source','destination')}}


def build_parser():
    from .... import eval_planning
    p=eval_planning.build_eval_parser();p.add_argument('--configuration',type=Path,required=True)
    p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--experiment-output',type=Path,required=True)
    return p


def main():
    from .... import eval_planning
    from ....main import _controller
    from ....models import JointQwenVLPlanner
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    from ....generalization.libero_protocol import abstract_score
    args=build_parser().parse_args();args.output=args.experiment_output
    config=read(args.configuration);verify_configuration(config);validate_inputs(config)
    args.knowledge='metadata';args.observation_mode='metadata-assisted';args.split='test';args.vlabench_planning_dir=str(args.planning_dir)
    apply_research_arguments(args);apply_knowledge_argument(args);random.seed(42);torch.manual_seed(42)
    observations=read(config['observations']);features=read(config['features'])
    if sha(config['observations'])!=config['observations_sha256'] or sha(config['features'])!=config['features_sha256']:
        raise ValueError('frozen feature/observation manifest changed')
    if features['observations_sha256']!=sha(config['observations']) or features['checkpoint_sha256']!=sha(args.checkpoint):
        raise ValueError('feature provenance differs')
    selected=episodes(read(config['libero_manifest']));bindings={r['episode_id']:r for r in observations['libero']}
    cached={r['episode_id']:r for r in features['episodes'] if r['dataset']=='libero'}
    runtime,_=eval_planning._prepare_eval(args,torch.device(args.device))
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,model_id=args.planner_model,
        load_in_4bit=args.load_in_4bit,decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller=_controller(args,torch.device(args.device))
    head,payload=load_snapshot(args.checkpoint,runtime=runtime,planner=planner,controller=controller,device=args.device)
    if features['encoder_sha256']!=head.visual_encoder_sha256:raise ValueError('feature encoder differs')
    planner.eval();head.eval();_ANONYMOUS_OBJECTS.set(True)
    settings={'configuration_sha256':sha(args.configuration),'checkpoint_sha256':sha(args.checkpoint),
        'head_sha256':payload['metrics']['grounding']['sha256'],'source_manifest_sha256':config['evaluation_source_manifest_sha256'],
        'manifest_sha256':sha(config['libero_manifest']),'observations_sha256':sha(config['observations']),
        'features_sha256':sha(config['features']),'condition':args.condition,'seed':42,'greedy':True,
        'training_updates':0,'capabilities':CAPABILITIES,'target_training':False}
    aligned={}
    if args.condition.startswith('aligned'):
        path=Path(config['output'])/'aligned/manifest.json';data=read(path)
        if data['settings']['configuration_sha256']!=sha(args.configuration):raise ValueError('paired retrieval configuration changed')
        aligned={r['episode_id']:r for r in data['episodes']};settings['aligned_manifest_sha256']=sha(path)
    freeze(args.output/'settings.json',settings)
    if (args.output/'predictions.jsonl').exists():raise ValueError('preserve previous predictions; use fresh output')
    rows=[]
    with (args.output/'predictions.jsonl').open('x') as stream,(args.output/'decisions.jsonl').open('x') as decisions:
        for i,episode in enumerate(selected):
            identity=episode['episode_id'];public=public_input(episode);binding=bindings[identity];f=cached[identity]
            if any(sha(path)!=expected for path,expected in binding['files'].items()) or sha(f['path'])!=f['sha256']:
                raise ValueError('frozen observation or feature changed')
            feature=torch.load(f['path'],map_location='cpu',weights_only=True);bound=context(public,'libero')
            planner.decision_observer=lambda r:decisions.write(__import__('json').dumps({'episode_id':identity,**r},default=str)+'\n')
            prediction,error,trace,rendered,roles,labels=generate(planner,head,runtime,bound,feature,public,args.condition,aligned.get(identity))
            freeze(args.output/'rendered'/f'{i:04d}.json',{'episode_id':identity,'input':rendered,
                'grounding_query':public['instruction'] if args.condition.startswith(('raw','aligned')) else bound['instruction']})
            # Scoring-private file is first opened here, after generation.
            if sha(episode['scoring_private'])!=episode['files'][episode['scoring_private']]:raise ValueError('reference changed')
            spec=read(episode['scoring_private']);score=abstract_score(prediction,spec)
            row={'episode_id':identity,'suite':episode['suite'],'prediction':prediction,'generation_error':error,
                'input_hash':digest(rendered),'observation_hash':digest(public),'predicted_roles':roles,'labels':labels,
                'role_scores':role_score(prediction,roles,spec),'rule_trace':trace,**score}
            rows.append(row);stream.write(__import__('json').dumps(row)+'\n');stream.flush();decisions.flush()
            save(args.output/'status.json',{'completed':i+1,'total':100,'updated_utc':utc()})
            if (i+1)%10==0:print(f'[object-fixes] {args.condition} {i+1}/100',flush=True)
    if args.condition=='baseline':
        gate=compare_baseline(rows,jsonl(config['archive_predictions']),[e['episode_id'] for e in selected])
        # Scores, including recall and hidden goal satisfaction, must also reproduce.
        old={r['episode_id']:r for r in jsonl(config['archive_predictions'])}
        for row in rows:
            for field in ('goal_success','goal_recall','violations'):
                if row[field]!=old[row['episode_id']][field]:gate['mismatches'].append({'episode_id':row['episode_id'],'field':field})
        gate['passed']=not gate['mismatches'];freeze(args.output/'baseline_gate.json',gate)
        if not gate['passed']:raise ValueError('baseline reproduction failed')
    freeze(args.output/'summary.json',{'overall':metrics(rows),'suites':{s:metrics([r for r in rows if r['suite']==s])
        for s in sorted({r['suite'] for r in rows})},'settings':settings})
    save(args.output/'status.json',{'stage':'complete','completed':100,'updated_utc':utc()})


if __name__=='__main__':main()
