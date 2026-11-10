"""Joint checkpoint planning on frozen public RLBench inputs, without goal access."""
from collections import Counter
import json
from pathlib import Path
import random
import torch

from . import eval_planning
from .eval_generalization import generate,input_record
from .checkpoint import load_joint_checkpoint
from .main import _controller
from .models import JointQwenVLPlanner
from .utils.generalization_protocol import file_hash,freeze_json
from .generalization.libero_protocol import digest
from .generalization.shared_constraints import SHARED_RULES
from .generalization.rlbench_protocol import planning_context,plan_validity,validate_calibration
from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument


def evaluate(planner,runtime,manifest,policy,output,settings):
    records=[row for row in manifest['episodes'] if row['split']=='evaluation']
    if len(records)!=100 or len({row['episode_id'] for row in records})!=100:
        raise ValueError('require exactly 100 unique frozen RLBench evaluation episodes')
    root=output/policy;root.mkdir(parents=True,exist_ok=True)
    freeze_json(root/'settings.json',{**settings,'constraint_policy':policy})
    if (root/'report.json').exists():return json.loads((root/'report.json').read_text())
    inputs={};valid=errors=0;violations=Counter();triggered=Counter();applicable=Counter()
    with (root/'predictions.jsonl').open('w') as stream,(root/'decisions.jsonl').open('w') as decisions:
        for index,episode in enumerate(records):
            # Never open state or scoring-private references for generation.
            public=json.loads(Path(episode['generation_input']).read_text())
            for path in [episode['generation_input'],*public['images']]:
                if file_hash(path)!=episode['files'][path]:raise ValueError('frozen public observation changed')
            context=planning_context(public)
            rendered=input_record(planner,context);inputs[episode['episode_id']]=digest(rendered)
            freeze_json(output/'rendered_inputs'/f'{index:03d}.json',{'episode_id':episode['episode_id'],**rendered})
            planner.decision_observer=lambda row:decisions.write(json.dumps({'episode_id':episode['episode_id'],**row},default=str)+'\n')
            prediction,error,trace,_=generate(planner,runtime,context,policy)
            validity=plan_validity(prediction,public)
            errors+=error is not None;valid+=validity['valid_plan'] and error is None
            violations.update(v['reason'] for v in validity['violations'])
            triggered.update(trace['triggered_rejections']);applicable.update(trace['applicable_events'])
            stream.write(json.dumps({'episode_id':episode['episode_id'],'prediction':prediction,
                'generation_error':error,'input_hash':inputs[episode['episode_id']],**validity,'rule_trace':trace})+'\n')
            stream.flush()
            print(f'[rlbench-planning] {policy} {index+1}/100',flush=True)
    result={'dataset':'rlbench_metadata_assisted_pick_place_subset','examples':100,'constraint_policy':policy,
        'valid_plan_rate':valid/100,'generation_errors':errors,'violations':dict(violations),
        'input_hashes':inputs,'declared_shared_rules':sorted(SHARED_RULES) if policy=='shared-verified' else [],
        'groundable_rule_episodes':{name:100 for name in SHARED_RULES} if policy=='shared-verified' else {},
        'applicable_events':dict(applicable),'triggered_rejections':dict(triggered),
        'goal_scoring':'separate native simulator evaluation; no abstract goals or VLABench Macro manufactured',
        'settings':settings}
    freeze_json(root/'report.json',result)
    return result


def main():
    parser=eval_planning.build_eval_parser()
    parser.set_defaults(split='test')
    parser.add_argument('--episode-manifest',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--experiment-output',type=Path,required=True)
    parser.add_argument('--constraint-policy',choices=['shared-verified','none'])
    args=parser.parse_args()
    if (args.no_dfa or getattr(args,'no_dfa_official',False) or args.input_probe!='original'
            or args.limit or args.num_shards!=1 or args.split!='test'
            or getattr(args,'pilot_manifest',None) or getattr(args,'audit_probes',False)):
        parser.error('RLBench uses its fixed public manifest; select --constraint-policy, with no native VLABench probes/filters')
    # Reference runtime matches the checkpoint. Actual decoding receives only
    # shared observation-bound constraints or FreeDecoding, never oracle tasks.
    args.knowledge='metadata';args.observation_mode='metadata-assisted';args.split='test'
    source=Path(__file__).parent
    validate_calibration(args.episode_manifest,args.calibration,source/'generalization')
    manifest=json.loads(args.episode_manifest.read_text())
    args.vlabench_planning_dir=str(args.planning_dir)
    apply_research_arguments(args);apply_knowledge_argument(args)
    random.seed(args.seed);torch.manual_seed(args.seed)
    runtime,_=eval_planning._prepare_eval(args,torch.device(args.device))
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,model_id=args.planner_model,
        load_in_4bit=args.load_in_4bit,decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller=_controller(args,torch.device(args.device))
    payload=load_joint_checkpoint(args.checkpoint,runtime=runtime,planner=planner,controller=controller,map_location=args.device)
    planner.eval()
    settings={'checkpoint_sha256':file_hash(args.checkpoint),'manifest_sha256':file_hash(args.episode_manifest),
        'calibration_sha256':file_hash(args.calibration),'seed':args.seed,'decoding':'greedy',
        'checkpoint_stage':payload['stage'],'checkpoint_epoch':payload['epoch'],
        'source_sha256':{str(p.relative_to(source)):file_hash(p) for p in [Path(__file__),
             source/'models.py',source/'eval_generalization.py',source/'generalization/rlbench_protocol.py',source/'generalization/shared_constraints.py']}}
    output=args.experiment_output;output.mkdir(parents=True,exist_ok=True)
    freeze_json(output/'settings.json',settings)
    reports={}
    for policy in [args.constraint_policy] if args.constraint_policy else ['shared-verified','none']:
        reports[policy]=evaluate(planner,runtime,manifest,policy,output,settings)
    if len(reports)==2 and reports['shared-verified']['input_hashes']!=reports['none']['input_hashes']:
        raise ValueError('constraint conditions changed actual generation inputs')
    planner.decision_observer=None
    freeze_json(output/'summary.json',{'settings':settings,'reports':reports})


if __name__=='__main__':main()
