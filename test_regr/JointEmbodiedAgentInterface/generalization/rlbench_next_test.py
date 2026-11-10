"""Freeze the RLBench follow-up design. Does not launch a simulator or model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

DEFAULT_CONFIG = Path(__file__).parent/'configs/rlbench_next_test.json'


def build_plan(config):
    if config['automatic_launch'] or config['retraining']:
        raise ValueError('follow-up is an evaluation design, not a training/launch request')
    if config['constraint_policies'] != ['shared-verified', 'none']:
        raise ValueError('RLBench must not alias privileged VLABench task metadata')
    splits = config['reset_indices']
    if set(splits) != {'evaluation', 'calibration'} or set(splits['evaluation']) & set(splits['calibration']):
        raise ValueError('evaluation/calibration reset indices must be disjoint')
    episodes = []; seeds = set(); cases = set()
    for task in config['tasks']:
        if set(task['skills']) != {'pick', 'place'}:
            raise ValueError('unsupported skills need a separate reviewed adapter')
        for variation in task['variations']:
            case = f"{task['task']}/variation{variation:02d}"
            if case in cases:raise ValueError('duplicate task variation')
            cases.add(case)
            for split, indices in splits.items():
                for index in indices:
                    episode_id = f'{case}/reset{index:02d}'
                    seed = int(hashlib.sha256(f"{config['seed']}:{episode_id}".encode()).hexdigest()[:8], 16)
                    if seed in seeds:raise ValueError('duplicate reset or seed')
                    seeds.add(seed)
                    episodes.append({'episode_id':episode_id, 'task':task['task'], 'variation':variation,
                                     'split':split, 'reset_index':index, 'seed':seed})
    counts = {split:sum(e['split']==split for e in episodes) for split in splits}
    if counts != {'evaluation':100, 'calibration':100}:
        raise ValueError('require exactly 100 evaluation and 100 calibration reset blueprints')
    return {'version':1, 'config':config,
            'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
            'native_task_classes':len(config['tasks']), 'task_variation_cases':len(cases),
            'counts':counts, 'episodes':episodes, 'ready_for_model_evaluation':False,
            'pending_gates':config['required_gates'],
            'notice':'Reset blueprints, not frozen simulator states. No RLBench adapter/executor has passed calibration.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rlbench-root',type=Path)
    args=parser.parse_args()
    result=build_plan(json.loads(args.config.read_text(encoding='utf-8')))
    if args.rlbench_root:
        revision=subprocess.check_output(['git','-C',str(args.rlbench_root),'rev-parse','HEAD'],text=True).strip()
        if revision != result['config']['source']['commit']:raise ValueError('RLBench source revision differs')
        sources={}
        for task in result['config']['tasks']:
            path=args.rlbench_root/'rlbench/tasks'/f"{task['task']}.py"
            sources[task['task']]=hashlib.sha256(path.read_bytes()).hexdigest()
        result['reviewed_task_source_hashes']=sources
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError('frozen follow-up design differs; preserve the original and use a new output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('native_task_classes','task_variation_cases','counts','ready_for_model_evaluation')}))


if __name__=='__main__':main()
