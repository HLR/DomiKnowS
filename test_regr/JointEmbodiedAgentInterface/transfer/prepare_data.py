"""Freeze separate target-training scenes using the retained native adapters.

Invoke directly inside each dataset's isolated simulator runtime. This prepares
supervision only; no model predictions or test outcomes choose scenes.
"""
import argparse
import importlib
import json
import random
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'transfer'

from .data import allocation, audit_disjoint, shard_cases
from .protocol import digest, freeze, read, sha


def native_json(value):
    """Convert native array/scalar observations before freezing JSON artifacts."""
    return json.loads(json.dumps(value, default=lambda item: item.tolist()))


def fingerprints(record, dataset):
    import numpy as np
    public = read(record['generation_input'])
    private = read(record['scoring_private'])
    if dataset == 'libero':
        physical = {'entities': private['entities'],
                    'state': np.round(np.load(record['state'], allow_pickle=False), 5).tolist()}
    else:
        state = {r['name']: r for r in read(record['state'])}
        names = sorted({name for entity in private['entities'] for name in entity['parts']})
        physical = {name: {'pose': np.round(state[name]['pose'], 5).tolist()}
                    for name in names}
    return {'episode_id': record['episode_id'], 'split': record['split'],
            'scene_fingerprint': digest(physical),
            'input_fingerprint': digest({**{k: v for k, v in public.items() if k != 'images'},
                                         'images': [sha(p) for p in public['images']]})}


def prepare_libero(config, output, shard_index=0, num_shards=1):
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark
    source = Path(config['datasets']['libero']['executor_source'])
    sys.path.insert(0, str(source))
    bridge = importlib.import_module('libero_campaign')
    base_manifest = read(config['datasets']['libero']['manifest'])
    bddl_by_case = {f"{e['suite']}/{e['task']}": e['bddl'] for e in base_manifest['episodes']}
    blueprints = allocation('libero', list(bddl_by_case), config['seed'])
    suite_cache = {}; records = []
    for case in shard_cases(list(bddl_by_case), shard_index, num_shards):
        suite_name, task_name = case.split('/', 1)
        if suite_name not in suite_cache:
            suite_cache[suite_name] = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        suite = suite_cache[suite_name]
        task_index = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name)
        states = suite.get_task_init_states(task_index)
        if len(states) < 40:
            raise ValueError('required initialization states unavailable: ' + case)
        spec = bridge.specification(bddl_by_case[case])
        env = bridge.environment(bddl_by_case[case], render=True)
        try:
            for blueprint in [e for e in blueprints if e['episode_id'].rsplit('/', 1)[0] == case]:
                folder = output/blueprint['split']/blueprint['episode_id']; folder.mkdir(parents=True, exist_ok=True)
                if (folder/'record.json').exists():
                    record = read(folder/'record.json')
                    if any(sha(p) != h for p, h in record['files'].items()):
                        raise ValueError('prepared scene changed')
                    records.append(record); continue
                # A deterministic reset, followed by the same settling as the archived preparation.
                random.seed(blueprint['seed']); np.random.seed(blueprint['seed'])
                env.seed(blueprint['seed']); env.reset()
                obs = env.set_init_state(states[blueprint['index']])
                for _ in range(10):
                    obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
                np.save(folder/'initial_state.npy', env.get_sim_state(), allow_pickle=False)
                rgb, numbered, visible = bridge.numbered_image(obs, env, spec['entities'])
                Image.fromarray(rgb).save(folder/'raw.png'); numbered.save(folder/'numbered.png')
                public = bridge.generation_input(spec, [folder/'raw.png', folder/'numbered.png'])
                public['visible_entity_ids'] = visible
                freeze(folder/'generation_input.json', public)
                freeze(folder/'scoring_private.json', spec)
                paths = [folder/n for n in ('initial_state.npy', 'raw.png', 'numbered.png',
                                           'generation_input.json', 'scoring_private.json')]
                record = {**blueprint, 'suite': suite_name, 'task': task_name,
                    'state_index': blueprint['index'], 'bddl': bddl_by_case[case],
                    'state': str(folder/'initial_state.npy'), 'generation_input': str(folder/'generation_input.json'),
                    'scoring_private': str(folder/'scoring_private.json'), 'files': {str(p): sha(p) for p in paths}}
                freeze(folder/'record.json', record); records.append(record)
                print(f'[transfer-data] libero {len(records)}/600', flush=True)
        finally:
            env.close()
    return records


def prepare_rlbench(config, output):
    source = Path(config['datasets']['rlbench']['executor_source'])
    sys.path.insert(0, str(source))
    bridge_module = importlib.import_module('rlbench_campaign')
    base = read(config['datasets']['rlbench']['manifest'])
    cases = {e['episode_id'].rsplit('/', 1)[0]: (e['task'], e['variation']) for e in base['episodes']}
    blueprints = allocation('rlbench', list(cases), config['seed'])
    bridge = bridge_module.RLBenchBridge(render=True); records = []
    try:
        for blueprint in blueprints:
            folder = output/blueprint['split']/blueprint['episode_id']; folder.mkdir(parents=True, exist_ok=True)
            if (folder/'record.json').exists():
                record = read(folder/'record.json')
                if any(sha(p) != h for p, h in record['files'].items()):
                    raise ValueError('prepared scene changed')
                records.append(record); continue
            task, variation = cases[blueprint['episode_id'].rsplit('/', 1)[0]]
            instruction = bridge.reset(task, variation, blueprint['seed'])
            state = bridge.snapshot_state(); bridge.restore_state(state)
            freeze(folder/'state.json', native_json(bridge.snapshot_state()))
            geometry = [bridge.geometry(i) for i in range(len(bridge.entities))]
            images, visible = bridge_module.image_inputs(bridge, folder)
            public = native_json(bridge_module.public_input(instruction, bridge.entities, geometry, images, visible))
            freeze(folder/'generation_input.json', public)
            # Labels are for target training/validation only, after the public record is frozen.
            freeze(folder/'scoring_private.json', {'entities': bridge.entities,
                                                  'reference': bridge.calibration_reference()})
            paths = [folder/n for n in ('state.json', 'raw.png', 'numbered.png',
                                       'generation_input.json', 'scoring_private.json')]
            record = {**blueprint, 'task': task, 'variation': variation, 'reset_index': blueprint['index'],
                'state': str(folder/'state.json'), 'generation_input': str(folder/'generation_input.json'),
                'scoring_private': str(folder/'scoring_private.json'), 'files': {str(p): sha(p) for p in paths}}
            freeze(folder/'record.json', record); records.append(record)
            print(f'[transfer-data] rlbench {len(records)}/300', flush=True)
    finally:
        bridge.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', choices=['libero', 'rlbench'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args(); config = read(args.config); dataset = args.dataset
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output/'settings.json', {'config_sha256': sha(args.config), 'dataset': dataset,
        'source_sha256': sha(__file__), 'seed': config['seed'], 'num_shards': args.num_shards})
    if args.finalize:
        records = [r for i in range(args.num_shards)
                   for r in read(args.output/'shards'/f'{i:03d}.json')['episodes']]
        records.sort(key=lambda r: r['episode_id'])
    elif dataset == 'libero':
        records = prepare_libero(config, args.output, args.shard_index, args.num_shards)
    else:
        if args.num_shards != 1:
            raise ValueError('RLBench preparation is not sharded')
        records = prepare_rlbench(config, args.output)
    reserved = read(config['datasets'][dataset]['manifest'])['episodes']
    integrity = audit_disjoint([fingerprints(r, dataset) for r in records],
                              [fingerprints(r, dataset) for r in reserved])
    if args.num_shards > 1 and not args.finalize:
        freeze(args.output/'shards'/f'{args.shard_index:03d}.json',
               {'episodes': records, 'integrity': integrity})
        return
    expected = 600 if dataset == 'libero' else 300
    if len(records) != expected:
        raise ValueError('target preparation incomplete')
    freeze(args.output/'integrity.json', integrity)
    freeze(args.output/'manifest.json', {'dataset': dataset, 'episodes': records,
        'label': 'separate target-domain training/validation scenes; seen-task adaptation',
        'reserved_manifest_sha256': sha(config['datasets'][dataset]['manifest']), 'integrity': integrity})


if __name__ == '__main__':
    main()
