"""Freeze validation 30--34 and confirmation 45--49 with the retained bridge."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def freeze(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read(path) != value: raise ValueError('frozen preparation changed: '+str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def pixels(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'))
    return digest([array.shape, hashlib.sha256(array.tobytes()).hexdigest()])


def assert_distinct(new, old, fields, label):
    if len(new) != 100 or len({r['episode_id'] for r in new}) != 100:
        raise ValueError('require exactly 100 unique '+label+' episodes')
    for field in fields:
        values = [r[field] for r in new]
        if len(set(values)) != 100: raise ValueError('duplicate '+label+' '+field)
        if set(values) & {r[field] for r in old}: raise ValueError(label+' overlaps reserved '+field)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); a = p.parse_args()
    c = read(a.config); root = Path(c['output']); base = Path(c['libero_base'])
    bridge_path = base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'
    sys.path.insert(0, str(bridge_path.parent)); bridge = importlib.import_module('libero_campaign')
    loader = importlib.util.spec_from_file_location('retained_scene_freezer', c['freezer'])
    helper = importlib.util.module_from_spec(loader); loader.loader.exec_module(helper)
    if sha(c['freezer']) != c['freezer_sha256']: raise ValueError('geometry resolver changed')
    for path, expected in c['input_hashes'].items():
        if sha(path) != expected: raise ValueError('preparation input changed: '+path)
    from libero.libero import benchmark
    from libero.libero.envs import SegmentationRenderEnv
    original = read(c['old_manifest']); current = read(c['current_manifest']); target = read(c['target_manifest'])
    calibration = [e for e in current['episodes'] if e['split'] == 'calibration']
    if len(calibration) != 100: raise ValueError('qualified calibration coverage changed')
    previous = original['episodes']+target['episodes']+[r for r in current['episodes'] if r['split'] == 'evaluation']
    prior = read(c['reserved_settled_audit'])
    if not prior['passed'] or prior['native_geometry_resolver_sha256'] != sha(c['freezer']):
        raise ValueError('reserved geometry audit incompatible')
    settled = {r['episode_id']: r for r in prior['records']}
    if len(settled) != 900: raise ValueError('expected 900 reserved states 0--44')
    reserved = []
    for e in previous:
        if any(sha(f) != h for f, h in e['files'].items()): raise ValueError('reserved episode changed')
        public = read(e['generation_input'])
        reserved.append({**settled[e['episode_id']], 'state_index': e['state_index'],
                         'raw_pixels': pixels(public['images'][0]), 'numbered_pixels': pixels(public['images'][1])})
    if len(reserved) != 900: raise ValueError('reserved split inventory differs')
    cases = {f"{e['suite']}/{e['task']}": e for e in current['episodes'] if e['split'] == 'evaluation'}
    if len(cases) != 20: raise ValueError('expected 20 fixed compatible tasks')
    cohorts = {'validation': [], 'confirmation': []}; fingerprints = {'validation': [], 'confirmation': []}
    official = []; dimensions = Image.open(read(next(iter(cases.values()))['generation_input'])['images'][0]).size
    for case, record in sorted(cases.items()):
        suite_name, task_name = case.split('/', 1); suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        index = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name)
        states = suite.get_task_init_states(index)
        if len(states) < 50: raise ValueError('required states 45--49 unavailable; no substitution')
        if sha(record['bddl']) != record['source_bddl_sha256']: raise ValueError('task specification changed')
        spec = bridge.specification(record['bddl'])
        env = SegmentationRenderEnv(bddl_file_name=record['bddl'], camera_heights=dimensions[1], camera_widths=dimensions[0],
                                   horizon=1000, control_freq=20, use_camera_obs=True)
        try:
            env.reset()
            for state_index in range(50):
                env.sim.set_state_from_flattened(np.asarray(states[state_index])); env.sim.forward()
                official.append({'episode_id': f'{case}/state{state_index:02d}', 'state_index': state_index,
                    'scene_sha256': helper.scene(env, spec),
                    'official_state_sha256': hashlib.sha256(np.asarray(states[state_index]).tobytes()).hexdigest()})
            for cohort, indices in [('validation', range(30, 35)), ('confirmation', range(45, 50))]:
                for state_index in indices:
                    identity = f'{case}/state{state_index:02d}'; folder = root/'data'/cohort/'evaluation'/identity
                    if (folder/'record.json').exists():
                        e = read(folder/'record.json')
                        if any(sha(f) != h for f, h in e['files'].items()): raise ValueError('partial preparation changed')
                    else:
                        folder.mkdir(parents=True, exist_ok=True)
                        seed = int(bridge.digest(identity)[:8], 16)
                        random.seed(seed); np.random.seed(seed); env.seed(seed); env.reset()
                        obs = env.set_init_state(states[state_index])
                        for _ in range(10): obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
                        state = folder/'initial_state.npy'; np.save(state, env.get_sim_state(), allow_pickle=False)
                        raw, numbered, visible = bridge.numbered_image(obs, env, spec['entities'])
                        raw_path = folder/'raw.png'; numbered_path = folder/'numbered.png'
                        Image.fromarray(raw).save(raw_path); numbered.save(numbered_path)
                        public = bridge.generation_input(spec, [raw_path, numbered_path]); public['visible_entity_ids'] = visible
                        freeze(folder/'generation_input.json', public); freeze(folder/'scoring_private.json', spec)
                        paths = [state, raw_path, numbered_path, folder/'generation_input.json', folder/'scoring_private.json']
                        e = {**{k:v for k,v in record.items() if k not in ('episode_id','state_index','split','state','generation_input','scoring_private','files')},
                            'episode_id': identity, 'state_index': state_index, 'cohort': cohort,
                            'split': 'evaluation', 'state': str(state), 'generation_input': str(folder/'generation_input.json'),
                            'scoring_private': str(folder/'scoring_private.json'), 'files': {str(f):sha(f) for f in paths}}
                        freeze(folder/'record.json', e)
                    public = read(e['generation_input'])
                    env.sim.set_state_from_flattened(np.load(e['state'], allow_pickle=False)); env.sim.forward()
                    raw_hash, numbered_hash = [pixels(f) for f in public['images']]
                    fingerprints[cohort].append({'episode_id': identity, 'state_index': state_index,
                        'scene_sha256': helper.scene(env, spec), 'raw_pixels': raw_hash, 'numbered_pixels': numbered_hash})
                    cohorts[cohort].append(e)
                    print(f'[verification-freeze] {cohort} {len(cohorts[cohort])}/100 {identity}', flush=True)
        finally: env.close()
    new = [r for r in official if r['state_index'] >= 45]
    old = [r for r in official if r['state_index'] < 45]
    assert_distinct(new, old, ('scene_sha256', 'official_state_sha256'), 'official confirmation')
    fields = ('scene_sha256', 'raw_pixels', 'numbered_pixels')
    assert_distinct(fingerprints['confirmation'], reserved+fingerprints['validation'], fields, 'settled confirmation')
    nonvalidation = [r for r in reserved if r['state_index'] not in range(30, 40)]
    assert_distinct(fingerprints['validation'], nonvalidation, fields, 'policy validation')
    for cohort, rows in cohorts.items():
        manifest = {**current, 'episodes': rows+calibration, 'cohort': cohort,
                    'state_indices': list(range(30,35) if cohort == 'validation' else range(45,50)),
                    'protocol': 'frozen policy validation' if cohort == 'validation' else 'preregistered confirmation; no selection or tuning'}
        path = root/'data'/cohort/'manifest.json'; freeze(path, manifest)
        certificate = helper.carry_calibration(current, manifest, read(c['calibration']), c['current_manifest'], path, bridge_path)
        freeze(path.parent/'calibration_reuse.json', certificate)
    freeze(root/'data/split_audit.json', {'passed': True, 'official_records': official, 'settled_records': fingerprints,
        'reserved_settled': 900, 'validation': 100, 'confirmation': 100,
        'source_sha256': sha(__file__), 'qualified_bridge_sha256': sha(bridge_path),
        'reserved_audit_sha256': sha(c['reserved_settled_audit']),
        'validation_role': 'reserved target-validation subset; never target training',
        'confirmation_role': 'unused official states 45--49, disjoint from all states 0--44'})
    print('[verification-freeze] split and calibration-reuse gates passed', flush=True)


if __name__ == '__main__': main()
