"""Freeze common source observations for matched ID-augmentation continuations.

Both arms use these same re-rendered RGB views. Source references are never read
by this renderer. Reset configuration stays private to the simulator; only the
entire declared inventory and observed segmentation determine marker bindings.
"""
import argparse
import random
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'transfer'

from .annotations import render
from .permutation import deterministic_permutations
from .protocol import digest, freeze, read, sha


def allocate(split_manifest):
    groups = split_manifest['splits']
    if {key: len(value) for key, value in groups.items()} != {'train': 3600, 'validation': 450, 'test': 450}:
        raise ValueError('historical 3600/450/450 split required')
    flat = [value for values in groups.values() for value in values]
    if len(set(flat)) != 4500 or split_manifest['seed'] != 42:
        raise ValueError('frozen source identities overlap or seed differs')
    return [{'episode_id': value, 'split': split} for split in ('train', 'validation') for value in groups[split]]


def native_reset_configuration(original):
    """Restore the explicitly saved background instead of the native empty default.

    Scene.save omits use_default_scene. Scene's constructor defaults it to True,
    which otherwise silently ignores a saved living_room/study/lab asset name.
    Native geometry must subsequently match the exact declared canonical name.
    """
    result=deepcopy(original); scene=result.get('task',result).get('scene')
    overrides={}
    if isinstance(scene,dict) and scene.get('name'):
        expected=scene['name']=='default'
        if 'use_default_scene' in scene and bool(scene['use_default_scene'])!=expected:
            raise ValueError('explicit background mode contradicts the saved scene name')
        if 'use_default_scene' not in scene:
            scene['use_default_scene']=expected;overrides['scene.use_default_scene']=expected
    return result,overrides


def observed_views(env, names, observation):
    owners = dict(env.task.entities)
    background = getattr(env.task, 'scene', None)
    if background is not None and background.name in names:
        if background.name in owners and owners[background.name] is not background:
            raise ValueError('background scene and object have the same canonical name')
        owners[background.name] = background
    if len(set(names)) != len(names) or set(names) != set(owners):
        raise ValueError('native inventory differs from the complete canonical source inventory; '
                         f'missing={sorted(set(names)-set(owners))} extra={sorted(set(owners)-set(names))}')
    rgb = np.asarray(observation['rgb']); segmentation = np.asarray(observation['segmentation'])
    if rgb.shape != (4, 480, 480, 3) or segmentation.shape != (4, 480, 480, 2):
        raise ValueError('source renderer requires all four original camera views')
    raw = Image.new('RGB', (960, 960)); centers = []
    for view in range(4):
        dx, dy = (view % 2)*480, (view//2)*480
        raw.paste(Image.fromarray(rgb[view].astype(np.uint8)), (dx, dy))
        for pointer, name in enumerate(names):
            owner = owners[name]
            geometry = owner.mjcf_model.find_all('geom') if owner is background else owner.geoms
            geoms = [int(env.physics.bind(g).element_id) for g in geometry]
            ys, xs = np.where(np.isin(segmentation[view, ..., 0], geoms))
            if len(xs) >= 8:
                centers.append((pointer, dx+int(np.median(xs)), dy+int(np.median(ys))))
    if not centers:
        raise ValueError('no observed canonical geometry; no reference-derived fallback')
    return raw, centers


def prepare(config, output, shard, shards):
    import signal
    from test_regr.VLABenchAgentInterface.dataset import _extract_entities
    from test_regr.VLABenchAgentInterface.simulation.environment import create_environment
    records = allocate(read(config['split_manifest']))
    selected = records[shard::shards]; result = []
    def timeout(_signal, _frame):
        raise TimeoutError('native source rendering exceeded 180 seconds; stop without filtering')
    signal.signal(signal.SIGALRM, timeout)
    output.mkdir(parents=True, exist_ok=True)
    freeze(output/'settings.json', {'split_sha256': sha(config['split_manifest']),
        'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'config_sha256': digest(config), 'shard': shard, 'shards': shards,
        'format': 'four re-rendered RGB views, numeric markers; identical format in both arms',
        'reset': 'native configuration, zero warm-up steps; no actions or success-based resets'})
    for index, record in enumerate(selected):
        identity = record['episode_id']; original = Path(config['planning_dir'])/identity
        configs = sorted((original/'env_config').glob('*.json'))
        if len(configs) != 1: raise ValueError('unambiguous native scene configuration required')
        scene_path = configs[0]; scene,overrides = native_reset_configuration(read(scene_path)); task = scene.get('task', scene)
        names = tuple(_extract_entities({k: task[k] for k in ('components', 'scene', 'entities') if k in task}))
        folder = output/'episodes'/digest(identity)[:24]; path = folder/'record.json'
        if path.exists():
            saved = read(path)
            if saved['scene_sha256'] != sha(scene_path) or any(sha(p) != h for p, h in saved['files'].items()):
                raise ValueError('resumed source observation or native scene differs')
            result.append(saved); continue
        seed = int(digest([42, identity, 'source-view-v1'])[:8], 16)
        # Native reset may already satisfy its condition. Skipping warm-up avoids
        # recursive success-triggered resets and preserves the saved initial pose.
        print(f'[source-views] starting shard={shard} {index+1}/{len(selected)} {identity}', flush=True)
        env = None; signal.alarm(180)
        try:
            random.seed(seed); np.random.seed(seed)
            env = create_environment(original.parent.name, episode_config=scene, run_mode='efficient',
                                     simulator_seed=seed, reset_wait_step=0)
            before_time = float(env.physics.time()); before_qpos = env.physics.data.qpos.copy()
            observation = env.get_observation(require_pcd=False)
            raw, centers = observed_views(env, names, observation)
            if float(env.physics.time()) != before_time or not np.array_equal(before_qpos, env.physics.data.qpos):
                raise ValueError('observation rendering advanced or changed the simulation')
            folder.mkdir(parents=True); raw.save(folder/'raw.png')
            freeze(folder/'private_native_configuration.json',scene)
            render(raw, centers, style='libero-v3').save(folder/'canonical.png')
            orders = deterministic_permutations(identity, len(names), seed=42, trials=4)
            augmentations = []
            for epoch, permutation in enumerate(orders):
                image_path = folder/f'epoch_{epoch:03d}.png'
                render(raw, centers, style='libero-v3', permutation=permutation).save(image_path)
                augmentations.append({'epoch': epoch, 'new_to_old': list(permutation.order), 'image': str(image_path)})
            saved = {**record, 'canonical_entities': list(names), 'centers': centers,
                'scene_path': str(scene_path), 'scene_sha256': sha(scene_path), 'simulator_seed': seed,
                'native_reset_overrides':overrides,
                'simulation_time': before_time, 'render_side_effect_free': True,
                'canonical_image': str(folder/'canonical.png'), 'augmentations': augmentations,
                'files': {str(p): sha(p) for p in [*folder.glob('*.png'),folder/'private_native_configuration.json']}}
            freeze(path, saved); result.append(saved)
        except Exception as exc:
            freeze(output/'failure.json', {'episode_id':identity,'scene_sha256':sha(scene_path),
                'error_type':type(exc).__name__,'error':str(exc),'episode_filtered':False})
            raise
        finally:
            signal.alarm(0)
            if env is not None: env.close()
        print(f'[source-views] shard={shard} {index+1}/{len(selected)} {identity}', flush=True)
    freeze(output/'manifest.json', {'episodes': result, 'complete': True})


def finalize(config, output, shards):
    expected = allocate(read(config['split_manifest'])); rows = []
    for shard in range(shards):
        proof = read(output/f'shard_{shard:03d}'/'manifest.json')
        if not proof['complete']: raise ValueError('source shard incomplete')
        rows.extend(proof['episodes'])
    indexed = {r['episode_id']: r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {r['episode_id'] for r in expected}:
        raise ValueError('source image coverage differs; no filtering of failed episodes')
    for row in rows:
        if sha(row['scene_path']) != row['scene_sha256'] or any(sha(p) != h for p, h in row['files'].items()):
            raise ValueError('source scene or image hash differs')
    freeze(output/'manifest.json', {'episodes': [indexed[r['episode_id']] for r in expected], 'complete': True,
        'split_sha256': sha(config['split_manifest']), 'train': 3600, 'validation': 450,
        'test_used': False, 'reference_plans_read': False, 'common_observation_format': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards: parser.error('invalid shard allocation')
    config = read(args.config)
    if args.finalize: finalize(config, args.output, args.shards)
    else: prepare(config, args.output/f'shard_{args.shard:03d}', args.shard, args.shards)


if __name__ == '__main__':
    main()
