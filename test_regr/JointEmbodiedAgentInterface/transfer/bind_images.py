"""Recover marker bindings from observed geometry, never scoring-private labels."""
import argparse
import importlib
from pathlib import Path
import sys
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'transfer'

from .annotations import verify, recover_centers
from .protocol import episodes, freeze, public_input, read, sha


def match_geometry(public, centers, movable, tolerance=1e-4):
    names = public['entities']
    if len(names) != len(centers) or len(centers) != len(movable):
        raise ValueError('observed entity inventory coverage differs')
    matches = []
    for name in names:
        candidates = [i for i, point in enumerate(centers)
            if bool(movable[i]) == (name in public['movable'])
            and np.max(np.abs(np.asarray(point)-np.asarray(public['positions'][name]))) <= tolerance]
        if len(candidates) != 1:
            raise ValueError('ambiguous or unmatched observed geometry; no reference fallback')
        matches.append(candidates[0])
    if len(set(matches)) != len(names):
        raise ValueError('observed geometry binding is not a bijection')
    return matches


def declared_libero_entities(path, protocol):
    groups = {g[0].lower(): g[1:] for g in protocol.sexpr(Path(path).read_text()) if isinstance(g, list)}
    # Objects and fixtures alone determine the mapping. Goals, task language,
    # target regions and objects of interest never select or order entities.
    return sorted({**protocol.typed_entities(groups[':objects']),
                   **protocol.typed_entities(groups.get(':fixtures', []))})


def bind_libero(config, selected, output):
    source = Path(config['datasets']['libero']['executor_source'])
    sys.path.insert(0, str(source))
    native = importlib.import_module('libero_campaign')
    protocol = importlib.import_module('libero_protocol')
    env = None; current = None; bindings = []
    try:
        for index, record in enumerate(selected):
            public = public_input(record)
            names = declared_libero_entities(record['bddl'], protocol)
            if len(names) != len(public['entities']): raise ValueError('all-entity declaration differs')
            if current != record['bddl']:
                if env is not None: env.close()
                env = native.environment(record['bddl'], render=True); current = record['bddl']
            obs = native.reset_episode(env, record)
            segmentation = np.flipud(obs['agentview_segmentation_instance']).squeeze()
            centers = []
            for tag, name in env.segmentation_id_mapping.items():
                if name not in names: continue
                ys, xs = np.where(segmentation == tag+1)
                if len(xs) >= 8: centers.append((names.index(name), int(np.median(xs)), int(np.median(ys))))
            if sorted(c[0] for c in centers) != sorted(public['visible_entity_ids']):
                raise ValueError('visible entity bindings differ from frozen observations')
            try:
                proof = verify(*public['images'], centers, style='libero-v3', entity_count=len(names))
            except ValueError:
                proof = recover_centers(*public['images'], centers, style='libero-v3', entity_count=len(names))
            freeze(output/f'{index:03d}.json', {'episode_id': record['episode_id'], **proof})
            bindings.append({'episode_id': record['episode_id'], **proof})
            print(f'[transfer-bindings] libero {index+1}/{len(selected)}', flush=True)
    finally:
        if env is not None: env.close()
    return bindings


def bind_rlbench(config, selected, output):
    from pyrep.objects.shape import Shape
    source = Path(config['datasets']['rlbench']['executor_source'])
    sys.path.insert(0, str(source))
    native = importlib.import_module('rlbench_campaign')
    bridge = native.RLBenchBridge(render=True); bindings = []
    try:
        for index, record in enumerate(selected):
            public = public_input(record)
            # Rebuild the entire scene inventory from geometry; do not read the
            # scoring-private entity table or reference plan.
            bridge.reset(record['task'], record['variation'], record['seed'], state=read(record['state']))
            observed = [bridge.geometry(i)['center'] for i in range(len(bridge.entities))]
            match = match_geometry(public, observed, [e['movable'] for e in bridge.entities])
            obs = bridge.task.get_observation(); centers = []
            for canonical, actual in enumerate(match):
                handles = [Shape(name).get_handle() for name in bridge.entities[actual]['parts']]
                ys, xs = np.where(np.isin(obs.front_mask, handles))
                if len(xs) >= 8: centers.append((canonical, int(np.median(xs)), int(np.median(ys))))
            if sorted(c[0] for c in centers) != sorted(public['visible_entity_ids']):
                raise ValueError('visible entity bindings differ from frozen observations')
            proof = verify(*public['images'], centers, style='rlbench-v33', entity_count=len(match))
            freeze(output/f'{index:03d}.json', {'episode_id': record['episode_id'], **proof})
            bindings.append({'episode_id': record['episode_id'], **proof})
            print(f'[transfer-bindings] rlbench {index+1}/{len(selected)}', flush=True)
    finally:
        bridge.close()
    return bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', choices=['libero', 'rlbench'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); config = read(args.config)
    selected = episodes(read(config['datasets'][args.dataset]['manifest']))
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output/'settings.json', {'config_sha256': sha(args.config),
        'source_sha256': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'dataset': args.dataset, 'manifest_sha256': sha(config['datasets'][args.dataset]['manifest'])})
    bindings = (bind_libero if args.dataset == 'libero' else bind_rlbench)(config, selected, args.output)
    freeze(args.output/'manifest.json', {'episodes': bindings, 'verified': True,
        'protocol': 'pixel-exact canonical marker reconstruction; observational bindings only'})


if __name__ == '__main__':
    main()
