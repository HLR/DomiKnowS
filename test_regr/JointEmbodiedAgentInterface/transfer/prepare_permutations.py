"""Materialize five pixel-verified ID perturbations without reading gold labels."""
import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'transfer'

from .annotations import write_counterfactual
from .permutation import deterministic_permutations
from .protocol import digest, episodes, freeze, indexed, public_input, read, sha


def prepare(config, dataset, bindings_path, output, trials=5):
    original_manifest = config['datasets'][dataset]['manifest']
    originals = episodes(read(original_manifest))
    bindings = read(bindings_path)
    if bindings.get('verified') is not True:
        raise ValueError('all original image bindings must pass before perturbation')
    bound = indexed(bindings['episodes'], [e['episode_id'] for e in originals])
    freeze(output/'settings.json', {'dataset': dataset, 'trials': trials, 'seed': config['seed'],
        'original_manifest_sha256': sha(original_manifest), 'bindings_sha256': sha(bindings_path),
        'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}})
    manifests = [[] for _ in range(trials)]
    for episode in originals:
        if any(sha(path) != expected for path, expected in episode['files'].items()):
            raise ValueError('original frozen episode or reference changed before perturbation')
        public = public_input(episode); binding = bound[episode['episode_id']]
        if binding['raw_sha256'] != sha(public['images'][0]) or binding['numbered_sha256'] != sha(public['images'][1]):
            raise ValueError('verified image pair differs from frozen episode')
        orders = deterministic_permutations(episode['episode_id'], len(public['entities']), seed=config['seed'], trials=trials)
        reference_hash = sha(episode['scoring_private'])  # hash only, never parse labels for a transformation
        for trial, perm in enumerate(orders):
            folder = output/f'trial_{trial:03d}'/'episodes'/digest(episode['episode_id'])[:24]
            folder.mkdir(parents=True, exist_ok=True)
            record_path = folder/'record.json'
            if record_path.exists():
                record = read(record_path)
                if record['new_to_old'] != list(perm.order) or record['original_reference_sha256'] != reference_hash:
                    raise ValueError('resumed permutation or frozen reference differs')
                if any(sha(path) != h for path, h in record['files'].items()):
                    raise ValueError('resumed permutation artifact differs')
                manifests[trial].append(record); continue
            proof = write_counterfactual(*public['images'], binding['centers'], style=binding['style'],
                permutation=perm, output=folder/'numbered.png')
            changed = perm.public(public)
            changed['images'] = [public['images'][0], str(folder/'numbered.png')]
            freeze(folder/'generation_input.json', changed)
            record = {**episode, 'generation_input': str(folder/'generation_input.json'),
                'canonical_generation_input': episode['generation_input'],
                'new_to_old': list(perm.order), 'old_to_new': list(perm.old_to_new),
                'trial': trial, 'original_reference_sha256': reference_hash,
                'annotation_proof': proof,
                'files': {**episode['files'], str(folder/'generation_input.json'): sha(folder/'generation_input.json'),
                          str(folder/'numbered.png'): sha(folder/'numbered.png')}}
            freeze(record_path, record); manifests[trial].append(record)
        print(f'[transfer-permutations] {dataset} {len(manifests[0])}/100', flush=True)
    for trial, records in enumerate(manifests):
        if len(records) != 100: raise ValueError('incomplete permutation coverage')
        freeze(output/f'trial_{trial:03d}'/'manifest.json', {'episodes': records,
            'dataset': dataset, 'trial': trial, 'original_manifest_sha256': sha(original_manifest),
            'protocol': 'same physical scene and frozen references; labels, pointers and observed bindings permuted together'})
    freeze(output/'manifest.json', {'dataset': dataset, 'trials': trials, 'episodes_per_trial': 100,
        'physical_episodes': 100, 'manifest_sha256': {str(i): sha(output/f'trial_{i:03d}'/'manifest.json') for i in range(trials)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', choices=['libero', 'rlbench'], required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(read(args.config), args.dataset, args.bindings, args.output)


if __name__ == '__main__':
    main()
