"""Freeze the independent-method campaign and its existing evaluation assets."""
import argparse
from pathlib import Path
from .protocol import METHODS, POLICIES, episodes, freeze, indexed, jsonl, read, sha


def prepare(config_path, output):
    config = read(config_path)
    if config['seed'] != 42 or config['epochs'] != 4 or config['batch_size'] != 8:
        raise ValueError('approved campaign training settings differ')
    proof = {'configuration': config, 'checkpoint_sha256': sha(config['checkpoint']),
             'methods': list(METHODS), 'datasets': {}, 'independent_methods': True}
    source = Path(config['source'])
    proof['source_sha256'] = {str(p.relative_to(source)): sha(p)
        for name in ('JointEmbodiedAgentInterface', 'VLABenchAgentInterface')
        for p in sorted((source/'test_regr'/name).rglob('*.py'))}
    for dataset in ('libero', 'rlbench'):
        item = config['datasets'][dataset]
        selected = episodes(read(item['manifest']))
        expected = [e['episode_id'] for e in selected]
        for episode in read(item['manifest'])['episodes']:
            for path, expected_hash in episode['files'].items():
                if sha(path) != expected_hash:
                    raise ValueError('frozen dataset hash differs: ' + path)
        gate = read(item['calibration'])
        if gate['episodes'] != 100 or gate['success_rate'] < .9 or gate['manifest_sha256'] != sha(item['manifest']):
            raise ValueError('retained executor calibration invalid')
        source = Path(item['executor_source'])
        fields = ({'executor_source_sha256': 'fixed_executor.py', 'bridge_source_sha256': 'libero_campaign.py'}
                  if dataset == 'libero' else {'executor_sha256': 'rlbench_executor.py',
                       'bridge_sha256': 'rlbench_bridge.py', 'runner_sha256': 'rlbench_campaign.py'})
        for key, name in fields.items():
            if sha(source/name) != gate[key]:
                raise ValueError('retained executor source differs: ' + name)
        if dataset == 'rlbench' and (source/'rlbench_planning.py').exists():
            raise ValueError('unqualified replacement must not be used in retained-executor campaign')
        if dataset == 'rlbench' and 'reset_validation' in gate:
            if sha(source/'rlbench_reset_validation.py') != gate['reset_validation']['validator_sha256']:
                raise ValueError('reset verifier changed')
        archives = {}
        for policy in POLICIES:
            path = Path(item['archive'])/policy/'predictions.jsonl'
            indexed(jsonl(path), expected)
            archives[policy] = sha(path)
        proof['datasets'][dataset] = {'manifest_sha256': sha(item['manifest']),
            'calibration_sha256': sha(item['calibration']), 'archive_predictions_sha256': archives,
            'episode_ids': expected, 'executor_source_sha256': {name: sha(source/name) for name in fields.values()}}
    freeze(Path(output)/'campaign_manifest.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)


if __name__ == '__main__':
    main()
