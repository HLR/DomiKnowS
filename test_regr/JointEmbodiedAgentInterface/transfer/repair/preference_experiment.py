"""Compare frozen executor DFA with and without the visual source preference."""

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'transfer.repair'

from ..anonymous_grounding.planner_binding.verification_control import native_command
from ..anonymous_grounding.planner_binding.verification_protocol import paired_interval
from ..campaign import run, save, utc
from ..protocol import episodes, freeze, read, sha
from .campaign import MODULE, gpu_job
from .protocol import verify


def indexed(path):
    with path.open() as stream:
        records = [json.loads(line) for line in stream]
    result = {row['episode_id']: row for row in records}
    if len(records) != len(result):
        raise ValueError(f'duplicate records in {path}')
    return result


def collect(root, baseline, output, identities):
    before, after = (indexed(folder/'predictions.jsonl') for folder in (baseline, output))
    native_before, native_after = (indexed(folder/'execution/episodes.jsonl') for folder in (baseline, output))
    for prediction, native in ((before, native_before), (after, native_after)):
        if set(identities) != set(prediction) or set(identities) != set(native):
            raise ValueError('prediction and execution identities must match the frozen manifest')
        if any(native[i]['steps'] > 1000 or native[i]['prediction'] != prediction[i]['prediction'] for i in identities):
            raise ValueError('native execution did not use the recorded prediction or horizon')

    before_settings, after_settings = (read(folder/'settings.json') for folder in (baseline, output))
    for key in ('checkpoint_sha256', 'manifest_sha256', 'policy', 'input_protocol',
                'executor_profile', 'compiled_rules', 'graph_budget', 'quant_matmul_policy', 'horizon'):
        if before_settings[key] != after_settings[key]:
            raise ValueError(f'preference changed the matched policy contract field {key}')
    for i in identities:
        if before[i]['input_hash'] != after[i]['input_hash']:
            raise ValueError(f'preference changed the model input for {i}')

    evidence = [read(path) for path in sorted((output/'preference_inputs').glob('*.json'))]
    if len(evidence) != len(identities) or {row['episode_id'] for row in evidence} != set(identities):
        raise ValueError('incomplete preference evidence')
    applied = {}
    with (output/'decisions.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('log_bias'):
                if not row['log_bias'].keys() <= set(row['allowed']):
                    raise ValueError('preference attempted to change a forbidden choice')
                applied.setdefault(row['episode_id'], []).append({
                    'prefix': row['prefix'], 'chosen': row['chosen'], 'log_bias': row['log_bias']})

    changes = [{'episode_id': i, 'before': before[i]['prediction'], 'after': after[i]['prediction'],
                'native_before': native_before[i]['success'], 'native_after': native_after[i]['success'],
                'preference_decisions': applied.get(i, [])}
               for i in identities if before[i]['prediction'] != after[i]['prediction']]
    if any(not row['preference_decisions'] for row in changes):
        raise ValueError('a prediction changed without an applied preference')

    summaries = []
    for name, prediction, native in (('executor DFA', before, native_before),
                                     ('executor DFA + visual source preference', after, native_after)):
        summaries.append({'condition': name, 'episodes': len(identities),
                          'physical_success': sum(row['success'] for row in native.values()),
                          'abstract_success': sum(row['goal_success'] for row in prediction.values()),
                          'valid': sum(row['valid_plan'] for row in prediction.values()),
                          'source_correct': sum(row['grounding']['source_correct'] for row in prediction.values()),
                          'generation_errors': sum(bool(row['generation_error']) for row in prediction.values())})
    result = {'scope': 'Exploratory evaluation on previously scored LIBERO states 35–39.',
              'baseline_execution': 'Reused verified matched executor-DFA archive.',
              'rows': summaries, 'changed_predictions': changes,
              'evidence_episodes': sum(bool(row['bound_evidence']) for row in evidence),
              'biased_episodes': len(applied),
              'physical_paired_interval': paired_interval(native_before, native_after, identities)}
    freeze(root/'results.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    c = read(root/'config.json')
    try:
        verify(c)
        manifest = root/'confirmation/manifest.json'
        calibration = root/'confirmation/calibration_reuse.json'
        baseline = Path(c['preference_baseline'])
        output = root/'evaluation'
        checkpoint = read(root/'frozen_selection.json')['systems']['joint-trained']['path']
        identities = [e['episode_id'] for e in episodes(read(manifest))]
        gate = read(calibration)
        executor = Path(c['libero_base'])/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'
        if (gate['manifest_sha256'] != sha(manifest)
                or gate['executor_source_sha256'] != sha(executor/'fixed_executor.py')
                or gate['bridge_source_sha256'] != sha(executor/'libero_campaign.py')):
            raise ValueError('qualified executor, bridge, or manifest changed')
        save(root/'status.json', {'stage': 'evaluating-preference', 'updated_utc': utc()})
        gpu_job(root, c, 'visual-source', MODULE+'.evaluate', ['--config', root/'config.json',
                '--arm', 'joint-trained', '--checkpoint', checkpoint, '--manifest', manifest,
                '--output', output, '--policy', 'shared-verified-executor', '--preference', 'visual-source'])
        save(root/'status.json', {'stage': 'native-execution', 'updated_utc': utc()})
        run(root, 'visual-source-native', native_command(root, c, 'visual-source-native', manifest,
            calibration, output/'predictions.jsonl', output/'execution'),
            {'physical_gpu': None, 'horizon': 1000})
        collect(root, baseline, output, identities)
        save(root/'status.json', {'stage': 'complete', 'results_sha256': sha(root/'results.json'), 'updated_utc': utc()})
    except Exception as exc:
        save(root/'status.json', {'stage': 'failed', 'error': str(exc), 'updated_utc': utc()})
        raise


if __name__ == '__main__':
    main()
