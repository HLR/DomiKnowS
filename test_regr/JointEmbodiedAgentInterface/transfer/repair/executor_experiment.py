"""Run and collect the matched, exploratory Shared DFA executor comparison."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

if not __package__:
    # The host supervisor only launches containers and reads JSON artifacts.
    # Keep model dependencies inside the pinned runtime, like the prior campaign.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'transfer.repair'
    from generalization.executor_policy import CAPABILITIES, EXECUTOR, POLICIES
else:
    from ...generalization.executor_policy import CAPABILITIES, EXECUTOR, POLICIES
from ..anonymous_grounding.planner_binding.verification_control import native_command
from ..anonymous_grounding.planner_binding.verification_protocol import paired_interval
from ..campaign import run, save, utc
from ..protocol import episodes, freeze, public_input, read, sha
from .campaign import MODULE, gpu_job, model_command
from .input_protocol import IMAGE_INSTRUCTION, constraint_context
from .protocol import verify


SYSTEM = 'joint-trained'


def indexed_jsonl(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    answer = {row['episode_id']: row for row in rows}
    if len(answer) != len(rows):
        raise ValueError(f'duplicate episode IDs in {path}')
    return answer


def archived_condition(config, policy):
    return Path(config['comparison_root'])/'evaluation'/SYSTEM/policy


def preflight(root, config):
    """Verify known failures and preservation without neural inference."""
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary, plan_to_tokens
    from ...generalization.executor_policy import bind_plan_policy, pick_place_profile

    vocabulary = PlanVocabulary.default()
    manifest = episodes(read(root/'confirmation/manifest.json'))
    archived = indexed_jsonl(archived_condition(config, 'shared-verified')/'predictions.jsonl')
    unsupported, preserved, changed = [], [], []
    for episode in manifest:
        identity = episode['episode_id']
        row = archived[identity]
        public = public_input(episode)
        context = constraint_context(public)
        profile = pick_place_profile()
        cap = bind_plan_policy(CAPABILITIES, vocabulary, 12, context, profile)
        state = bind_plan_policy(EXECUTOR, vocabulary, 12, context, profile)
        labels = [vocabulary.label_for_token(token) for token in
                  plan_to_tokens(row['prediction'], public['entities'], world=cap.adapter.world)]
        is_unsupported = any(v['reason'] == 'unsupported_skill' for v in row['violations'])
        if is_unsupported:
            if cap.accepts(labels) or state.accepts(labels):
                raise ValueError('compiled capability rule accepted an unsupported archived operation')
            unsupported.append(identity)
        elif not cap.accepts(labels):
            raise ValueError('capability layer rejected an archived plan with supported operations')
        elif state.accepts(labels):
            preserved.append(identity)
        else:
            changed.append(identity)
    if len(unsupported) != 9 or changed or len(preserved) != 91:
        raise ValueError(f'archived replay differs: {len(unsupported)} unsupported, {len(preserved)} preserved, {len(changed)} other')
    freeze(root/'replay.json', {'passed': True, 'unsupported_rejected': unsupported,
                              'accepted_by_both_extensions': preserved, 'other_state_rejections': changed})


def decision_summary(path):
    choices, rejected, events = {}, Counter(), Counter()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row['episode_id']
            if 'chosen' in row:
                sequence = choices.setdefault(identity, [])
                sequence.append(row)
                if not row['raw_argmax_allowed']:
                    rejected[identity] += 1
            else:
                events.update(row.get('compiled_rule_trace', {}).get('triggered_rejections', {}))
    return choices, rejected, events


def collect(root, config):
    folders = root/'evaluation'/SYSTEM
    manifest = episodes(read(root/'confirmation/manifest.json'))
    ids = [episode['episode_id'] for episode in manifest]
    snapshots = read(root/'frozen_selection.json')['systems']
    summaries, outcomes, plans, decisions = [], {}, {}, {}
    archived_hashes = read(archived_condition(config, 'none')/'report.json')['input_hashes']
    for policy in POLICIES:
        folder = folders/policy
        report = read(folder/'report.json')
        settings = read(folder/'settings.json')
        outcomes[policy] = native = indexed_jsonl(folder/'execution/episodes.jsonl')
        plans[policy] = predicted = indexed_jsonl(folder/'predictions.jsonl')
        if (settings['checkpoint_sha256'] != snapshots[SYSTEM]['sha256']
                or report['input_hashes'] != archived_hashes or set(predicted) != set(ids) or set(native) != set(ids)):
            raise ValueError(f'{policy} changed frozen checkpoint, inputs, or episodes')
        if any(native[i]['steps'] > 1000 or native[i]['prediction'] != predicted[i]['prediction'] for i in ids):
            raise ValueError(f'{policy} execution used different plans or horizon')
        if policy in POLICIES[:2]:
            archived = indexed_jsonl(archived_condition(config, policy)/'predictions.jsonl')
            if any((predicted[i]['prediction'], predicted[i]['generation_error'])
                   != (archived[i]['prediction'], archived[i]['generation_error']) for i in ids):
                raise ValueError(f'{policy} archive predictions did not reproduce')
        choices, rejected, events = decision_summary(folder/'decisions.jsonl')
        decisions[policy] = choices
        summaries.append({'policy': policy, 'episodes': len(ids),
            'native_success': sum(row['success'] for row in native.values()),
            'completed_success': sum(row['success'] and row['termination_reason'] == 'completed'
                                     for row in native.values()),
            'valid': report['valid_percent'], 'abstract_success': report['abstract_percent'],
            'generation_errors': report['generation_errors'],
            'legal_wrong_source': sum(row['valid_plan'] and not row['grounding']['source_correct']
                                      for row in predicted.values()),
            'legal_wrong_destination': sum(row['valid_plan'] and not row['grounding']['destination_correct']
                                           for row in predicted.values()),
            'unsupported': sum(any(v['reason'] == 'unsupported_skill' for v in row['violations'])
                               for row in predicted.values()),
            'raw_top_rejections': sum(rejected.values()), 'rejected_alternatives_by_rule': events})

    preserved_ids = read(root/'replay.json')['accepted_by_both_extensions']
    for policy in (CAPABILITIES, EXECUTOR):
        if any(plans[policy][i]['labels'] != plans['shared-verified'][i]['labels'] for i in preserved_ids):
            raise ValueError(f'{policy} changed a compliant Shared greedy plan')

    paired = {}
    for before, after in zip(POLICIES, POLICIES[1:]):
        changes = []
        for i in ids:
            if plans[before][i]['prediction'] == plans[after][i]['prediction']:
                continue
            first = next(({'step': n, 'before': a['chosen'], 'after': b['chosen'],
                           'raw_top_after': b['raw_argmax'], 'rejection_after': b.get('rejection')}
                          for n, (a, b) in enumerate(zip(decisions[before].get(i, ()), decisions[after].get(i, ())))
                          if a['chosen'] != b['chosen']), None)
            changes.append({'episode_id': i, 'first_divergence': first,
                            'before_plan': plans[before][i]['prediction'], 'after_plan': plans[after][i]['prediction'],
                            'before_success': outcomes[before][i]['success'], 'after_success': outcomes[after][i]['success']})
        paired[f'{before} -> {after}'] = {'changed_plans': changes,
            'native_paired_interval': paired_interval(outcomes[before], outcomes[after], ids)}

    source = root/'vlabench'/EXECUTOR
    old_source = Path(config['source_comparison_root'])/'evaluation'/SYSTEM/'shared-verified'
    new_report, old_report = read(source/'report.json'), read(old_source/'report.json')
    new_predictions = indexed_jsonl(source/'raw_predictions.jsonl')
    old_predictions = indexed_jsonl(old_source/'raw_predictions.jsonl')
    if (len(new_predictions) != 450 or new_predictions != old_predictions
            or new_report['input_hashes'] != old_report['input_hashes']
            or sha(source/'references.json') != sha(old_source/'references.json')
            or any(new_report[key] != old_report[key] for key in ('Macro', 'progress_score', 'generation_errors'))):
        raise ValueError('strict VLABench regression differs')
    result = {'scope': 'Exploratory comparison on previously scored LIBERO states 35–39.',
              'rows': summaries, 'paired': paired, 'replay_sha256': sha(root/'replay.json'),
              'vlabench_regression': {key: new_report[key] for key in ('examples', 'Macro', 'progress_score', 'generation_errors')}}
    freeze(root/'results.json', result)
    return result


def campaign(root, config):
    manifest = root/'confirmation/manifest.json'
    calibration = root/'confirmation/calibration_reuse.json'
    gate = read(calibration)
    executor = Path(config['libero_base'])/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'
    if (gate['manifest_sha256'] != sha(manifest)
            or gate['executor_source_sha256'] != sha(executor/'fixed_executor.py')
            or gate['bridge_source_sha256'] != sha(executor/'libero_campaign.py')):
        raise ValueError('qualified executor, bridge, or manifest changed')
    checkpoint = read(root/'frozen_selection.json')['systems'][SYSTEM]['path']
    run(root, 'replay', model_command(root, config, 'replay', MODULE+'.executor_experiment',
        ['--root', root, '--replay']), {'physical_gpu': None})
    save(root/'status.json', {'stage': 'evaluating', 'updated_utc': utc()})

    def evaluation(policy, simulators):
        name = 'libero-' + policy
        output = root/'evaluation'/SYSTEM/policy
        gpu_job(root, config, name, MODULE+'.evaluate', ['--config', root/'config.json', '--arm', SYSTEM,
            '--checkpoint', checkpoint, '--manifest', manifest, '--output', output,
            '--policy', policy, '--input-protocol', IMAGE_INSTRUCTION])
        return simulators.submit(run, root, name+'-native', native_command(root, config,
            name+'-native', manifest, calibration, output/'predictions.jsonl', output/'execution'),
            {'physical_gpu': None, 'horizon': 1000})

    with ThreadPoolExecutor(max_workers=4) as models, ThreadPoolExecutor(max_workers=2) as simulators:
        source = models.submit(gpu_job, root, config, 'vlabench-regression', MODULE+'.source_test',
                               ['--config', root/'config.json', '--arm', SYSTEM, '--policy', EXECUTOR,
                                '--output', root/'vlabench'/EXECUTOR])
        tasks = [models.submit(evaluation, policy, simulators) for policy in POLICIES]
        executions, failures = [], []
        for task in as_completed([source, *tasks]):
            try:
                completed = task.result()
                if completed:
                    executions.append(completed)
            except Exception as exc:
                failures.append(str(exc))
        for execution in as_completed(executions):
            try:
                execution.result()
            except Exception as exc:
                failures.append(str(exc))
    if failures:
        raise RuntimeError(' | '.join(failures))
    collect(root, config)
    save(root/'status.json', {'stage': 'complete', 'results_sha256': sha(root/'results.json'), 'updated_utc': utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    config = read(args.root/'config.json')
    verify(config)
    if args.replay:
        preflight(args.root, config)
        return
    try:
        campaign(args.root, config)
    except Exception as exc:
        save(args.root/'status.json', {'stage': 'failed', 'error': str(exc), 'updated_utc': utc()})
        raise


if __name__ == '__main__':
    main()
