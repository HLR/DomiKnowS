"""Validate complete paired transfers before publishing any success claims."""
import argparse
from collections import Counter
from pathlib import Path
import random
from .prepare import prepare
from .protocol import POLICIES, digest, episodes, freeze, indexed, jsonl, read, sha


def wilson(successes, count):
    z = 1.96; p = successes/count; den = 1+z*z/count
    mid = (p+z*z/(2*count))/den
    half = z*((p*(1-p)/count+z*z/(4*count*count))**.5)/den
    return [max(0., mid-half), min(1., mid+half)]


def roles(plan):
    value = {'source': [], 'destination': []}
    for op in plan if isinstance(plan, list) else []:
        if not isinstance(op, dict) or op.get('name') not in ('pick', 'place'):
            continue
        key, parameter = (('source', 'target_entity_name') if op['name'] == 'pick'
                          else ('destination', 'target_container_name'))
        params = op.get('params', {})
        pointer = params.get(parameter) if isinstance(params, dict) else None
        value[key].append(pointer if type(pointer) is int else None)
    return value


def reference_roles(private, dataset):
    if dataset == 'rlbench':
        return roles(private['reference'])
    # Scoring only: independently held BDDL relations, never generation inputs.
    return {'source': [private['entities'].index(g[1]) for g in private['goals']],
            'destination': [private['entities'].index(g[2]) for g in private['goals']]}


def summarize(root):
    root = Path(root)
    proof = read(root/'campaign_manifest.json'); config = proof['configuration']
    # Recheck every frozen dataset, checkpoint, source and calibration hash.
    prepare(root/'config.json', root)
    result = {'scope': 'Independent semantic-prompt transfer from fixed metadata epoch-4; 100 episodes per dataset/condition',
              'campaign_manifest_sha256': sha(root/'campaign_manifest.json'), 'rows': [], 'paired': []}
    for dataset in ('libero', 'rlbench'):
        records = episodes(read(config['datasets'][dataset]['manifest']))
        expected = [e['episode_id'] for e in records]
        refs = {e['episode_id']: reference_roles(read(e['scoring_private']), dataset) for e in records}
        conditions = {}
        for method in ('baseline', 'semantic-prompt'):
            reports = {}
            for policy in POLICIES:
                folder = root/dataset/method/policy
                planning = indexed(jsonl(folder/'predictions.jsonl'), expected)
                native = indexed(jsonl(folder/'execution/episodes.jsonl'), expected)
                summary = read(folder/'execution/summary.json')
                if summary['episodes'] != 100 or summary['predictions_sha256'] != sha(folder/'predictions.jsonl'):
                    raise ValueError('execution predictions or coverage differ')
                if any(native[k]['prediction'] != planning[k]['prediction'] for k in expected):
                    raise ValueError('executed prediction differs from generated plan')
                for suffix in ('planning', policy+'-execution'):
                    if read(root/'jobs'/f'{dataset}-{method}-{suffix}.json')['exit_code'] != 0:
                        raise ValueError('job did not finish successfully')
                report = read(folder/'report.json'); reports[policy] = report
                if report['input_hashes'] != {k: r['input_hash'] for k, r in planning.items()}:
                    raise ValueError('reported input hashes differ from predictions')
                for index, record in enumerate(records):
                    rendered = read(folder.parent/'rendered_inputs'/f'{index:03d}.json')
                    if rendered['episode_id'] != record['episode_id'] or digest({
                        k: v for k, v in rendered.items() if k != 'episode_id'}) != planning[record['episode_id']]['input_hash']:
                        raise ValueError('saved rendered prompt differs from prediction fingerprint')
                success = sum(bool(r['success']) for r in native.values())
                completed = sum(r['success'] and r['termination_reason'] == 'completed'
                                and r.get('holding_at_end') is None for r in native.values())
                row = {'dataset': dataset, 'method': method, 'policy': policy, 'n': 100,
                       'native_successes': success, 'native_ci95': wilson(success, 100),
                       'completed_successes': completed, 'valid_plan_rate': report['valid_plan_rate'],
                       'generation_errors': report['generation_errors'],
                       'termination_counts': dict(Counter(r['termination_reason'] for r in native.values())),
                       'triggered_rejections': report['triggered_rejections'],
                       'applicable_candidate_events': dict(sum((Counter(r['rule_trace']['applicable_events'])
                            for r in planning.values()), Counter())),
                       'source_role_exact_rate': sum(roles(planning[k]['prediction'])['source'] == refs[k]['source'] for k in expected)/100,
                       'destination_role_exact_rate': sum(roles(planning[k]['prediction'])['destination'] == refs[k]['destination'] for k in expected)/100}
                if dataset == 'libero': row['abstract_goal_success'] = report['abstract_goal_success']
                result['rows'].append(row); conditions[method, policy] = (planning, native)
            if reports['none']['input_hashes'] != reports['shared-verified']['input_hashes']:
                raise ValueError('constraint conditions have different rendered inputs')
        for policy in POLICIES:
            bp, bn = conditions['baseline', policy]; mp, mn = conditions['semantic-prompt', policy]
            ids = sorted(expected); changes = [int(mn[k]['success'])-int(bn[k]['success']) for k in ids]
            rng = random.Random(42)
            samples = sorted(sum(changes[rng.randrange(100)] for _ in range(100)) for _ in range(10000))
            result['paired'].append({'dataset': dataset, 'policy': policy,
                'native_gain_pp': sum(changes), 'bootstrap95_pp': [samples[249], samples[9749]],
                'identical_predictions': sum(bp[k]['prediction'] == mp[k]['prediction'] for k in ids),
                'identical_prediction_native_discordance': sum(bp[k]['prediction'] == mp[k]['prediction']
                    and bn[k]['success'] != mn[k]['success'] for k in ids),
                'native_gains': sum(v == 1 for v in changes), 'native_losses': sum(v == -1 for v in changes),
                'examples': [{'episode_id': k, 'baseline_prediction': bp[k]['prediction'],
                    'method_prediction': mp[k]['prediction'], 'reference_roles': refs[k],
                    'baseline_native_success': bn[k]['success'], 'method_native_success': mn[k]['success'],
                    'method_termination': mn[k]['termination_reason']}
                    for k in [k for k in ids if bp[k]['prediction'] != mp[k]['prediction']][:3]]})
    freeze(root/'results.json', result)
    lines = ['# Semantic prompt transfer', '', result['scope'], '',
             '| Dataset | Method | Constraints | Native success | Completed success | Valid plans |',
             '|---|---|---|---:|---:|---:|']
    for row in result['rows']:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['policy']} | {row['native_successes']}/100 | "
                     f"{row['completed_successes']}/100 | {100*row['valid_plan_rate']:.1f}% |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)


if __name__ == '__main__':
    main()
