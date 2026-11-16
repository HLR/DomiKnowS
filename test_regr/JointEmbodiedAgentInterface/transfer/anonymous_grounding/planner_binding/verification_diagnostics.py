"""Post-scoring input gates and factual token/physical failure diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

if not __package__:
    root = Path(sys.argv[sys.argv.index('--root') + 1])
    sys.path.insert(0, str(root/'source'))
    __package__ = 'test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding'
from ...protocol import read, freeze, sha


def same_inputs(reports):
    if not reports or not reports[0]['input_hashes']:
        raise ValueError('missing rendered-input hashes')
    expected = reports[0]['input_hashes']
    if any(r['input_hashes'] != expected for r in reports[1:]):
        raise ValueError('training comparison changed generation inputs')
    return expected


def factual_decisions(rows, vocabulary):
    """Join each actual prefix to rule traces, without replaying another prefix."""
    tokens = defaultdict(list); traces = {}
    for row in rows:
        identity = row['episode_id']
        if 'raw_argmax_allowed' in row:
            tokens[identity].append(row)
        else:
            traces[identity] = row.get('rule_trace', {})
    for identity, decisions in tokens.items():
        by_step = defaultdict(lambda: defaultdict(set))
        for name in ('events', 'executor_events'):
            for event in traces.get(identity, {}).get(name, []):
                token = vocabulary.token_for_label(event['candidate_label'])
                by_step[event['step']][token].update(event['rejected_by'])
        for step, row in enumerate(decisions):
            reasons = {t: sorted(names) for t, names in by_step[step].items()}
            raw_reasons = reasons.get(row['raw_argmax'], []) if not row['raw_argmax_allowed'] else []
            if not row['raw_argmax_allowed'] and not raw_reasons:
                raw_reasons = ['grammar_or_productive_completion']
            yield {**row, 'step': step, 'rejected_alternatives_by_rule': reasons,
                   'raw_top_rejected_by': raw_reasons}


def failure_partition(plans, outcomes):
    if set(plans) != set(outcomes):
        raise ValueError('physical failure denominators differ')
    result = dict(generation_error=0, syntax_capability_or_precondition=0,
                  legal_wrong_object_or_destination=0,
                  abstract_success_physical_failure=0, native_success=0)
    for identity, row in plans.items():
        outcome = outcomes[identity]
        if row['prediction'] != outcome['prediction']:
            raise ValueError('execution prediction differs')
        # Native predicates are the physical outcome authority. An abstract
        # scoring disagreement must never hide a native success.
        if outcome['success']:
            result['native_success'] += 1
        elif row['generation_error']:
            result['generation_error'] += 1
        elif not row['valid_plan']:
            result['syntax_capability_or_precondition'] += 1
        elif not row['goal_success']:
            result['legal_wrong_object_or_destination'] += 1
        elif not outcome['success']:
            result['abstract_success_physical_failure'] += 1
    return result


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True)
    a = p.parse_args();root = a.root
    from .verification_analysis import runtime_for
    from .epoch_evaluation import verify_inputs
    c = read(root/'confirmation_config.json');verify_inputs(c)
    if not read(root/'validation_gate.json')['passed'] or not read(root/'control_complete.json')['passed']:
        raise ValueError('prior correctness gates incomplete')
    if not (root/'results.json').exists():raise ValueError('confirmation scoring incomplete')
    certificates = {};physical = {};disagreements = {};paths = []
    for dataset in ('vlabench', 'libero'):
        policies = ('none','shared-verified','full-metadata') if dataset == 'vlabench' else ('none','shared-verified')
        reports = [root/'control'/snapshot/policy/dataset/'report.json'
                   for snapshot in ('initialization','reproduction') for policy in policies]
        certificates['control:'+dataset] = same_inputs([read(path) for path in reports]);paths.extend(reports)
    reports = [root/'confirmation'/snapshot/policy/'libero/report.json'
               for snapshot in ('initialization','epoch-1') for policy in ('none','shared-verified','executor')]
    certificates['confirmation:libero'] = same_inputs([read(path) for path in reports]);paths.extend(reports)
    vocabulary = runtime_for(c).vlabench_vocabulary
    for snapshot in ('initialization','epoch-1'):
        for policy in ('none','shared-verified','executor'):
            folder = root/'confirmation'/snapshot/policy/'libero'
            decisions = folder/'decisions.jsonl';paths.append(decisions)
            rows = [json.loads(line) for line in decisions.read_text().splitlines()]
            target = root/'analysis/factual_decisions'/snapshot/(policy+'.jsonl');target.parent.mkdir(parents=True,exist_ok=True)
            text = ''.join(json.dumps(row)+'\n' for row in factual_decisions(rows,vocabulary))
            if target.exists() and target.read_text() != text:raise ValueError('factual decision resume differs')
            if not target.exists():target.write_text(text)
            files = [folder/'predictions.jsonl',folder/'execution/episodes.jsonl'];paths.extend(files)
            plans, outcomes = [{r['episode_id']:r for r in [json.loads(s) for s in f.read_text().splitlines()]} for f in files]
            if len(plans)!=100 or len(outcomes)!=100:raise ValueError('confirmation denominator changed')
            physical[snapshot+':'+policy] = failure_partition(plans,outcomes)
            disagreements[snapshot+':'+policy] = [i for i in plans
                if outcomes[i]['success'] and not plans[i]['goal_success']]
    freeze(root/'publication_gate.json',{'passed':True,'source_sha256':sha(__file__),
        'results_sha256':sha(root/'results.json'),'frozen_inputs':{str(f):sha(f) for f in paths},
        'matched_inputs':certificates,'failure_partition':physical,
        'native_success_abstract_disagreement':disagreements,
        'scope':'post-scoring diagnostics; no change to generation, executor, selection or policy'})


if __name__ == '__main__':main()
