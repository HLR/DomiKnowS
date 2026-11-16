"""Recover an EAI search-budget failure without changing its accepted language."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import ast
import copy
import csv
import json
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'transfer.anonymous_grounding.planner_binding'
from ...protocol import read, freeze, sha, episodes


def canonical_eai_record(record):
    """Compare all semantic facts while ignoring Python set repr ordering."""
    value = copy.deepcopy(record)
    for key in ('predicted_state', 'gold_state', 'initial_state'):
        raw = value.get('goal', {}).get(key)
        if isinstance(raw, str):
            facts = set() if raw == 'set()' else ast.literal_eval(raw)
            if not isinstance(facts, set): raise ValueError('expected a serialized EAI fact set')
            value['goal'][key] = sorted(facts)
    return value


@contextmanager
def retry_eai_search(runtime, records, *, maximum=1000000):
    from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
    original = runtime.dfa_for; patched = []
    def factory(domain, context):
        policy = original(domain, context)
        if domain != 'eai' or any(p is policy for p, _, _ in patched): return policy
        search = getattr(policy, '_reachability', None)
        if search is None: return policy
        allowed = policy.allowed_tokens; initial = search.max_state_expansions
        def bounded(state, remaining_steps=None):
            while True:
                try: return allowed(state, remaining_steps=remaining_steps)
                except SemanticSearchLimitExceeded:
                    before = search.max_state_expansions
                    if before >= maximum: raise
                    search.max_state_expansions = min(maximum, before*10)
                    event = {'old_cap': before, 'new_cap': search.max_state_expansions,
                             'remaining_steps': remaining_steps, 'hard_constraints_relaxed': False}
                    records.append(event)
                    print('[eai-search-retry] '+json.dumps(event), flush=True)
        patched.append((policy, allowed, initial)); policy.allowed_tokens = bounded
        return policy
    runtime.dfa_for = factory
    try: yield
    finally:
        runtime.dfa_for = original
        for policy, allowed, cap in patched:
            policy.allowed_tokens = allowed; policy._reachability.max_state_expansions = cap


def evaluation():
    from .... import main as joint
    from . import epoch_evaluation
    out = Path(sys.argv[sys.argv.index('--experiment-output')+1])
    freeze(out/'recovery.json', {'module_sha256': sha(__file__), 'original_cap': 10000,
        'retry_caps': [100000, 1000000], 'constraint_language_changed': False,
        'policy': 'retry exact reachability queries only; fail on unknown after final cap'})
    original = joint._evaluate_eai
    def recovered(planner, runtime, examples, **kwargs):
        records = []
        with retry_eai_search(runtime, records): report = original(planner, runtime, examples, **kwargs)
        freeze(out/'eai_search_retries.json', records)
        return {**report, 'exact_search_budget_retries': len(records), 'hard_constraints_relaxed': False}
    joint._evaluate_eai = recovered
    try: epoch_evaluation.evaluate()
    finally: joint._evaluate_eai = original


def collect(root, config, retry_folder):
    """Coverage checks also include the original successful EAI prefix."""
    from .epoch_campaign import POLICIES
    expected = {e['episode_id'] for e in episodes(read(config['external_manifest']))}
    rows = []; common = {}; references = None
    for epoch in range(1, 5):
        base = root/'evaluation'/f'epoch_{epoch:02d}'
        def folder(policy): return retry_folder if epoch == 3 and policy == 'none' else base/policy
        eai = read(folder('none')/'eai.json')
        if eai['examples'] != 88: raise ValueError('incomplete EAI result')
        for policy in POLICIES:
            out = folder(policy); vla = read(out/'vlabench/report.json')
            if vla['examples'] != 450: raise ValueError('incomplete VLABench result')
            refs = read(out/'references.json')
            if references is None: references = refs
            if refs != references: raise ValueError('references changed across conditions')
            for dataset in ('vlabench', 'libero'):
                if dataset == 'libero' and policy == 'full-metadata': continue
                hashes = read(out/dataset/'report.json')['input_hashes']
                if dataset not in common: common[dataset] = hashes
                if hashes != common[dataset]: raise ValueError('rendered inputs differ')
            row = {'epoch': epoch, 'constraint_policy': policy,
                   'eai_goal_percent': eai['goal_success']*100 if policy == 'none' else None,
                   'vla_macro': vla['Macro'], 'progress_score': vla['progress_score'],
                   'libero_abstract_percent': None, 'libero_physical_percent': None}
            if policy != 'full-metadata':
                lib = out/'libero'; report = read(lib/'report.json'); physical = read(lib/'execution/summary.json')
                plans = [json.loads(s) for s in (lib/'predictions.jsonl').read_text().splitlines()]
                native = [json.loads(s) for s in (lib/'execution/episodes.jsonl').read_text().splitlines()]
                p = {r['episode_id']: r for r in plans}; n = {r['episode_id']: r for r in native}
                if len(plans) != 100 or len(native) != 100 or set(p) != expected or set(n) != expected or physical['episodes'] != 100:
                    raise ValueError('native execution coverage differs')
                if any(p[k]['prediction'] != n[k]['prediction'] for k in expected): raise ValueError('executed plan differs')
                row.update(libero_abstract_percent=report['abstract_goal_success']*100,
                           libero_physical_percent=physical['success_rate']*100,
                           libero_success_ci95=physical['success_ci95'])
            rows.append(row)
    previous = [json.loads(s) for s in (root/'evaluation/epoch_03/none/eai_predictions.jsonl').read_text().splitlines()]
    recovered = [json.loads(s) for s in (retry_folder/'eai_predictions.jsonl').read_text().splitlines()]
    if (len(previous) != 70 or len(recovered) != 88 or
            [canonical_eai_record(r) for r in previous] !=
            [canonical_eai_record(r) for r in recovered[:len(previous)]]):
        raise ValueError('recovered EAI evaluation changed archived successful prefix')
    freeze(root/'results.json', {'rows': rows, 'config_sha256': sha(root/'config.json'),
        'coverage_passed': True, 'identical_inputs_passed': True, 'identical_references_passed': True,
        'eai_successful_prefix_reproduced': len(previous), 'eai_fact_sets_compared_without_repr_order': True,
        'epoch3_recovery': str(retry_folder),
        'eai_search_retries': read(retry_folder/'eai_search_retries.json'),
        'selection': 'all epochs diagnostic; original source-loss best unchanged', 'libero_states': [40, 41, 42, 43, 44]})
    columns = ['epoch', 'constraint_policy', 'eai_goal_percent', 'vla_macro', 'progress_score',
               'libero_abstract_percent', 'libero_physical_percent']
    with (root/'table_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    labels = {'none': 'No DFA', 'shared-verified': 'Shared verified', 'full-metadata': 'Full metadata'}
    lines = ['# Masked CE: four operation-binding continuation epochs', '',
        '| Epoch | Constraint policy | EAI goal (%) | VLA Macro (0–80) | Progress Score (0–100) | LIBERO abstract (%) | LIBERO physical (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values = ['—' if row[k] is None else f'{row[k]:.2f}' for k in columns[2:]]
        lines.append('| '+str(row['epoch'])+' | '+labels[row['constraint_policy']]+' | '+' | '.join(values)+' |')
    lines += ['', 'Fresh LIBERO states 40–44; 450 internal VLABench test cases; 88 canonical EAI validation cases. '
        'Fixed calibrated executor, horizon 1000. All epochs are diagnostic, without test-based selection. '
        'Epoch 3 EAI retried exact semantic reachability with larger computational caps; hard constraints unchanged, '
        'and all 70 archived prefix predictions and semantic scores reproduced (fact-set repr order ignored).']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def host(root):
    from concurrent.futures import ThreadPoolExecutor
    from ...campaign import run, save, utc
    from .epoch_campaign import unused_gpu
    c = read(root/'config.json'); original = read(root/'jobs/planning-e03-none.json')
    if original.get('exit_code') != 1 or 'SemanticSearchLimitExceeded' not in Path(original['log']).read_text():
        raise ValueError('unexpected failure; preserve artifacts')
    retry = root/'evaluation/epoch_03/none_retry001'; name = 'planning-e03-none-retry001'
    def model():
        job = root/'jobs'/f'{name}.json'
        if job.exists():
            run(root, name, read(job)['command'], {})
            return
        index, uuid, lock = unused_gpu(root, name)
        try:
            command = list(original['command']); command[command.index('--name')+1] = root.name+'-'+name
            command[command.index('--gpus')+1] = 'device='+str(index)
            command = [('CUDA_VISIBLE_DEVICES='+uuid if s.startswith('CUDA_VISIBLE_DEVICES=') else
                        'BINDING_GPU_UUID='+uuid if s.startswith('BINDING_GPU_UUID=') else s) for s in command]
            command[command.index('-m')+1] = 'test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding.epoch_recovery'
            command[command.index('--experiment-output')+1] = str(retry)
            run(root, name, command, {'physical_gpu': index, 'gpu_uuid': uuid, 'retry_of': 'planning-e03-none',
                                    'recovery_module_sha256': sha(__file__)})
        finally: lock.close()
    def execution(epoch, policy, folder=None):
        logical = f'execution-e{epoch:02d}-{policy}'; name = logical if folder is None else logical+'-retry001'
        path = root/'jobs'/f'{name}.json'
        if path.exists(): run(root, name, read(path)['command'], {}); return
        template = list(read(root/'jobs/execution-e01-none.json')['command'])
        template[template.index('--name')+1] = root.name+'-'+name
        output = (folder or root/'evaluation'/f'epoch_{epoch:02d}'/policy)/'libero'
        template[template.index('--predictions')+1] = str(output/'predictions.jsonl')
        template[template.index('--output')+1] = str(output/'execution')
        run(root, name, template, {'physical_gpu': None, 'executor': 'unchanged calibrated executor'})
    save(root/'status.json', {'stage': 'recovering-epoch3-and-completing-native-evaluations', 'updated_utc': utc()})
    try:
        with ThreadPoolExecutor(max_workers=1) as gpu, ThreadPoolExecutor(max_workers=2) as native:
            pending = gpu.submit(model)
            futures = [native.submit(execution, e, p) for e in range(1, 5) for p in ('none', 'shared-verified') if (e, p) != (3, 'none')]
            pending.result(); futures.append(native.submit(execution, 3, 'none', retry))
            for f in futures: f.result()
        collect(root, c, retry)
        save(root/'status.json', {'stage': 'complete', 'updated_utc': utc()})
    except Exception as exc:
        save(root/'status.json', {'stage': 'failed-recovery', 'error': str(exc), 'updated_utc': utc()}); raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'host':
        p = argparse.ArgumentParser(); p.add_argument('mode'); p.add_argument('--root', type=Path, required=True)
        host(p.parse_args().root)
    else: evaluation()
