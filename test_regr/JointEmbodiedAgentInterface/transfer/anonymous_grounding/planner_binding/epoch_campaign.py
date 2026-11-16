"""Isolated all-epoch evaluation; unchanged training and best-checkpoint rule."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'transfer.anonymous_grounding.planner_binding'
from ...campaign import run, save, utc
from ...protocol import read, freeze, sha, episodes
from ..campaign import runtime_environment

POLICIES = ('none', 'shared-verified', 'full-metadata')


def finish_evaluations(futures, native, execute):
    """An isolated planner failure must not discard other completed executions."""
    errors = []; executions = []
    for future in as_completed(futures):
        try: epoch, policy = future.result()
        except Exception as exc:
            errors.append(str(exc))
            continue
        if policy != 'full-metadata': executions.append(native.submit(execute, epoch, policy))
    for future in executions:
        try: future.result()
        except Exception as exc: errors.append(str(exc))
    if errors: raise RuntimeError('evaluation jobs failed; successful conditions preserved: '+'; '.join(errors))


def unused_gpu(root, name):
    import fcntl
    lockroot = Path('/home/auszok/experiments/.gpu_locks'); lockroot.mkdir(exist_ok=True)
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
        occupied = {line.strip() for line in processes.splitlines()}
        for row in rows.splitlines():
            index, uuid, memory, utilization = [x.strip() for x in row.split(',')]
            if int(memory) >= 100 or int(utilization) != 0 or uuid in occupied: continue
            lock = (lockroot/(uuid+'.lock')).open('a')
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: lock.close(); continue
            return int(index), uuid, lock
        save(root/'queues'/f'{name}.json', {'stage': 'waiting-for-unused-gpu', 'updated_utc': utc()})
        time.sleep(5)


def collect(root, config):
    rows = []; common = {}; references = None
    expected = {e['episode_id'] for e in episodes(read(config['external_manifest']))}
    for epoch in range(1, 5):
        base = root/'evaluation'/f'epoch_{epoch:02d}'
        eai = read(base/'none/eai.json')
        if eai['examples'] != 88: raise ValueError('incomplete EAI result')
        for policy in POLICIES:
            out = base/policy; vla = read(out/'vlabench/report.json')
            if vla['examples'] != 450: raise ValueError('incomplete VLABench result')
            refs = read(out/'references.json')
            if references is None: references = refs
            if refs != references: raise ValueError('evaluation references changed between conditions')
            for dataset in ('vlabench', 'libero'):
                if dataset == 'libero' and policy == 'full-metadata': continue
                report = read(out/dataset/'report.json')
                hashes = report['input_hashes']
                if dataset not in common: common[dataset] = hashes
                if hashes != common[dataset]: raise ValueError('rendered input hashes differ across epochs or policies')
            result = {'epoch': epoch, 'constraint_policy': policy,
                      'eai_goal_percent': eai['goal_success']*100 if policy == 'none' else None,
                      'vla_macro': vla['Macro'], 'progress_score': vla['progress_score'],
                      'libero_abstract_percent': None, 'libero_physical_percent': None}
            if policy != 'full-metadata':
                folder = out/'libero'; lib = read(folder/'report.json'); physical = read(folder/'execution/summary.json')
                predictions = {r['episode_id']: r for r in (json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines())}
                outcomes = {r['episode_id']: r for r in (json.loads(s) for s in (folder/'execution/episodes.jsonl').read_text().splitlines())}
                if set(predictions) != expected or set(outcomes) != expected or physical['episodes'] != 100:
                    raise ValueError('incomplete native LIBERO coverage')
                if any(predictions[k]['prediction'] != outcomes[k]['prediction'] for k in expected):
                    raise ValueError('executor used different predictions')
                result.update(libero_abstract_percent=lib['abstract_goal_success']*100,
                              libero_physical_percent=physical['success_rate']*100,
                              libero_success_ci95=physical['success_ci95'])
            rows.append(result)
    freeze(root/'results.json', {'rows': rows, 'config_sha256': sha(root/'config.json'),
        'coverage_passed': True, 'identical_inputs_passed': True, 'identical_references_passed': True,
        'selection': 'all-epoch diagnostic; source-loss best and automatic campaign selection unchanged',
        'libero_states': [40, 41, 42, 43, 44]})
    columns = ['epoch', 'constraint_policy', 'eai_goal_percent', 'vla_macro', 'progress_score',
               'libero_abstract_percent', 'libero_physical_percent']
    with (root/'table_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    labels = {'none': 'No DFA', 'shared-verified': 'Shared verified', 'full-metadata': 'Full metadata'}
    lines = ['# Masked CE operation-binding continuation: all four epochs', '',
        '| Snapshot | Constraint policy | EAI goal (%) | VLA Macro (0–80) | Progress Score (0–100) | LIBERO abstract (%) | LIBERO physical (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [('—' if row[k] is None else f'{row[k]:.2f}') for k in columns[2:]]
        lines.append('| Epoch '+str(row['epoch'])+' | '+labels[row['constraint_policy']]+' | '+' | '.join(values)+' |')
    lines += ['', 'Internal source-only experiment. Four additional epochs continue from the selected anonymous-ID grounding snapshot. '
        'Joint Qwen LoRA/label heads and causal operation-grounding heads train on VLABench+EAI only. '
        'Anonymous objects and consistent ID/image/output-pointer permutations; masked VLABench CE, allowed-mass penalty 0.5, '
        'operation-grounding loss 1.0; frozen paired-SigLIP primary hints and fixed physical controller. '
        'Seed 42, batch 8, four-bit Qwen loading, planner LR 2e-5, grounding LR 1e-3, AdamW/PCGrad; no Stage 2 RL.', '',
        'VLABench: frozen 450 internal test episodes, scene-verified density references; EAI: canonical 88 validation examples. '
        'LIBERO: 100 fresh Spatial/Object episodes, states 40–44. The older 0–4 table is a different set. '
        'Shared verified means the original whitelist, without the additional pick/place executor capability overlay. '
        'Full metadata is a privileged VLABench condition and is not applied to LIBERO. '
        'Native success uses the unchanged independently calibrated fixed executor and 1,000-step horizon. '
        'These external per-epoch results are diagnostic and must not choose a checkpoint or tune training.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    root = a.root; c = read(root/'config.json'); source = Path(c['source'])
    module = 'test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding'
    prefix = ['docker', 'run', '-d', '--ipc=host', '--cpus=8', '--memory=64g',
              '-v', '/home/auszok:/home/auszok', '-v', '/home/auszok/DomiKnowS:/workspace', '-w', str(source)]
    for k, v in runtime_environment(source, c['planning_dir']).items(): prefix += ['-e', k+'='+v]
    def cpu(name, args):
        command = prefix+['--name', root.name+'-'+name, '--entrypoint', '/opt/dominows/venv/bin/python',
                          c['model_image'], '-u', '-m', *args]
        run(root, name, command, {'physical_gpu': None})
    def execute(epoch, policy):
        name = f'execution-e{epoch:02d}-{policy}'; base = Path(c['libero_base'])
        folder = root/'evaluation'/f'epoch_{epoch:02d}'/policy/'libero'
        command = ['docker', 'run', '-d', '--name', root.name+'-'+name, '--user', '0', '--cpus=6', '--memory=12g',
            '-v', '/home/auszok:/home/auszok', '-e', 'MUJOCO_GL=egl', '-e', 'PYOPENGL_PLATFORM=egl',
            '-e', 'LIBERO_CONFIG_PATH='+str(base/'libero_config'),
            '-e', 'PYTHONPATH='+str(base/'libero_runtime/lib/python3.10/site-packages')+':'+str(base/'tooling/LIBERO'),
            '--entrypoint', str(base/'libero_container_runtime/bin/python'), c['executor_image'], '-u',
            str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'), 'evaluate',
            '--manifest', c['external_manifest'], '--calibration', c['calibration'],
            '--predictions', str(folder/'predictions.jsonl'), '--output', str(folder/'execution')]
        run(root, name, command, {'physical_gpu': None, 'executor': 'retained independently calibrated fixed executor; horizon 1000'})
    def model(epoch, policy):
        name = f'planning-e{epoch:02d}-{policy}'; job = root/'jobs'/f'{name}.json'
        if job.exists():
            previous = read(job); run(root, name, previous['command'], {})
            return epoch, policy
        index, uuid, lock = unused_gpu(root, name)
        try:
            save(root/'queues'/f'{name}.json', {'stage': 'launching', 'physical_gpu': index, 'gpu_uuid': uuid, 'updated_utc': utc()})
            command = prefix+['--name', root.name+'-'+name, '--gpus', 'device='+str(index),
                '-e', 'CUDA_VISIBLE_DEVICES='+uuid, '-e', 'BINDING_GPU_UUID='+uuid,
                '--entrypoint', '/opt/dominows/venv/bin/python', c['model_image'], '-u', '-m', module+'.epoch_evaluation',
                '--epoch-config', str(root/'config.json'), '--epoch', str(epoch), '--checkpoint', c['snapshots'][str(epoch)]['path'],
                '--constraint-policy', policy, '--experiment-output', str(root/'evaluation'/f'epoch_{epoch:02d}'/policy),
                '--planning-dir', c['planning_dir'], '--planner-model', '/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct',
                '--vision-model', c['vision_model'], '--image-size', '384', '384', '--device', 'cuda:0', '--seed', '42', '--load-in-4bit']
            run(root, name, command, {'physical_gpu': index, 'gpu_uuid': uuid, 'epoch': epoch,
                'checkpoint_sha256': c['snapshots'][str(epoch)]['sha256'], 'policy': policy})
            save(root/'queues'/f'{name}.json', {'stage': 'complete', 'updated_utc': utc()})
            return epoch, policy
        finally: lock.close()
    try:
        save(root/'status.json', {'stage': 'testing-evaluation-adapter', 'updated_utc': utc()})
        cpu('tests', ['pytest', 'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding/test_epoch_evaluation.py',
                      'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding/test_binding.py', '-q'])
        save(root/'status.json', {'stage': 'preparing-frozen-evaluation-inputs', 'updated_utc': utc()})
        cpu('prepare', [module+'.epoch_evaluation', 'prepare', '--config', str(root/'config.json')])
        save(root/'status.json', {'stage': 'queued-or-evaluating', 'jobs': 12, 'updated_utc': utc()})
        with ThreadPoolExecutor(max_workers=2) as gpu, ThreadPoolExecutor(max_workers=2) as native:
            futures = [gpu.submit(model, e, policy) for e in range(1, 5) for policy in POLICIES]
            finish_evaluations(futures, native, execute)
        collect(root, c)
        save(root/'status.json', {'stage': 'complete', 'updated_utc': utc()})
    except Exception as exc:
        save(root/'status.json', {'stage': 'failed', 'error': str(exc), 'updated_utc': utc()}); raise


if __name__ == '__main__': main()
