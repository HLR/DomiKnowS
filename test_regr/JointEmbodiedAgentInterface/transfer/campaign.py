"""Host launcher for one sequential transfer method, with owned jobs only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import sys
from concurrent.futures import ThreadPoolExecutor

if not __package__:
    # The host orchestrator must not import Joint's GPU/ontology dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'transfer'

from .prepare import prepare
from .protocol import POLICIES, freeze, read, require_predecessor, sha


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def idle_service_available(index, uuid):
    """Permit only the inspected idle Celery service, never arbitrary resident jobs."""
    name = 'ics-stancer-celery-worker-gpu'
    worker = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
    if not worker['State']['Running']:
        return False
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    pids = {int(parts[1]) for line in rows.splitlines()
            if len(parts := [x.strip() for x in line.split(',')]) == 2 and parts[0] == uuid}
    if pids != {worker['State']['Pid']}:
        return False
    for state in ('active', 'reserved', 'scheduled'):
        reply = subprocess.run(['docker', 'exec', '-w', '/tmp', name, 'celery', '-b',
            'redis://ics-stancer-redis:6379/0', 'inspect', state, '--destination',
            'celery@'+worker['Id'][:12], '--timeout', '2', '--json'], capture_output=True, text=True, timeout=8)
        if reply.returncode or json.loads(reply.stdout).get('celery@'+worker['Id'][:12]) != []:
            return False
    return True


def idle_gpu(root, idle_service_gpu=None):
    import fcntl
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
        for row in sorted(rows.splitlines(), key=lambda line: int(line.split(',')[2].strip()) >= 100):
            index, uuid, memory, total, utilization = [x.strip() for x in row.split(',')]
            if int(utilization) != 0:
                continue
            if int(memory) >= 100:
                if int(index) != idle_service_gpu or int(memory) > 4096 or int(total)-int(memory) < 81920:
                    continue
                try:
                    if not idle_service_available(int(index), uuid):
                        continue
                except (subprocess.SubprocessError, ValueError):
                    continue
            lock = (root/f'{uuid}.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close(); continue
            return int(index), uuid, lock
        save(root/'status.json', {'stage': 'waiting-for-free-gpu', 'updated_utc': utc()})
        time.sleep(1)


def run(root, name, command, metadata):
    jobpath = root/'jobs'/f'{name}.json'
    resumed = False
    if jobpath.exists():
        old = read(jobpath)
        if old['command'] != command:
            raise ValueError('job resume command differs')
        if old.get('exit_code') == 0:
            return
        if old.get('exit_code') is not None or not old.get('container_id'):
            raise ValueError('prior failed attempt preserved; use a fresh campaign root')
        container = old['container_id']; resumed = True
    else:
        freeze(jobpath, {'command': command, **metadata, 'started_utc': utc()})
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode:
            save(jobpath, {**read(jobpath), 'launch_error': proc.stderr, 'exit_code': proc.returncode})
            raise RuntimeError('container launch failed: ' + proc.stderr)
        container = proc.stdout.strip()
    worker = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
    if worker['Name'].lstrip('/') != command[command.index('--name')+1]:
        raise ValueError('owned container identity differs')
    if resumed and worker['Id'] != container:
        raise ValueError('resumed container ID differs')
    logpath = root/'logs'/f'{name}.log'; logpath.parent.mkdir(exist_ok=True)
    if resumed: logpath = logpath.with_name(logpath.stem+f'.resume-{os.getpid()}.log')
    previous = read(jobpath)
    save(jobpath, {**previous, 'container_id': container, 'pid': previous.get('pid', worker['State']['Pid']),
                   'image_id': worker['Image'], 'log': str(logpath),
                   'previous_logs': previous.get('previous_logs', [])+([previous['log']] if resumed else [])})
    with logpath.open('w') as log:
        follow = subprocess.Popen(['docker', 'logs', '-f', container], stdout=log, stderr=subprocess.STDOUT)
        code = int(subprocess.check_output(['docker', 'wait', container], text=True).strip())
        follow.wait()
    save(jobpath, {**read(jobpath), 'exit_code': code, 'finished_utc': utc()})
    if code:
        raise RuntimeError(f'job {name} exited {code}; see {logpath}')


def launch_planning(root, config, dataset, method):
    output = root/dataset/method
    source = Path(config['source'])
    jobpath = root/'jobs'/f'{dataset}-{method}-planning.json'
    if jobpath.exists() and read(jobpath).get('exit_code') == 0:
        settings = read(output/'settings.json')
        if settings['checkpoint_sha256'] != sha(config['checkpoint']):
            raise ValueError('completed planner checkpoint differs')
        if not (output/'summary.json').exists():
            raise ValueError('completed planner job has no summary')
        return
    index, uuid, lock = idle_gpu(root, config.get('idle_service_gpu'))
    try:
        save(root/'status.json', {'stage': 'planning', 'method': method, 'dataset': dataset,
             'physical_gpu': index, 'gpu_uuid': uuid, 'updated_utc': utc()})
        item = config['datasets'][dataset]
        command = ['docker', 'run', '-d', '--name', root.name+'-'+dataset+'-'+method,
            '--gpus', 'device='+str(index), '--ipc=host', '-v', '/home/auszok:/home/auszok',
            '-v', '/home/auszok/DomiKnowS:/workspace', '-e', 'CUDA_VISIBLE_DEVICES='+uuid,
            '-e', 'PYTHONPATH='+str(source)+':/opt/dominows/VLABench:/home/auszok/domiknowsrelated/src:/opt/dominows/PreciseRoIPooling/pytorch',
            '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4',
            '-w', str(source), '--entrypoint', '/opt/dominows/venv/bin/python', config['model_image'],
            '-u', '-m', 'test_regr.JointEmbodiedAgentInterface.transfer.evaluate',
            '--checkpoint', config['checkpoint'], '--dataset', dataset, '--method', method,
            '--manifest', item['manifest'], '--archive', item['archive'], '--output', str(output),
            '--planning-dir', config['planning_dir'], '--planner-model', config['planner_model'],
            '--vision-model', 'google/siglip-base-patch16-384', '--image-size', '384', '384',
            '--seed', str(config['seed']), '--device', 'cuda:0', '--split', 'test', '--load-in-4bit']
        run(root, dataset+'-'+method+'-planning', command, {'physical_gpu': index, 'gpu_uuid': uuid,
            'idle_service_gpu_allowed': config.get('idle_service_gpu')})
    finally:
        lock.close()


def launch_execution(root, config, dataset, method, policy):
    item = config['datasets'][dataset]
    folder = root/dataset/method/policy
    script = str(Path(item['executor_source'])/(dataset+'_campaign.py'))
    args = ['evaluate', '--manifest', item['manifest'], '--calibration', item['calibration'],
            '--predictions', str(folder/'predictions.jsonl'), '--output', str(folder/'execution')]
    name = root.name+'-'+dataset+'-'+method+'-'+policy
    prefix = ['docker', 'run', '-d', '--name', name, '--cpus=6', '--memory=12g',
              '-v', '/home/auszok:/home/auszok']
    if dataset == 'rlbench':
        command = prefix+[item['runtime_image'], script, *args]
    else:
        command = prefix+['-e', 'MUJOCO_GL=egl', '-e', 'PYOPENGL_PLATFORM=egl',
            '-e', 'LIBERO_CONFIG_PATH='+config['libero_config'], '-e', 'PYTHONPATH='+config['libero_pythonpath'],
            '--entrypoint', config['libero_python'], item['runtime_image'], '-u', script, *args]
    run(root, dataset+'-'+method+'-'+policy+'-execution', command, {'physical_gpu': None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['semantic-prompt'], required=True)
    parser.add_argument('--idle-service-gpu', type=int, choices=[5], help='use idle capacity beside the inspected Celery service without stopping it')
    args = parser.parse_args(); root = args.output; root.mkdir(parents=True, exist_ok=True)
    try:
        proof = prepare(args.config, root); config = {**proof['configuration'], 'idle_service_gpu': args.idle_service_gpu}
        require_predecessor(args.method, root)
        for dataset in ['libero', 'rlbench']:
            launch_planning(root, config, dataset, 'baseline')
        # Both baseline prediction gates pass before treatment generation. CPU
        # baseline execution and the fixed treatment may overlap within method 1;
        # no comparison or later method is allowed until all executions finish.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(launch_execution, root, config, dataset, 'baseline', policy)
                       for dataset in ('libero', 'rlbench') for policy in POLICIES]
            for dataset in ('libero', 'rlbench'):
                for future in futures:
                    if future.done(): future.result()
                launch_planning(root, config, dataset, args.method)
            futures += [pool.submit(launch_execution, root, config, dataset, args.method, policy)
                        for dataset in ('libero', 'rlbench') for policy in POLICIES]
            save(root/'status.json', {'stage': 'physical-evaluation', 'method': args.method,
                'conditions': [d+':'+m+':'+p for d in ('libero', 'rlbench')
                               for m in ('baseline', args.method) for p in POLICIES],
                'supervisor_pid': os.getpid(), 'updated_utc': utc()})
            for future in futures:
                future.result()
        from .report import summarize
        summarize(root)
        freeze(root/args.method/'qualification.json', {'technical_validation_passed': True,
            'campaign_manifest_sha256': sha(root/'campaign_manifest.json'), 'results_sha256': sha(root/'results.json')})
        save(root/'status.json', {'stage': 'complete', 'method': args.method, 'updated_utc': utc()})
    except Exception as exc:
        save(root/'status.json', {'stage': 'stopped-on-validation-failure', 'error': str(exc), 'updated_utc': utc()})
        raise


if __name__ == '__main__':
    main()
