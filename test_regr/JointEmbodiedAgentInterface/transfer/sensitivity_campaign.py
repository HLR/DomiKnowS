"""Validate image bindings and run five fixed ID trials on unused GPUs only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'transfer'

from .campaign import idle_gpu, run, save, utc
from .protocol import freeze, read, sha


def wait_bindings(root):
    pending = ['libero', 'rlbench']
    while pending:
        for dataset in pending[:]:
            path = root/'jobs'/f'bindings-{dataset}.json'
            job = read(path)
            state = json.loads(subprocess.check_output(['docker', 'inspect', job['container_id']], text=True))[0]
            if state['Name'] != '/'+root.name+'-bindings-'+dataset:
                raise ValueError('binding container ownership differs')
            if state['State']['Running']:
                continue
            result = subprocess.run(['docker', 'logs', job['container_id']], capture_output=True, text=True)
            log = root/'logs'/f'bindings-{dataset}.log'; log.parent.mkdir(exist_ok=True)
            log.write_text(result.stdout+result.stderr)
            save(path, {**job, 'exit_code': state['State']['ExitCode'], 'log': str(log), 'finished_utc': utc()})
            if state['State']['ExitCode']:
                raise RuntimeError('binding gate failed; preserve attempt and stop: '+dataset)
            proof = read(root/'bindings'/dataset/'manifest.json')
            if not proof['verified'] or len(proof['episodes']) != 100:
                raise ValueError('all 100 original observations must pass exact reconstruction')
            pending.remove(dataset)
        if pending:
            save(root/'status.json', {'stage': 'verifying-image-bindings', 'datasets': pending, 'updated_utc': utc()})
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    config = read(root/'sensitivity_config.json')
    if 'idle_service_gpu' in config:
        raise ValueError('this phase requires completely unused GPUs')
    wait_bindings(root)
    source = Path(config['source']); package = source/'test_regr/JointEmbodiedAgentInterface/transfer'
    prior = Path(read(root/'phase_manifest.json')['predecessor'])
    for dataset in ('libero', 'rlbench'):
        name = root.name+'-permutations-'+dataset
        item = config['datasets'][dataset]
        command = ['docker', 'run', '-d', '--name', name, '--cpus=4', '--memory=12g',
                   '-v', '/home/auszok:/home/auszok']
        argv = [str(package/'prepare_permutations.py'), '--config', str(root/'sensitivity_config.json'),
                '--dataset', dataset, '--bindings', str(root/'bindings'/dataset/'manifest.json'),
                '--output', str(root/'permutations'/dataset)]
        if dataset == 'libero':
            command += ['-e', 'PYTHONPATH='+config['libero_pythonpath'], '--entrypoint', config['libero_python'],
                        'sha256:90c8c650ddb38a07fd71304f41fbdc12f0b2f8beeca9f7da04a5122afd3c67b0', '-u', *argv]
        else:
            command += [item['runtime_image'], *argv]
        save(root/'status.json', {'stage': 'materializing-id-trials', 'dataset': dataset, 'updated_utc': utc()})
        run(root, 'permutations-'+dataset, command, {'physical_gpu': None})
    for dataset in ('libero', 'rlbench'):
        item = config['datasets'][dataset]
        index, uuid, lock = idle_gpu(root)
        try:
            command = ['docker', 'run', '-d', '--name', root.name+'-sensitivity-'+dataset,
                '--gpus', 'device='+str(index), '--ipc=host', '--cpus=8',
                '-v', '/home/auszok:/home/auszok', '-v', '/home/auszok/DomiKnowS:/workspace',
                '-e', 'CUDA_VISIBLE_DEVICES='+uuid,
                '-e', 'PYTHONPATH='+str(source)+':/opt/dominows/VLABench:/home/auszok/domiknowsrelated/src:/opt/dominows/PreciseRoIPooling/pytorch',
                '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4', '-w', str(source),
                '--entrypoint', '/opt/dominows/venv/bin/python', config['model_image'], '-u',
                '-m', 'test_regr.JointEmbodiedAgentInterface.transfer.evaluate',
                '--checkpoint', config['checkpoint'], '--dataset', dataset, '--method', 'id-sensitivity',
                '--manifest', item['manifest'], '--archive', item['archive'],
                '--permutation-root', str(root/'permutations'/dataset),
                '--baseline-output', str(prior/dataset/'baseline'),
                '--output', str(root/'sensitivity'/dataset), '--planning-dir', config['planning_dir'],
                '--planner-model', config['planner_model'], '--vision-model', 'google/siglip-base-patch16-384',
                '--image-size', '384', '384', '--seed', '42', '--device', 'cuda:0', '--split', 'test', '--load-in-4bit']
            save(root/'status.json', {'stage': 'id-sensitivity', 'dataset': dataset,
                 'physical_gpu': index, 'gpu_uuid': uuid, 'updated_utc': utc()})
            run(root, 'sensitivity-'+dataset, command, {'physical_gpu': index, 'gpu_uuid': uuid,
                'gpu_policy': 'completely-unused-only', 'checkpoint_sha256': sha(config['checkpoint'])})
        finally:
            lock.close()
    freeze(root/'sensitivity_complete.json', {'datasets': ['libero', 'rlbench'], 'trials': 5,
        'physical_episodes_per_dataset': 100, 'conditions': ['none', 'shared-verified'],
        'execution': 'planning sensitivity only; physical evaluation follows matched continuation training',
        'no_training_started': True})
    save(root/'status.json', {'stage': 'id-sensitivity-complete', 'updated_utc': utc()})


if __name__ == '__main__':
    main()
