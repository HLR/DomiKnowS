"""Freeze the fresh Stage 1 recipe, audit it on CPU, then acquire an unused GPU."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'transfer.fresh_metadata'

from ..campaign import run, save, utc
from ..protocol import freeze, read, sha
from ..repair.campaign import gpu_job, model_command


RECIPE = 'metadata-pmd-pcgrad-grounding-preference'
MODULE = 'test_regr.JointEmbodiedAgentInterface.transfer.fresh_metadata.train'
DEFAULT_INPUTS = Path('/home/auszok/experiments/joint_decoder_repair_20261118_attempt9/config.json')


def prepare(root, inputs, repository):
    if root.exists():
        raise ValueError('choose a fresh experiment directory')
    root.mkdir(parents=True)
    (root/'jobs').mkdir()
    (root/'logs').mkdir()
    source = root/'source'
    shutil.copytree(repository, source, ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__',
                                                                     '.pytest_cache', 'logs'))
    paths = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*.py') if p.is_file()}
    manifest = root/'source_manifest.json'
    freeze(manifest, {'files': paths, 'checkout': str(repository),
                      'base_commit': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'],
                                                              text=True).strip()})
    config = read(inputs)
    hashes = dict(config['input_hashes'])
    hashes[str(inputs)] = sha(inputs)
    model = Path(config['model'])
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json'):
        path = model/name
        if path.is_file():
            hashes[str(path)] = sha(path)
    # Hash the actual pretrained shards, once, independently of any Joint checkpoint.
    shards = sorted(model.glob('*.safetensors'))
    if not shards:
        raise ValueError('pinned pretrained model must be locally available')
    for shard in shards:
        hashes[str(shard)] = sha(shard)
    config.update(source=str(source), output=str(root), source_manifest=str(manifest),
                  source_manifest_sha256=sha(manifest), input_hashes=hashes,
                  fresh_recipe=RECIPE, internal_training_approved=True)
    freeze(root/'config.json', config)


def supervise(root):
    config = read(root/'config.json')
    try:
        run(root, 'source-audit', model_command(root, config, 'source-audit', MODULE,
            ['--root', root, '--audit-only']), {'physical_gpu': None, 'optimizer_updates': 0})
        if not read(root/'permutation_audit.json')['passed']:
            raise ValueError('source audit did not pass')
        save(root/'status.json', {'stage': 'waiting-for-unused-gpu', 'updated_utc': utc()})
        gpu_job(root, config, 'training', MODULE, ['--root', root])
        if read(root/'status.json')['stage'] != 'training-complete':
            raise ValueError('training exited before completing')
    except Exception as exc:
        save(root/'status.json', {'stage': 'failed', 'error': str(exc), 'updated_utc': utc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=[RECIPE], default=RECIPE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage1-epochs', type=int, choices=[4], default=4)
    parser.add_argument('--batch-size', type=int, choices=[8], default=8)
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    parser.add_argument('--gpu', choices=['auto'], default='auto')
    parser.add_argument('--allow-internal-evaluation-training', action='store_true')
    parser.add_argument('--input-config', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.supervise:
        supervise(root)
        return
    if not args.allow_internal_evaluation_training:
        parser.error('the historical VLABench split requires --allow-internal-evaluation-training')
    repository = Path(__file__).resolve().parents[4]
    prepare(root, args.input_config.resolve(), repository)
    script = root/'source'/Path(__file__).resolve().relative_to(repository)
    command = ['python3', '-u', str(script), '--output', str(root), '--supervise']
    with (root/'launcher.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root/'source', stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    freeze(root/'supervisor.json', {'pid': process.pid, 'command': command, 'started_utc': utc(),
                                   'launcher_pid': os.getpid()})
    print(f'Campaign {root} launched with supervisor PID {process.pid}', flush=True)


if __name__ == '__main__':
    main()
