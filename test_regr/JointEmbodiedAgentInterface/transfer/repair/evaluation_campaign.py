"""Adopt existing training; replace its unopened evaluations with strict inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if not __package__:
    experiment = Path(sys.argv[sys.argv.index('--root')+1])
    sys.path.insert(0, str(experiment/'source/test_regr/JointEmbodiedAgentInterface'))
    __package__ = 'transfer.repair'

from ..protocol import read, freeze, sha
from ..campaign import run, save, utc
from ..anonymous_grounding.planner_binding.verification_control import native_command
from .campaign import gpu_job, model_command, MODULE
from .protocol import verify
from .input_protocol import IMAGE_INSTRUCTION


def require_unopened_training(status, prediction_files):
    if status.get('stage') != 'two-fresh-four-epoch-source-arms' or prediction_files:
        raise ValueError('handoff requires active source training and unopened confirmation')


def freeze_parent_selection(root, parent):
    systems = {'base': {'path': str(parent/'native_base/initialization.pt'), 'qualified': True},
        'joint-initialization': {'path': str(parent/'arms/joint/initialization.pt'), 'qualified': True}}
    for arm in ('native', 'joint'):
        path = parent/'arms'/arm/'selection.json'; selected = read(path)
        chosen = selected['selected'] or selected['diagnostic_best']
        systems[arm+'-trained'] = {'path': chosen['checkpoint'], 'qualified': selected['qualified'],
            'source_epoch': chosen['epoch'], 'source_progress_score': chosen['progress_score'],
            'selection_sha256': sha(path)}
    for spec in systems.values():
        spec['sha256'] = sha(spec['path'])
    freeze(root/'frozen_selection.json', {'systems': systems,
        'selection_data': 'Unchanged parent source-validation protocol and thresholds; no target scoring',
        'input_protocol': IMAGE_INSTRUCTION,
        'primary_contrasts': ['native-trained minus base under none', 'joint-trained minus base under none'],
        'bootstrap': {'samples': 10000, 'seed': 42, 'confidence': .975, 'cluster': 'task, retaining five scenes'}})
    return systems


def handoff(root, c):
    parent = Path(c['training_parent'])
    require_unopened_training(read(parent/'status.json'), list((parent/'evaluation').rglob('predictions.jsonl')))
    expected = c['training_supervisor_pid']
    arguments = Path(f'/proc/{expected}/cmdline').read_bytes().decode().strip('\0').split('\0')
    script = str(parent/'source/test_regr/JointEmbodiedAgentInterface/transfer/repair/campaign.py')
    if arguments != ['python3', '-u', script, '--root', str(parent)]:
        raise ValueError('training supervisor identity differs')
    owned = {}
    for arm in ('native', 'joint'):
        job = read(parent/'jobs'/f'train-{arm}.json')
        state = json.loads(subprocess.check_output(['docker', 'inspect', job['container_id']], text=True))[0]
        if state['Name'].lstrip('/') != parent.name+'-train-'+arm or state['Id'] != job['container_id']:
            raise ValueError('owned training container identity differs')
        owned[arm] = {'container_id': state['Id'], 'pid': state['State']['Pid'],
            'running': state['State']['Running'], 'gpu_uuid': job['gpu_uuid'], 'image_id': state['Image']}
    reporter = read(parent/'reporting_overlay_v2/supervisor.json')
    reporter_args = Path(f'/proc/{reporter["pid"]}/cmdline').read_bytes().decode().strip('\0').split('\0')
    if reporter_args != reporter['command']:
        raise ValueError('owned historical reporting supervisor identity differs')
    freeze(root/'handoff_intent.json', {'training_parent': str(parent), 'supervisor_pid': expected,
        'supervisor_command': arguments, 'reporting_supervisor': reporter, 'containers': owned,
        'confirmation_predictions_before_handoff': 0, 'reason': 'User requested knowledge confined to optional DFA'})
    # Kill only the verified host orchestration PIDs. Docker-managed training
    # processes and their GPU contexts are never signalled or restarted.
    os.kill(reporter['pid'], signal.SIGTERM)
    os.kill(expected, signal.SIGTERM)
    for _ in range(50):
        if not Path(f'/proc/{expected}').exists():
            break
        time.sleep(.1)
    if Path(f'/proc/{expected}').exists():
        raise ValueError('old orchestration did not exit; no competing confirmation launcher allowed')
    for arm, original in owned.items():
        state = json.loads(subprocess.check_output(['docker', 'inspect', original['container_id']], text=True))[0]
        if original['running'] and (not state['State']['Running'] or state['State']['Pid'] != original['pid']):
            raise ValueError('training process changed during handoff')
    freeze(root/'handoff_complete.json', {'passed': True, 'training_containers_unchanged': owned,
        'old_pending_metadata_evaluation_disabled': True, 'training_restarted': False})
    freeze(parent/'evaluation_handoff.json', {'evaluation_root': str(root),
        'input_protocol': IMAGE_INSTRUCTION, 'old_pending_evaluation_disabled': True,
        'training_restarted': False, 'handoff_sha256': sha(root/'handoff_complete.json')})
    save(parent/'status.json', {'stage': 'training-continues-strict-evaluation-handoff',
        'evaluation_root': str(root), 'updated_utc': utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adopt', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(); root = args.root; c = read(root/'config.json')
    parent = Path(c['training_parent'])
    try:
        verify(c)
        if c['evaluation_input_protocol'] != IMAGE_INSTRUCTION or sha(parent/'config.json') != c['training_parent_config_sha256']:
            raise ValueError('frozen evaluation/training contract differs')
        run(root, 'tests', model_command(root, c, 'tests', 'pytest', [
            'test_regr/JointEmbodiedAgentInterface/transfer/repair',
            'test_regr/JointEmbodiedAgentInterface/transfer/matched_baseline/test_protocol.py',
            'test_regr/JointEmbodiedAgentInterface/transfer/test_native_baseline.py', '-q']), {'physical_gpu': None})
        run(root, 'model-input-audit', model_command(root,c,'model-input-audit',MODULE+'.input_audit',
            ['--root',root]), {'physical_gpu':None,'model_weights_loaded':False})
        if not read(root/'confirmation/model_input_audit.json')['passed']:
            raise ValueError('rendered model input audit failed')
        for name in ('manifest.json', 'split_audit.json', 'calibration_reuse.json', 'frozen_input_audit.json'):
            if sha(root/'confirmation'/name) != sha(parent/'confirmation'/name):
                raise ValueError('frozen confirmation observation/calibration changed')
        gate = read(root/'confirmation/calibration_reuse.json')
        executor = Path(c['libero_base'])/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'
        if (gate['manifest_sha256'] != sha(root/'confirmation/manifest.json') or
            gate['executor_source_sha256'] != sha(executor/'fixed_executor.py') or
            gate['bridge_source_sha256'] != sha(executor/'libero_campaign.py')):
            raise ValueError('qualified executor or bridge changed')
        if args.validate_only:
            save(root/'status.json', {'stage': 'validated-awaiting-owned-handoff', 'updated_utc': utc()})
            return
        if not (root/'handoff_complete.json').exists():
            if not args.adopt:
                raise ValueError('explicit --adopt required for the verified orchestration handoff')
            handoff(root, c)
        save(root/'status.json', {'stage': 'waiting-for-existing-source-training', 'updated_utc': utc()})
        def adopt(arm):
            job = read(parent/'jobs'/f'train-{arm}.json')
            run(parent, 'train-'+arm, job['command'], {'physical_gpu': job['physical_gpu'],
                'gpu_uuid': job['gpu_uuid'], 'config_sha256': job['config_sha256']})
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(adopt, arm) for arm in ('native', 'joint')]):
                future.result()
        if not read(parent/'native_base/complete.json')['base_reference']:
            raise ValueError('source Base validation incomplete')
        systems = freeze_parent_selection(root, parent)
        save(root/'status.json', {'stage': 'strict-eight-condition-confirmation', 'updated_utc': utc()})
        errors = []; executions = []
        with ThreadPoolExecutor(max_workers=2) as planning, ThreadPoolExecutor(max_workers=2) as physical:
            def condition(system, policy):
                folder = root/'evaluation'/system/policy; name = 'confirm-'+system+'-'+policy
                gpu_job(root, c, name, MODULE+'.evaluate', ['--config', root/'config.json', '--arm', system,
                    '--checkpoint', systems[system]['path'], '--manifest', root/'confirmation/manifest.json',
                    '--output', folder, '--policy', policy, '--input-protocol', IMAGE_INSTRUCTION, '--seed', '42'])
                executions.append(physical.submit(run, root, name+'-native', native_command(root, c, name+'-native',
                    root/'confirmation/manifest.json', root/'confirmation/calibration_reuse.json',
                    folder/'predictions.jsonl', folder/'execution'), {'physical_gpu': None, 'horizon': 1000}))
            for future in as_completed([planning.submit(condition, system, policy)
                    for system in systems for policy in ('none', 'shared-verified')]):
                try: future.result()
                except Exception as exc: errors.append(str(exc))
            for future in executions:
                try: future.result()
                except Exception as exc: errors.append(str(exc))
        if errors:
            raise RuntimeError('Completed conditions retained: '+'; '.join(errors))
        run(root, 'analysis', model_command(root, c, 'analysis', MODULE+'.analysis', ['--root', root]), {'physical_gpu': None})
        freeze(root/'complete.json', {'results_sha256': sha(root/'results.json'), 'input_protocol': IMAGE_INSTRUCTION,
            'training_restarted': False, 'native_executions': 800})
        save(root/'status.json', {'stage': 'complete-awaiting-google-publication', 'updated_utc': utc()})
    except Exception as exc:
        save(root/'status.json', {'stage': 'failed', 'error': str(exc), 'updated_utc': utc()})
        raise


if __name__ == '__main__':
    main()
