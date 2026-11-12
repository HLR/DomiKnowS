"""Deploy a frozen first-method source overlay to an isolated server directory."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpu2')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume-host', action='store_true', help='replace only an idle, owned host supervisor; preserve frozen model source')
    parser.add_argument('--idle-service-gpu', type=int, choices=[5])
    parser.add_argument('--resume-native', action='store_true', help='adopt owned native containers without restarting their simulations')
    args = parser.parse_args()
    files = {p.name: base64.b64encode(p.read_bytes()).decode() for p in Path(__file__).parent.glob('*.py')}
    payload = json.dumps({'root': args.output, 'files': files, 'resume_host': args.resume_host,
                          'idle_service_gpu': args.idle_service_gpu, 'resume_native': args.resume_native})
    script = r'''
from pathlib import Path
import base64,json,subprocess,shutil,os,signal,time,hashlib
data=json.loads(PAYLOAD)
root=Path(data['root'])
if root.parent != Path('/home/auszok/experiments'):
 raise ValueError('campaign must be a direct child of /home/auszok/experiments')
if data['resume_host']:
 stage=json.loads((root/'status.json').read_text())['stage']
 if stage!='waiting-for-free-gpu' and not (data['resume_native'] and stage=='physical-evaluation'):
  raise ValueError('only an idle GPU-wait supervisor can be replaced')
 jobs={p.name:json.loads(p.read_text()) for p in (root/'jobs').glob('*.json')}
 if any(j.get('exit_code') not in (None,0) or (j.get('exit_code') is None and
    not (data['resume_native'] and name.endswith('-execution.json') and j.get('container_id')))
    for name,j in jobs.items()):
  raise ValueError('a model job is active or a job failed; do not replace supervisor')
 supervisor=root/'supervisor_current.json'
 if not supervisor.exists():supervisor=root/'supervisor.json'
 old=json.loads(supervisor.read_text());pid=old['pid']
 cmdline=Path(f'/proc/{pid}/cmdline').read_text().replace('\0',' ')
 if str(root) not in cmdline or 'transfer/campaign.py' not in cmdline:
  raise ValueError('supervisor ownership could not be verified')
 children=subprocess.check_output(['ps','--ppid',str(pid),'-o','pid='],text=True).split()
 os.kill(pid,signal.SIGTERM)
 owned_ids={j.get('container_id') for j in jobs.values()}
 for child in children:
  try:
   args=Path(f'/proc/{child}/cmdline').read_bytes().decode().strip('\0').split('\0')
   if Path(args[0]).name=='docker' and len(args)>2 and args[1] in ('logs','wait') and args[-1] in owned_ids:
    os.kill(int(child),signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
 runner=root/('host_runner_'+str(time.time_ns()));package=runner/'transfer';package.mkdir(parents=True)
 for name,value in data['files'].items():
  (package/name).write_bytes(base64.b64decode(value))
 provenance={'previous_supervisor':old,'reason':'host-only queue/report updates; model source and running native simulations unchanged',
  'adopted_native_containers':[j['container_id'] for name,j in jobs.items() if name.endswith('-execution.json') and j.get('exit_code') is None],
  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')}}
 (runner/'provenance.json').write_text(json.dumps(provenance,indent=2))
 cmd=['python3','-u',str(package/'campaign.py'),'--config',str(root/'config.json'),'--output',str(root),'--method','semantic-prompt']
 if data['idle_service_gpu'] is not None:cmd+=['--idle-service-gpu',str(data['idle_service_gpu'])]
 with (runner/'launcher.log').open('w') as stream:
  proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (root/'supervisor_current.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'log':str(runner/'launcher.log')},indent=2))
 print(json.dumps({'pid':proc.pid,'root':str(root),'log':str(runner/'launcher.log')}));raise SystemExit(0)
if root.exists():
 raise ValueError('use a new direct child of /home/auszok/experiments')
base=Path('/home/auszok/experiments/joint_stage1_metadata_generalization_20261108')
rl=Path('/home/auszok/experiments/rlbench_model_evaluation_v33_20261110_attempt2')
root.mkdir();source=root/'source'
subprocess.run(['git','clone','--shared',str(base/'evaluation_source'),str(source)],check=True,capture_output=True)
package=source/'test_regr/JointEmbodiedAgentInterface'
(package/'transfer').mkdir()
for name,value in data['files'].items():
 (package/'transfer'/name).write_bytes(base64.b64decode(value))
shutil.copy2(rl/'source/test_regr/JointEmbodiedAgentInterface/generalization/rlbench_protocol.py',package/'generalization/rlbench_protocol.py')
model_image='sha256:08b4f594076cd4d4e0e392ef0c7934b82e525dbe07bd0a1ac005b7f710682d92'
config={'seed':42,'epochs':4,'batch_size':8,'source':str(source),
 'checkpoint':str(base/'checkpoints/joint_stage1_epoch_003.pt'),'model_image':model_image,
 'planning_dir':'/home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data/planning',
 'planner_model':'/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct',
 'libero_config':str(base/'libero_config'), 'libero_python':str(base/'libero_container_runtime/bin/python'),
 'libero_pythonpath':str(base/'libero_runtime/lib/python3.10/site-packages')+':'+str(base/'tooling/LIBERO'),
 'datasets':{'libero':{'manifest':str(base/'libero_frozen_v3/manifest.json'),
   'calibration':str(base/'calibration_v3/summary.json'),'executor_source':str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'),
   'archive':str(base/'evaluation/epoch4/libero'),'runtime_image':model_image},
 'rlbench':{'manifest':'/home/auszok/experiments/rlbench_followup_20261109/frozen_v2/manifest.json',
   'calibration':str(rl/'calibration_replay_v33/summary.json'),'executor_source':str(rl/'native_replay_source_v33'),
   'archive':str(rl/'planning/epoch4'),'runtime_image':'sha256:5fff7d4968af50c6278328c2086aa6747f9fbbf0523765ae9c389dee83b55a58'}}}
(root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
env={**os.environ,'PYTHONPATH':str(source)}
cmd=['python3','-u',str(package/'transfer/campaign.py'),'--config',str(root/'config.json'),'--output',str(root),'--method','semantic-prompt']
if data['idle_service_gpu'] is not None:cmd+=['--idle-service-gpu',str(data['idle_service_gpu'])]
with (root/'launcher.log').open('w') as stream:
 proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(root/'supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'log':str(root/'launcher.log')},indent=2))
print(json.dumps({'pid':proc.pid,'root':str(root),'log':str(root/'launcher.log')}))
'''.replace('PAYLOAD', repr(payload))
    result = subprocess.run(['ssh', '-o', 'ProxyJump=none', args.host, 'python3', '-'],
                            input=script, capture_output=True, text=True)
    print(result.stdout); print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
