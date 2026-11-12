"""Deploy an independent matched continuation after the fixed ID diagnostic."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='gpu2');parser.add_argument('--predecessor',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    payload={'prior':args.predecessor,'root':args.output,'files':{
        p.name:base64.b64encode(p.read_bytes()).decode() for p in Path(__file__).parent.glob('*.py')}}
    script=r'''
from pathlib import Path
import json,base64,subprocess,shutil,hashlib
data=json.loads(PAYLOAD);prior=Path(data['prior']);root=Path(data['root'])
if root.parent!=Path('/home/auszok/experiments') or root.exists():raise ValueError('new isolated experiment root required')
gate=json.loads((prior/'sensitivity_complete.json').read_text())
if gate['trials']!=5 or gate['physical_episodes_per_dataset']!=100 or not gate['no_training_started']:
 raise ValueError('fixed ID diagnostic gate differs')
for dataset in ['libero','rlbench']:
 job=json.loads((prior/'jobs'/('sensitivity-'+dataset+'.json')).read_text())
 if job['exit_code']!=0:raise ValueError('diagnostic job failed')
 for trial in range(5):
  p=prior/'sensitivity'/dataset/f'trial_{trial:03d}'/'summary.json'
  summary=json.loads(p.read_text())
  if any(v['episodes']!=100 or v['generation_errors'] for v in summary['reports'].values()):
   raise ValueError('diagnostic generation integrity gate failed')
root.mkdir();config=json.loads((prior/'sensitivity_config.json').read_text());original=Path(config['source']);source=root/'source'
subprocess.run(['git','clone','--shared',str(original),str(source)],check=True,capture_output=True)
package=source/'test_regr/JointEmbodiedAgentInterface';transfer=package/'transfer';transfer.mkdir(exist_ok=True)
shutil.copy2(original/'test_regr/JointEmbodiedAgentInterface/generalization/rlbench_protocol.py',package/'generalization/rlbench_protocol.py')
for name,value in data['files'].items():(transfer/name).write_bytes(base64.b64decode(value))
base=Path(config['checkpoint']).parent
config.update(source=str(source),predecessor=str(prior),split_manifest=str(base/'split_identities.json'),
 original_configuration=str(base/'campaign_configuration.json'))
config.pop('idle_service_gpu',None)
# OSMesa-installed runtime also supports EGL. Its immutable image contains the
# original model environment and the native source simulator/assets.
config['model_image']='sha256:90c8c650ddb38a07fd71304f41fbdc12f0b2f8beeca9f7da04a5122afd3c67b0'
(root/'config.json').write_text(json.dumps(config,indent=2))
files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface'] for p in parent.rglob('*.py')}
(root/'provenance.json').write_text(json.dumps({'source_sha256':files,
 'initialization_sha256':hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest(),
 'diagnostic_gate_sha256':hashlib.sha256((prior/'sensitivity_complete.json').read_bytes()).hexdigest(),
 'configuration_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
 'gpu_policy':'completely-unused-only','independent_of_semantic_prompt':True,
 'source_observations':'common regenerated RGB format in both arms; zero simulation warm-up',
 'controller':'fixed; no controller optimizer or warm-up','epochs':4,'seed':42,'batch_size':8},indent=2))
cmd=['python3','-u',str(transfer/'continuation_campaign.py'),'--root',str(root)]
with (root/'supervisor.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'log':str(root/'supervisor.log')},indent=2))
print(json.dumps({'root':str(root),'supervisor_pid':proc.pid,'log':str(root/'supervisor.log')}))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
    result=subprocess.run(['ssh','-o','ProxyJump=none',args.host,'python3','-'],input=script,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()
