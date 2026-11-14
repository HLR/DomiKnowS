"""Freeze one selected-snapshot evaluation on gpu2 without changing training."""
import argparse
import json
from pathlib import Path
import subprocess
from .deploy import export_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='gpu2');p.add_argument('--output',required=True)
    p.add_argument('--training-run',required=True);a=p.parse_args()
    payload=json.dumps({'output':a.output,'training':a.training_run,
        'files':export_files(Path(__file__).resolve().parents[2])})
    script=r'''
from pathlib import Path
import json,subprocess,base64,hashlib,shutil
data=json.loads(PAYLOAD);root=Path(data['output']);training=Path(data['training'])
if root.parent!=Path('/home/auszok/experiments') or root.exists():raise ValueError('fresh evaluation root required')
selected=json.loads((training/'selected_snapshot_registration.json').read_text())
original=json.loads((training/'config.json').read_text());root.mkdir();source=root/'source'
subprocess.run(['git','clone','--shared',original['source'],str(source)],check=True,capture_output=True)
package=source/'test_regr/JointEmbodiedAgentInterface'
for name,value in data['files'].items():
 if name in ('main.py','checkpoint.py'):continue
 if '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe source export')
 path=package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value))
shutil.copy2(Path(original['source'])/'test_regr/JointEmbodiedAgentInterface/generalization/rlbench_protocol.py',package/'generalization/rlbench_protocol.py')
for name in ('main.py','checkpoint.py'):
 shutil.copy2(Path(original['source'])/'test_regr/JointEmbodiedAgentInterface'/name,package/name)
manifest={'files':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface',source/'domiknows/program',source/'domiknows/generation']
 for p in parent.rglob('*') if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}
mp=root/'source_manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
base=Path('/home/auszok/experiments/joint_stage1_metadata_generalization_20261108')
config={'source':str(source),'evaluation_source_manifest':str(mp),
 'evaluation_source_manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
 'training_configuration':str(training/'config.json'),'training_run':str(training),
 'checkpoint':selected['checkpoint'],'checkpoint_sha256':selected['sha256'],
 'grounding_head':selected['grounding_head'],'grounding_sha256':selected['grounding_sha256'],
 'vision_model':original['vision_model'],'planning_dir':original['planning_dir'],
 'libero_base':str(base),'libero_manifest':str(base/'libero_frozen_v3/manifest.json'),
 'model_image':original['model_image'],
 'executor_image':'sha256:08b4f594076cd4d4e0e392ef0c7934b82e525dbe07bd0a1ac005b7f710682d92',
 'selection':'source validation only; continuation epoch 2','seed':42,
 'policies':['none','shared-verified','full-metadata'],
 'protocol':'internal seen-task VLA + source-only planner transfer; no target training'}
(root/'config.json').write_text(json.dumps(config,indent=2,sort_keys=True))
cmd=['python3','-u',str(package/'transfer/anonymous_grounding/eval_campaign.py'),'--root',str(root)]
with (root/'supervisor.log').open('w') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':cmd},indent=2))
print(json.dumps({'root':str(root),'supervisor_pid':proc.pid,'checkpoint':selected['checkpoint']}))
'''.replace('PAYLOAD',repr(payload))
    result=subprocess.run(['ssh','-o','ProxyJump=none',a.host,'python3','-'],input=script,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()
