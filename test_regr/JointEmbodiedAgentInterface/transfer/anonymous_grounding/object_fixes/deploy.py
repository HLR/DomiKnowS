"""Export only these fixes onto the verified archived evaluation runtime."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='gpu2');p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(__file__).parent;package=folder.parents[2]
    files={f.relative_to(package).as_posix():base64.b64encode(f.read_bytes()).decode()
        for f in folder.rglob('*') if f.suffix in ('.py','.md') and '__pycache__' not in f.parts}
    agent=folder.parent/'agent.py';files[agent.relative_to(package).as_posix()]=base64.b64encode(agent.read_bytes()).decode()
    protocol=package/'transfer/protocol.py';files[protocol.relative_to(package).as_posix()]=base64.b64encode(protocol.read_bytes()).decode()
    script=r'''
import json,base64,hashlib,subprocess,shutil
from pathlib import Path
d=json.loads(PAYLOAD);root=Path(d['output'])
if root.parent!=Path('/home/auszok/experiments') or root.exists():raise ValueError('fresh isolated experiment required')
previous=Path('/home/auszok/experiments/joint_anonymous_grounding_evaluation_20261113')
c=json.loads((previous/'config_v2.json').read_text());old=Path(c['source'])
root.mkdir();source=root/'source';subprocess.run(['git','clone','--shared',str(old),str(source)],check=True,capture_output=True)
prior=json.loads(Path(c['evaluation_source_manifest']).read_text())
for name,expected in prior['files'].items():
 raw=(old/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('archived runtime changed '+name)
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
package=source/'test_regr/JointEmbodiedAgentInterface'
for name,value in d['files'].items():
 if '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe export')
 target=package/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(value))
manifest={'files':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface',source/'domiknows/program',source/'domiknows/generation']
 for p in parent.rglob('*') if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}
mp=root/'source_manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
c.update(source=str(source),evaluation_source_manifest=str(mp),evaluation_source_manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
 output=str(root),observations=str(previous/'observations/manifest.json'),features=str(previous/'features/manifest.json'),
 archive_predictions=str(previous/'evaluation/libero/none/predictions.jsonl'),
 archive_execution=str(previous/'evaluation/libero/none/execution'),
 conditions=['baseline','raw','contract','raw-contract','aligned','aligned-contract'],
 scope='diagnostic regression on previously inspected test scenes; no training or target labels in generation')
for name in ('observations','features','libero_manifest','archive_predictions'):
 c[name+'_sha256']=hashlib.sha256(Path(c[name]).read_bytes()).hexdigest()
cp=root/'config.json';cp.write_text(json.dumps(c,indent=2,sort_keys=True))
cmd=['python3','-u',str(package/'transfer/anonymous_grounding/object_fixes/preflight.py'),'--root',str(root)]
with (root/'supervisor.log').open('w') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'log':str(root/'supervisor.log')},indent=2))
print(json.dumps({'root':str(root),'supervisor_pid':proc.pid,'log':str(root/'supervisor.log')}))
'''.replace('PAYLOAD',repr(json.dumps({'output':a.output,'files':files})))
    result=subprocess.run(['ssh','-o','ProxyJump=none','-o','ConnectTimeout=8',a.host,'python3','-'],input=script,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()
