"""Adopt immutable CPU preparation and freeze an updated pre-training source."""
import argparse
import json
from pathlib import Path
import subprocess
from .deploy import export_files


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--host',default='gpu2')
    parser.add_argument('--root',required=True);args=parser.parse_args()
    payload=json.dumps({'root':args.root,'files':export_files(Path(__file__).resolve().parents[2])})
    script=r'''
from pathlib import Path
import json,base64,hashlib,subprocess,shutil
data=json.loads(PAYLOAD);root=Path(data['root'])
if root.parent!=Path('/home/auszok/experiments'):raise ValueError('invalid owned experiment root')
if not (root/'supervisor_retired_before_training.json').exists() or (root/'jobs/training.json').exists():
 raise ValueError('handoff requires retired owned supervisor and no model job')
old=json.loads((root/'config.json').read_text());source=root/'training_source_v2'
if source.exists():raise ValueError('preserve existing frozen training overlay')
subprocess.run(['git','clone','--shared',str(Path(old['source'])),str(source)],check=True,capture_output=True)
package=source/'test_regr/JointEmbodiedAgentInterface'
for name,value in data['files'].items():
 if '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('invalid source path')
 path=package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value))
original=Path(old['source'])/'test_regr/JointEmbodiedAgentInterface'
shutil.copy2(original/'generalization/rlbench_protocol.py',package/'generalization/rlbench_protocol.py')
# The running producer is unchanged; reject handoffs that alter its contract.
for name in ['prepare.py','markers.py','glyph_atlas.json']:
 before=original/'transfer/anonymous_grounding'/name;after=package/'transfer/anonymous_grounding'/name
 if hashlib.sha256(before.read_bytes()).digest()!=hashlib.sha256(after.read_bytes()).digest():
  raise ValueError('running preparation source differs: '+name)
manifest={'files':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface',source/'domiknows/program',source/'domiknows/generation']
 for p in parent.rglob('*') if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}
mp=root/'training_source_manifest_v2.json';mp.write_text(json.dumps(manifest,indent=2))
config={**old,'source':str(source),'source_manifest':str(mp),'source_manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
 'preparation_config':str(root/'config.json'),'augmentation_policy':'balanced identity for two-object scenes; identity once per four epochs otherwise'}
cp=root/'training_config_v2.json';cp.write_text(json.dumps(config,indent=2))
(root/'training_overlay_v2_provenance.json').write_text(json.dumps({'preparation_config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
 'training_config_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'reason':'freeze balanced identity/permutation schedule before any model updates',
 'preparation_restarted':False,'source_manifest_sha256':config['source_manifest_sha256']},indent=2))
cmd=['python3','-u',str(package/'transfer/anonymous_grounding/campaign.py'),'--root',str(root),'--config',str(cp)]
with (root/'supervisor_training_v2.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor_current.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'log':str(root/'supervisor_training_v2.log')},indent=2))
print(json.dumps({'supervisor_pid':proc.pid,'config':str(cp),'preparation_container_preserved':True}))
'''.replace('PAYLOAD',repr(payload))
    result=subprocess.run(['ssh','-o','ProxyJump=none',args.host,'python3','-'],input=script,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()
