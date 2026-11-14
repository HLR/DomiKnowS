"""Freeze a matched source-only campaign onto the existing scene-preparation root."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',default='gpu2')
    p.add_argument('--prepared-root');a=p.parse_args()
    folder=Path(__file__).parent
    files={f.name:base64.b64encode(f.read_bytes()).decode() for f in folder.iterdir() if f.suffix in ('.py','.md')}
    script=r'''
from pathlib import Path
import json,hashlib,base64,subprocess
d=json.loads(PAYLOAD);root=Path(d['root'])
if d.get('prepared_root'):
 prepared=Path(d['prepared_root'])
 if root.parent!=Path('/home/auszok/experiments') or root.exists():raise ValueError('new attempt root required')
 gate=json.loads((prepared/'fresh_libero/settled_split_audit.json').read_text())
 if not gate['passed'] or gate['manifest_sha256']!=hashlib.sha256((prepared/'fresh_libero/manifest.json').read_bytes()).hexdigest():raise ValueError('previous settled-scene gate changed')
 root.mkdir();(root/'prepare_config.json').write_bytes((prepared/'prepare_config.json').read_bytes())
if root.parent!=Path('/home/auszok/experiments') or not (root/'prepare_config.json').exists() or (root/'source').exists():raise ValueError('fresh prepared campaign root required')
previous=Path('/home/auszok/experiments/joint_object_transfer_fixes_20261113_attempt2')
old=json.loads((previous/'config.json').read_text());old_source=Path(old['source']);source=root/'source'
parent_commit=subprocess.check_output(['git','-C',str(old_source),'rev-parse','HEAD'],text=True).strip()
# Clone from the root object store to avoid deeply nested --shared alternates.
result=subprocess.run(['git','clone','--shared','--no-checkout','/home/auszok/DomiKnowS',str(source)],capture_output=True,text=True)
if result.returncode:raise RuntimeError(result.stderr)
subprocess.run(['git','-C',str(source),'checkout','--detach',parent_commit],check=True,capture_output=True)
prior=json.loads(Path(old['evaluation_source_manifest']).read_text())
for name,expected in prior['files'].items():
 raw=(old_source/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('archived source changed '+name)
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
package=source/'test_regr/JointEmbodiedAgentInterface';folder=package/'transfer/anonymous_grounding/planner_binding';folder.mkdir()
for name,value in d['files'].items():
 if Path(name).name!=name or '\\' in name:raise ValueError('unsafe source filename')
 (folder/name).write_bytes(base64.b64decode(value))
training=Path('/home/auszok/experiments/joint_anonymous_id_grounding_source_20261111_attempt6')
c=json.loads((training/'config.json').read_text());selected=json.loads((training/'selected_snapshot_registration.json').read_text())
original_manifest=json.loads(Path(c['source_manifest']).read_text())
dependency_names=[name for name in original_manifest['files'] if name.startswith(('test_regr/VLABenchAgentInterface/','domiknows/generation/')) or name in (
 'test_regr/JointEmbodiedAgentInterface/main.py','test_regr/JointEmbodiedAgentInterface/world_graph.py',
 'test_regr/JointEmbodiedAgentInterface/transfer/train.py','test_regr/JointEmbodiedAgentInterface/transfer/source_transform.py',
 'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/audits.py','test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/train.py',
 'test_regr/JointEmbodiedAgentInterface/utils/planner_context.py','test_regr/JointEmbodiedAgentInterface/utils/generalization_protocol.py')]
deps={}
for name in dependency_names:
 expected=original_manifest['files'][name]
 if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise ValueError('model-independent audit dependencies changed '+name)
 deps[name]=expected
manifest={'files':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface',source/'domiknows/program',source/'domiknows/generation']
 for p in parent.rglob('*') if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}
mp=root/'source_manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
c.update(source=str(source),source_manifest=str(mp),source_manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
 checkpoint=selected['checkpoint'],checkpoint_sha256=selected['sha256'],output=str(root),
 observations=str(training/'data/manifest.json'),features=str(training/'features/manifest.json'),
 grounding_head=selected['grounding_head'],grounding_sha256=selected['grounding_sha256'],
 audit_cache=str(training/'checkpoints/audits'),audit_dependency_hashes=deps,
 libero_base=old['libero_base'],libero_manifest=old['libero_manifest'],executor_image=old['executor_image'],
 arms=['masked','full'],ce_scope='VLABench graph labels; matched EAI objective',
 operation_grounding_weight=1.,allowed_mass_weight=.5,max_pointer_residual_gain=4.,
 evaluation_plan={'states':[40,41,42,43,44],'snapshots':['initialization','masked source-val best','full source-val best'],'policies':['none','executor'],'target_training':False})
if d.get('prepared_root'):
 c['prepared_external_root']=d['prepared_root']
 prior=Path(d['prepared_root'])/'text_priors/manifest.json'
 if prior.exists():
  (root/'text_priors').mkdir();(root/'text_priors/manifest.json').write_bytes(prior.read_bytes())
for name in ('observations','features'):
 c[name+'_sha256']=hashlib.sha256(Path(c[name]).read_bytes()).hexdigest()
if not Path(c['audit_cache']).exists():raise ValueError('completed source audits unavailable')
(root/'launch_config.json').write_text(json.dumps(c,indent=2,sort_keys=True))
command=['python3','-u',str(folder/'campaign.py'),'--root',str(root)]
with (root/'supervisor.log').open('w') as log:proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':command,'log':str(root/'supervisor.log')},indent=2))
print(json.dumps({'root':str(root),'supervisor_pid':proc.pid,'checkpoint':c['checkpoint'],'matched_arms':c['arms']}))
'''.replace('PAYLOAD',repr(json.dumps({'root':a.root,'files':files,'prepared_root':a.prepared_root})))
    result=subprocess.run(['ssh','-o','ProxyJump=none','-o','ConnectTimeout=8',a.host,'python3','-'],input=script,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()
