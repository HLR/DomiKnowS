"""Freeze a compatible evaluation overlay before scoring; retain input producers."""
import argparse,json,subprocess
from pathlib import Path
from .deploy import export_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',default='gpu2');a=p.parse_args()
    payload=json.dumps({'root':a.root,'files':export_files(Path(__file__).resolve().parents[2])})
    script=r'''
from pathlib import Path
import json,subprocess,hashlib,base64,shutil
data=json.loads(PAYLOAD);root=Path(data['root'])
if not (root/'supervisor_retired_before_planning.json').exists() or list((root/'jobs').glob('planning*.json')):
 raise ValueError('handoff allowed only before planning')
old=json.loads((root/'config.json').read_text());training=json.loads(Path(old['training_configuration']).read_text())
source=root/'source_v2'
if source.exists():raise ValueError('preserve frozen overlay')
subprocess.run(['git','clone','--shared',training['source'],str(source)],check=True,capture_output=True)
package=source/'test_regr/JointEmbodiedAgentInterface'
for name,value in data['files'].items():
 if name in ('main.py','checkpoint.py'):continue
 if '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe export')
 path=package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value))
for name in ('main.py','checkpoint.py','generalization/rlbench_protocol.py'):
 shutil.copy2(Path(training['source'])/'test_regr/JointEmbodiedAgentInterface'/name,package/name)
for name in ('eval_observations.py','features.py','markers.py','glyph_atlas.json'):
 before=Path(old['source'])/'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding'/name
 after=package/'transfer/anonymous_grounding'/name
 if before.read_bytes()!=after.read_bytes():raise ValueError('running input producer contract differs: '+name)
manifest={'files':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface',source/'domiknows/program',source/'domiknows/generation']
 for p in parent.rglob('*') if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}
mp=root/'source_manifest_v2.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
config={**old,'source':str(source),'evaluation_source_manifest':str(mp),
 'evaluation_source_manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest()}
cp=root/'config_v2.json';cp.write_text(json.dumps(config,indent=2,sort_keys=True))
cmd=['docker','run','--rm','--cpus=2','--memory=8g','-v','/home/auszok:/home/auszok','-w',str(source),
 '-e','PYTHONPATH='+str(source)+':/opt/dominows/VLABench:/home/auszok/domiknowsrelated/src:/opt/dominows/PreciseRoIPooling/pytorch',
 '--entrypoint','/opt/dominows/venv/bin/python',old['model_image'],'-m','pytest',
 'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding','-q']
result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
(root/'evaluation_tests_v2.log').write_text(result.stdout+result.stderr)
print('tests',result.returncode,(result.stdout+result.stderr)[-1500:])
if result.returncode:raise RuntimeError('frozen runtime preflight failed')
(root/'overlay_v2_provenance.json').write_text(json.dumps({'reason':'retain exact training main/checkpoint dependency interfaces; updated evaluation-only modules',
 'training_runtime_source':training['source'],'source_manifest_sha256':config['evaluation_source_manifest_sha256'],
 'input_producers_preserved':True,'evaluation_started':False,'tests_passed':True},indent=2))
command=['python3','-u',str(package/'transfer/anonymous_grounding/eval_campaign.py'),'--root',str(root),'--config',str(cp)]
with (root/'supervisor_v2.log').open('w') as log:proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor_current.json').write_text(json.dumps({'pid':proc.pid,'command':command},indent=2))
print('supervisor',proc.pid)
'''.replace('PAYLOAD',repr(payload))
    r=subprocess.run(['ssh','-o','ProxyJump=none',a.host,'python3','-'],input=script,capture_output=True,text=True)
    print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)


if __name__=='__main__':main()
