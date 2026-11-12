"""Freeze the combined experiment in a fresh server source directory."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def export_files(package):
    files={p.relative_to(package).as_posix():base64.b64encode(p.read_bytes()).decode()
           for p in (package/'transfer').rglob('*') if p.suffix in ('.py','.json','.md') and '__pycache__' not in p.parts}
    files['checkpoint.py']=base64.b64encode((package/'checkpoint.py').read_bytes()).decode()
    files['main.py']=base64.b64encode((package/'main.py').read_bytes()).decode()
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--host',default='gpu2')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    package=Path(__file__).resolve().parents[2]
    files=export_files(package)
    payload=json.dumps({'root':args.output,'files':files})
    script=r'''
from pathlib import Path
import subprocess,json,base64,hashlib,shutil
data=json.loads(PAYLOAD);root=Path(data['root'])
if any('\\' in name or Path(name).is_absolute() or '..' in Path(name).parts for name in data['files']):
 raise ValueError('source exports must use safe POSIX-relative paths')
if root.parent!=Path('/home/auszok/experiments') or root.exists():raise ValueError('new isolated experiment required')
base=Path('/home/auszok/experiments/joint_stage1_metadata_generalization_20261108')
original=base/'evaluation_source';root.mkdir();source=root/'source'
subprocess.run(['git','clone','--shared',str(original),str(source)],check=True,capture_output=True)
package=source/'test_regr/JointEmbodiedAgentInterface'
for name,value in data['files'].items():
 path=package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value))
rl=Path('/home/auszok/experiments/rlbench_model_evaluation_v33_20261110_attempt2/source/test_regr/JointEmbodiedAgentInterface/generalization/rlbench_protocol.py')
shutil.copy2(rl,package/'generalization/rlbench_protocol.py')
manifest={'files':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface',source/'domiknows/program',source/'domiknows/generation']
 for p in parent.rglob('*') if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}
source_manifest=root/'source_manifest.json';source_manifest.write_text(json.dumps(manifest,indent=2))
config={'source':str(source),'source_manifest':str(source_manifest),
 'source_manifest_sha256':hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
 'checkpoint':str(base/'checkpoints/joint_stage1_epoch_003.pt'),
 'split_manifest':str(base/'checkpoints/split_identities.json'),
 'original_configuration':str(base/'checkpoints/campaign_configuration.json'),
 'model_source':str(base/'checkpoints/model_source.json'),
 'planning_dir':'/home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data/planning',
 'official_evaluation_root':'/home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data/planning',
 'vision_model':'/home/auszok/.cache/huggingface/hub/models--google--siglip-base-patch16-384/snapshots/41aec1c83b32e0a6fca20ad88ba058aa5b5ea394',
 'seed':42,'epochs':4,'batch_size':8,
 'source_only':True,'target_training_used':False,
 'model_image':'sha256:90c8c650ddb38a07fd71304f41fbdc12f0b2f8beeca9f7da04a5122afd3c67b0'}
for key in ['checkpoint','split_manifest','original_configuration','model_source']:
 config[key+'_sha256']=hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
(root/'config.json').write_text(json.dumps(config,indent=2))
(root/'provenance.json').write_text(json.dumps({'initialization_sha256':hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest(),
 'experiment':'NEW COMBINATION: anonymous IDs + source/destination grounding',
 'data':'source VLABench historical internal split + canonical EAI; no target training',
 'source_parent_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
 'source_manifest_sha256':config['source_manifest_sha256'],'gpu_policy':'completely-unused-only',
 'controller':'fixed; no warm-up; no Stage 2 RL','epochs':4,'seed':42,'batch_size':8,
 'input':'original RGB plus verified renumbered RGB; no simulator reconstruction or segmentation color IDs',
 'loss':'source corrected masked CE + mass 0.5 + grounding 1.0; unknown pointer labels masked',
 'selection':'source validation only; no zero-regression gate'},indent=2))
cmd=['python3','-u',str(package/'transfer/anonymous_grounding/campaign.py'),'--root',str(root)]
with (root/'supervisor.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'log':str(root/'supervisor.log')},indent=2))
print(json.dumps({'root':str(root),'supervisor_pid':proc.pid,'log':str(root/'supervisor.log')}))
'''.replace('PAYLOAD',repr(payload))
    result=subprocess.run(['ssh','-o','ProxyJump=none',args.host,'python3','-'],input=script,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()
