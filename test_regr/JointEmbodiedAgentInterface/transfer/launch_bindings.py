"""Start method 2 only after method 1 is qualified and its report published."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpu2')
    parser.add_argument('--predecessor', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    names = ('__init__.py', 'protocol.py', 'permutation.py', 'annotations.py', 'bind_images.py')
    payload = json.dumps({'predecessor': args.predecessor, 'output': args.output,
        'files': {n: base64.b64encode((Path(__file__).parent/n).read_bytes()).decode() for n in names}})
    script = r'''
from pathlib import Path
import json,base64,hashlib,subprocess,datetime
data=json.loads(PAYLOAD);prior=Path(data['predecessor']);root=Path(data['output'])
if root.parent!=Path('/home/auszok/experiments') or root.exists():raise ValueError('new isolated experiment required')
qual=json.loads((prior/'semantic-prompt/qualification.json').read_text())
published=json.loads((prior/'semantic-prompt/report_update.json').read_text())
if not qual['technical_validation_passed']:raise ValueError('previous method not qualified')
digest=hashlib.sha256((prior/'results.json').read_bytes()).hexdigest()
if qual['results_sha256']!=digest or published['results_sha256']!=digest:raise ValueError('previous report/evaluation differs')
root.mkdir();package=root/'binding_source/transfer';package.mkdir(parents=True)
for name,value in data['files'].items():(package/name).write_bytes(base64.b64decode(value))
config=json.loads((prior/'config.json').read_text())
# This independent method uses completely unused GPUs, not the predecessor's
# optional permission to share an idle service GPU.
config.pop('idle_service_gpu',None)
(root/'binding_config.json').write_text(json.dumps(config,indent=2))
proof={'method':'id-augmentation','phase':'verify-observational-image-bindings','predecessor':str(prior),
 'previous_results_sha256':digest,'checkpoint_sha256':hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest(),
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')},
 'permutation_trials':5,'seed':42,'epochs':4,'batch_size':8,'independent_of_semantic_prompt':True,
 'gpu_policy':'completely-unused-only; select and recheck at launch'}
(root/'phase_manifest.json').write_text(json.dumps(proof,indent=2))
for dataset in ('libero','rlbench'):
 name=root.name+'-bindings-'+dataset;item=config['datasets'][dataset]
 prefix=['docker','run','-d','--name',name,'--cpus=6','--memory=12g','-v','/home/auszok:/home/auszok']
 args=['--config',str(root/'binding_config.json'),'--dataset',dataset,'--output',str(root/'bindings'/dataset)]
 if dataset=='libero':
  image='sha256:90c8c650ddb38a07fd71304f41fbdc12f0b2f8beeca9f7da04a5122afd3c67b0'
  cmd=prefix+['-e','LP_NUM_THREADS=4','-e','OMP_NUM_THREADS=2','-e','LIBERO_CONFIG_PATH='+config['libero_config'],
   '-e','PYTHONPATH='+config['libero_pythonpath'],'--entrypoint',config['libero_python'],image,'-u',str(package/'bind_images.py'),*args]
 else:
  cmd=prefix+[item['runtime_image'],str(package/'bind_images.py'),*args]
 cid=subprocess.check_output(cmd,text=True).strip();d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
 (root/'jobs').mkdir(exist_ok=True)
 (root/'jobs'/f'bindings-{dataset}.json').write_text(json.dumps({'command':cmd,'container_id':cid,
  'physical_gpu':None,'pid':d['State']['Pid'],'image_id':d['Image'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
 print(dataset,'CPU PID',d['State']['Pid'])
'''.replace('PAYLOAD', repr(payload))
    result = subprocess.run(['ssh', '-o', 'ProxyJump=none', args.host, 'python3', '-'],
                            input=script, capture_output=True, text=True)
    print(result.stdout); print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
