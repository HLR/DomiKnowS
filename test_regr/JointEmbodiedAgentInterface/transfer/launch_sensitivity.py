"""Freeze and launch the method-2 sensitivity phase without sharing GPUs."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpu2')
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    payload = {'root': args.root, 'files': {p.name: base64.b64encode(p.read_bytes()).decode()
               for p in Path(__file__).parent.glob('*.py')}}
    script = r'''
from pathlib import Path
import json,base64,subprocess,shutil,hashlib
data=json.loads(PAYLOAD);root=Path(data['root'])
if root.parent!=Path('/home/auszok/experiments'):raise ValueError('invalid campaign root')
config=json.loads((root/'binding_config.json').read_text());source=root/'sensitivity_source'
if source.exists():raise ValueError('preserve prior source and jobs; use a fresh attempt')
original=Path(config['source'])
subprocess.run(['git','clone','--shared',str(original),str(source)],check=True,capture_output=True)
package=source/'test_regr/JointEmbodiedAgentInterface'
shutil.copy2(original/'test_regr/JointEmbodiedAgentInterface/generalization/rlbench_protocol.py',package/'generalization/rlbench_protocol.py')
transfer=package/'transfer';transfer.mkdir(exist_ok=True)
for name,value in data['files'].items():(transfer/name).write_bytes(base64.b64decode(value))
config['source']=str(source);config.pop('idle_service_gpu',None)
(root/'sensitivity_config.json').write_text(json.dumps(config,indent=2))
(root/'sensitivity_source_provenance.json').write_text(json.dumps({
 'parent_source':str(original),'checkpoint_sha256':hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest(),
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in transfer.glob('*.py')},
 'gpu_policy':'completely-unused-only'},indent=2))
cmd=['python3','-u',str(transfer/'sensitivity_campaign.py'),'--root',str(root)]
with (root/'sensitivity_supervisor.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'sensitivity_supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'log':str(root/'sensitivity_supervisor.log')},indent=2))
print(json.dumps({'pid':proc.pid,'log':str(root/'sensitivity_supervisor.log'),'gpu_policy':'completely-unused-only'}))
'''.replace('PAYLOAD', repr(json.dumps(payload)))
    result = subprocess.run(['ssh', '-o', 'ProxyJump=none', args.host, 'python3', '-'],
                            input=script, capture_output=True, text=True)
    print(result.stdout); print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
