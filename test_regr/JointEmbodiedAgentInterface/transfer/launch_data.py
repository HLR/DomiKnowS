"""Launch four bounded CPU LIBERO preparation shards, then audit their union."""
import argparse
import base64
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpu2')
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    names = ('__init__.py', 'protocol.py', 'data.py', 'prepare_data.py')
    files = {n: base64.b64encode((Path(__file__).parent/n).read_bytes()).decode() for n in names}
    payload = json.dumps({'campaign': args.campaign, 'files': files})
    script = r'''
from pathlib import Path
import json,subprocess,base64,hashlib,datetime
data=json.loads(PAYLOAD);root=Path(data['campaign']);config=json.loads((root/'config.json').read_text())
if root.parent!=Path('/home/auszok/experiments'):raise ValueError('invalid experiment root')
source=root/'libero_data_source_v2'/'transfer';output=root/'target_data/libero_sharded'
if source.exists() or output.exists():raise ValueError('preserve prior attempts; preparation already exists')
source.mkdir(parents=True);output.mkdir(parents=True)
for name,value in data['files'].items():(source/name).write_bytes(base64.b64decode(value))
settings={'config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
 'dataset':'libero','source_sha256':hashlib.sha256((source/'prepare_data.py').read_bytes()).hexdigest(),
 'seed':config['seed'],'num_shards':4}
(output/'settings.json').write_text(json.dumps(settings,sort_keys=True,indent=2)+'\n')
oldpath=root/'target_data/libero_job.json'
if oldpath.exists():
 old=json.loads(oldpath.read_text());d=json.loads(subprocess.check_output(['docker','inspect',old['container_id']],text=True))[0]
 if d['Name']!='/'+root.name+'-prepare-libero':raise ValueError('old preparation ownership differs')
 if d['State']['Running']:subprocess.run(['docker','stop','-t','10',d['Id']],check=True,capture_output=True)
 old.update(stopped_for_parallel_preparation=True,exit_code=json.loads(subprocess.check_output(['docker','inspect',d['Id']],text=True))[0]['State']['ExitCode'])
 oldpath.write_text(json.dumps(old,indent=2))
image='sha256:90c8c650ddb38a07fd71304f41fbdc12f0b2f8beeca9f7da04a5122afd3c67b0'
prefix=['docker','run','-d','--cpus=4','--memory=12g','-v','/home/auszok:/home/auszok',
 '-e','LP_NUM_THREADS=4','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
 '-e','LIBERO_CONFIG_PATH='+config['libero_config'],'-e','PYTHONPATH='+config['libero_pythonpath'],
 '--entrypoint',config['libero_python']]
baseargs=['-u',str(source/'prepare_data.py'),'--config',str(root/'config.json'),'--dataset','libero','--output',str(output),'--num-shards','4']
jobs=[]
for i in range(4):
 name=root.name+'-prepare-libero-shard'+str(i)
 cmd=prefix+['--name',name,image,*baseargs,'--shard-index',str(i)]
 cid=subprocess.check_output(cmd,text=True).strip();d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
 job={'command':cmd,'container_id':cid,'physical_gpu':None,'pid':d['State']['Pid'],'image_id':d['Image'],
  'renderer':'CPU OSMesa','source_sha256':settings['source_sha256']}
 path=root/'target_data'/f'libero_shard_{i:03d}_job.json';path.write_text(json.dumps(job,indent=2));jobs.append(str(path))
finalcmd=['docker','run','--rm','--cpus=4','--memory=12g','-v','/home/auszok:/home/auszok',
 '-e','PYTHONPATH='+config['libero_pythonpath'],
 '--entrypoint',config['libero_python'],image,*baseargs,'--finalize']
monitor="""from pathlib import Path
import json,subprocess,time
root=Path(ROOT);jobs=JOBS
while True:
 active=False;failed=[]
 for name in jobs:
  p=Path(name);j=json.loads(p.read_text());state=json.loads(subprocess.check_output(['docker','inspect',j['container_id']],text=True))[0]['State']
  if state['Running']:active=True;continue
  j['exit_code']=state['ExitCode'];p.write_text(json.dumps(j,indent=2))
  a=subprocess.run(['docker','logs',j['container_id']],capture_output=True,text=True);p.with_suffix('.log').write_text(a.stdout+a.stderr)
  if state['ExitCode']:failed.append(name)
 if failed:
  (root/'target_data/libero_sharded_status.json').write_text(json.dumps({'stage':'failed','jobs':failed}));break
 if not active:
  with (root/'target_data/libero_finalize.log').open('w') as f:r=subprocess.run(FINAL,stdout=f,stderr=subprocess.STDOUT)
  (root/'target_data/libero_sharded_status.json').write_text(json.dumps({'stage':'complete' if r.returncode==0 else 'failed','exit_code':r.returncode,'command':FINAL}));break
 time.sleep(10)
""".replace('ROOT',repr(str(root))).replace('JOBS',repr(jobs)).replace('FINAL',repr(finalcmd))
monitorpath=root/'libero_data_source_v2/monitor.py';monitorpath.write_text(monitor)
with (root/'target_data/libero_monitor.log').open('w') as log:proc=subprocess.Popen(['python3','-u',str(monitorpath)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'target_data/libero_monitor_job.json').write_text(json.dumps({'pid':proc.pid,'script':str(monitorpath)},indent=2))
print(json.dumps({'monitor_pid':proc.pid,'shards':4,'output':str(output),'physical_gpu':None}))
'''.replace('PAYLOAD', repr(payload))
    result = subprocess.run(['ssh', '-o', 'ProxyJump=none', args.host, 'python3', '-'],
                            input=script, capture_output=True, text=True)
    print(result.stdout); print(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
