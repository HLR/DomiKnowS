"""Deploy and orchestrate the fixed-snapshot RLBench follow-up."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='transfer.anonymous_grounding'
from ..campaign import run,idle_gpu,save,utc
from ..protocol import read,freeze,sha
from .campaign import runtime_environment


def campaign(root):
    c=read(root/'config.json');source=Path(c['source']);module='test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.eval_rlbench'
    prefix=['docker','run','-d','--ipc=host','--cpus=8','--memory=64g','-v','/home/auszok:/home/auszok',
        '-v','/home/auszok/DomiKnowS:/workspace','-w',str(source)]
    for k,v in runtime_environment(source,c['planning_dir']).items():prefix+=['-e',k+'='+v]
    def gpu_job(name,args):
        index,uuid,lock=idle_gpu(root)
        try:
            cmd=prefix+['--name',root.name+'-'+name,'--gpus','device='+str(index),'-e','CUDA_VISIBLE_DEVICES='+uuid,
                '--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m',module,*args]
            run(root,name,cmd,{'physical_gpu':index,'gpu_uuid':uuid,'checkpoint_sha256':c['checkpoint_sha256']})
        finally:lock.close()
    def condition(policy):
        gpu_job('planning-'+policy,['evaluate','--config',str(root/'config.json'),'--output',str(root/'planning'),
            '--observations',str(root/'observations/manifest.json'),'--features',str(root/'features/manifest.json'),'--policy',policy])
        cmd=['docker','run','-d','--name',root.name+'-execution-'+policy,'--user','0','--cpus=6','--memory=12g',
            '-v','/home/auszok:/home/auszok',c['executor_image'],str(Path(c['executor_source'])/'rlbench_campaign.py'),
            'evaluate','--manifest',c['rlbench_manifest'],'--calibration',c['rlbench_calibration'],
            '--predictions',str(root/'planning'/policy/'predictions.jsonl'),'--output',str(root/'execution'/policy)]
        run(root,'execution-'+policy,cmd,{'physical_gpu':None,'executor':'retained source_v33; calibration replay 91/100'})
    try:
        save(root/'status.json',{'stage':'preparing-observation-only-inputs','updated_utc':utc()})
        cmd=prefix+['--name',root.name+'-prepare','--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],
            '-u','-m',module,'prepare','--config',str(root/'config.json'),'--output',str(root/'observations')]
        run(root,'prepare',cmd,{'physical_gpu':None})
        save(root/'status.json',{'stage':'extracting-frozen-features','updated_utc':utc()})
        gpu_job('features',['features','--config',str(root/'config.json'),'--observations',str(root/'observations/manifest.json'),
            '--output',str(root/'features')])
        save(root/'status.json',{'stage':'planning-and-native-execution','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks=[pool.submit(condition,p) for p in ('none','shared-verified')]
            for task in tasks:task.result()
        reports={}
        for policy in ('none','shared-verified'):
            reports[policy]=read(root/'planning'/policy/'report.json')
            reports[policy]['execution']=read(root/'execution'/policy/'summary.json')
            rows=[json.loads(s) for s in (root/'execution'/policy/'episodes.jsonl').read_text().splitlines()]
            reports[policy]['completed_successes']=sum(r['success'] and r['termination_reason']=='completed' and r['holding_at_end'] is None for r in rows)
            if len(rows)!=100 or len({r['episode_id'] for r in rows})!=100:raise ValueError('native coverage differs')
        if reports['none']['input_hashes']!=reports['shared-verified']['input_hashes']:
            raise ValueError('constraint conditions changed neural generation inputs')
        freeze(root/'results.json',{'checkpoint':c['checkpoint'],'checkpoint_sha256':c['checkpoint_sha256'],
            'reports':reports,'coverage':read(root/'observations/manifest.json')['coverage'],
            'prior_evaluation':c['prior_evaluation'],'abstract_goal_scoring':'not defined for RLBench'})
        save(root/'status.json',{'stage':'complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


def deploy(output,previous,host):
    from .deploy import export_files
    files={k:v for k,v in export_files(Path(__file__).resolve().parents[2]).items() if k.startswith('transfer/anonymous_grounding/')}
    payload=json.dumps({'root':output,'previous':previous,'files':files})
    script=r'''
from pathlib import Path
import json,hashlib,subprocess,shutil,base64,sys
data=json.loads(PAYLOAD);root=Path(data['root']);previous=Path(data['previous'])
if root.parent!=Path('/home/auszok/experiments') or root.exists():raise ValueError('fresh experiment required')
old=json.loads((previous/'config_v2.json').read_text());prior_source=Path(old['source']);root.mkdir();source=root/'source'
subprocess.run(['git','clone','--shared',str(prior_source),str(source)],check=True,capture_output=True)
prior_manifest=json.loads(Path(old['evaluation_source_manifest']).read_text())
for name,expected in prior_manifest['files'].items():
 original=prior_source/name
 if hashlib.sha256(original.read_bytes()).hexdigest()!=expected:raise ValueError('prior frozen evaluator changed')
 path=source/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,path)
package=source/'test_regr/JointEmbodiedAgentInterface'
for name,value in data['files'].items():
 if '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe export')
 path=package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value))
manifest={'files':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
 for parent in [package,source/'test_regr/VLABenchAgentInterface',source/'domiknows/program',source/'domiknows/generation']
 for p in parent.rglob('*') if p.suffix in ('.py','.json') and '__pycache__' not in p.parts}}
mp=root/'source_manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
native=Path('/home/auszok/experiments/rlbench_model_evaluation_v33_20261110_attempt2')
c={**old,'source':str(source),'prior_evaluation':str(previous),'evaluation_source_manifest':str(mp),
 'evaluation_source_manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
 'rlbench_manifest':'/home/auszok/experiments/rlbench_followup_20261109/frozen_v2/manifest.json',
 'rlbench_calibration':str(native/'calibration_replay_v33/summary.json'),
 'executor_source':str(native/'native_replay_source_v33'),
 'executor_image':'sha256:5fff7d4968af50c6278328c2086aa6747f9fbbf0523765ae9c389dee83b55a58',
 'policies':['none','shared-verified'],'dataset':'rlbench'}
(root/'config.json').write_text(json.dumps(c,indent=2,sort_keys=True))
sys.path.insert(0,str(package))
from transfer.anonymous_grounding.campaign import runtime_environment
from generalization.rlbench_protocol import validate_calibration
gate=validate_calibration(c['rlbench_manifest'],c['rlbench_calibration'],c['executor_source'])
(root/'calibration_gate.json').write_text(json.dumps(gate,indent=2))
cmd=['docker','run','--rm','--cpus=2','--memory=8g','-v','/home/auszok:/home/auszok','-w',str(source),
 ]
for name,value in runtime_environment(source,c['planning_dir']).items():cmd+=['-e',name+'='+value]
cmd+=['--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-m','pytest',
 'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding','-q']
result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
(root/'tests.log').write_text(result.stdout+result.stderr);print('tests',result.returncode,(result.stdout+result.stderr)[-1400:])
if result.returncode:raise ValueError('frozen evaluation tests failed')
command=['python3','-u',str(package/'transfer/anonymous_grounding/eval_rlbench_host.py'),'--root',str(root)]
with (root/'supervisor.log').open('w') as log:proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'supervisor.json').write_text(json.dumps({'pid':proc.pid,'command':command},indent=2));print('supervisor',proc.pid)
'''.replace('PAYLOAD',repr(payload))
    result=subprocess.run(['ssh','-o','ProxyJump=none',host,'python3','-'],input=script,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--deploy',type=str);p.add_argument('--previous',type=str)
    p.add_argument('--host',default='gpu2');a=p.parse_args()
    if a.deploy:deploy(a.deploy,a.previous,a.host)
    else:campaign(a.root)


if __name__=='__main__':main()
