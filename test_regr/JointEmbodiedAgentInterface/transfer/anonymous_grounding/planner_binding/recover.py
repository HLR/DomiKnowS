"""Host recovery: adopt live masked training and retry a pre-update full-CE OOM.

Model code, data, configuration and the running container remain unchanged.
Failed artifacts stay in place; explicit read/job aliases let the original
campaign use the successful retry for its already-frozen evaluations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def retry_command(original,root,job_name,index,uuid):
    root=Path(root);output=root/'arms'/job_name/'checkpoints';command=list(original)
    command[command.index('--name')+1]=root.name+'-'+job_name
    command[command.index('--gpus')+1]='device='+str(index)
    command[command.index('--output')+1]=str(output)
    for i,value in enumerate(command):
        if value.startswith(('CUDA_VISIBLE_DEVICES=','BINDING_GPU_UUID=')):command[i]=value.split('=')[0]+'='+uuid
    return command,output


def redirected_path(path,root,retry_name):
    path=Path(path);root=Path(root)
    if path==root/'jobs/training-full.json':return root/'jobs'/f'{retry_name}.json'
    old=root/'arms/full/checkpoints'
    if path.is_relative_to(old):return root/'arms'/retry_name/'checkpoints'/path.relative_to(old)
    return path


def unused_gpu(root):
    import fcntl
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    occupied={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    for line in rows.splitlines():
        index,uuid,memory,utilization=[x.strip() for x in line.split(',')]
        if int(memory)>=100 or int(utilization)!=0 or uuid in occupied:continue
        lock=(root/(uuid+'.lock')).open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        return int(index),uuid,lock
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--supersede-supervisor',action='store_true')
    a=p.parse_args();root=a.root.resolve();c=json.loads((root/'config.json').read_text());source=Path(c['source'])
    sys.path.insert(0,str(source/'test_regr/JointEmbodiedAgentInterface'))
    from transfer.campaign import run,save,utc
    from transfer.protocol import read,freeze,sha
    old=read(root/'jobs/training-full.json');masked=read(root/'jobs/training-masked.json')
    old_output=root/'arms/full/checkpoints'
    if old.get('exit_code')!=1 or 'OutOfMemoryError' not in Path(old['log']).read_text():raise ValueError('only a recorded pre-update OOM may be retried')
    if (old_output/'first_update.json').exists() or list(old_output.glob('progress_epoch*.pt')) or list(old_output.glob('joint_stage1_epoch_*.pt')):
        raise ValueError('failed arm has updates; explicit checkpoint recovery required')
    # Ensure this is the observed backward failure, before the optimizer step.
    if "gradients.capture('eai'" not in Path(old['log']).read_text():raise ValueError('unverified failure phase')
    state=read(root/'supervisor.json');old_pid=state['pid']
    if a.supersede_supervisor and Path(f'/proc/{old_pid}/cmdline').exists():
        cmd=Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(root) not in cmd or 'planner_binding/campaign.py' not in cmd:raise ValueError('supervisor ownership differs')
        os.kill(old_pid,signal.SIGTERM) # detached model containers are deliberately preserved
    freeze(root/'supervisor_before_recovery.json',state)
    supervisor={'pid':os.getpid(),'command':[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--supersede-supervisor'],
        'log':str(root/'logs/recovery-supervisor.log'),'role':'adopt live masked arm, retry pre-update full OOM, retain original evaluation plan'}
    save(root/'supervisor.json',supervisor);save(root/'supervisor_current.json',supervisor)
    name='training-full-retry001'
    if (root/'jobs'/f'{name}.json').exists():raise ValueError('preserve earlier retry; use a separately registered recovery')
    plan={'masked_container_id':masked['container_id'],'masked_pid':masked['pid'],'superseded_host_pid':old_pid,
        'failed_full_job_sha256':sha(root/'jobs/training-full.json'),'failed_full_log_sha256':sha(old['log']),
        'full_configuration_sha256':sha(root/'full.json'),'frozen_source_manifest_sha256':c['source_manifest_sha256'],
        'retry_job':name,'retry_output':str(root/'arms'/name/'checkpoints'),
        'recipe_changed':False,'model_updates_in_failed_attempt':0,'failed_artifacts_preserved':True}
    freeze(root/'recovery_plan.json',plan)
    def retry():
        queue=root/'queues'/f'{name}.json'
        while True:
            available=unused_gpu(root)
            if available:break
            save(queue,{'stage':'waiting-for-unused-gpu','updated_utc':utc(),'previous_failure':'external GPU contention; pre-update OOM'})
            time.sleep(5)
        index,uuid,lock=available
        try:
            command,output=retry_command(old['command'],root,name,index,uuid)
            save(queue,{'stage':'launching','physical_gpu':index,'gpu_uuid':uuid,'updated_utc':utc()})
            run(root,name,command,{'physical_gpu':index,'gpu_uuid':uuid,'retry_of':'training-full',
                'initialization_sha256':c['checkpoint_sha256'],'configuration_sha256':sha(root/'full.json')})
            if read(output/'complete.json')['epochs']!=4:raise ValueError('retry did not finish four epochs')
            return output
        finally:lock.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        live=pool.submit(run,root,'training-masked',masked['command'],{})
        full=pool.submit(retry)
        while not (live.done() and full.done()):
            arms={}
            for arm,folder in [('masked',root/'arms/masked/checkpoints'),('full',root/'arms'/name/'checkpoints')]:
                path=folder/'status.json';progress=read(path) if path.exists() else {'stage':'waiting-for-unused-gpu'}
                arms[arm]=progress
            for arm,future in [('masked',live),('full',full)]:
                if future.done() and future.exception() is not None:arms[arm]={'stage':'failed','error':str(future.exception())}
            save(root/'status.json',{'stage':'training-recovery','arms':arms,'full_retry':name,'updated_utc':utc()})
            time.sleep(10)
        live.result();full.result()
    initial=[read(folder/'initialization_parameters.json') for folder in (root/'arms/masked/checkpoints',root/'arms'/name/'checkpoints')]
    if initial[0]!=initial[1]:raise ValueError('retry initialization differs from matched control')
    freeze(root/'recovery_training_complete.json',{'matched_initialization':initial[0],'retry_job':name,'epochs':4})
    # Reuse the immutable original evaluator/orchestrator with explicit aliases.
    # No failed files, model sources, settings or original commands are overwritten.
    campaign=importlib.import_module('transfer.anonymous_grounding.planner_binding.campaign')
    original_read=campaign.read;original_run=campaign.run
    campaign.read=lambda path:original_read(redirected_path(path,root,name))
    def adopted_run(job_root,job_name,command,metadata):
        return original_run(job_root,name if job_name=='training-full' else job_name,command,metadata)
    campaign.run=adopted_run
    sys.argv=[sys.argv[0],'--root',str(root)];campaign.main()


if __name__=='__main__':main()
