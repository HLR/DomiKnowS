"""Launch exactly one fresh four-epoch campaign after its CPU preflight passes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from run_campaign import free_gpu, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--source',required=True)
    args=parser.parse_args();root=Path(args.root);source=Path(args.source)
    while not (root/'preflight.json').exists():time.sleep(15)
    preflight=json.loads((root/'preflight.json').read_text())
    if not preflight['passed']:raise ValueError('preflight failed')
    audit=json.loads((root/'checkpoints/training_constraint_audit.json').read_text())
    if audit['examples']!=4500 or audit['conflicts']:raise ValueError('reference audit incomplete or conflicting')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('release source must be clean')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    training_args=json.loads((root/'arguments.json').read_text())
    if any(x in training_args for x in ('--resume','--resume-adapter','--initialize-from')):
        raise ValueError('fresh campaign must not resume adapted weights')
    gpu=None
    while gpu is None:
        gpu=free_gpu(root,[2])
        if gpu is None:time.sleep(15)
    index,uuid,lock=gpu
    name='metadata-stage1-four-epochs-20261108'
    command=['/opt/dominows/venv/bin/python','-u','-m','test_regr.JointEmbodiedAgentInterface.main',*training_args]
    configuration={'source_commit':commit,'arguments':training_args,'physical_gpu':index,'gpu_uuid':uuid,
        'model_image':'dominows:latest','internal_experiment':True,'snapshots':5,
        'evaluation_corpus_exposure':True,'initialization':'base Qwen with untrained Joint heads'}
    config_path=root/'checkpoints/campaign_configuration.json'
    if config_path.exists():raise ValueError('campaign already launched; use recorded recovery protocol')
    write(config_path,configuration)
    launch=['docker','run','-d','--name',name,'--gpus','device='+str(index),'--ipc=host',
        '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace',
        '-e','CUDA_VISIBLE_DEVICES='+uuid,'-e','PYTHONPATH='+str(source)+':/opt/dominows/VLABench:/home/auszok/domiknowsrelated/src:/opt/dominows/PreciseRoIPooling/pytorch',
        '-e','VLABENCH_OFFICIAL_EVALUATION_ROOT=/home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data/planning',
        '-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4','-w',str(source),
        '--entrypoint',command[0],'dominows:latest',*command[1:]]
    subprocess.run(launch,check=True)
    worker=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    write(root/'training_job.json',dict(configuration,container=name,pid=worker['State']['Pid'],
        host_monitor_pid=os.getpid(),command=command,docker_command=launch,image_id=worker['Image'],
        log=str(root/'training.log'),started_utc=datetime.now(timezone.utc).isoformat()))
    queue=['python3','-u',str(source/'test_regr/JointEmbodiedAgentInterface/generalization/run_campaign.py'),
        '--root',str(root),'--source',str(source),'--training-container',name,'--training-gpu',str(index)]
    with (root/'campaign_queue.log').open('w') as log:
        process=subprocess.Popen(queue,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'queue_job.json',{'pid':process.pid,'command':queue,'log':str(root/'campaign_queue.log')})
    try:
        with (root/'training.log').open('w') as log:
            follow=subprocess.Popen(['docker','logs','-f',name],stdout=log,stderr=subprocess.STDOUT)
            code=int(subprocess.check_output(['docker','wait',name],text=True).strip());follow.wait()
        (root/'training_exit_code.txt').write_text(str(code)+'\n')
    finally:lock.close()


if __name__=='__main__':main()
