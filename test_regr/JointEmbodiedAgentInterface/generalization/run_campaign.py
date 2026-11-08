"""Host-side snapshot queue. Uses only this experiment's containers/artifacts."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
import hashlib


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def free_gpu(root, reserved):
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'], text=True)
    for row in rows.splitlines():
        index, uuid, memory = [x.strip() for x in row.split(',')]
        if int(index) in reserved or int(memory) >= 100:
            continue
        lock = (root/f'gpu-{uuid}.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close(); continue
        return int(index), uuid, lock
    return None


def run_model(args, checkpoint, label):
    root = Path(args.root); source = Path(args.source); output=root/'evaluation'/label
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        settings = json.loads((output/'summary.json').read_text())['settings']
        if settings['checkpoint_sha256'] != sha(checkpoint) or settings['manifest_sha256'] != sha(root/'libero_frozen_v3/manifest.json'):
            raise ValueError('completed evaluation checkpoint/manifest changed')
        return
    gpu = None
    while gpu is None:
        gpu = free_gpu(root, [args.training_gpu, args.render_gpu])
        if gpu is None:time.sleep(15)
    index, uuid, lock = gpu
    name = f'metadata-generalization-{label}-20261108'
    command = ['/opt/dominows/venv/bin/python','-u','-m','test_regr.JointEmbodiedAgentInterface.eval_generalization',
        '--checkpoint',str(checkpoint),'--episode-manifest',str(root/'libero_frozen_v3/manifest.json'),
        '--experiment-output',str(output),'--planning-dir','/home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data/planning',
        '--planner-model','/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct',
        '--vision-model','google/siglip-base-patch16-384','--image-size','384','384','--device','cuda:0','--seed','42']
    inspect = subprocess.run(['docker','inspect','-f','{{.State.Running}}',name],capture_output=True,text=True)
    if inspect.returncode:
        launch=['docker','run','-d','--name',name,'--gpus','device='+str(index),'--ipc=host',
            '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace',
            '-e','CUDA_VISIBLE_DEVICES='+uuid,'-e','PYTHONPATH='+str(source)+':/opt/dominows/VLABench:/home/auszok/domiknowsrelated/src:/opt/dominows/PreciseRoIPooling/pytorch',
            '-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4','-w',str(source),
            '--entrypoint',command[0],'dominows:latest',*command[1:]]
        subprocess.run(launch,check=True)
    job_path = output/'job.json'
    if job_path.exists():
        old = json.loads(job_path.read_text())
        if old['command'] != command:raise ValueError('existing worker command changed')
    else:
        write(job_path,{'container':name,'physical_gpu':index,'uuid':uuid,'command':command,
            'checkpoint':str(checkpoint),'started_utc':datetime.now(timezone.utc).isoformat()})
    worker = json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    job = json.loads(job_path.read_text());job['pid'] = worker['State']['Pid'];write(job_path,job)
    try:
        with (output/'eval.log').open('w') as log:
            follow=subprocess.Popen(['docker','logs','-f',name],stdout=log,stderr=subprocess.STDOUT)
            result=int(subprocess.check_output(['docker','wait',name],text=True).strip())
            follow.wait()
        (output/'exit_code.txt').write_text(str(result)+'\n')
        if result or not (output/'summary.json').exists():
            raise RuntimeError(f'generalization model evaluation failed: {label}, exit={result}')
    finally:
        lock.close()


def run_execution(args, label):
    root=Path(args.root);source=Path(args.source);output=root/'evaluation'/label
    script=source/'test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'
    python=root/'libero_container_runtime/bin/python'
    for policy in ('shared-verified','none'):
        folder=output/'libero'/policy
        if (folder/'execution/summary.json').exists():continue
        cmd=['docker','exec',args.libero_container,str(python),'-u',str(script),'evaluate',
            '--manifest',str(root/'libero_frozen_v3/manifest.json'),
            '--calibration',str(root/args.calibration_directory/'summary.json'),
            '--predictions',str(folder/'predictions.jsonl'),'--output',str(folder/'execution')]
        with (folder/'execution.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'physical execution failed: {label}/{policy}')


def summarize(root):
    snapshots=[]
    for label in ('initialization','epoch1','epoch2','epoch3','epoch4'):
        output=root/'evaluation'/label
        if not (output/'summary.json').exists():continue
        summary=json.loads((output/'summary.json').read_text()); rows=[]
        for key,report in summary['reports'].items():
            row={'condition':key,**{k:report[k] for k in ('Macro','progress_score','exact_match','abstract_goal_success','valid_plan_rate','generation_errors') if k in report}}
            dataset,policy=key.split(':')
            execution=output/dataset/policy/'execution/summary.json'
            if execution.exists():row['fixed_executor']=json.loads(execution.read_text())
            rows.append(row)
        snapshots.append({'snapshot':label,'eai':summary['eai'],'rows':rows})
    report={'internal_experiment':True,'snapshots':snapshots,'scope':'450 internal VLABench / 100 external LIBERO; fixed executor, not learned controller'}
    write(root/'campaign_results.json',report)
    text=['# Full-metadata Stage 1 generalization campaign','',report['scope'],'',
          '| Snapshot | Condition | Macro | Progress | Abstract goal success | Physical success |','|---|---|---:|---:|---:|---:|']
    for snapshot in snapshots:
        for row in snapshot['rows']:
            def number(key):return f'{row[key]:.2f}' if key in row else 'N/A'
            execution=row.get('fixed_executor',{}).get('success_rate')
            text.append(f"| {snapshot['snapshot']} | {row['condition']} | {number('Macro')} | {number('progress_score')} | {number('abstract_goal_success')} | {execution if execution is not None else 'pending'} |")
    (root/'campaign_results.md').write_text('\n'.join(text)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--training-container',required=True);parser.add_argument('--training-gpu',type=int,required=True)
    parser.add_argument('--libero-container',default='metadata-libero-runtime-20261108')
    parser.add_argument('--render-gpu',type=int,default=2)
    parser.add_argument('--calibration-directory',default='calibration_v3')
    args=parser.parse_args();root=Path(args.root)
    while not (root/args.calibration_directory/'summary.json').exists():time.sleep(15)
    calibration=json.loads((root/args.calibration_directory/'summary.json').read_text())
    if calibration['success_rate']<.9:raise RuntimeError('calibration has not passed')
    for label,file in [('initialization','joint_initialization.pt')]+[(f'epoch{i+1}',f'joint_stage1_epoch_{i:03d}.pt') for i in range(4)]:
        checkpoint=root/'checkpoints'/file
        while not checkpoint.exists():
            state=subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',args.training_container],text=True).strip()
            if state!='true':raise RuntimeError(f'training ended without {file}')
            time.sleep(15)
        write(root/'campaign_status.json',{'stage':'evaluating','snapshot':label,'checkpoint':str(checkpoint)})
        run_model(args,checkpoint,label);run_execution(args,label);summarize(root)
        subprocess.run(['docker','exec',args.libero_container,str(root/'libero_container_runtime/bin/python'),
            str(Path(args.source)/'test_regr/JointEmbodiedAgentInterface/generalization/report_campaign.py'),
            '--root',str(root)],check=True)
    write(root/'campaign_status.json',{'stage':'complete','snapshots':5,'completed_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
