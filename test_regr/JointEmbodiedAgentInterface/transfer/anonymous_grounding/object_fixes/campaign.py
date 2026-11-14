"""Host-only diagnostic runner; unchanged snapshot, inputs, and calibrated executor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]));__package__='transfer.anonymous_grounding.object_fixes'
from ...campaign import run,idle_gpu,save,utc
from ...protocol import read,freeze
from ..campaign import runtime_environment

CONDITIONS=('baseline','raw','contract','raw-contract','aligned','aligned-contract')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;cp=root/'config.json';c=read(cp);source=Path(c['source'])
    module='test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.object_fixes'
    prefix=['docker','run','-d','--ipc=host','--cpus=8','--memory=64g','-v','/home/auszok:/home/auszok',
        '-v','/home/auszok/DomiKnowS:/workspace','-w',str(source)]
    for k,v in runtime_environment(source,c['planning_dir']).items():prefix+=['-e',k+'='+v]
    def model(name,args):
        index,uuid,lock=idle_gpu(root)
        try:
            command=prefix+['--name',root.name+'-'+name,'--gpus','device='+str(index),
                '-e','CUDA_VISIBLE_DEVICES='+uuid,'--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m',*args]
            run(root,name,command,{'physical_gpu':index,'gpu_uuid':uuid,'checkpoint_sha256':c['checkpoint_sha256']})
        finally:lock.close()
    def planning(variant):
        model('planning-'+variant,[module+'.evaluate','--configuration',str(cp),'--condition',variant,
            '--experiment-output',str(root/'planning'/variant),'--checkpoint',c['checkpoint'],'--planning-dir',c['planning_dir'],
            '--planner-model','/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct','--vision-model',c['vision_model'],
            '--image-size','384','384','--device','cuda:0','--seed','42','--load-in-4bit'])
    def execution(variant):
        if variant=='baseline':return # reproduced plans; archived native executions explicitly identified
        base=Path(c['libero_base'])
        command=['docker','run','-d','--name',root.name+'-execution-'+variant,'--user','0','--cpus=6','--memory=12g',
            '-v','/home/auszok:/home/auszok','-e','MUJOCO_GL=egl','-e','PYOPENGL_PLATFORM=egl',
            '-e','LIBERO_CONFIG_PATH='+str(base/'libero_config'),
            '-e','PYTHONPATH='+str(base/'libero_runtime/lib/python3.10/site-packages')+':'+str(base/'tooling/LIBERO'),
            '--entrypoint',str(base/'libero_container_runtime/bin/python'),c['executor_image'],'-u',
            str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'),
            'evaluate','--manifest',c['libero_manifest'],'--calibration',str(base/'calibration_v3/summary.json'),
            '--predictions',str(root/'planning'/variant/'predictions.jsonl'),'--output',str(root/'execution'/variant)]
        run(root,'execution-'+variant,command,{'physical_gpu':None,'executor':'retained qualified LIBERO; 96/100 calibration'})
    try:
        save(root/'status.json',{'stage':'baseline-reproduction-and-paired-feature-extraction','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(planning,'baseline'),pool.submit(model,'aligned-features',[module+'.features','--configuration',str(cp)])]
            for job in jobs:job.result()
        if not read(root/'planning/baseline/baseline_gate.json')['passed']:raise ValueError('baseline gate failed')
        save(root/'status.json',{'stage':'testing-five-fixed-conditions','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=2) as gpu_pool,ThreadPoolExecutor(max_workers=2) as cpu_pool:
            def experiment(variant):
                planning(variant);return cpu_pool.submit(execution,variant)
            gpu_jobs=[gpu_pool.submit(experiment,v) for v in CONDITIONS[1:]]
            executions=[job.result() for job in gpu_jobs]
            for job in executions:job.result()
        from .report import report
        report(root,c)
        save(root/'status.json',{'stage':'complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
