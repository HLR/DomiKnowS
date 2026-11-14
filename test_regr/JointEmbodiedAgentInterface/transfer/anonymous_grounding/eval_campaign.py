"""Host orchestration for one selected grounding snapshot and fixed executions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='transfer.anonymous_grounding'
from ..campaign import run,idle_gpu,save,utc
from ..protocol import read,sha,freeze
from .campaign import runtime_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path);a=p.parse_args()
    root=a.root;config_path=a.config or root/'config.json';config=read(config_path);source=Path(config['source'])
    env=runtime_environment(source,config['planning_dir'])
    prefix=['docker','run','-d','--ipc=host','--cpus=8','--memory=64g',
            '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',str(source)]
    for k,v in env.items():prefix+=['-e',k+'='+v]
    module='test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding'
    def model_job(name,arguments):
        if name=='features' and (root/'jobs/features.json').exists():
            # An immutable feature producer can be adopted during a host handoff.
            previous=read(root/'jobs/features.json');run(root,name,previous['command'],{})
            return
        index,uuid,lock=idle_gpu(root)
        try:
            command=prefix+['--name',root.name+'-'+name,'--gpus','device='+str(index),
                '-e','CUDA_VISIBLE_DEVICES='+uuid,'--entrypoint','/opt/dominows/venv/bin/python',
                config['model_image'],'-u','-m',*arguments]
            run(root,name,command,{'physical_gpu':index,'gpu_uuid':uuid,'checkpoint_sha256':config['checkpoint_sha256']})
        finally:lock.close()
    def policy_job(policy):
        arguments=[module+'.evaluate','--checkpoint',config['checkpoint'],'--configuration',str(config_path),
            '--observations',str(root/'observations/manifest.json'),'--features',str(root/'features/manifest.json'),
            '--constraint-policy',policy,'--experiment-output',str(root/'evaluation'),
            '--planning-dir',config['planning_dir'],'--planner-model','/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct',
            '--vision-model',config['vision_model'],'--image-size','384','384','--device','cuda:0','--seed','42','--load-in-4bit']
        if policy=='none':arguments+=['--evaluate-eai']
        model_job('planning-'+policy,arguments)
        if policy=='full-metadata':return
        # Identical retained bridge and fixed executor used by Section 12.
        base=Path(config['libero_base']);folder=root/'evaluation/libero'/policy
        command=['docker','run','-d','--name',root.name+'-execution-'+policy,'--user','0','--cpus=6','--memory=12g',
            '-v','/home/auszok:/home/auszok','-e','MUJOCO_GL=egl','-e','PYOPENGL_PLATFORM=egl',
            '-e','LIBERO_CONFIG_PATH='+str(base/'libero_config'),
            '-e','PYTHONPATH='+str(base/'libero_runtime/lib/python3.10/site-packages')+':'+str(base/'tooling/LIBERO'),
            '--entrypoint',str(base/'libero_container_runtime/bin/python'),config['executor_image'],'-u',
            str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'),
            'evaluate','--manifest',config['libero_manifest'],'--calibration',str(base/'calibration_v3/summary.json'),
            '--predictions',str(folder/'predictions.jsonl'),'--output',str(folder/'execution')]
        run(root,'execution-'+policy,command,{'physical_gpu':None,'executor':'retained 96/100 calibration; unchanged'})
    try:
        save(root/'status.json',{'stage':'preparing-frozen-observations','updated_utc':utc()})
        command=prefix+['--name',root.name+'-prepare','--entrypoint','/opt/dominows/venv/bin/python',config['model_image'],
            '-u','-m',module+'.eval_observations','prepare','--config',str(config_path),'--output',str(root/'observations')]
        if (root/'jobs/prepare.json').exists():command=read(root/'jobs/prepare.json')['command']
        run(root,'prepare',command,{'physical_gpu':None})
        coverage=read(root/'observations/manifest.json')['coverage']
        if any(c['visible_entities']==0 for c in coverage.values()):raise ValueError('image marker adapter has no verified bindings')
        save(root/'status.json',{'stage':'extracting-frozen-features','coverage':coverage,'updated_utc':utc()})
        model_job('features',[module+'.eval_observations','features','--config',str(config_path),
            '--manifest',str(root/'observations/manifest.json'),'--output',str(root/'features')])
        save(root/'status.json',{'stage':'evaluating-planning-and-execution','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(policy_job,policy) for policy in ('none','shared-verified','full-metadata')]
            for future in futures:future.result()
        reports={}
        for dataset,policies in [('vlabench',('none','shared-verified','full-metadata')),('libero',('none','shared-verified'))]:
            hashes=[]
            for policy in policies:
                folder=root/'evaluation'/dataset/policy;report=read(folder/'report.json');hashes.append(report['input_hashes'])
                reports[dataset+':'+policy]={k:v for k,v in report.items() if k not in ('original','corrected','input_hashes')}
                if dataset=='libero':reports[dataset+':'+policy]['execution']=read(folder/'execution/summary.json')
            if any(h!=hashes[0] for h in hashes[1:]):raise ValueError('constraint policy changed rendered input hashes')
        freeze(root/'results.json',{'checkpoint':config['checkpoint'],'checkpoint_sha256':config['checkpoint_sha256'],
            'reports':reports,'eai':read(root/'evaluation/eai.json'),'coverage':coverage})
        save(root/'status.json',{'stage':'complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
