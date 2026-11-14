"""Host-only matched training, followed by one frozen fresh-state evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]));__package__='transfer.anonymous_grounding.planner_binding'
from ...campaign import run,idle_gpu,save,utc
from ...protocol import read,freeze,sha
from ..campaign import runtime_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    c=read(root/'launch_config.json');source=Path(c['source']);module='test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding'
    prefix=['docker','run','-d','--ipc=host','--cpus=8','--memory=64g','-v','/home/auszok:/home/auszok',
        '-v','/home/auszok/DomiKnowS:/workspace','-w',str(source)]
    for k,v in runtime_environment(source,c['planning_dir']).items():prefix+=['-e',k+'='+v]
    def cpu(name,args):
        command=prefix+['--name',root.name+'-'+name,'--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m',*args]
        run(root,name,command,{'physical_gpu':None})
    def gpu(name,args):
        job=root/'jobs'/f'{name}.json'
        if job.exists():
            previous=read(job);run(root,name,previous['command'],{});return
        index,uuid,lock=idle_gpu(root)
        try:
            command=prefix+['--name',root.name+'-'+name,'--gpus','device='+str(index),
                '-e','CUDA_VISIBLE_DEVICES='+uuid,'-e','BINDING_GPU_UUID='+uuid,
                '--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m',*args]
            run(root,name,command,{'physical_gpu':index,'gpu_uuid':uuid,'initialization_sha256':c['checkpoint_sha256']})
        finally:lock.close()
    def execute(name,predictions):
        base=Path(c['libero_base']);command=['docker','run','-d','--name',root.name+'-'+name,'--user','0','--cpus=6','--memory=12g',
            '-v','/home/auszok:/home/auszok','-e','MUJOCO_GL=egl','-e','PYOPENGL_PLATFORM=egl',
            '-e','LIBERO_CONFIG_PATH='+str(base/'libero_config'),
            '-e','PYTHONPATH='+str(base/'libero_runtime/lib/python3.10/site-packages')+':'+str(base/'tooling/LIBERO'),
            '--entrypoint',str(base/'libero_container_runtime/bin/python'),c['executor_image'],'-u',
            str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'),
            'evaluate','--manifest',c['external_manifest'],'--calibration',str(Path(c.get('prepared_external_root',root))/'fresh_libero/calibration_reuse.json'),
            '--predictions',str(predictions),'--output',str(predictions.parent/'execution')]
        run(root,name,command,{'physical_gpu':None,'executor':'unchanged, qualified original calibration records'})
    try:
        save(root/'status.json',{'stage':'waiting-for-fresh-scene-integrity-gate','updated_utc':utc()})
        prepared=Path(c.get('prepared_external_root',root))
        job_name=read(prepared/'freeze_supervisor_current.json')['job_name']
        while read(prepared/'jobs'/f'{job_name}.json').get('exit_code') is None:time.sleep(2)
        if read(prepared/'jobs'/f'{job_name}.json')['exit_code']!=0:raise ValueError('fresh scene preparation failed; no training authorized past correctness gate')
        if not read(prepared/'fresh_libero/split_audit.json')['passed']:raise ValueError('fresh split integrity failed')
        c['external_manifest']=str(prepared/'fresh_libero/manifest.json');c['external_manifest_sha256']=sha(c['external_manifest'])
        freeze(root/'config.json',c)
        base=Path(c['libero_base']);folder=source/'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding'
        settled=['docker','run','-d','--name',root.name+'-settled-audit','--user','0','--cpus=6','--memory=12g',
            '-v','/home/auszok:/home/auszok','-e','LIBERO_CONFIG_PATH='+str(base/'libero_config'),
            '-e','PYTHONPATH='+str(base/'libero_runtime/lib/python3.10/site-packages')+':'+str(base/'tooling/LIBERO'),
            '--entrypoint',str(base/'libero_container_runtime/bin/python'),c['executor_image'],'-u',
            str(folder/'settled_audit.py'),'--config',str(root/'config.json'),'--freezer',str(folder/'freeze_external.py')]
        if prepared==root:
            run(root,'settled-audit',settled,{'physical_gpu':None,'gate':'actual frozen 100 fresh versus 800 reserved scenes'})
        settled_gate=read(prepared/'fresh_libero/settled_split_audit.json')
        if not settled_gate['passed'] or settled_gate['manifest_sha256']!=c['external_manifest_sha256']:raise ValueError('settled-scene integrity failed')
        freeze(root/'settled_audit_reuse.json',{'source':str(prepared/'fresh_libero/settled_split_audit.json'),
            'sha256':sha(prepared/'fresh_libero/settled_split_audit.json'),'manifest_sha256':c['external_manifest_sha256']})
        save(root/'status.json',{'stage':'testing-source-only-binding-implementation','updated_utc':utc()})
        cpu('tests',['pytest','test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding',
            'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/test_training.py',
            'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/test_evaluation.py','-q'])
        save(root/'status.json',{'stage':'preparing-public-source-and-fresh-observation-features','updated_utc':utc()})
        gpu('public-features',[module+'.prepare_inputs','--config',str(root/'config.json')])
        c['text_manifest']=str(root/'text_priors/manifest.json');c['text_manifest_sha256']=sha(c['text_manifest'])
        c['inputs_complete_sha256']=sha(root/'inputs_complete.json');freeze(root/'training_common.json',c)
        for arm in ('masked','full'):
            freeze(root/f'{arm}.json',{**c,'arm':arm})
            save(root/'status.json',{'stage':'auditing-source-references-and-permutations','arm':arm,'updated_utc':utc()})
            cpu('audit-'+arm,[module+'.train','--config',str(root/f'{arm}.json'),
                '--output',str(root/'arms'/arm/'checkpoints'),'--audit-only'])
        save(root/'status.json',{'stage':'training-two-matched-four-epoch-arms','updated_utc':utc()})
        def train(arm):
            gpu('training-'+arm,[module+'.train','--config',str(root/f'{arm}.json'),'--output',str(root/'arms'/arm/'checkpoints')])
            if not read(root/'arms'/arm/'checkpoints/complete.json')['epochs']==4:raise ValueError('four epochs incomplete')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train,arm) for arm in ('masked','full')]
            for future in futures:future.result()
        initial=[read(root/'arms'/arm/'checkpoints/initialization_parameters.json') for arm in ('masked','full')]
        if initial[0]!=initial[1]:raise ValueError('matched arms did not start from identical parameters')
        snapshots={'initialization':str(root/'arms/masked/checkpoints/joint_initialization.pt'),
            **{arm:read(root/'arms'/arm/'checkpoints/best.json')['checkpoint'] for arm in ('masked','full')}}
        freeze(root/'selected_snapshots.json',{'snapshots':snapshots,'selection':'source validation loss only within each arm',
            'identical_initialization':initial[0],'warning':'validation losses from different CE denominators are not cross-arm rankings'})
        save(root/'status.json',{'stage':'fresh-held-out-planning-and-native-execution','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=2) as gpu_pool,ThreadPoolExecutor(max_workers=2) as cpu_pool:
            def evaluate(name,checkpoint,policy):
                folder=root/'evaluation'/name/policy
                arguments=[module+'.evaluate','--config',str(root/'training_common.json'),'--checkpoint',checkpoint,
                    '--constraint-policy',policy,'--experiment-output',str(folder),'--planning-dir',c['planning_dir'],
                    '--planner-model','/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct','--vision-model',c['vision_model'],
                    '--image-size','384','384','--device','cuda:0','--seed','42','--load-in-4bit']
                if policy=='none':arguments+=['--evaluate-eai']
                gpu('planning-'+name+'-'+policy,arguments)
                return cpu_pool.submit(execute,'execution-'+name+'-'+policy,folder/'predictions.jsonl')
            jobs=[gpu_pool.submit(evaluate,name,checkpoint,policy) for name,checkpoint in snapshots.items() for policy in ('none','executor')]
            native=[job.result() for job in jobs]
            for job in native:job.result()
        from .report import report
        report(root,c)
        save(root/'status.json',{'stage':'complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
