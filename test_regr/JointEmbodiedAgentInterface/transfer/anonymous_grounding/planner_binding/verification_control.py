"""Host-only matched initialization controls and archived Epoch-1 reproduction."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

if not __package__:
    root = Path(sys.argv[sys.argv.index('--root')+1])
    sys.path.insert(0, str(root/'source/test_regr/JointEmbodiedAgentInterface'))
    __package__ = 'transfer.anonymous_grounding.planner_binding'
from ...protocol import read, freeze, sha
from ...campaign import run, save, utc
from ..campaign import runtime_environment
from .epoch_campaign import unused_gpu
from .epoch_recovery import canonical_eai_record

MODULE = 'test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding'


def verify_reproduction(archive, actual, policy):
    original = Path(archive)/'evaluation/epoch_01'/policy; actual = Path(actual)
    datasets = ('vlabench',) if policy == 'full-metadata' else ('vlabench','libero')
    certificates = {}
    for dataset in datasets:
        first = read(original/dataset/'report.json'); second = read(actual/dataset/'report.json')
        if first['input_hashes'] != second['input_hashes']: raise ValueError('reproduction neural inputs differ')
        a = [json.loads(s) for s in (original/dataset/'predictions.jsonl').read_text().splitlines()]
        b = [json.loads(s) for s in (actual/dataset/'predictions.jsonl').read_text().splitlines()]
        if len(a) != len(b) or len(a) != (450 if dataset == 'vlabench' else 100): raise ValueError('reproduction coverage differs')
        if any(x['episode_id'] != y['episode_id'] or x['prediction'] != y['prediction'] or x['generation_error'] != y['generation_error'] for x,y in zip(a,b)):
            raise ValueError('archived Epoch1 predictions do not reproduce')
        keys = ('Macro','progress_score','exact_match','generation_errors') if dataset == 'vlabench' else ('abstract_goal_success','valid_plan_rate','generation_errors')
        if any(abs(first[k]-second[k]) > 1e-9 for k in keys): raise ValueError('reproduced scores differ')
        certificates[dataset] = {'episodes':len(a), 'archive_predictions_sha256':sha(original/dataset/'predictions.jsonl'),
                                 'reproduced_predictions_sha256':sha(actual/dataset/'predictions.jsonl')}
    if policy == 'none':
        a = [canonical_eai_record(json.loads(s)) for s in (original/'eai_predictions.jsonl').read_text().splitlines()]
        b = [canonical_eai_record(json.loads(s)) for s in (actual/'eai_predictions.jsonl').read_text().splitlines()]
        if len(a) != 88 or a != b or read(original/'eai.json') != read(actual/'eai.json'):
            raise ValueError('canonical EAI reproduction differs')
    # Model containers own their output directories; host-side gates live separately.
    freeze(actual.parents[2]/'control_reproduction'/f'{policy}.json', {'passed':True, 'policy':policy, 'datasets':certificates,
                                      'eai_semantic_fact_order_ignored_only':True})


def native_command(root, config, name, manifest, calibration, predictions, output):
    base = Path(config['libero_base'])
    return ['docker','run','-d','--name',Path(root).name+'-'+name,'--user','0','--cpus=6','--memory=12g',
        '-v','/home/auszok:/home/auszok','-e','MUJOCO_GL=egl','-e','PYOPENGL_PLATFORM=egl',
        '-e','LIBERO_CONFIG_PATH='+str(base/'libero_config'),
        '-e','PYTHONPATH='+str(base/'libero_runtime/lib/python3.10/site-packages')+':'+str(base/'tooling/LIBERO'),
        '--entrypoint',str(base/'libero_container_runtime/bin/python'),config['executor_image'],'-u',
        str(base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization/libero_campaign.py'),'evaluate',
        '--manifest',str(manifest),'--calibration',str(calibration),'--predictions',str(predictions),'--output',str(output)]


def model_command(root, config, name, config_path, snapshot, policy, output, gpu_index, uuid, datasets=None, skip_eai=False):
    source = Path(config['source'])
    command = ['docker','run','-d','--ipc=host','--cpus=8','--memory=64g',
        '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',str(source)]
    for k,v in runtime_environment(source,config['planning_dir']).items(): command += ['-e',k+'='+v]
    command += ['--name',Path(root).name+'-'+name,'--gpus','device='+str(gpu_index),
        '-e','CUDA_VISIBLE_DEVICES='+uuid,'-e','BINDING_GPU_UUID='+uuid,
        '--entrypoint','/opt/dominows/venv/bin/python',config['model_image'],'-u','-m',MODULE+'.epoch_evaluation',
        '--epoch-config',str(config_path),'--snapshot',snapshot,'--checkpoint',config['snapshots']['initialization' if snapshot=='initialization' else snapshot[-1]]['path'],
        '--constraint-policy',policy,'--experiment-output',str(output),'--planning-dir',config['planning_dir'],
        '--planner-model','/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct','--vision-model',config['vision_model'],
        '--image-size','384','384','--device','cuda:0','--seed','42','--load-in-4bit']
    if datasets: command += ['--datasets',*datasets]
    if skip_eai: command += ['--skip-eai']
    return command


def run_model(root, config_path, name, snapshot, policy, output, datasets=None, skip_eai=False):
    root = Path(root); c = read(config_path); job = root/'jobs'/f'{name}.json'
    if job.exists(): run(root,name,read(job)['command'],{}); return
    index,uuid,lock = unused_gpu(root,name)
    try:
        command = model_command(root,c,name,config_path,snapshot,policy,output,index,uuid,datasets,skip_eai)
        run(root,name,command,{'physical_gpu':index,'gpu_uuid':uuid,'snapshot':snapshot,'policy':policy,
                             'configuration_sha256':sha(config_path)})
    finally: lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    c=read(root/'control_config.json');folder=Path(c['source'])/'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding'
    prefix=['docker','run','-d','--ipc=host','--cpus=8','--memory=64g','-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',c['source']]
    for k,v in runtime_environment(Path(c['source']),c['planning_dir']).items():prefix+=['-e',k+'='+v]
    tests=prefix+['--name',root.name+'-control-tests','--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m','pytest',
        str(folder/'test_verification.py'),str(folder/'test_binding.py'),str(folder/'test_epoch_evaluation.py'),str(folder/'test_epoch_recovery.py'),'-q']
    try:
        save(root/'control_status.json',{'stage':'testing-control-evaluator','updated_utc':utc()})
        run(root,'control-tests',tests,{'physical_gpu':None})
        save(root/'control_status.json',{'stage':'matched-controls-and-epoch1-reproduction','updated_utc':utc()})
        def work(snapshot,policy):
            kind='reproduction' if snapshot=='epoch-1' else 'initialization';out=root/'control'/kind/policy
            run_model(root,root/'control_config.json','control-'+kind+'-'+policy,snapshot,policy,out)
            if snapshot=='epoch-1': verify_reproduction(read(root/'setup.json')['archive'],out,policy)
            return snapshot,policy,out
        with ThreadPoolExecutor(max_workers=2) as gpu, ThreadPoolExecutor(max_workers=2) as cpu:
            futures=[gpu.submit(work,snapshot,policy) for policy in ('none','shared-verified','full-metadata') for snapshot in ('epoch-1','initialization')]
            native=[];errors=[]
            for f in as_completed(futures):
                try:snapshot,policy,out=f.result()
                except Exception as exc:errors.append(str(exc));continue
                if snapshot=='initialization' and policy!='full-metadata':
                    name='control-execution-init-'+policy
                    command=native_command(root,c,name,c['external_manifest'],c['calibration'],out/'libero/predictions.jsonl',out/'libero/execution')
                    native.append(cpu.submit(run,root,name,command,{'physical_gpu':None,'executor':'unchanged qualified executor'}))
            for f in native:
                try:f.result()
                except Exception as exc:errors.append(str(exc))
            if errors:raise RuntimeError('; '.join(errors))
        freeze(root/'control_complete.json',{'passed':True,'configuration_sha256':sha(root/'control_config.json'),
            'reproduction_policies':['none','shared-verified','full-metadata'],'initialization_own_head':True})
        save(root/'control_status.json',{'stage':'complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'control_status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
