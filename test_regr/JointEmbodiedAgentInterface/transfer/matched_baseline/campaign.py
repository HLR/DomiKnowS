"""Host-only, isolated, eight-condition evaluation; never starts training."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys

if not __package__:
    root=Path(sys.argv[sys.argv.index('--root')+1]);sys.path.insert(0,str(root/'source/test_regr/JointEmbodiedAgentInterface'))
    __package__='transfer.matched_baseline'
from ..protocol import read,freeze,sha,episodes
from ..campaign import run,save,utc
from ..anonymous_grounding.campaign import runtime_environment
from ..anonymous_grounding.planner_binding.epoch_campaign import unused_gpu
from ..anonymous_grounding.planner_binding.verification_control import native_command
from .protocol import MODELS,POLICIES

MODULE='test_regr.JointEmbodiedAgentInterface.transfer.matched_baseline'


def model_command(root,c,key,policy,gpu,uuid,*,preflight=False):
    name=('preflight-' if preflight else 'planning-')+key+'-'+policy
    command=['docker','run','-d','--name',root.name+'-'+name,'--ipc=host','--cpus=8','--memory=64g',
        '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',c['source']]
    for k,v in runtime_environment(Path(c['source']),c['planning_dir']).items():command+=['-e',k+'='+v]
    command+=['--gpus','device='+str(gpu),'-e','CUDA_VISIBLE_DEVICES='+uuid,
        '--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m',MODULE+'.evaluate',
        '--config',str(root/'config.json'),'--model-key',key,'--policy',policy]
    if preflight:command+=['--preflight']
    return command


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;c=read(root/'config.json')
    try:
        save(root/'status.json',{'stage':'frozen-protocol-tests','updated_utc':utc()})
        if len(episodes(read(c['manifest'])))!=100:raise ValueError('coverage changed')
        for path,expected in c['input_hashes'].items():
            if sha(path)!=expected:raise ValueError('input changed '+path)
        base=Path(c['libero_base']);gate=read(c['calibration'])
        if gate['successes']!=96 or gate['manifest_sha256']!=sha(c['manifest']):raise ValueError('qualified calibration differs')
        folder=base/'evaluation_source/test_regr/JointEmbodiedAgentInterface/generalization'
        if sha(folder/'fixed_executor.py')!=gate['executor_source_sha256'] or sha(folder/'libero_campaign.py')!=gate['bridge_source_sha256']:
            raise ValueError('qualified executor/bridge changed')
        source=Path(c['source']);command=['docker','run','-d','--name',root.name+'-tests','--cpus=8','--memory=32g',
            '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',str(source)]
        for k,v in runtime_environment(source,c['planning_dir']).items():command+=['-e',k+'='+v]
        command+=['--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m','pytest',
            'test_regr/JointEmbodiedAgentInterface/transfer/matched_baseline/test_protocol.py',
            'test_regr/JointEmbodiedAgentInterface/transfer/test_native_baseline.py','-q']
        run(root,'tests',command,{'physical_gpu':None})
        def preflight(key):
            gpu,uuid,lock=unused_gpu(root,'preflight-'+key)
            try:run(root,'preflight-'+key+'-none',model_command(root,c,key,'none',gpu,uuid,preflight=True),
                {'physical_gpu':gpu,'gpu_uuid':uuid,'model':key,'scored_episodes':0})
            finally:lock.close()
        save(root/'status.json',{'stage':'checkpoint-loader-and-head-preflight','updated_utc':utc()})
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(preflight,k) for k in MODELS]):f.result()
        proofs=[read(root/('preflight_'+k+'.json')) for k in MODELS]
        if any(p['vocabulary_tokens']!=proofs[0]['vocabulary_tokens'] for p in proofs):raise ValueError('model output vocabularies differ')
        freeze(root/'preflight_passed.json',{'models':list(MODELS),'scored_episodes':0,'passed':True})
        save(root/'status.json',{'stage':'planning-and-native-execution','updated_utc':utc()})
        futures=[];errors=[]
        with ThreadPoolExecutor(max_workers=4) as cpu,ThreadPoolExecutor(max_workers=4) as gpu_pool:
            def execute(key,policy):
                name='execution-'+key+'-'+policy;folder=root/'evaluation'/key/policy
                command=native_command(root,c,name,c['manifest'],c['calibration'],folder/'predictions.jsonl',folder/'execution')
                # Same qualified bridge and CPU software rendering as the archive.
                if '--user' not in command:
                    command[3:3]=['--user','0']
                run(root,name,command,{'physical_gpu':None,'model':key,'policy':policy,'horizon':1000})
            def model(key):
                gpu,uuid,lock=unused_gpu(root,'model-'+key)
                try:
                    for policy in POLICIES:
                        try:
                            run(root,'planning-'+key+'-'+policy,model_command(root,c,key,policy,gpu,uuid),
                                {'physical_gpu':gpu,'gpu_uuid':uuid,'model':key,'policy':policy,'config_sha256':sha(root/'config.json')})
                            futures.append(cpu.submit(execute,key,policy))
                        except Exception as exc:errors.append(str(exc))
                finally:lock.close()
            for f in as_completed([gpu_pool.submit(model,k) for k in MODELS]):
                try:f.result()
                except Exception as exc:errors.append(str(exc))
            for f in futures:
                try:f.result()
                except Exception as exc:errors.append(str(exc))
        if errors:raise RuntimeError('; '.join(errors))
        save(root/'status.json',{'stage':'paired-reporting','updated_utc':utc()})
        command=['docker','run','-d','--name',root.name+'-analysis','--cpus=4','--memory=8g',
            '-v','/home/auszok:/home/auszok','-w',c['source'],'-e','PYTHONPATH='+c['source'],
            '--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u','-m',MODULE+'.analysis','--root',str(root)]
        run(root,'analysis',command,{'physical_gpu':None,'bootstrap_samples':10000,'seed':42})
        save(root/'status.json',{'stage':'complete','updated_utc':utc(),'results':str(root/'results.json')})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
