"""Prepare all frozen source scenes, then launch the two matched four-epoch arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import subprocess
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent)); __package__='transfer'

from .campaign import idle_gpu, run, save, utc
from .protocol import read, freeze, sha


def launch(root, config, phase, number):
    index, uuid, lock = idle_gpu(root)
    try:
        source=Path(config['source']); name=root.name+'-'+phase+'-'+str(number)
        command=['docker','run','-d','--name',name,'--gpus','device='+str(index),'--ipc=host','--cpus=6',
            '--memory=40g','-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace',
            '-e','CUDA_VISIBLE_DEVICES='+uuid,'-e','MUJOCO_GL=egl','-e','PYOPENGL_PLATFORM=egl',
            '-e','OMP_NUM_THREADS=2','-e','LP_NUM_THREADS=4',
            '-e','PYTHONPATH='+str(source)+':/opt/dominows/VLABench:/home/auszok/domiknowsrelated/src:/opt/dominows/PreciseRoIPooling/pytorch',
            '-w',str(source),'--entrypoint','/opt/dominows/venv/bin/python',config['model_image'],'-u']
        if phase=='source':
            command += ['-m','test_regr.JointEmbodiedAgentInterface.transfer.source_views',
                '--config',str(root/'config.json'),'--output',str(root/'source_views'),'--shard',str(number),'--shards','4']
        else:
            command += ['-m','test_regr.JointEmbodiedAgentInterface.transfer.train',
                '--config',str(root/'config.json'),'--manifest',str(root/'source_views/manifest.json'),
                '--output',str(root/'training'/number),'--arm',number]
        run(root,phase+'-'+str(number),command,{'physical_gpu':index,'gpu_uuid':uuid,'phase':phase,
            'gpu_policy':'completely-unused-only','checkpoint_sha256':sha(config['checkpoint'])})
    finally: lock.close()


def stop_owned_phase(root, phase):
    for path in (root/'jobs').glob(phase+'-*.json'):
        job=read(path)
        if not job.get('container_id'): continue
        state=json.loads(subprocess.check_output(['docker','inspect',job['container_id']],text=True))[0]
        if state['Name'] != '/'+root.name+'-'+path.stem: raise ValueError('owned phase container identity differs')
        if state['State']['Running']:
            subprocess.run(['docker','stop','-t','10',state['Id']],check=True,capture_output=True)


def parallel(root, config, phase, numbers):
    with ThreadPoolExecutor(max_workers=len(numbers)) as executor:
        futures=[executor.submit(launch,root,config,phase,number) for number in numbers]
        try:
            for future in as_completed(futures): future.result()
        except Exception:
            stop_owned_phase(root,phase)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root;config=read(root/'config.json')
    if config.get('idle_service_gpu') is not None: raise ValueError('unused GPUs required')
    try:
        save(root/'status.json',{'stage':'preparing-source-observations','updated_utc':utc()})
        parallel(root,config,'source',list(range(4)))
        source=Path(config['source'])
        command=['docker','run','-d','--name',root.name+'-source-finalize','--cpus=4',
            '-v','/home/auszok:/home/auszok','-e','PYTHONPATH='+str(source),'-w',str(source),
            '--entrypoint','/opt/dominows/venv/bin/python',config['model_image'],'-u','-m',
            'test_regr.JointEmbodiedAgentInterface.transfer.source_views','--config',str(root/'config.json'),
            '--output',str(root/'source_views'),'--shards','4','--finalize']
        run(root,'source-finalize',command,{'physical_gpu':None})
        save(root/'status.json',{'stage':'matched-continuation-training','epochs':4,'arms':['control','augmented'],'updated_utc':utc()})
        parallel(root,config,'training',['control','augmented'])
        freeze(root/'training_complete.json',{'epochs':4,'arms':['control','augmented'],
            'selection':'source validation only','controller':'fixed','external_evaluation_started':False})
        save(root/'status.json',{'stage':'matched-continuations-complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'stopped-on-correctness-or-runtime-failure','error':str(exc),'updated_utc':utc()})
        raise


if __name__=='__main__': main()
