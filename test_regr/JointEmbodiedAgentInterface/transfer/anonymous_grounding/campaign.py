"""One source-only combined experiment; original-image preparation then training."""
import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='transfer.anonymous_grounding'

from ..campaign import run,idle_gpu,save,utc
from ..protocol import read,freeze,sha
from .provenance import validate_source


def runtime_environment(source, official_root):
    """Bind corpus identity independently of the isolated code checkout."""
    return {
        'PYTHONPATH':str(source)+':/opt/dominows/VLABench:/home/auszok/domiknowsrelated/src:/opt/dominows/PreciseRoIPooling/pytorch',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'HF_HOME':'/home/auszok/.cache/huggingface',
        'VLABENCH_OFFICIAL_EVALUATION_ROOT':str(official_root),
        # Trust only this owned checkout for checkpoint commit provenance.
        'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'safe.directory',
        'GIT_CONFIG_VALUE_0':str(source),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args();root=args.root;config_path=args.config or root/'config.json';config=read(config_path)
    source=Path(config['source'])
    prefix=['docker','run','-d','--ipc=host','--cpus=8','--memory=64g',
        '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace']
    for name,value in runtime_environment(source,config['official_evaluation_root']).items():
        prefix+=['-e',name+'='+value]
    prefix+=['-w',str(source)]
    module='test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding'
    try:
        save(root/'status.json',{'stage':'verifying-frozen-source-and-models','updated_utc':utc()})
        validate_source(config)
        save(root/'status.json',{'stage':'preparing-original-source-images','updated_utc':utc()})
        command=prefix+['--name',root.name+'-prepare','--entrypoint','/opt/dominows/venv/bin/python',
            config['model_image'],'-u','-m',module+'.prepare','--config',str(root/'config.json'),
            '--output',str(root/'data'),'--workers','4']
        # A host handoff adopts the identical, owned CPU preparation command;
        # its source/config stay immutable while training uses a new overlay.
        if (root/'jobs/prepare.json').exists():command=read(root/'jobs/prepare.json')['command']
        run(root,'prepare',command,{'physical_gpu':None,'input':'original images; no simulator regeneration'})
        proof=read(root/'data/manifest.json')
        if not proof['complete'] or proof['test_used'] or proof['references_read']:raise ValueError('source input gate failed')
        index,uuid,lock=idle_gpu(root)
        try:
            save(root/'status.json',{'stage':'features-and-training','physical_gpu':index,'gpu_uuid':uuid,'updated_utc':utc()})
            command=prefix+['--name',root.name+'-training','--gpus','device='+str(index),
                '-e','CUDA_VISIBLE_DEVICES='+uuid,'--entrypoint','/opt/dominows/venv/bin/python',
                config['model_image'],'-u','-m',module+'.train','--config',str(config_path),
                '--manifest',str(root/'data/manifest.json'),'--features',str(root/'features'),
                '--output',str(root/'checkpoints')]
            run(root,'training',command,{'physical_gpu':index,'gpu_uuid':uuid,'epochs':4,
                'initialization_sha256':sha(config['checkpoint']),'source_only':True,'gpu_policy':'completely-unused-only'})
        finally:lock.close()
        if not (root/'checkpoints/complete.json').exists():raise ValueError('training job ended without four completed epochs')
        freeze(root/'complete.json',{'epochs':4,'source_only':True,'best':read(root/'checkpoints/best.json'),
            'target_training':False,'target_test_scoring':False})
        save(root/'status.json',{'stage':'complete','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
