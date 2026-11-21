"""Fixed source-test supplement; never trains or selects a new checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

if not __package__:
    root=Path(sys.argv[sys.argv.index('--root')+1])
    sys.path.insert(0,str(root/'source/test_regr/JointEmbodiedAgentInterface'));__package__='transfer.repair'
from ..protocol import read,freeze,sha
from ..campaign import run,save,utc
from .campaign import gpu_job,model_command,MODULE
from .protocol import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root;c=read(root/'config.json')
    try:
        verify(c)
        run(root,'tests',model_command(root,c,'tests','pytest',[
            'test_regr/JointEmbodiedAgentInterface/transfer/repair/test_source_test.py',
            'test_regr/JointEmbodiedAgentInterface/transfer/repair/test_input_protocol.py','-q']),{'physical_gpu':None})
        save(root/'status.json',{'stage':'strict-450-episode-test-evaluation','updated_utc':utc()})
        def condition(system,policy):
            out=root/'evaluation'/system/policy
            gpu_job(root,c,'vla-'+system+'-'+policy,MODULE+'.source_test',[
                '--config',root/'config.json','--arm',system,'--policy',policy,'--output',out])
        errors=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(condition,s,p) for s in ('base','joint-initialization','native-trained','joint-trained')
                     for p in ('none','shared-verified')]
            for f in as_completed(futures):
                try:f.result()
                except Exception as exc:errors.append(str(exc))
        if errors:raise RuntimeError('completed conditions preserved: '+'; '.join(errors))
        run(root,'collect',model_command(root,c,'collect',MODULE+'.source_test',[
            '--config',root/'config.json','--collect']),{'physical_gpu':None})
        freeze(root/'complete.json',{'table_results_sha256':sha(root/'table_results.json'),'training_launched':False})
        save(root/'status.json',{'stage':'complete-awaiting-google-table-fill','updated_utc':utc()})
    except Exception as exc:
        save(root/'status.json',{'stage':'failed','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
