"""Wait for fixed scoring, then run publication correctness checks on CPU."""
import argparse
import json
from pathlib import Path
import sys
import time

if not __package__:
    root = Path(sys.argv[sys.argv.index('--root') + 1])
    sys.path.insert(0, str(root/'source/test_regr/JointEmbodiedAgentInterface'))
    __package__ = 'transfer.anonymous_grounding.planner_binding'
from ...protocol import read, sha
from ...campaign import run, save, utc
from ..campaign import runtime_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root;c=read(root/'control_config.json')
    try:
        save(root/'publication_status.json',{'stage':'waiting-for-fixed-evaluations','updated_utc':utc()})
        while not (root/'results.json').exists():
            for name, failure in [('status.json','blocked-correctness'),('control_status.json','failed')]:
                path=root/name
                if path.exists() and read(path)['stage']==failure:
                    raise RuntimeError('upstream correctness gate failed: '+read(path).get('error',''))
            time.sleep(5)
        for path,expected in read(root/'post_scoring_diagnostics_manifest.json').items():
            if sha(path)!=expected:raise ValueError('diagnostic implementation changed')
        if read(root/'jobs/diagnostics-tests.json').get('exit_code')!=0:
            raise ValueError('diagnostic regression checks incomplete')
        save(root/'publication_status.json',{'stage':'checking-matched-inputs-and-factual-failures','updated_utc':utc()})
        command=['docker','run','-d','--name',root.name+'-publication-gate','--cpus=4','--memory=24g',
            '-v','/home/auszok:/home/auszok','-v','/home/auszok/DomiKnowS:/workspace','-w',c['source']]
        for k,v in runtime_environment(Path(c['source']),c['planning_dir']).items():command+=['-e',k+'='+v]
        command+=['--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-u',
            '-m','test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding.verification_diagnostics',
            '--root',str(root)]
        run(root,'publication-gate',command,{'physical_gpu':None,'purpose':'post-scoring correctness; no tuning'})
        if not read(root/'publication_gate.json')['passed']:raise ValueError('publication gate failed')
        save(root/'publication_status.json',{'stage':'ready-for-google-report','results_sha256':sha(root/'results.json'),
            'gate_sha256':sha(root/'publication_gate.json'),'updated_utc':utc()})
    except Exception as exc:
        save(root/'publication_status.json',{'stage':'blocked-correctness','error':str(exc),'updated_utc':utc()});raise


if __name__=='__main__':main()
