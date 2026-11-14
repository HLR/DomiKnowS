"""Host-only qualification before any experimental predictions."""
import argparse
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]));__package__='transfer.anonymous_grounding.object_fixes'
from ...protocol import read,freeze
from ...campaign import run
from ..campaign import runtime_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;c=read(root/'config.json');source=Path(c['source'])
    command=['docker','run','-d','--name',root.name+'-tests','--cpus=4','--memory=12g',
        '-v','/home/auszok:/home/auszok','-w',str(source)]
    for k,v in runtime_environment(source,c['planning_dir']).items():command+=['-e',k+'='+v]
    command+=['--entrypoint','/opt/dominows/venv/bin/python',c['model_image'],'-m','pytest',
        'test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding','-q']
    run(root,'tests',command,{'physical_gpu':None})
    freeze(root/'tests_passed.json',{'passed':True,'source_manifest_sha256':c['evaluation_source_manifest_sha256']})
    from .campaign import main as campaign
    campaign()


if __name__=='__main__':main()
