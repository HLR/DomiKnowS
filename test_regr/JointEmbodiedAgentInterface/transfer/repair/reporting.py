"""Freeze a reporting overlay without editing an active campaign's source."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

if not __package__:
    experiment = Path(sys.argv[sys.argv.index('--root')+1])
    configuration = json.loads((experiment/'config.json').read_text())
    sys.path.insert(0, str(Path(configuration['source'])/'test_regr/JointEmbodiedAgentInterface'))
    importlib.import_module('transfer.repair').__path__.insert(0, str(Path(__file__).parent))
    __package__ = 'transfer.repair'

from ..protocol import read, freeze, sha
from ..campaign import run, save, utc
from .campaign import model_command
from .protocol import verify

PACKAGE = 'test_regr.JointEmbodiedAgentInterface.transfer.repair'
FILES = ('analysis.py', 'preservation_audit.py', 'test_preservation_audit.py', 'reporting.py', 'input_protocol.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = args.root; config = read(root/'config.json')
    verify(config)
    output = args.output or root/'reporting_overlay'
    if output.resolve() == root.resolve() or not output.resolve().is_relative_to(root.resolve()):
        raise ValueError('reporting artifacts must be a separate directory inside the experiment')
    code = Path(__file__).resolve().parent
    proof = {'config_sha256': sha(root/'config.json'),
        'campaign_source_manifest_sha256': sha(config['source_manifest']),
        'files': {str(code/name): sha(code/name) for name in FILES},
        'scope': 'Post-generation preservation audit and reporting only; no training or inference changes'}
    freeze(output/'source_manifest.json', proof)
    bootstrap = ('import importlib,pathlib,json; '
        f'p=importlib.import_module({PACKAGE!r}); p.__path__.insert(0,{str(code)!r}); '
        f'from {PACKAGE}.protocol import verify; '
        f'from test_regr.JointEmbodiedAgentInterface.transfer.protocol import read,sha; '
        f'verify(read({str(root/"config.json")!r})); '
        f'proof=read({str(output/"source_manifest.json")!r}); '
        'assert all(sha(path)==expected for path,expected in proof["files"].items()); ')

    def command(name, program):
        cmd = model_command(root, config, name+'-'+output.name, 'unused', [])
        return cmd[:cmd.index('-u')] + ['-u', '-c', bootstrap + program]

    run(output, 'tests', command('reporting-tests',
        f'import pytest; raise SystemExit(pytest.main(["--pyargs",{(PACKAGE+".test_preservation_audit")!r},"-q"]))'),
        {'physical_gpu': None, 'reporting_source_sha256': sha(output/'source_manifest.json')})
    if args.validate_only:
        save(output/'status.json', {'stage': 'validated-awaiting-campaign', 'updated_utc': utc()})
        return
    try:
        while not (root/'complete.json').exists():
            status = read(root/'status.json')
            if status['stage'] == 'failed':
                raise ValueError('campaign failed; reporting cannot qualify incomplete results: '+str(status))
            if not args.wait:
                raise ValueError('campaign incomplete; use --wait for its fixed evaluations')
            save(output/'status.json', {'stage': 'waiting-for-fixed-campaign', 'updated_utc': utc()})
            time.sleep(30)
        verify(config)
        run(output, 'audited-analysis', command('reporting-analysis',
            f'from {PACKAGE}.analysis import collect; '
            f'collect(pathlib.Path({str(root)!r}),pathlib.Path({str(output/"results")!r}))'),
            {'physical_gpu': None, 'reporting_source_sha256': sha(output/'source_manifest.json')})
        freeze(output/'complete.json', {'passed': True,
            'campaign_complete_sha256': sha(root/'complete.json'),
            'preservation_audit_sha256': sha(root/'confirmation/preservation_audit.json'),
            'audited_results_sha256': sha(output/'results/results.json')})
        save(output/'status.json', {'stage': 'complete-awaiting-google-publication', 'updated_utc': utc()})
    except Exception as exc:
        save(output/'status.json', {'stage': 'failed', 'error': str(exc), 'updated_utc': utc()})
        raise


if __name__ == '__main__':
    main()
