"""Run from a physical script so DomiKnowS logs stay beside this suite.

python ilp_repro_20260928/run_suite.py [additional pytest arguments]
Native opt-in: DOMIKNOWS_REPRO_NATIVE=1 (requires a valid license).
"""
import contextlib
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import pytest
import torch
torch.set_num_threads(2)

class Results:
    def __init__(self):
        self.rows = []
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or (report.when == 'setup' and report.outcome != 'passed'):
            self.rows.append(dict(test=report.nodeid, stage=report.when, outcome=report.outcome,
                                 seconds=report.duration, properties=dict(report.user_properties),
                                 detail=str(report.longrepr) if report.longrepr else ''))

class Tee:
    def __init__(self, a, b): self.a, self.b = a, b
    def write(self, data):
        self.a.write(data); self.b.write(data)
        return len(data)
    def flush(self): self.a.flush(); self.b.flush()
    def isatty(self): return False
    def __getattr__(self, name): return getattr(self.a, name)

if __name__ == '__main__':
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = Results()
    with (out / (stamp+'.log')).open('w') as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            code = pytest.main([str(HERE), '-q', '--tb=short', '-o', 'addopts=', '-o', 'junit_family=xunit1',
                                '--junitxml='+str(out/(stamp+'.xml')), *sys.argv[1:]], plugins=[report])
    versions = {p: importlib.metadata.version(p) for p in ('torch','pytest','gurobipy')}
    git=os.environ.get('REPRO_GIT','git')
    def git_read(*args):
        p=subprocess.run([git,*args],cwd=ROOT,text=True,capture_output=True)
        return p.stdout.strip() if p.returncode==0 else 'unavailable'
    result = dict(timestamp_utc=stamp, python=sys.version, versions=versions,
                  commit=git_read('rev-parse','HEAD'),branch=git_read('branch','--show-current'),
                  tracked_diff=git_read('diff','--stat','HEAD'),
                  native_opt_in=os.environ.get('DOMIKNOWS_REPRO_NATIVE') == '1', exit_code=int(code), tests=report.rows)
    (out/(stamp+'.json')).write_text(json.dumps(result, indent=2))
    print('Evidence:', out/(stamp+'.json'))
    sys.exit(code)
