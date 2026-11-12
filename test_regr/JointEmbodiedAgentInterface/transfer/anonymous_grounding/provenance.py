"""Reject source or observation drift before preparation, training or resume."""
from pathlib import Path
from ..protocol import read,sha


def validate_source(config):
    for key in ('checkpoint','split_manifest','original_configuration','model_source'):
        if sha(config[key])!=config[key+'_sha256']:raise ValueError('frozen experiment input changed: '+key)
    manifest=Path(config['source_manifest'])
    if sha(manifest)!=config['source_manifest_sha256']:raise ValueError('frozen source manifest changed')
    root=Path(config['source'])
    for name,expected in read(manifest)['files'].items():
        if sha(root/name)!=expected:raise ValueError('frozen runtime source changed: '+name)
    model=read(config['model_source'])
    if not model['planner_files']:raise ValueError('starting backbone must have frozen file hashes')
    for name,expected in model['planner_files'].items():
        if sha(Path(model['planner_model'])/name)!=expected:raise ValueError('starting Qwen backbone changed: '+name)
