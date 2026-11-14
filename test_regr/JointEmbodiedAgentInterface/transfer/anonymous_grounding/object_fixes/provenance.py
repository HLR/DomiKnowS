"""Reject changed external manifests, cached observations, or archived baselines."""
from ...protocol import sha


def validate_inputs(config):
    for name in ('observations','features','libero_manifest','archive_predictions'):
        if sha(config[name])!=config[name+'_sha256']:
            raise ValueError('frozen diagnostic input changed: '+name)
