"""Persistent external-holdout exclusions for future target preparation/use."""
import os
from pathlib import Path
from ..protocol import read
from .protocol import reject_reserved


def guard(manifest,directory=None):
    root=Path(directory or os.environ.get('DOMIKNOWS_TARGET_RESERVATIONS',
                                         '/home/auszok/experiments/target_reservations'))
    if root.is_dir():
        for path in sorted(root.glob('*.json')):
            reservation=read(path);reject_reserved(manifest,reservation)
            ids={r['episode_id'] for r in reservation['episodes']}
            if any(r.get('split') in ('train','training','validation','valid') and r.get('episode_id') in ids
                   for r in manifest['episodes']):raise ValueError('reserved confirmation identity in target training/validation')
