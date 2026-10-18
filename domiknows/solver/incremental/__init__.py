"""Persistent, transactional inference over changing DataNode graphs."""
from .result import IncrementalInferenceResult

__all__ = ['IncrementalInferenceResult', 'IncrementalInferenceSession']


def __getattr__(name):
    if name == 'IncrementalInferenceSession':
        from .session import IncrementalInferenceSession
        return IncrementalInferenceSession
    raise AttributeError(name)
