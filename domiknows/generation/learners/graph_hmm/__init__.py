"""Graph-HMM learner heads."""

from .hmm import GraphHMMGenerationHead
from .spectral import GraphSpectralGenerationHead

__all__ = [
    "GraphHMMGenerationHead",
    "GraphSpectralGenerationHead",
]
