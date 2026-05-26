"""Automata-backed learner heads."""

from .hmm import HMMGenerationHead
from .losses import allowed_mass_loss, hmm_sequence_nll, wfa_sequence_energy_loss
from .prompt_conditioned_hmm import PromptConditionedHMMGenerationHead
from .prompt_conditioned_spectral_wfa import PromptConditionedSpectralWFAGenerationHead
from .prompt_encoders import FrozenBackbonePromptEncoder, PromptEmbeddingEncoder
from .spectral_wfa import SpectralWFAGenerationHead

__all__ = [
    "FrozenBackbonePromptEncoder",
    "HMMGenerationHead",
    "PromptConditionedHMMGenerationHead",
    "PromptConditionedSpectralWFAGenerationHead",
    "PromptEmbeddingEncoder",
    "SpectralWFAGenerationHead",
    "allowed_mass_loss",
    "hmm_sequence_nll",
    "wfa_sequence_energy_loss",
]
