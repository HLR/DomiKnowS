"""Compact-label learner heads."""

from ..common.base import CompactLabelGenerationHead, CompactLabelSequenceModel
from .crf import CRFCompactLabelScorer
from .energy import EnergyCompactLabelGenerationHead
from .gru import GRUCompactLabelGenerationHead
from .neural_ngram import NeuralNGramCompactLabelGenerationHead
from .transformer import TransformerCompactLabelGenerationHead

__all__ = [
    "CompactLabelGenerationHead",
    "CompactLabelSequenceModel",
    "CRFCompactLabelScorer",
    "EnergyCompactLabelGenerationHead",
    "GRUCompactLabelGenerationHead",
    "NeuralNGramCompactLabelGenerationHead",
    "TransformerCompactLabelGenerationHead",
]
