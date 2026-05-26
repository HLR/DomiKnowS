"""Generation learner heads."""

from .automata import (
    FrozenBackbonePromptEncoder,
    HMMGenerationHead,
    PromptConditionedHMMGenerationHead,
    PromptConditionedSpectralWFAGenerationHead,
    PromptEmbeddingEncoder,
    SpectralWFAGenerationHead,
    allowed_mass_loss,
    hmm_sequence_nll,
    wfa_sequence_energy_loss,
)
from .compact import (
    CompactLabelGenerationHead,
    CompactLabelSequenceModel,
    CRFCompactLabelScorer,
    EnergyCompactLabelGenerationHead,
    GRUCompactLabelGenerationHead,
    NeuralNGramCompactLabelGenerationHead,
    TransformerCompactLabelGenerationHead,
)
from .factors import (
    HMMFactorGraphBundle,
    HMMFactorGraphContext,
    HMMFactorGraphEncoder,
    HMMFactorGraphHead,
    SpectralWFAFactorGraphBundle,
    SpectralWFAFactorGraphContext,
    SpectralWFAFactorGraphEncoder,
    SpectralWFAFactorGraphHead,
    apply_hmm_dp_consistency_constraints,
    apply_wfa_factor_consistency_constraints,
    hmm_dp_factor_consistency_loss,
    hmm_factor_sequence_nll,
    hmm_forward_backward_factors,
    wfa_factor_consistency_loss,
    wfa_factor_sequence_energy_loss,
)
from .graph_hmm import GraphHMMGenerationHead, GraphSpectralGenerationHead

__all__ = [name for name in globals() if not name.startswith("_")]
