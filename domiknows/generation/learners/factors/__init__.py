"""Factor-graph learner heads."""

from .hmm import (
    HMMFactorGraphBundle,
    HMMFactorGraphContext,
    HMMFactorGraphEncoder,
    HMMFactorGraphHead,
    apply_hmm_dp_consistency_constraints,
    hmm_dp_factor_consistency_loss,
    hmm_factor_sequence_nll,
    hmm_forward_backward_factors,
)
from .wfa import (
    SpectralWFAFactorGraphBundle,
    SpectralWFAFactorGraphContext,
    SpectralWFAFactorGraphEncoder,
    SpectralWFAFactorGraphHead,
    apply_wfa_factor_consistency_constraints,
    wfa_factor_consistency_loss,
    wfa_factor_sequence_energy_loss,
)

__all__ = [
    "HMMFactorGraphBundle",
    "HMMFactorGraphContext",
    "HMMFactorGraphEncoder",
    "HMMFactorGraphHead",
    "SpectralWFAFactorGraphBundle",
    "SpectralWFAFactorGraphContext",
    "SpectralWFAFactorGraphEncoder",
    "SpectralWFAFactorGraphHead",
    "apply_hmm_dp_consistency_constraints",
    "apply_wfa_factor_consistency_constraints",
    "hmm_dp_factor_consistency_loss",
    "hmm_factor_sequence_nll",
    "hmm_forward_backward_factors",
    "wfa_factor_consistency_loss",
    "wfa_factor_sequence_energy_loss",
]
