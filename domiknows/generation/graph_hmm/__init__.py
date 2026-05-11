"""DomiKnowS-aware discrete HMM utilities.

This package provides a Torch-first HMM learner that can project learned
transition and emission distributions through masks derived from a DomiKnowS
graph, explicit task constraints, or caller-supplied masks.
"""

from .constraints import (
    AllowedEmissionsSpec,
    AllowedTransitionsSpec,
    ConstraintApplicationReport,
    ConstraintDFAExportSpec,
    EmissionMaskSpec,
    ForbiddenEmissionsSpec,
    ForbiddenTransitionsSpec,
    StatePredicateTransitionSpec,
    TransitionMaskSpec,
    combine_masks,
    normalize_matrix_rows,
    project_distribution,
    project_matrix,
    project_matrix_rows,
    validate_mask,
)
from .dynamic import (
    DynamicConstraintContext,
    FactorizedStateSpace,
    FiniteStateDynamicConstraint,
    apply_transition_energy,
    transition_energy_matrix,
)
from .graph_adapter import DomiKnowSGraphAdapter
from .graph_hmm import DomiKnowSAwareHMM, HMMFitResult, ViterbiResult
from .spectral import GraphSpectralAutomaton, GraphSpectralFitResult, masked_empirical_initialization, sequence_has_legal_path
from .torch_learners import GraphHMMGenerationHead, GraphSpectralGenerationHead

__all__ = [
    "AllowedEmissionsSpec",
    "AllowedTransitionsSpec",
    "ConstraintApplicationReport",
    "ConstraintDFAExportSpec",
    "DomiKnowSAwareHMM",
    "DomiKnowSGraphAdapter",
    "DynamicConstraintContext",
    "EmissionMaskSpec",
    "FactorizedStateSpace",
    "FiniteStateDynamicConstraint",
    "ForbiddenEmissionsSpec",
    "ForbiddenTransitionsSpec",
    "GraphSpectralAutomaton",
    "GraphSpectralFitResult",
    "GraphHMMGenerationHead",
    "GraphSpectralGenerationHead",
    "HMMFitResult",
    "StatePredicateTransitionSpec",
    "TransitionMaskSpec",
    "ViterbiResult",
    "apply_transition_energy",
    "combine_masks",
    "masked_empirical_initialization",
    "normalize_matrix_rows",
    "project_distribution",
    "project_matrix",
    "project_matrix_rows",
    "sequence_has_legal_path",
    "transition_energy_matrix",
    "validate_mask",
]
