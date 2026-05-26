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
from .constraint_compiler import (
    ConstraintHMMCompilation,
    ConstraintHMMState,
    compile_generation_constraints_to_hmm_support,
    domiknows_hmm_from_generation_constraints,
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


def __getattr__(name):
    if name == "GraphHMMGenerationHead":
        from ..learners.graph_hmm import GraphHMMGenerationHead

        return GraphHMMGenerationHead
    if name == "GraphSpectralGenerationHead":
        from ..learners.graph_hmm import GraphSpectralGenerationHead

        return GraphSpectralGenerationHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AllowedEmissionsSpec",
    "AllowedTransitionsSpec",
    "ConstraintApplicationReport",
    "ConstraintDFAExportSpec",
    "ConstraintHMMCompilation",
    "ConstraintHMMState",
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
    "compile_generation_constraints_to_hmm_support",
    "domiknows_hmm_from_generation_constraints",
    "masked_empirical_initialization",
    "normalize_matrix_rows",
    "project_distribution",
    "project_matrix",
    "project_matrix_rows",
    "sequence_has_legal_path",
    "transition_energy_matrix",
    "validate_mask",
]
