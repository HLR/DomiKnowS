"""Graph-aware HMM package exports."""

from .constraint_compiler import (
    ConstraintHMMCompilation,
    ConstraintHMMState,
    compile_generation_constraints_to_hmm_support,
    domiknows_hmm_from_generation_constraints,
)
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
from .graphAwareHMM import DomiKnowSAwareHMM, HMMFitResult, ViterbiResult
from .graphAwareHMMLearner import GraphHMMGenerationHead

__all__ = [
    "AllowedEmissionsSpec",
    "AllowedTransitionsSpec",
    "ConstraintApplicationReport",
    "ConstraintDFAExportSpec",
    "ConstraintHMMCompilation",
    "ConstraintHMMState",
    "DomiKnowSAwareHMM",
    "DynamicConstraintContext",
    "EmissionMaskSpec",
    "FactorizedStateSpace",
    "FiniteStateDynamicConstraint",
    "ForbiddenEmissionsSpec",
    "ForbiddenTransitionsSpec",
    "GraphHMMGenerationHead",
    "HMMFitResult",
    "StatePredicateTransitionSpec",
    "TransitionMaskSpec",
    "ViterbiResult",
    "apply_transition_energy",
    "combine_masks",
    "compile_generation_constraints_to_hmm_support",
    "domiknows_hmm_from_generation_constraints",
    "normalize_matrix_rows",
    "project_distribution",
    "project_matrix",
    "project_matrix_rows",
    "transition_energy_matrix",
    "validate_mask",
]
