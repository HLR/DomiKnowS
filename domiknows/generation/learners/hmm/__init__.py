"""Lazy HMM package exports."""

from importlib import import_module

_EXPORTS = {
    'AllowedEmissionsSpec': '.constraints',
    'AllowedTransitionsSpec': '.constraints',
    'BaumWelchResult': '.core',
    'ConstraintApplicationReport': '.constraints',
    'ConstraintDFAExportSpec': '.constraints',
    'ConstraintHMMCompilation': '.constraint_compiler',
    'ConstraintHMMState': '.constraint_compiler',
    'DiscreteHMM': '.core',
    'DomiKnowSAwareHMM': '.graph',
    'DomiKnowSGraphAdapter': '.graph_adapter',
    'DynamicConstraintContext': '.dynamic',
    'EmissionMaskSpec': '.constraints',
    'FactorizedStateSpace': '.dynamic',
    'FiniteStateDynamicConstraint': '.dynamic',
    'ForbiddenEmissionsSpec': '.constraints',
    'ForbiddenTransitionsSpec': '.constraints',
    'GraphHMMGenerationHead': '.graph_head',
    'HMMFactorGraphBundle': '.factors',
    'HMMFactorGraphContext': '.factors',
    'HMMFactorGraphEncoder': '.factors',
    'HMMFactorGraphHead': '.factors',
    'HMMFitResult': '.graph',
    'HMMForwardBackward': '.core',
    'HMMGenerationHead': '.head',
    'HMMParameters': '.core',
    'PromptConditionedHMMGenerationHead': '.prompt_conditioned_head',
    'StatePredicateTransitionSpec': '.constraints',
    'TransitionMaskSpec': '.constraints',
    'ViterbiResult': '.graph',
    'all_sequences': '.core',
    'apply_hmm_dp_consistency_constraints': '.factors',
    'apply_transition_energy': '.dynamic',
    'baum_welch_train': '.core',
    'combine_masks': '.constraints',
    'compare_hmm_dfa': '.core',
    'compile_generation_constraints_to_hmm_support': '.constraint_compiler',
    'domiknows_hmm_from_generation_constraints': '.constraint_compiler',
    'hmm_dp_factor_consistency_loss': '.factors',
    'hmm_factor_sequence_nll': '.factors',
    'hmm_forward_backward_factors': '.factors',
    'normalize_matrix_rows': '.constraints',
    'project_distribution': '.constraints',
    'project_matrix': '.constraints',
    'project_matrix_rows': '.constraints',
    'transition_energy_matrix': '.dynamic',
    'validate_mask': '.constraints',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value
