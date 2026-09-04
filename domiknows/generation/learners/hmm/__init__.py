"""Lazy HMM package exports."""

from importlib import import_module

_EXPORTS = {
    'AllowedEmissionsSpec': '.graph.constraints',
    'AllowedTransitionsSpec': '.graph.constraints',
    'BaumWelchResult': '.discrete.discreteHMM',
    'ConstraintApplicationReport': '.graph.constraints',
    'ConstraintDFAExportSpec': '.graph.constraints',
    'ConstraintHMMCompilation': '.graph.constraint_compiler',
    'ConstraintHMMState': '.graph.constraint_compiler',
    'DiscreteHMM': '.discrete.discreteHMM',
    'DomiKnowSAwareHMM': '.graph',
    'DynamicConstraintContext': '.graph.dynamic',
    'EmissionMaskSpec': '.graph.constraints',
    'FactorizedStateSpace': '.graph.dynamic',
    'FiniteStateDynamicConstraint': '.graph.dynamic',
    'ForbiddenEmissionsSpec': '.graph.constraints',
    'ForbiddenTransitionsSpec': '.graph.constraints',
    'GraphHMMGenerationHead': '.graph.graphAwareHMMLearner',
    'HMMFactorGraphBundle': '.discrete.factors',
    'HMMFactorGraphContext': '.discrete.factors',
    'HMMFactorGraphEncoder': '.discrete.factors',
    'HMMFactorGraphHead': '.discrete.factors',
    'HMMFitResult': '.graph',
    'HMMForwardBackward': '.discrete.discreteHMM',
    'HMMGenerationHead': '.discrete.discreteHMMLearner',
    'HMMParameters': '.discrete.discreteHMM',
    'StatePredicateTransitionSpec': '.graph.constraints',
    'TransitionMaskSpec': '.graph.constraints',
    'ViterbiResult': '.graph',
    'apply_hmm_dp_consistency_constraints': '.discrete.factors',
    'apply_transition_energy': '.graph.dynamic',
    'baum_welch_train': '.discrete.discreteHMM',
    'combine_masks': '.graph.constraints',
    'compare_hmm_dfa': '.discrete.discreteHMM',
    'compile_generation_constraints_to_hmm_support': '.graph.constraint_compiler',
    'domiknows_hmm_from_generation_constraints': '.graph.constraint_compiler',
    'hmm_dp_factor_consistency_loss': '.discrete.factors',
    'hmm_factor_sequence_nll': '.discrete.factors',
    'hmm_forward_backward_factors': '.discrete.factors',
    'normalize_matrix_rows': '.graph.constraints',
    'project_distribution': '.graph.constraints',
    'project_matrix': '.graph.constraints',
    'project_matrix_rows': '.graph.constraints',
    'transition_energy_matrix': '.graph.dynamic',
    'validate_mask': '.graph.constraints',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value
