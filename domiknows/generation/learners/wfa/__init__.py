"""Lazy WFA package exports."""

from importlib import import_module

_EXPORTS = {
    'GraphSpectralAutomaton': '.graph',
    'GraphSpectralFitResult': '.graph',
    'GraphSpectralGenerationHead': '.graph_head',
    'ProductDecoderState': '.hankel',
    'PromptConditionedSpectralWFAGenerationHead': '.prompt_conditioned_head',
    'SpectralBasis': '.spectral_learning',
    'SpectralLearningResult': '.spectral_learning',
    'SpectralWFAFactorGraphBundle': '.factors',
    'SpectralWFAFactorGraphContext': '.factors',
    'SpectralWFAFactorGraphEncoder': '.factors',
    'SpectralWFAFactorGraphHead': '.factors',
    'SpectralWFAGenerationHead': '.head',
    'WeightedFiniteAutomaton': '.hankel',
    'allowed_product_symbols': '.hankel',
    'apply_wfa_factor_consistency_constraints': '.factors',
    'build_spectral_basis': '.spectral_learning',
    'constrained_hankel_matrix': '.hankel',
    'hankel_matrix': '.hankel',
    'masked_empirical_initialization': '.graph',
    'projection_summary': '.hankel',
    'sequence_has_legal_path': '.graph',
    'spectral_learn_from_counts': '.spectral_learning',
    'spectral_learn_from_oracle': '.spectral_learning',
    'spectral_learn_from_samples': '.spectral_learning',
    'start_product_state': '.hankel',
    'step_product_state': '.hankel',
    'wfa_factor_consistency_loss': '.factors',
    'wfa_factor_sequence_energy_loss': '.factors',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value
