"""DFA primitives, compilers, graph discovery, and tracing helpers."""

from importlib import import_module

_EXPORTS = {
    "ConstrainedGenerationResult": ".decoder",
    "DFA": ".core",
    "DFATrace": ".visualization",
    "DFATraceStep": ".visualization",
    "DotSvgRenderResult": ".visualization",
    "GenerationBundle": ".encoder",
    "GenerationConstraintAnalysis": ".graph_discovery",
    "GenerationEncoder": ".encoder",
    "GenerationGraphContext": ".encoder",
    "ProductAutomatonTrace": ".visualization",
    "ProductReachabilityGraph": ".visualization",
    "ProductTraceStep": ".visualization",
    "State": ".core",
    "Symbol": ".core",
    "TokenVocabulary": ".vocabulary",
    "apply_all_constraints": ".generation_constraints",
    "apply_conditional_max_non_eos_constraint": ".generation_constraints",
    "apply_conditional_max_non_eos_constraints": ".generation_constraints",
    "apply_eos_closure_constraint": ".generation_constraints",
    "apply_forbidden_token_constraint": ".generation_constraints",
    "apply_forbidden_token_constraints": ".generation_constraints",
    "apply_max_non_eos_constraint": ".generation_constraints",
    "apply_required_token_constraint": ".generation_constraints",
    "apply_required_token_constraints": ".generation_constraints",
    "analyze_generation_constraints": ".graph_discovery",
    "complement_dfa": ".core",
    "constrained_beam_search_decode": ".decoder",
    "constrained_greedy_decode": ".decoder",
    "constrained_label_beam_search_decode": ".decoder",
    "constrained_label_greedy_decode": ".decoder",
    "constrained_label_sample_decode": ".decoder",
    "constrained_sample_decode": ".decoder",
    "constraints_to_dfa_from_graph": ".graph_discovery",
    "create_generation_debug_app": ".visualization",
    "dfa_to_dot": ".visualization",
    "dot_to_svg": ".visualization",
    "explain_dfa_rejection": ".visualization",
    "generation_bundle_from_graph": ".encoder",
    "mask_label_logits_for_dfa": ".decoder",
    "mask_logits_for_dfa": ".decoder",
    "product_dfa": ".core",
    "product_trace_to_dot": ".visualization",
    "reachable_product_graph": ".visualization",
    "render_dot_svg": ".visualization",
    "run_generation_debug_server": ".visualization",
    "trace_dfa": ".visualization",
    "trace_discrete_hmm": ".visualization",
    "trace_product_automaton": ".visualization",
    "union_dfa": ".core",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value
