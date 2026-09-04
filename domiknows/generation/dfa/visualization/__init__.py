"""DFA tracing, DOT export, and optional debug viewer helpers."""

from .server import (
    DotSvgRenderResult,
    create_generation_debug_app,
    dot_to_svg,
    render_dot_svg,
    run_generation_debug_server,
)
from .tracing import (
    DFATrace,
    DFATraceStep,
    ProductAutomatonTrace,
    ProductReachabilityGraph,
    ProductTraceStep,
    dfa_to_dot,
    explain_dfa_rejection,
    product_trace_to_dot,
    reachable_product_graph,
    trace_dfa,
    trace_discrete_hmm,
    trace_product_automaton,
)

__all__ = [
    "DFATrace",
    "DFATraceStep",
    "DotSvgRenderResult",
    "ProductAutomatonTrace",
    "ProductReachabilityGraph",
    "ProductTraceStep",
    "create_generation_debug_app",
    "dfa_to_dot",
    "dot_to_svg",
    "explain_dfa_rejection",
    "product_trace_to_dot",
    "reachable_product_graph",
    "render_dot_svg",
    "run_generation_debug_server",
    "trace_dfa",
    "trace_discrete_hmm",
    "trace_product_automaton",
]
