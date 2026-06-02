"""Graph and bundle builder for the nested-constraints demo.

The graph mirrors :mod:`Tasks.real_hmm_pmd_learning.graph` (string contains
positions, each position carries a ``generated_symbol`` enum, ``precedes``
relates two positions with ``earlier`` and ``later`` roles).  The constraint
set is intentionally larger and uses path-aware shapes -- see
:mod:`Tasks.nested_constraints_demo.constraints` for the LC tree.
"""
from __future__ import annotations

from domiknows.generation import generation_bundle_from_graph
from domiknows.graph import Concept, EnumConcept, Graph, Relation


VOCAB = ("A", "B", "C", "D", "END")
EOS_TOKEN = "END"
OTHER_TOKEN = "_other"
ENUM_VALUES = (*VOCAB, OTHER_TOKEN)


GENERATION_SCHEMA = {
    "sequence": {
        "graph_name": "string",
        "bundle_field": "text",
        "builder_arg": "text_name",
        "description": "Whole generated string.",
    },
    "position": {
        "graph_name": "position",
        "bundle_field": "token",
        "builder_arg": "token_name",
        "description": "One generated symbol position.",
    },
    "label": {
        "graph_name": "generated_symbol",
        "bundle_field": "generated_token",
        "builder_arg": "generated_token_name",
        "description": "Compact enum label: A, B, C, D, END, or _other.",
    },
    "order": {
        "graph_name": "precedes",
        "bundle_field": "is_before_rel",
        "builder_arg": "before_relation_name",
        "description": "Ordering relation between two positions.",
    },
    "earlier": {
        "graph_name": "earlier",
        "bundle_field": "first_token",
        "builder_arg": "first_role_name",
        "description": "Earlier endpoint of precedes.",
    },
    "later": {
        "graph_name": "later",
        "bundle_field": "second_token",
        "builder_arg": "second_role_name",
        "description": "Later endpoint of precedes.",
    },
    "vocabulary": {
        "graph_name": None,
        "bundle_field": "vocabulary",
        "builder_arg": "vocab",
        "description": "Compact surface-symbol vocabulary.",
    },
}


def build_graph():
    """Build the DomiKnowS graph for the nested-constraints demo.

    The graph carries no logical constraints itself -- :func:`apply_constraints`
    in :mod:`Tasks.nested_constraints_demo.constraints` registers the three
    head LCs on top of the returned graph.
    """
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("nested_constraints_demo") as graph:
        string = Concept(name="string")
        position = Concept(name="position")
        symbol = Concept(name="symbol")

        (contains,) = string.contains(position)
        precedes = Concept(name="precedes")
        earlier, later = precedes.has_a(earlier=position, later=position)

        generated_symbol = position(
            name="generated_symbol",
            ConceptClass=EnumConcept,
            values=list(ENUM_VALUES),
        )

    return graph, (string, position, symbol, contains, generated_symbol, precedes, earlier, later)


def build_bundle():
    """Return ``(graph, bundle)`` with constraints registered on the graph.

    The bundle wires graph concepts to the generation-side identifiers used by
    DFA discovery, the PMD program, and decoders.  Constraints are applied
    immediately so the bundle is fully usable by callers (matching the shape
    of :mod:`Tasks.real_hmm_pmd_learning.graph.build_bundle`).
    """
    graph, _parts = build_graph()
    bundle = generation_bundle_from_graph(
        graph,
        vocab=VOCAB,
        eos_token=EOS_TOKEN,
        text_name="string",
        token_name="position",
        generated_token_name="generated_symbol",
        before_relation_name="precedes",
        first_role_name="earlier",
        second_role_name="later",
    )

    # Apply the head LCs *after* the bundle wiring exists so the constraints
    # can reference ``bundle.context.is_before_rel`` etc. through the canonical
    # helper API rather than the raw graph relation handles.
    try:
        from .constraints import apply_constraints  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - direct script execution fallback
        from constraints import apply_constraints  # type: ignore[no-redef]
    apply_constraints(graph, bundle)
    return graph, bundle
