"""Self-contained one-rule graph for the real PMD learning demo."""
from __future__ import annotations

from domiknows.generation import generation_bundle_from_graph
from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import atMostAL


VOCAB = ("A", "B", "C", "D", "END")
EOS_TOKEN = "END"
OTHER_TOKEN = "_other"
ENUM_VALUES = (*VOCAB, OTHER_TOKEN)

CANDIDATES = {
    "valid": ("A", "B", "C", "D", "END"),
    "invalid": ("A", "B", "C", "B", "END"),
}

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
    """Build the tiny DomiKnowS graph: token B may appear at most once."""
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("real_hmm_pmd_learning") as graph:
        string = Concept(name="string")
        position = Concept(name="position")
        symbol = Concept(name="symbol")

        contains, = string.contains(position)
        precedes = Concept(name="precedes")
        earlier, later = precedes.has_a(earlier=position, later=position)

        generated_symbol = position(
            name="generated_symbol",
            ConceptClass=EnumConcept,
            values=list(ENUM_VALUES),
        )

        # The one domain rule in this beginner demo.
        atMostAL(generated_symbol.B("x"), 1)

    return graph, (string, position, symbol, contains, generated_symbol, precedes, earlier, later)


def build_bundle():
    """Adapt the graph to the generation helpers used by DFA/PMD code.

    ``GENERATION_SCHEMA`` is the single map between human graph names,
    ``generation_bundle_from_graph`` arguments, and returned bundle fields.
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
    return graph, bundle
