"""Declarative graph for the real HMM comparison visualization demo."""
from __future__ import annotations

from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import atLeastAL, atMostAL


VOCAB = ("A", "B", "C", "END")
EOS_TOKEN = "END"
OTHER_TOKEN = "_other"
ENUM_VALUES = (*VOCAB, OTHER_TOKEN)


def build_graph():
    """Build a tiny graph with one DomiKnowS generation constraint."""

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("real_hmm_comparison_viz") as graph:
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

        # The only graph rule in this demo.
        atMostAL(generated_symbol.B("x"), 1)

    return graph, (string, position, symbol, contains, generated_symbol, precedes, earlier, later)


def build_two_constraint_graph():
    """Build a tiny graph with two DomiKnowS generation constraints."""

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("real_hmm_comparison_viz_two_constraints") as graph:
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

        # Rule 1: token B may appear at most once.
        atMostAL(generated_symbol.B("x"), 1)
        # Rule 2: token C must appear at least once.
        atLeastAL(generated_symbol.C("x"), 1)

    return graph, (string, position, symbol, contains, generated_symbol, precedes, earlier, later)
