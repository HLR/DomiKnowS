"""Declarative graph for the simple HMM + DFA visualization demo."""
from __future__ import annotations

from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import atLeastAL, atMostAL


VOCAB = ("A", "B", "C", "END")
EOS_TOKEN = "END"
OTHER_TOKEN = "_other"
ENUM_VALUES = (*VOCAB, OTHER_TOKEN)


def build_graph():
    """Build a tiny DomiKnowS graph with one human-readable constraint.

    The only domain rule is:

    ``B`` may appear at most once in the generated string.
    """

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("simple_hmm_dfa_viz") as graph:
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

        # The single demo constraint: token B may appear at most once.
        atMostAL(generated_symbol.B("x"), 1)

    parts = (string, position, symbol, contains, generated_symbol, precedes, earlier, later)
    return graph, parts


def build_two_constraint_graph():
    """Build a tiny graph with two readable constraints.

    The domain rules are:

    - ``B`` may appear at most once.
    - ``C`` must appear at least once.
    """

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("simple_hmm_dfa_viz_two_constraints") as graph:
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

        # Constraint 1: token B may appear at most once.
        atMostAL(generated_symbol.B("x"), 1)
        # Constraint 2: token C must appear at least once.
        atLeastAL(generated_symbol.C("x"), 1)

    parts = (string, position, symbol, contains, generated_symbol, precedes, earlier, later)
    return graph, parts
