"""Ordinal and named-card selection, declared as logical constraints over image tags.

Instructions such as "pick the 2nd poker from left to right", "fetch the first book from the
left on the bottom layer", "add 3rd condiment from bottom" or "pick the 2nd largest poker" name
a group of objects, an order and a position in it. The observations (which tags belong to the
group, which tag precedes which, which card face a tag shows) come from image perception and
the instruction (``selection_observations.py``). These formulas decide which tag may be chosen:

* ``ordinal_rank_<k>``: the chosen tag is in the group and exactly ``k - 1`` group members
  precede it;
* ``named_card_face``: the chosen tag shows the named card face.

``SelectionGraph.admitted`` grounds the formulas for every visible tag; the admitted tags become
the instruction's candidates (``instruction_candidates_bound_targets`` in ``decoding_graph.py``).
"""
from dataclasses import dataclass

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import andL, exactL, ifL
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding,
    PredicateEvent,
    compile_predicate_constraints,
)

DOMAIN = "selection"
MAX_GROUP = 8  # the largest group an ordinal rule is declared for


@dataclass(frozen=True)
class SelectionGraph:
    graph: Graph
    bindings: tuple
    program: object

    def recompiled(self):
        """Snapshot the active formulas again; rebuild after editing or deactivating a declaration."""
        return SelectionGraph(self.graph, self.bindings, compile_predicate_constraints(self.graph, list(self.bindings)))

    def admitted(self, tags, rank=None, members=(), precedes=(), face_requested=False, face_matches=()):
        """The tags the compiled formulas allow as the chosen object (all of them when nothing is requested)."""
        tags = tuple(tags)
        members, face_matches, precedes = set(members), set(face_matches), set(precedes)
        values = {(f"rank_{k}", "request"): rank == k for k in range(1, MAX_GROUP + 1)}
        values[("face_requested", "request")] = bool(face_requested)
        for tag in tags:
            values[("member", tag)] = tag in members
            values[("face_matches", tag)] = tag in face_matches
            for other in tags:
                values[("precedes", other, tag)] = (other, tag) in precedes
        result = []
        for tag in tags:
            for other in tags:
                values[("chosen", other)] = other == tag
            event = PredicateEvent(DOMAIN, values, variable_domains={"q": ("request",), "s": (tag,), "o": tags})
            if self.program.check(event):
                result.append(tag)
        return tuple(result)


def build_selection_graph(name="vlabench_selection_knowledge"):
    with Graph(name) as graph:
        request = Concept(name="selection_request")
        tag = Concept(name="selection_tag")
        precedes = Concept(name="selection_precedes")
        precedes.has_a(selection_before=tag, selection_after=tag)
        ranks = {k: request(name=f"rank_{k}") for k in range(1, MAX_GROUP + 1)}
        face_requested = request(name="face_requested")
        chosen, member, face_matches = (tag(name=n) for n in ("chosen", "member", "face_matches"))

        for k, rank in ranks.items():
            # The k-th of the group: a member with exactly k - 1 members before it.
            ifL(andL(rank("q"), chosen("s")),
                andL(member("s"), exactL(andL(member("o"), precedes("o", "s")), limit=k - 1)),
                name=f"ordinal_rank_{k}")
        ifL(andL(face_requested("q"), chosen("s")), face_matches("s"), name="named_card_face")

    bindings = [PredicateBinding(concept, concept.name, DOMAIN)
                for concept in (*ranks.values(), face_requested, chosen, member, face_matches)]
    bindings.append(PredicateBinding(precedes, "precedes", DOMAIN))
    return SelectionGraph(graph, tuple(bindings), compile_predicate_constraints(graph, bindings))
