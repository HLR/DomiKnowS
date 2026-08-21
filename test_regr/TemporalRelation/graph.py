from dataclasses import dataclass
import re

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, exactL, ifL, notL


TEMPORAL_LABELS = ("Before", "After", "Equal", "Vague")


def safe_name(value):
    value = str(value)
    safe = re.sub(r"\W", "_", value)
    if not safe or safe[0].isdigit():
        safe = f"v_{safe}"
    return safe


@dataclass
class TemporalRelationContext:
    graph: Graph
    document: Concept
    sentence: Concept
    token: Concept
    event: Concept
    query_event1: Concept
    query_event2: Concept
    event_pair: Concept
    temporal_relation: Concept
    document_contains_sentence: Relation
    sentence_contains_token: Relation
    pair_event1: Relation
    pair_event2: Relation
    label_concepts: dict
    event_concepts: dict
    concepts: dict
    namespace: dict


def create_temporal_graph(
    instance_or_dataset=None,
    graph_name="temporal_relation",
    include_global_constraints=True,
    include_transitive_constraints=True,
):
    """
    Build a MATRES-style temporal relation graph.

    Event mentions are token/span data instances, not graph concepts. The graph
    therefore contains generic token -> event predicates, learned query_event1 /
    query_event2 marker predicates over events, and a multiclass temporal label
    over each EventPair. MATRES annotations provide oracle labels for the marker
    predicates, while the normal learner predicts the temporal relation label.
    """
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph(graph_name) as graph:
        document = Concept(name="document")
        sentence = Concept(name="sentence")
        token = Concept(name="token")
        document_contains_sentence, = document.contains(sentence)
        sentence_contains_token, = sentence.contains(token)

        event = token(name="event")
        query_event1 = event(name="query_event1")
        query_event2 = event(name="query_event2")

        event_pair = Concept(name="EventPair")
        pair_event1, pair_event2 = event_pair.has_a(e1=token, e2=token)

        temporal_relation = event_pair(name="temporal_relation")
        label_concepts = {label: temporal_relation(name=label) for label in TEMPORAL_LABELS}
        before = label_concepts["Before"]
        after = label_concepts["After"]
        equal = label_concepts["Equal"]
        vague = label_concepts["Vague"]

        if include_global_constraints:
            # Graph-level temporal consistency losses. Disable these for the
            # first CLEVR-style executable-query-only baseline.
            ifL(
                event_pair("p"),
                exactL(before("p"), after("p"), equal("p"), vague("p"), limit=1),
                name="temporal_exactly_one_label",
            )
            ifL(
                andL(
                    before("p"),
                    event("p1", path=("p", pair_event1)),
                    event("p2", path=("p", pair_event2)),
                ),
                after("p_rev", path=(("p2", pair_event1.reversed), ("p1", pair_event2.reversed))),
                name="temporal_before_inverse_after",
            )
            ifL(
                andL(
                    after("p"),
                    event("p1", path=("p", pair_event1)),
                    event("p2", path=("p", pair_event2)),
                ),
                before("p_rev", path=(("p2", pair_event1.reversed), ("p1", pair_event2.reversed))),
                name="temporal_after_inverse_before",
            )
            ifL(
                andL(
                    equal("p"),
                    event("p1", path=("p", pair_event1)),
                    event("p2", path=("p", pair_event2)),
                ),
                equal("p_rev", path=(("p2", pair_event1.reversed), ("p1", pair_event2.reversed))),
                name="temporal_equal_symmetric",
            )
            ifL(
                andL(
                    vague("p"),
                    event("p1", path=("p", pair_event1)),
                    event("p2", path=("p", pair_event2)),
                ),
                vague("p_rev", path=(("p2", pair_event1.reversed), ("p1", pair_event2.reversed))),
                name="temporal_vague_symmetric",
            )
            ifL(
                andL(
                    before("p"),
                    event("p1", path=("p", pair_event1)),
                    event("p2", path=("p", pair_event2)),
                ),
                notL(before("p_rev", path=(("p2", pair_event1.reversed), ("p1", pair_event2.reversed)))),
                name="temporal_before_no_cycle_2",
            )
            # The 7 transitive/substitution rules below all chain a *second*
            # EventPair-typed variable (yz) off an event derived from the first
            # (xy) via a `.reversed` path hop. This currently crashes
            # domiknows' calculateLcLoss with a tensor shape mismatch (the
            # chained variable resolves to the full candidate-pair count
            # instead of being scoped to its binding) — see temporal_improve
            # investigation notes. The 6 simpler rules above (single EventPair
            # variable, direct event paths, no chaining) are unaffected.
            # include_transitive_constraints lets callers exclude just these 7
            # as an interim workaround while that's traced/fixed upstream.
            if include_transitive_constraints:
                ifL(
                    andL(
                        before("xy"),
                        event("x", path=("xy", pair_event1)),
                        event("y", path=("xy", pair_event2)),
                        before("yz", path=("y", pair_event1.reversed)),
                        event("z", path=("yz", pair_event2)),
                    ),
                    before("xz", path=(("x", pair_event1.reversed), ("z", pair_event2.reversed))),
                    name="temporal_before_transitive",
                )
                ifL(
                    andL(
                        after("xy"),
                        event("x", path=("xy", pair_event1)),
                        event("y", path=("xy", pair_event2)),
                        after("yz", path=("y", pair_event1.reversed)),
                        event("z", path=("yz", pair_event2)),
                    ),
                    after("xz", path=(("x", pair_event1.reversed), ("z", pair_event2.reversed))),
                    name="temporal_after_transitive",
                )
                ifL(
                    andL(
                        equal("xy"),
                        event("x", path=("xy", pair_event1)),
                        event("y", path=("xy", pair_event2)),
                        equal("yz", path=("y", pair_event1.reversed)),
                        event("z", path=("yz", pair_event2)),
                    ),
                    equal("xz", path=(("x", pair_event1.reversed), ("z", pair_event2.reversed))),
                    name="temporal_equal_transitive",
                )
                ifL(
                    andL(
                        equal("xy"),
                        event("x", path=("xy", pair_event1)),
                        event("y", path=("xy", pair_event2)),
                        before("yz", path=("y", pair_event1.reversed)),
                        event("z", path=("yz", pair_event2)),
                    ),
                    before("xz", path=(("x", pair_event1.reversed), ("z", pair_event2.reversed))),
                    name="temporal_equal_before_left_substitution",
                )
                ifL(
                    andL(
                        equal("xy"),
                        event("x", path=("xy", pair_event1)),
                        event("y", path=("xy", pair_event2)),
                        after("yz", path=("y", pair_event1.reversed)),
                        event("z", path=("yz", pair_event2)),
                    ),
                    after("xz", path=(("x", pair_event1.reversed), ("z", pair_event2.reversed))),
                    name="temporal_equal_after_left_substitution",
                )
                ifL(
                    andL(
                        before("xy"),
                        event("x", path=("xy", pair_event1)),
                        event("y", path=("xy", pair_event2)),
                        equal("yz", path=("y", pair_event1.reversed)),
                        event("z", path=("yz", pair_event2)),
                    ),
                    before("xz", path=(("x", pair_event1.reversed), ("z", pair_event2.reversed))),
                    name="temporal_equal_before_right_substitution",
                )
                ifL(
                    andL(
                        after("xy"),
                        event("x", path=("xy", pair_event1)),
                        event("y", path=("xy", pair_event2)),
                        equal("yz", path=("y", pair_event1.reversed)),
                        event("z", path=("yz", pair_event2)),
                    ),
                    after("xz", path=(("x", pair_event1.reversed), ("z", pair_event2.reversed))),
                    name="temporal_equal_after_right_substitution",
                )

    concepts = {
        "document": document,
        "sentence": sentence,
        "token": token,
        "event": event,
        "query_event1": query_event1,
        "query_event2": query_event2,
        "document_contains_sentence": document_contains_sentence,
        "sentence_contains_token": sentence_contains_token,
        "EventPair": event_pair,
        "event_pair": event_pair,
        "pair_event1": pair_event1,
        "pair_event2": pair_event2,
        "temporal_relation": temporal_relation,
        **label_concepts,
    }
    namespace = dict(concepts)
    namespace.update({"graph": graph, "iota_target": event_pair})
    _register_namespace(graph, namespace)

    return TemporalRelationContext(
        graph=graph,
        document=document,
        sentence=sentence,
        token=token,
        event=event,
        query_event1=query_event1,
        query_event2=query_event2,
        event_pair=event_pair,
        temporal_relation=temporal_relation,
        document_contains_sentence=document_contains_sentence,
        sentence_contains_token=sentence_contains_token,
        pair_event1=pair_event1,
        pair_event2=pair_event2,
        label_concepts=label_concepts,
        event_concepts={},
        concepts=concepts,
        namespace=namespace,
    )


def unpack_pair(pair):
    if isinstance(pair, dict):
        return pair.get("e1"), pair.get("e2"), pair.get("label")
    if len(pair) == 2:
        return pair[0], pair[1], None
    return pair[0], pair[1], pair[2]


def _register_namespace(graph, namespace):
    var_map = graph.varNameReversedMap
    for name, value in namespace.items():
        var_map[name] = value
        if hasattr(value, "reversed"):
            var_map[f"{name}.reversed"] = value.reversed
