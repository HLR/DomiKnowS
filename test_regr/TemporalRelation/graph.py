from collections.abc import Sequence
from dataclasses import dataclass
import re

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.dataNode import DataNode
from domiknows.graph.logicalConstrain import andL, exactL, ifL, notL
from domiknows.solver import ilpOntSolverFactory


#: MATRES's four relations — the default, so existing runs are unchanged.
MATRES_LABELS = ("Before", "After", "Equal", "Vague")
#: Relations TB-Dense adds on top. ``Simultaneous`` is deliberately distinct from
#: MATRES's ``Equal``: MATRES compares start-points under its multi-axis scheme,
#: while TB-Dense's SIMULTANEOUS asserts full interval identity. Collapsing them
#: (as ``_normalize_label`` used to) discards the containment structure the new
#: constraints rely on.
TBDENSE_EXTRA_LABELS = ("Includes", "IsIncluded", "Simultaneous")
EXTENDED_LABELS = MATRES_LABELS + TBDENSE_EXTRA_LABELS


class _LabelSet(Sequence):
    """A label tuple whose contents can be swapped *in place*.

    ``TEMPORAL_LABELS`` is consumed through ``from .graph import TEMPORAL_LABELS``
    in five modules, so rebinding the module global would leave every importer
    holding the old tuple. Mutating one shared object instead lets
    ``create_temporal_graph(labels=...)`` switch the label space without
    touching ~20 call sites, all of which only need ``len``, ``index``,
    ``[]``, ``in`` and iteration.
    """

    def __init__(self, labels):
        self._labels = tuple(labels)

    def set(self, labels):
        self._labels = tuple(labels)
        return self._labels

    def __getitem__(self, index):
        return self._labels[index]

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        return iter(self._labels)

    def __contains__(self, value):
        return value in self._labels

    def index(self, value):
        return self._labels.index(value)

    def __eq__(self, other):
        return tuple(self._labels) == tuple(other)

    def __hash__(self):
        return hash(self._labels)

    def __repr__(self):
        return repr(self._labels)


TEMPORAL_LABELS = _LabelSet(MATRES_LABELS)


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


def create_temporal_graph(instance_or_dataset=None, graph_name="temporal_relation",
                          include_global_constraints=True,
                          include_exactly_one=True, include_transitivity=True,
                          labels=None):
    """
    Build a MATRES-style temporal relation graph.

    Event mentions are token/span data instances, not graph concepts. The graph
    therefore contains generic token -> event predicates, learned query_event1 /
    query_event2 marker predicates over events, and a multiclass temporal label
    over each EventPair. MATRES annotations provide oracle labels for the marker
    predicates, while the normal learner predicts the temporal relation label.

    ``include_exactly_one`` / ``include_transitivity`` gate two constraints that
    behave differently than they appear under the default training setup; both
    default to on, so the constraint set is unchanged unless asked. See the
    notes at their definitions below.

    ``labels`` selects the relation vocabulary — ``MATRES_LABELS`` (default, K=4)
    or ``EXTENDED_LABELS`` (K=7, needed for TB-Dense's containment and
    simultaneity relations). It is applied to the shared ``TEMPORAL_LABELS``
    before any concept is built, so every consumer sees the same vocabulary.
    """
    TEMPORAL_LABELS.set(MATRES_LABELS if labels is None else labels)

    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    # This example has no ontology, so the factory's generic cache key would
    # otherwise reuse a solver built for the previous label vocabulary.
    ilpOntSolverFactory.clear()

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
            if include_exactly_one:
                # Kept on by default; use
                # ``--no-exactly-one-label`` to measure its effect.
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
                    before("p"),
                    event("p1", path=("p", pair_event1)),
                    event("p2", path=("p", pair_event2)),
                ),
                notL(before("p_rev", path=(("p2", pair_event1.reversed), ("p1", pair_event2.reversed)))),
                name="temporal_before_no_cycle_2",
            )
            # ---- relations only the extended (TB-Dense) vocabulary provides ----
            includes = label_concepts.get("Includes")
            is_included = label_concepts.get("IsIncluded")
            simultaneous = label_concepts.get("Simultaneous")

            if includes is not None and is_included is not None:
                # Containment is an inverse pair exactly like before/after.
                for source, target, rule in (
                    (includes, is_included, "temporal_includes_inverse_is_included"),
                    (is_included, includes, "temporal_is_included_inverse_includes"),
                ):
                    ifL(
                        andL(
                            source("p"),
                            event("p1", path=("p", pair_event1)),
                            event("p2", path=("p", pair_event2)),
                        ),
                        target("p_rev", path=(("p2", pair_event1.reversed),
                                              ("p1", pair_event2.reversed))),
                        name=rule,
                    )

            if simultaneous is not None:
                # Simultaneity is symmetric: it holds in both directions.
                ifL(
                    andL(
                        simultaneous("p"),
                        event("p1", path=("p", pair_event1)),
                        event("p2", path=("p", pair_event2)),
                    ),
                    simultaneous("p_rev", path=(("p2", pair_event1.reversed),
                                                ("p1", pair_event2.reversed))),
                    name="temporal_simultaneous_symmetric",
                )

            if include_transitivity:
                # Kept on by default; ``--no-transitivity``
                # disables it, and the trainer warns when it cannot ground.
                ifL(
                    andL(
                        before("xy"),
                        before(
                            "yz",
                            path=("xy", pair_event2, pair_event1.reversed),
                        ),
                    ),
                    before(
                        "xz",
                        path=(
                            ("xy", pair_event1, pair_event1.reversed),
                            ("yz", pair_event2, pair_event2.reversed),
                        ),
                    ),
                    name="temporal_before_transitive",
                )

                # Allen composition is *disjunctive* in general: from
                # Includes(a,b) and Before(b,c) it does NOT follow that
                # Before(a,c) — c may fall inside a. Writing such an entry as a
                # definite implication would train the model toward a false
                # rule, so only single-valued compositions appear here. Anything
                # ambiguous is omitted rather than approximated.
                compositions = [
                    (simultaneous, simultaneous, simultaneous,
                     "temporal_simultaneous_transitive"),
                    (simultaneous, before, before,
                     "temporal_simultaneous_before_is_before"),
                    (includes, includes, includes,
                     "temporal_includes_transitive"),
                    (is_included, is_included, is_included,
                     "temporal_is_included_transitive"),
                    (before, includes, before,
                     "temporal_before_includes_is_before"),
                ]
                for first, second, result, rule_name in compositions:
                    if first is None or second is None or result is None:
                        continue  # relation absent from the active vocabulary
                    ifL(
                        andL(
                            first("xy"),
                            second(
                                "yz",
                                path=("xy", pair_event2, pair_event1.reversed),
                            ),
                        ),
                        result(
                            "xz",
                            path=(
                                ("xy", pair_event1, pair_event1.reversed),
                                ("yz", pair_event2, pair_event2.reversed),
                            ),
                        ),
                        name=rule_name,
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
