"""VLABench poker categories, dominance and first-combination ties as LCs.

Reference: OpenMOSS/VLABench cf588fe60c0c7282174fe979f5913170cfe69017,
VLABench/tasks/hierarchical_tasks/poker_utils.py. In particular, this is NOT
standard kicker tie-breaking or an exactly-five-card destination requirement.
The adapter enumerates candidates and observes equality/count/adjacency facts;
the logical formulas, rather than Python winner selection, decide obligations.
"""
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations, count

from domiknows.generation.dfa.predicate_monitor import PredicateEvent
from domiknows.generation.grounded_facts import Check
from domiknows.generation.lc_relay import RelayedKnowledge

UPSTREAM_REVISION = "cf588fe60c0c7282174fe979f5913170cfe69017"
_IDS = count()
FEATURES = ("ace_high", "quad", "trip", "two_pairs", "one_pair", "no_pairs",
            *(f"suit_equal_{i}" for i in range(4)), *(f"rank_adjacent_{i}" for i in range(4)))


def category_expression(features, rank, x):
    from domiknows.graph.logicalConstrain import andL, notL, orL

    def body(r):
        f = lambda n: features[n](x)
        flush = lambda: andL(*(f(f"suit_equal_{i}") for i in range(4)))
        straight = lambda: andL(*(f(f"rank_adjacent_{i}") for i in range(4)))
        return {10: lambda: andL(flush(), straight(), f("ace_high")), 9: lambda: andL(flush(), straight()),
                8: lambda: f("quad"), 7: lambda: andL(f("trip"), f("one_pair")), 6: flush, 5: straight,
                4: lambda: f("trip"), 3: lambda: f("two_pairs"), 2: lambda: f("one_pair"),
                1: lambda: f("no_pairs")}[r]()

    higher = [body(k) for k in range(rank + 1, 11)]
    return andL(body(rank), notL(orL(*higher))) if higher else body(rank)


def declare_poker(rules):
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import andL, existsL, notL, orL
    serial = next(_IDS)
    request, candidate, card = (Concept(name=n) for n in ("poker_request", "poker_candidate", "poker_card"))
    known = request(name="observations_known")
    ending = request(name="at_completion")
    features = {n: candidate(name=n) for n in FEATURES}
    categories = {r: candidate(name=f"category_{r}") for r in range(1, 11)}
    placed, location_known = (card(name=n) for n in ("at_destination", "location_known"))

    def relation(name, left, right):
        rel = Concept(name="poker_" + name)
        rel.has_a(**{f"{name}_left_{serial}": left, f"{name}_right_{serial}": right})
        return rel

    earlier = relation("earlier", candidate, candidate)
    members = {n: relation(n, candidate, card) for n in ("member", "count4", "count3", "count2", "largest")}

    def category(rank, x):
        return categories[rank](x)

    # Absent faces/order/destination are unknown evidence, not a guessed hand.
    rules.rule("poker__observations", "goal_completion", lambda: ending("q"),
               lambda: orL(known("q"), notL(known("q"))), lambda: known("q"))
    for rank in range(1, 11):
        def winner(r=rank):
            superior = [category(k, "d") for k in range(r + 1, 11)]
            superior.append(andL(category(r, "d"), earlier("d", "c")))
            return andL(ending("q"), known("q"), candidate("c"), category(r, "c"),
                        notL(existsL(andL(candidate("d"), orL(*superior)))))

        member_kind = {8: "count4", 4: "count3", 3: "count2", 2: "count2", 1: "largest"}.get(rank, "member")
        rules.rule(f"poker__category_{rank}__required_cards", "goal_completion",
                   lambda w=winner, m=member_kind: andL(w(), members[m]("c", "e")),
                   lambda: orL(notL(location_known("e")), placed("e")),
                   lambda: andL(location_known("e"), placed("e")))
    rows = [(known, "observations_known"), (ending, "at_completion"), (candidate, "candidate"),
            (earlier, "earlier"), (placed, "at_destination"), (location_known, "location_known")]
    rows += [(c, n) for n, c in (*features.items(), *members.items())]
    rows += [(c, f"category_{r}") for r, c in categories.items()]
    return [(c, n, "poker_selection", "completion") for c, n in rows]


class PokerKnowledge(RelayedKnowledge):
    """Classification queries feed the finite-grounded dominance/goal formulas.

    The ten query formulas are auxiliary computations, like the relay's own
    applies/decided views; they are not ten mutually incompatible plan demands.
    """
    def __init__(self, active=True):
        from domiknows.graph import Concept, Graph
        from domiknows.graph.logicalConstrain import ifL
        from domiknows.generation.dfa.predicate_monitor import PredicateBinding, compile_predicate_constraints, PredicateConstraintProgram
        with Graph(f"poker_classification_{next(_IDS)}") as graph:
            hand = Concept(name="hand")
            features = {n: hand(name=n) for n in FEATURES}
            for rank in range(1, 11):
                ifL(hand("c"), category_expression(features, rank, "c"), name=f"category_{rank}")
        program = compile_predicate_constraints(graph,
            [PredicateBinding(hand, "candidate", "poker_selection", "completion")]
            + [PredicateBinding(c, n, "poker_selection", "completion") for n, c in features.items()])
        self.classification_graph = graph
        self.classifiers = {r.name: PredicateConstraintProgram((r,)) for r in program.rules}
        self._classification_cache = {}
        super().__init__("vlabench_poker", declare_poker, active=active)

    def relay(self, event, names=None):
        values = dict(event.values)
        for candidate in event.variable_domains["c"]:
            key = tuple(values[(n, candidate)] for n in FEATURES)
            if key not in self._classification_cache:
                snapshot = PredicateEvent("poker_selection", values, "completion", variable_domains={"c": (candidate,)})
                self._classification_cache[key] = {n: p.check(snapshot) for n, p in self.classifiers.items()}
            values.update({(n, candidate): truth for n, truth in self._classification_cache[key].items()})
        derived = PredicateEvent(event.domain, values, event.phase, variable_domains=event.variable_domains)
        yield from super().relay(derived, names)


def build_poker_knowledge(active=True):
    return PokerKnowledge(active=active)


@dataclass(frozen=True)
class PokerSelection:
    cards: tuple = ()  # Ordered (canonical entity, observed rank or None, observed suit or None).
    destination: str | None = None
    known: bool = False


def parse_poker_selection(raw, resolve):
    if raw is None:
        return PokerSelection()
    if not isinstance(raw, dict) or set(raw) - {"cards", "destination", "order_known"}:
        raise ValueError("poker_selection accepts only cards, destination and order_known observations")
    cards = []
    ranks = {"ace": 14, "king": 13, "queen": 12, "jack": 11, **{str(i): i for i in range(2, 11)}}
    observed = raw.get("cards", ())
    if not isinstance(observed, (tuple, list)) or len(observed) > 7:
        raise ValueError("poker_selection.cards must be an ordered list of at most seven cards")
    for item in observed:
        if not isinstance(item, dict) or set(item) - {"entity", "rank", "suit", "face_observed"}:
            raise ValueError("card observations accept entity, rank, suit and face_observed")
        face = item.get("face_observed", False)
        if type(face) is not bool:
            raise ValueError("face_observed must be Boolean")
        rank, suit = item.get("rank"), item.get("suit")
        if face:
            rank = ranks.get(rank.lower()) if isinstance(rank, str) else rank
            if rank is not None and (type(rank) is not int or not 2 <= rank <= 14):
                raise ValueError("observed card rank must be 2..14")
            if suit is not None and suit not in ("clubs", "diamonds", "hearts", "spades"):
                raise ValueError("unknown observed card suit")
        cards.append((resolve(item["entity"]), rank if face else None, suit if face else None))
    if len({c[0] for c in cards}) != len(cards):
        raise ValueError("duplicate observed card entity")
    faces = [(r, s) for _, r, s in cards if r is not None and s is not None]
    if len(set(faces)) != len(faces):
        raise ValueError("duplicate observed card face")
    order = raw.get("order_known", True)
    if type(order) is not bool:
        raise ValueError("order_known must be Boolean")
    destination = resolve(raw["destination"]) if raw.get("destination") is not None else None
    known = order and 5 <= len(cards) <= 7 and len(faces) == len(cards) and destination is not None
    return PokerSelection(tuple(cards), destination, known)


def poker_event(selection, state):
    cards = selection.cards
    combos = tuple(combinations(range(len(cards)), 5)) if selection.known else ()
    candidates = tuple(f"hand_{i}" for i in range(len(combos)))
    entities = tuple(c[0] for c in cards)
    values = {("observations_known", "request"): selection.known, ("at_completion", "request"): True}
    for i, combo in enumerate(combos):
        c = candidates[i]
        ranks = [cards[j][1] for j in combo]
        suits = [cards[j][2] for j in combo]
        counts = Counter(ranks)
        ordered = sorted(1 if r == 14 and set(ranks) == {14, 2, 3, 4, 5} else r for r in ranks)
        facts = {"ace_high": max(ranks) == 14, "quad": 4 in counts.values(), "trip": 3 in counts.values(),
                 "one_pair": list(counts.values()).count(2) == 1, "two_pairs": list(counts.values()).count(2) == 2,
                 "no_pairs": len(counts) == 5}
        facts.update({f"suit_equal_{j}": suits[j] == suits[j + 1] for j in range(4)})
        facts.update({f"rank_adjacent_{j}": ordered[j] + 1 == ordered[j + 1] for j in range(4)})
        values[("candidate", c)] = True
        values.update({(n, c): v for n, v in facts.items()})
        for e, (name, rank, _suit) in enumerate(cards):
            values[("member", c, name)] = e in combo
            for number in (2, 3, 4):
                values[(f"count{number}", c, name)] = e in combo and counts.get(rank, 0) == number
            values[("largest", c, name)] = e in combo and rank == max(ranks)
        for j, d in enumerate(candidates):
            values[("earlier", c, d)] = i < j
    for entity in entities:
        truth = state.facts.truth(("at", entity, selection.destination)) if selection.destination is not None else None
        values[("location_known", entity)] = truth is not None
        values[("at_destination", entity)] = truth is True
    return PredicateEvent("poker_selection", values, "completion",
                          variable_domains={"q": ("request",), "c": candidates, "d": candidates, "e": entities})


@dataclass(eq=False)
class PokerMonitor:
    knowledge: object
    selection: PokerSelection
    _cache: dict = field(default_factory=dict, repr=False)

    def operation(self, state, op):
        return ()

    def completion(self, state):
        if state.facts not in self._cache:
            observations = tuple(o for _, o in self.knowledge.relay(poker_event(self.selection, state)))
            truths = [o.truth for o in observations]
            truth = False if False in truths else None if None in truths else True
            self._cache[state.facts] = (Check("goal_completion", bool(observations), truth, (), observations),)
        return self._cache[state.facts]
