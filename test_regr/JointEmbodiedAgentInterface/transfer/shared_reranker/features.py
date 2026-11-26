"""Public, anonymous features for a plan executed in the compiled symbolic model."""

from dataclasses import dataclass

from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
from domiknows.generation.grounded_facts import GroundedState


ACTIONS = ("move", "observe", "acquire", "place", "release", "open", "close",
           "activate", "deactivate", "transfer", "manipulate", "other", "start")
_ACTION_ALIASES = {
    "move": ("walk", "run"),
    "observe": ("find", "lookat", "watch", "turnto", "read", "pointat"),
    "acquire": ("pick", "grab", "left_grasp", "right_grasp"),
    "place": ("place", "insert", "put", "putback", "putin", "puton",
              *(f"{hand}_place_{relation}" for hand in ("left", "right")
                for relation in ("inside", "nextto", "nextto_ontop", "on_top", "ontop", "under"))),
    "release": ("drop", "putobjback", "left_release", "right_release", "right_realease"),
    "open": ("open", "open_door", "open_laptop"),
    "close": ("close", "close_door"),
    "activate": ("switch_on", "switchon", "toggle_on", "turn_on", "press"),
    "deactivate": ("switch_off", "switchoff", "toggle_off", "turn_off"),
    "transfer": ("pour", "right_transfer_contents_inside"),
    "manipulate": ("push", "pull", "lift", "touch"),
}
_FAMILIES = {name: family for family, names in _ACTION_ALIASES.items() for name in names}
GOAL_RELATIONS = {"inside", "ontop", "under", "nextto", "at"}
GOAL_PROPERTIES = {"open", "closed", "on", "off", "clean", "cooked", "plugged_in", "frozen",
                   "dusty", "stained"}
PROGRESS = ("unknown", "steady", "advance", "regress", "mixed")
TRUTH = (None, False, True)


@dataclass(frozen=True)
class PublicGoal:
    """Explicit literals grounded from public instructions. Empty means unavailable."""

    literals: tuple[tuple[tuple[str, ...], bool], ...] = ()

    def __post_init__(self):
        seen = set()
        for fact, expected in self.literals:
            arity = 3 if fact and fact[0] in GOAL_RELATIONS else 2
            if (len(fact) != arity or fact[0] not in GOAL_RELATIONS | GOAL_PROPERTIES
                    or any(not isinstance(value, str) or not value for value in fact)
                    or type(expected) is not bool or fact in seen):
                raise ValueError("expected distinct signed public goal literals")
            seen.add(fact)

    def truth(self, state):
        return tuple(None if (observed := state.facts.truth(fact)) is None else observed == expected
                     for fact, expected in self.literals)


@dataclass(frozen=True)
class OperationTrace:
    action: str
    source: str | None
    destination: str | None
    before: GroundedState
    after: GroundedState


@dataclass(frozen=True)
class StepFeatures:
    action: str
    roles: tuple[bool, ...]
    before: tuple[float, float, float]
    after: tuple[float, float, float]
    progress: str
    transitions: tuple[float, ...]


def _fractions(values):
    return tuple(values.count(truth) / len(values) if values else float(truth is None)
                 for truth in TRUTH)


def _transitions(before, after):
    counts = [0.] * (len(TRUTH) ** 2)
    for old, new in zip(before, after):
        counts[len(TRUTH) * TRUTH.index(old) + TRUTH.index(new)] += 1. / len(before)
    return tuple(counts)


def anonymous_features(initial, operations, goal=PublicGoal()):
    """Keep equality and role evidence while discarding all raw entity names."""
    truth = goal.truth(initial)
    steps = [StepFeatures("start", (False,) * 8, _fractions(truth), _fractions(truth),
                          "unknown", _transitions(truth, truth))]
    seen = set()
    for op in operations:
        following = goal.truth(op.after)
        advances = any(new is True and old is not True for old, new in zip(truth, following))
        regresses = any(old is True and new is False for old, new in zip(truth, following))
        if advances and regresses:
            progress = "mixed"
        elif advances:
            progress = "advance"
        elif regresses:
            progress = "regress"
        else:
            progress = "steady" if any(value is not None for value in (*truth, *following)) else "unknown"
        source, destination = op.source, op.destination
        roles = (
            source is not None, destination is not None,
            source is not None and source == destination,
            source in seen, destination in seen,
            source is not None and source in dict(op.before.holdings).values(),
            source is not None and source in dict(op.after.holdings).values(),
            any(len(fact) == 3 and fact[1:] == (source, destination) and expected
                for fact, expected in goal.literals),
        )
        steps.append(StepFeatures(_FAMILIES.get(op.action, "other"), roles,
                                  _fractions(truth), _fractions(following), progress,
                                  _transitions(truth, following)))
        seen.update(entity for entity in (source, destination) if entity is not None)
        truth = following
    return tuple(steps)


def eai_public_goal(observed_goal, world):
    """Ground the public SimpleTL conjunction fragment with open-world truth."""
    from test_regr.EmbodiedAgentInterface.reward import (
        _And, _Atom, _Not, _canonical_predicate, _is_variable, _normalize_name, _using_world_bundle,
    )

    if observed_goal.unresolved:
        return PublicGoal()

    def literals(node):
        if isinstance(node, _And):
            for child in node.args:
                yield from literals(child)
            return
        expected = not isinstance(node, _Not)
        atom = node if expected else node.arg
        if not isinstance(atom, _Atom) or any(_is_variable(value) for value in atom.args):
            raise ValueError("unsupported SimpleTL goal expression")
        relation = _canonical_predicate(atom.name)
        if relation.startswith("not_"):
            relation, expected = relation[4:], not expected
        entities = tuple(observed_goal.aliases.get(name, name) for name in map(_normalize_name, atom.args))
        if not all(entity in observed_goal.types for entity in entities):
            raise ValueError("unbound public goal entity")
        yield (relation, *entities), expected

    with _using_world_bundle(world):
        try:
            return PublicGoal(tuple(dict.fromkeys(literals(observed_goal.ast))))
        except ValueError:
            # Disjunctions, quantifiers and temporal goals need a separate
            # public monitor. Flattening them would change their semantics.
            return PublicGoal()


def _eai_trace(labels, vocabulary, adapter):
    from test_regr.EmbodiedAgentInterface.observed_reward import _events
    from test_regr.EmbodiedAgentInterface.reward import _side_for_action

    executor = adapter.executor
    initial = state = executor.planning.initial
    events = _events((vocabulary.token_for_label(label) for label in labels), adapter.aliases)
    if events is None:
        raise ValueError("accepted EAI plan could not be interpreted")
    steps = []
    for event in events:
        successor, _, _ = executor.advance(state, event.name, event.args)
        entity = event.args[0] if event.args else None
        family = _FAMILIES.get(event.name)
        source, destination = entity, None
        if family in {"place", "transfer"}:
            destination = entity
            hand = _side_for_action(event.name)
            available = dict(state.holdings)
            if hand is not None:
                source = available.get(hand)
            else:
                changed = {obj for side, obj in available.items() if obj is not None
                           and dict(successor.holdings).get(side) != obj}
                held = changed or {obj for obj in available.values() if obj is not None}
                source = next(iter(held)) if len(held) == 1 else None
        steps.append(OperationTrace(event.name, source, destination, state, successor))
        state = successor
    return initial, steps


def _graph_trace(labels, vocabulary, adapter):
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan

    semantics = adapter.semantics
    state = adapter.start_state.semantic
    initial = state.grounded
    steps = []
    for operation in labels_to_plan(labels, vocabulary):
        action, params = operation["name"], operation["params"]
        entity, destination = semantics._arguments(params)
        source = (semantics._subject(state, action, entity, destination)
                  if _FAMILIES.get(action) in {"place", "transfer", "release"} else entity)
        successor, _ = semantics.grounded_transition(state, action, params)
        steps.append(OperationTrace(action, source, destination, state.grounded, successor))
        state = semantics.advance(state, action, params, grounded=successor)
    return initial, steps


def candidate_features(labels, policy, vocabulary, *, goal=PublicGoal()):
    """Encode just a complete accepting sequence and its declared symbolic effects.

    The caller binds the DFA from permitted public observations. For transfer,
    use the public executor policy with its anonymous canonical pointers.
    """
    if not labels or labels[-1] != vocabulary.eos_label or vocabulary.eos_label in labels[:-1]:
        raise ValueError("shared reranker requires exactly one terminal EOS")
    monitor = find_predicate_monitor(policy)
    if monitor is None or not policy.accepts(labels):
        raise ValueError("shared reranker requires a completed plan accepted by a bound DomiKnowS DFA")

    from test_regr.EmbodiedAgentInterface.utils.constraint_compiler import EAIAdapter
    from test_regr.VLABenchAgentInterface.constraint_compiler import VLABenchPredicateAdapter

    if isinstance(monitor.adapter, EAIAdapter):
        initial, operations = _eai_trace(labels, vocabulary, monitor.adapter)
    elif isinstance(monitor.adapter, VLABenchPredicateAdapter):
        initial, operations = _graph_trace(labels, vocabulary, monitor.adapter)
    else:
        raise ValueError("shared reranker needs an EAI or Joint graph-label adapter")
    return anonymous_features(initial, operations, goal)
