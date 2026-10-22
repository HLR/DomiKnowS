"""Immutable, open-world planning facts and checks, independent of any benchmark.

Adapters supply action meanings. This module never guesses observations from
reference trajectories, entity names, or a desired answer.
"""
from dataclasses import dataclass, field, replace
from functools import cached_property
from math import isfinite
from typing import Mapping

FAMILIES = ("destination", "holding", "capacity", "affordance", "compatibility",
            "milestone_order", "goal_completion", "preservation")


@dataclass(frozen=True)
class FactState:
    positive: frozenset = frozenset()
    negative: frozenset = frozenset()

    def __post_init__(self):
        if self.positive & self.negative:
            raise ValueError("contradictory positive and negative planning facts")

    def truth(self, fact):
        fact = tuple(fact)
        return True if fact in self.positive else False if fact in self.negative else None

    def update(self, positive=(), negative=()):
        yes, no = frozenset(map(tuple, positive)), frozenset(map(tuple, negative))
        if yes & no:
            raise ValueError("contradictory declared action effects")
        return FactState((self.positive - no) | yes, (self.negative - yes) | no)


@dataclass(frozen=True)
class GroundedState:
    facts: FactState = FactState()
    holdings: tuple = ()  # Presence of (hand, None) means known empty.
    milestones: frozenset = frozenset()


@dataclass(frozen=True)
class Operation:
    action: str
    subject: str | None = None
    destination: str | None = None
    hand: str = "right"
    grasp: bool = False
    requires_held: bool = False
    release: bool = False
    affordances: tuple = ()  # (entity, property) requirements
    writes: tuple = ()
    relation: str | None = None
    idempotent: bool = False
    preconditions: tuple = ()
    container_required: bool = False


@dataclass(frozen=True)
class RequirementObservation:
    """One observed requirement, before any logical conjunction is evaluated.

    ``truth=None`` is an unobserved fact, never evidence of satisfaction.
    Names describe supplied facts/comparisons for diagnostics, not policy.
    """

    name: str
    truth: bool | None
    inputs: tuple = ()


@dataclass(frozen=True)
class Check:
    family: str
    applicable: bool = False
    satisfied: bool | None = None
    missing: tuple = ()
    requirements: tuple[RequirementObservation, ...] = ()

    @property
    def status(self):
        if not self.applicable:
            return "not_applicable"
        if self.satisfied is False:
            return "violated"
        if self.missing:
            return "partial" if self.satisfied is True else "unavailable"
        return "checked" if self.satisfied is not None else "unavailable"


def observations(checks):
    values = {}
    for check in checks:
        if check.family not in FAMILIES:
            # Detailed diagnostic checks (for example numeric selection or
            # missing effects) are folded into declared LC families upstream.
            continue
        # Unknowns are disclosed separately; they are not observations of truth.
        values[check.family + "__applicable"] = check.applicable and check.satisfied is not None
        values[check.family + "__satisfied"] = check.satisfied is not False
    return values


def report(checks):
    rows = [{"family": c.family, "status": c.status, "missing": list(c.missing)} for c in checks]
    return {"checks": rows, "fully_grounded": not any(c.applicable and (c.satisfied is None or c.missing) for c in checks),
            "violations": sum(c.applicable and c.satisfied is False for c in checks)}


def combined(family, values, *, applicable=True, missing=(), labels=(), inputs=()):
    """Retain individual facts; aggregate status is only a diagnostic summary.

    Known false requirements violate a family. Unknown requirements remain
    disclosed as missing, even if other requirements are known true. The graph
    can enforce each retained fact directly instead of trusting this summary.
    """
    if not applicable:
        return Check(family)
    values = tuple(values)
    unknown = tuple(missing) + tuple("observation" for value in values if value is None)
    satisfied = False if False in values else True if True in values else None
    labels = tuple(labels)
    requirements = tuple(RequirementObservation(
        labels[index] if index < len(labels) else f"{family} requirement {index + 1}", value, tuple(inputs),
    ) for index, value in enumerate(values))
    return Check(family, applicable, satisfied, unknown, requirements)


@dataclass(frozen=True)
class SelectionRequirement:
    """Action-target eligibility from an explicit candidate universe and scores.

    Scores are exposed observations in one declared property/unit. No entity
    order, task name, reference answer, or geometric model supplies a score.
    """

    action: str
    candidates: tuple = ()
    scores: tuple = ()
    criterion: str | None = None
    property: str = "value"
    argument: str = "subject"
    threshold: float | None = None
    rank: int | None = None  # Zero-based ascending order statistic; ties share eligibility.

    @classmethod
    def parse(cls, raw, resolve):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("action"), str) or not raw["action"]:
            raise ValueError("selection requirements need an action name")
        criterion = raw.get("criterion")
        if criterion not in (None, "min", "max", "below", "above", "rank"):
            raise ValueError("selection criterion must be min, max, below, above, or rank")
        argument = raw.get("argument", "subject")
        if argument not in ("subject", "destination"):
            raise ValueError("selection argument must be subject or destination")
        scores = raw.get("scores", {})
        if not isinstance(scores, Mapping):
            raise ValueError("selection scores must map candidates to observed numbers")
        candidates = raw.get("candidates")
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("selection candidates must be a nonempty entity sequence")
        candidates = tuple(resolve(candidate) for candidate in candidates)
        if len(set(candidates)) != len(candidates):
            raise ValueError("duplicate selection candidate")
        rank = raw.get("rank")
        if criterion == "rank":
            if type(rank) is not int or not 0 <= rank < len(candidates):
                raise ValueError("selection rank must be a zero-based integer within the candidate universe")
        elif rank is not None:
            raise ValueError("selection rank requires the rank criterion")
        def number(value):
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("selection scores and thresholds must be finite numbers or unknown")
            try:
                converted = float(value)
            except OverflowError as error:
                raise ValueError("selection scores and thresholds must be finite numbers or unknown") from error
            if not isfinite(converted):
                raise ValueError("selection scores and thresholds must be finite numbers or unknown")
            return converted
        observed = tuple((resolve(candidate), number(value)) for candidate, value in scores.items())
        if len({candidate for candidate, _ in observed}) != len(observed):
            raise ValueError("duplicate selection score observation")
        if set(candidate for candidate, _ in observed) - set(candidates):
            raise ValueError("selection score refers to an undeclared candidate")
        return cls(raw["action"], candidates, observed, criterion, str(raw.get("property", "value")),
                   argument, number(raw.get("threshold")), rank)

    def check(self, op):
        if op.action != self.action:
            return Check("selection")
        chosen = getattr(op, self.argument)
        scores = dict(self.scores)
        value = scores.get(chosen)
        missing = []
        if not self.candidates:
            missing.append("selection candidate universe")
        if self.criterion is None:
            missing.append("selection criterion")
        if chosen is None:
            missing.append("selection " + self.argument)
        if value is None:
            missing.append(f"{self.property} of selected candidate")
        if self.candidates and chosen is not None and chosen not in self.candidates:
            return Check("selection", True, False, tuple(missing),
                         (RequirementObservation("selected entity belongs to the supplied candidate universe", False),))
        satisfied = None
        if self.criterion in ("min", "max"):
            missing.extend(f"{self.property} of {candidate}" for candidate in self.candidates
                           if scores.get(candidate) is None)
            if value is not None:
                better = any(other is not None and (other < value if self.criterion == "min" else other > value)
                             for candidate in self.candidates for other in (scores.get(candidate),))
                satisfied = False if better else True if not missing else None
        elif self.criterion == "rank":
            missing.extend(f"{self.property} of {candidate}" for candidate in self.candidates
                           if scores.get(candidate) is None)
            if value is not None and self.rank is not None:
                lower = sum(scores.get(c) is not None and scores[c] < value for c in self.candidates)
                greater = sum(scores.get(c) is not None and scores[c] > value for c in self.candidates)
                # Even unknown scores cannot rescue a candidate with too many
                # known values on either side of the requested order statistic.
                impossible = lower > self.rank or greater > len(self.candidates) - self.rank - 1
                satisfied = False if impossible else True if not missing else None
        elif self.criterion in ("below", "above"):
            if self.threshold is None:
                missing.append("selection threshold")
            elif value is not None:
                satisfied = value < self.threshold if self.criterion == "below" else value > self.threshold
        # Retain primitive numeric comparisons instead of collapsing the
        # selection policy into a single pass/fail observation. The graph's
        # universal requirement LC enforces their conjunction, including known
        # counterexamples in an otherwise incompletely measured scene.
        requirements = [RequirementObservation(
            "selected entity belongs to the supplied candidate universe",
            chosen in self.candidates if chosen is not None and self.candidates else None,
        )]
        if self.criterion in ("min", "max"):
            for candidate in self.candidates:
                other = scores.get(candidate)
                truth = None if value is None or other is None else (value <= other if self.criterion == "min" else value >= other)
                requirements.append(RequirementObservation(
                    f"selected {self.property} is {'no greater' if self.criterion == 'min' else 'no smaller'} than {candidate!r}", truth))
        elif self.criterion == "rank":
            known = value is not None and self.rank is not None
            lower = sum(scores.get(c) is not None and scores[c] < value for c in self.candidates) if known else 0
            greater = sum(scores.get(c) is not None and scores[c] > value for c in self.candidates) if known else 0
            requirements.extend((
                RequirementObservation("known lower-valued candidates do not exceed the requested rank",
                                       lower <= self.rank if known else None),
                RequirementObservation("known higher-valued candidates leave room for the requested rank",
                                       greater <= len(self.candidates) - self.rank - 1 if known else None),
            ))
        else:
            requirements.append(RequirementObservation(f"selected {self.property} satisfies the supplied threshold", satisfied))
        return Check("selection", True, satisfied, tuple(dict.fromkeys(missing)), tuple(requirements))


@dataclass(frozen=True)
class PlanningFacts:
    initial: GroundedState = GroundedState()
    properties: tuple = ()
    capacities: tuple = ()
    destinations: tuple = ()
    default_destination: str | None = None
    compatibility: tuple = ()
    goals: FactState = FactState()
    preserve: FactState = FactState()
    protected: frozenset = frozenset()
    milestones: tuple = ()
    effects: tuple = ()
    completed_skill_prefix: tuple = ()
    supplied: bool = False
    selections: tuple = ()
    reusable_entities: frozenset = frozenset()
    action_destinations: tuple = ()
    skill_cycle: tuple = ()
    required_skill_sequence: tuple = ()

    def destination_for(self, action, subject):
        return self._action_destination_map.get((action, subject), self._destination_map.get(subject, self.default_destination))

    @cached_property
    def _action_destination_map(self):
        return {(a, s): d for a, s, d in self.action_destinations}

    @cached_property
    def _destination_map(self):
        return dict(self.destinations)

    @classmethod
    def parse(cls, raw=None, *, entities=(), resolve=None, default_destination=None, constants=()):
        if isinstance(raw, cls):
            return raw
        if raw is not None and not isinstance(raw, Mapping):
            raise ValueError("planning_facts must be an object")
        raw = raw or {}
        known = set(entities)
        role_constants = set(constants) | set((raw.get("initial_state") or {}).get("held", ())) | set(raw.get("capacities", ()))
        def entity(value):
            if resolve is not None:
                return resolve(value)
            value = str(value)
            if value not in known:
                raise ValueError(f"planning entity {value!r} is absent from the current scene")
            return value
        def fact(value, *, template=False):
            if not isinstance(value, (tuple, list)) or not value or not isinstance(value[0], str):
                raise ValueError("a planning fact must be [predicate, entity, ...]")
            if template and any(str(arg).startswith("$") and str(arg) not in {"$subject", "$destination", "$hand"} for arg in value[1:]):
                raise ValueError("unknown effect placeholder")
            return (value[0], *(str(arg) if str(arg) in role_constants or (template and str(arg).startswith("$"))
                                 else entity(arg) for arg in value[1:]))
        def facts(value):
            value = value or {}
            return FactState(frozenset(fact(x) for x in value.get("positive", ())),
                             frozenset(fact(x) for x in value.get("negative", ())))
        initial = raw.get("initial_state") or {}
        holdings = tuple(sorted((str(hand), None if obj is None else entity(obj))
                                for hand, obj in initial.get("held", {}).items()))
        capacities = tuple(sorted((str(hand), count) for hand, count in raw.get("capacities", {}).items()))
        if any(isinstance(count, bool) or not isinstance(count, int) or count not in (0, 1) for _, count in capacities):
            raise ValueError("hand capacities must be 0 or 1")
        properties = []
        for obj, values in raw.get("entity_properties", {}).items():
            for name, value in values.items():
                if not isinstance(value, bool):
                    raise ValueError("entity capabilities must be Boolean")
                properties.append((entity(obj), str(name), value))
        compatibility = []
        for item in raw.get("compatibility", ()):
            if not isinstance(item.get("allowed"), bool):
                raise ValueError("compatibility.allowed must be Boolean")
            compatibility.append((entity(item["subject"]), entity(item["destination"]), item["allowed"]))
        if len({(s,d) for s,d,_ in compatibility}) != len(compatibility):
            raise ValueError("duplicate compatibility observation")
        milestones = []
        for item in raw.get("milestones", ()):
            milestones.append((str(item["name"]), str(item["action"]),
                               entity(item["subject"]) if item.get("subject") is not None else None,
                               entity(item["destination"]) if item.get("destination") is not None else None,
                               tuple(item.get("requires", ())), bool(item.get("required", True))))
        names = {x[0] for x in milestones}
        if len(names) != len(milestones) or any(set(x[4]) - names for x in milestones):
            raise ValueError("milestones need unique names and declared predecessors")
        # Reject cycles instead of turning them into an inexplicable dead end.
        pending = {x[0]: set(x[4]) for x in milestones}
        seen = set()
        while pending:
            ready = {name for name, parents in pending.items() if parents <= seen}
            if not ready:
                raise ValueError("cyclic milestone requirements")
            seen |= ready
            pending = {name: parents for name, parents in pending.items() if name not in ready}
        completed = frozenset(initial.get("milestones", ()))
        if completed - names:
            raise ValueError("unknown completed milestone")
        if any(name in completed and not set(parents) <= completed for name, _, _, _, parents, _ in milestones):
            raise ValueError("completed milestone is missing its declared predecessor progress")
        effects = []
        for action, effect in raw.get("effects", {}).items():
            effects.append((str(action), tuple(fact(x, template=True) for x in effect.get("positive", ())),
                            tuple(fact(x, template=True) for x in effect.get("negative", ())),
                            tuple(map(str, effect.get("release_hands", ())))))
        if any(set(additions) & set(removals) for _, additions, removals, _ in effects):
            raise ValueError("contradictory declared action effects")
        action_destinations = tuple((str(item["action"]), entity(item["subject"]), entity(item["destination"])) for item in raw.get("action_destinations", ()))
        if len({(a, s) for a, s, _ in action_destinations}) != len(action_destinations):
            raise ValueError("duplicate action-specific destination requirement")
        return cls(GroundedState(facts(initial), holdings, completed), tuple(sorted(properties)), capacities,
                   tuple(sorted((entity(s), entity(d)) for s,d in raw.get("destinations", {}).items())),
                   entity(default_destination) if default_destination is not None else None,
                   tuple(compatibility), facts(raw.get("goals")), facts(raw.get("preserve")),
                   frozenset(entity(x) for x in raw.get("protected_entities", ())), tuple(milestones), tuple(effects),
                   tuple(raw.get("completed_skill_prefix", ())), bool(raw) or default_destination is not None,
                   selections=tuple(SelectionRequirement.parse(item, entity) for item in raw.get("selections", ())),
                   reusable_entities=frozenset(entity(x) for x in raw.get("reusable_entities", ())),
                   action_destinations=action_destinations,
                   skill_cycle=tuple(raw.get("skill_cycle", ())), required_skill_sequence=tuple(raw.get("required_skill_sequence", ())))

    def completion(self, state):
        values = [state.facts.truth(f) for f in self.goals.positive]
        values += [None if state.facts.truth(f) is None else not state.facts.truth(f) for f in self.goals.negative]
        required = {m[0] for m in self.milestones if m[5]}
        values += [name in state.milestones for name in required]
        labels = [f"goal fact {fact!r} holds" for fact in self.goals.positive]
        labels += [f"goal fact {fact!r} is false" for fact in self.goals.negative]
        labels += [f"required milestone {name!r} is completed" for name in required]
        return combined("goal_completion", values, applicable=bool(values), labels=labels)

    # The tuples above are immutable; their lookup tables are built once per instance, not once per operation.
    @cached_property
    def _capacity_map(self):
        return dict(self.capacities)

    @cached_property
    def _property_map(self):
        return {(s, k): v for s, k, v in self.properties}

    @cached_property
    def _compatibility_map(self):
        return {(s, d): v for s, d, v in self.compatibility}

    @cached_property
    def _goal_and_preserved_facts(self):
        return self.goals.positive | self.goals.negative | self.preserve.positive | self.preserve.negative

    def evaluate(self, state, op, *, positive=(), negative=(), clear_relations=(), holding_fact=None):
        """Source checks followed by symbolic effects and invariant checks."""
        holdings = dict(state.holdings)
        held = holdings.get(op.hand)
        hand_known = op.hand in holdings
        subject = op.subject if op.subject is not None else held
        expected = self.destination_for(op.action, subject)
        checks = [combined("destination", [op.destination == expected] if expected is not None else [None],
                           applicable=op.destination is not None, labels=("destination equals the requested destination",))]
        checks.append(combined("holding", [True if op.idempotent else
                                           (held is not None and (op.subject is None or held == op.subject)) if hand_known else None],
                               applicable=op.requires_held, labels=("hand holds the required subject, or the action is idempotent",),
                               inputs=(("idempotent", op.idempotent), ("hand_known", hand_known),
                                       ("held_exists", held is not None), ("subject_unspecified", op.subject is None),
                                       ("held_matches_subject", held == op.subject))))
        capacity = self._capacity_map.get(op.hand)
        checks.append(combined("capacity", [False if capacity == 0 or (capacity is not None and hand_known and held is not None and held != subject) else
                                           capacity > 0 if capacity is not None and hand_known else None], applicable=op.grasp,
                               labels=("hand has capacity for this grasp",),
                               inputs=(("capacity_known", capacity is not None), ("capacity_zero", capacity == 0),
                                       ("capacity_positive", capacity is not None and capacity > 0),
                                       ("hand_known", hand_known), ("held_exists", held is not None),
                                       ("held_matches_subject", held == subject))))
        properties = self._property_map
        selections = tuple(selection.check(replace(op, subject=subject)) for selection in self.selections
                           if selection.action == op.action)
        # The compatibility summary preserves the established diagnostic API.
        # Detailed graph enforcement receives each underlying comparison below.
        capability_values = [properties.get(pair) for pair in op.affordances]
        precondition_values = [state.facts.truth(f) for f in op.preconditions]
        labels = ([f"{entity!r} has capability {capability!r}" for entity, capability in op.affordances]
                  + [f"precondition {fact!r} holds" for fact in op.preconditions])
        source_values = capability_values + precondition_values
        summary = combined(
            "affordance", source_values + [selection.satisfied for selection in selections],
            applicable=bool(op.affordances or op.preconditions or selections),
            missing=tuple(reason for selection in selections for reason in selection.missing),
        )
        requirements = tuple(RequirementObservation(label, value) for label, value in zip(labels, source_values))
        requirements += tuple(term for selection in selections for term in selection.requirements)
        checks.append(replace(summary, requirements=requirements))
        checks.extend(selections)
        pairs = self._compatibility_map
        checks.append(combined("compatibility", [pairs.get((subject, op.destination))] + ([properties.get((op.destination,"container"))] if op.container_required else []),
                               applicable=op.destination is not None,
                               labels=("subject and destination are compatible", "destination has container capability")))
        matched = [m for m in self.milestones if m[1] == op.action and (m[2] is None or m[2] == subject)
                   and (m[3] is None or m[3] == op.destination) and m[0] not in state.milestones]
        enabled = [m for m in matched if set(m[4]) <= state.milestones]
        # A repeated action may describe successive milestones. One source-state
        # event advances enabled milestones only, never their future successors.
        checks.append(combined("milestone_order", [bool(enabled)], applicable=bool(matched),
                               labels=("a matching milestone has its required predecessor progress",)))
        yes, no = set(map(tuple, positive)), set(map(tuple, negative))
        # Functional spatial updates invalidate known/required alternative locations.
        def candidate_facts():
            return state.facts.positive | self._goal_and_preserved_facts
        candidates = candidate_facts() if clear_relations or holding_fact is not None else frozenset()
        for relation, obj in clear_relations:
            no.update(f for f in candidates if len(f) >= 2 and f[:2] == (relation, obj) and f not in yes)
        if op.grasp and subject is not None:
            holdings[op.hand] = subject
        elif op.release and hand_known:
            holdings[op.hand] = None
        bindings = {"$subject": subject, "$destination": op.destination, "$hand": op.hand}
        missing_effects = []
        for action, additions, removals, released in self.effects:
            if action != op.action:
                continue
            for source, target in ((additions, yes), (removals, no)):
                for f in source:
                    instantiated = tuple(bindings.get(x, x) for x in f)
                    if None in instantiated:
                        missing_effects.append("effect argument")
                    else:
                        (no if target is yes else yes).discard(instantiated)
                        target.add(instantiated)
            for hand in released:
                holdings[hand] = None
        if holding_fact is not None:
            for hand, obj in holdings.items():
                old = dict(state.holdings).get(hand)
                if old is not None and old != obj:
                    no.add(holding_fact(hand, old))
                if obj is not None:
                    yes.add(holding_fact(hand, obj))
                for fact in candidates:
                    if len(fact) > 1 and fact == holding_fact(hand, fact[-1]) and fact[-1] != obj:
                        no.add(fact)
            no.difference_update(yes)
        next_facts = state.facts.update(yes, no)
        done = state.milestones | {m[0] for m in enabled}
        successor = GroundedState(next_facts, tuple(sorted(holdings.items())), frozenset(done))
        changed_subjects = {f[1] for f in yes if len(f) > 1 and state.facts.truth(f) is not True}
        changed_subjects.update(f[1] for f in no if len(f) > 1 and state.facts.truth(f) is not False)
        protected = [False for entity in set(op.writes) | changed_subjects if entity in self.protected]
        protected += [next_facts.truth(f) for f in self.preserve.positive]
        protected += [None if next_facts.truth(f) is None else not next_facts.truth(f) for f in self.preserve.negative]
        if self.protected and not protected:
            protected = [True]
        if self.protected and op.requires_held and subject is None:
            missing_effects.append("unknown manipulated entity")
        checks.append(combined("preservation", protected, applicable=bool(self.protected or self.preserve.positive or self.preserve.negative),
                               missing=missing_effects,
                               labels=([f"protected entity {entity!r} is not modified" for entity in set(op.writes) | changed_subjects
                                        if entity in self.protected]
                                       + [f"preserved fact {fact!r} still holds" for fact in self.preserve.positive]
                                       + [f"preserved fact {fact!r} remains false" for fact in self.preserve.negative])))
        if missing_effects:
            checks.append(Check("effects", True, None, tuple(missing_effects)))
        return successor, tuple(checks)
