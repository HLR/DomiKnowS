"""Articulated task contracts and explicit joint observations.

Symbolic effects predict successful skill execution. Only fresh measurements
from the simulator are used to seed a subsequent replanning episode.
"""
from dataclasses import dataclass, field, replace
from math import isfinite
import numpy as np

from domiknows.generation.grounded_facts import Check, FactState
from domiknows.generation.dfa.predicate_monitor import PredicateEvent
from domiknows.generation.lc_relay import RelayedKnowledge

TASKS = {"cool_drink", "set_study_table", "manage_study_table"}
KINDS = ("drink_inside", "fridge_closed", "book_outside", "joint_target")


@dataclass(frozen=True)
class ArticulationTarget:
    entity: str
    joint: str
    lower: float
    upper: float

    @property
    def fact(self):
        return (f"joint_in_range:{self.joint}:{self.lower:g}:{self.upper:g}", self.entity)


def parse_targets(raw, resolve):
    targets = []
    for item in raw.get("articulation_targets", ()):
        lower, upper = item["lower"], item["upper"]
        if (type(lower) not in (int, float) or type(upper) not in (int, float)
                or not isfinite(lower) or not isfinite(upper) or lower > upper):
            raise ValueError("articulation intervals must be finite and ordered")
        joint = item["joint"]
        if not isinstance(joint, str) or not joint.strip():
            raise ValueError("articulation target needs a named joint")
        targets.append(ArticulationTarget(resolve(item["entity"]), joint, float(lower), float(upper)))
    if len({(t.entity, t.joint) for t in targets}) != len(targets):
        raise ValueError("duplicate articulation target")
    return tuple(targets)


def declare_articulation(rules):
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import andL, notL, orL
    requirement = Concept(name="articulation_requirement")
    kinds = {k: requirement(name=k) for k in KINDS}
    known, holds = (requirement(name=n) for n in ("known", "holds"))
    for kind in KINDS:
        rules.rule("articulation__" + kind, "goal_completion", lambda k=kind: kinds[k]("r"),
                   lambda: orL(notL(known("r")), holds("r")),
                   lambda: andL(known("r"), holds("r")))
    return [(c, n, "articulation", "completion") for n, c in (*kinds.items(), ("known", known), ("holds", holds))]


def build_articulation_knowledge(active=True):
    return RelayedKnowledge("vlabench_articulation", declare_articulation, active=active)


@dataclass(eq=False)
class ArticulationMonitor:
    knowledge: object
    targets: tuple
    requirements: tuple  # (kind, fact, desired truth); None fact denotes missing bindings
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def tracked_locations(self):
        return tuple(fact for _, fact, _ in self.requirements if fact and fact[0] == "inside")

    def initial(self, state, raw, resolve):
        observations = {}
        for item in raw.get("joint_observations", ()):
            key = resolve(item["entity"]), item["joint"]
            value = item.get("value")
            if value is not None and (type(value) not in (int, float) or not isfinite(value)):
                raise ValueError("joint observations must be finite numbers or null")
            if key in observations:
                raise ValueError("duplicate joint observation")
            observations[key] = value
        yes, no = set(state.facts.positive), set(state.facts.negative)
        for target in self.targets:
            value = observations.get((target.entity, target.joint))
            if value is not None:
                yes.discard(target.fact)
                no.discard(target.fact)
                (yes if target.lower <= value <= target.upper else no).add(target.fact)
        return replace(state, facts=FactState(frozenset(yes), frozenset(no)))

    def effects(self, skill, entity):
        return tuple(t.fact for t in self.targets if skill == "open_laptop" and t.entity == entity)

    def operation(self, state, op):
        return ()

    def completion(self, state):
        if state.facts not in self._cache:
            rows = tuple(range(len(self.requirements)))
            values = {}
            for i, (kind, fact, desired) in enumerate(self.requirements):
                truth = state.facts.truth(fact) if fact else None
                values.update({(k, i): kind == k for k in KINDS})
                values[("known", i)] = truth is not None
                values[("holds", i)] = truth is desired
            event = PredicateEvent("articulation", values, "completion", variable_domains={"r": rows})
            relayed = tuple(o for _, o in self.knowledge.relay(event))
            truths = [o.truth for o in relayed]
            result = False if False in truths else None if None in truths else True
            self._cache[state.facts] = (Check("goal_completion", bool(rows), result, (), relayed),)
        return self._cache[state.facts]


def articulation_monitor(world, task, raw, resolve, context, entities):
    targets = parse_targets(raw, resolve)
    if not targets and task not in TASKS:
        return None
    block = raw.get("articulation_task") or {}
    get = context.get if hasattr(context, "get") else lambda k, d=None: getattr(context, k, d)

    def binding(key, fallback=None):
        value = block.get(key, fallback)
        return resolve(value) if value is not None else None

    subject = binding("subject", get("task_target_entity"))
    requirements = []
    if task == "cool_drink":
        fridge = binding("container", get("task_target_container"))
        requirements.extend((("drink_inside", ("inside", subject, fridge) if subject and fridge else None, True),
                             ("fridge_closed", ("closed", fridge) if fridge else None, True)))
    elif task in ("set_study_table", "manage_study_table"):
        source = binding("source_container")
        laptop = binding("laptop")
        if laptop is None:
            candidates = [name for name in entities if "laptop" in name.lower()]
            laptop = candidates[0] if len(candidates) == 1 else None
        requirements.append(("book_outside", ("inside", subject, source) if subject and source else None, False))
        matching = [t for t in targets if t.entity == laptop]
        if not matching:
            requirements.append(("joint_target", None, True))
        elif any((t.lower, t.upper) != (1.2, 1.5) for t in matching):
            raise ValueError("study-table laptop contract requires interval [1.2, 1.5]")
    requirements.extend(("joint_target", t.fact, True) for t in targets)
    return ArticulationMonitor(world.articulation_knowledge.get(), targets, tuple(requirements))


def refreshed_articulation_context(env, context, entities):
    """Replace joint/door evidence with actual state before each replanning call."""
    from copy import deepcopy
    context = deepcopy(context or {})
    raw = context.get("planning_facts")
    task = context.get("task") or context.get("task_name")
    if task not in TASKS and not (isinstance(raw, dict) and raw.get("articulation_targets")):
        return context
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("live articulation refresh requires mapping planning_facts")
    context["planning_facts"] = raw
    table = dict(enumerate(entities))
    resolve = lambda x: table[x] if type(x) is int else str(x)
    targets = parse_targets(raw, resolve)
    task_entities = getattr(getattr(env, "task", None), "entities", {})
    physics = getattr(env, "physics", None)
    measured = []
    initial = raw.setdefault("initial_state", {})
    # Predicted contract facts are never recycled into fresh observations.
    for polarity in ("positive", "negative"):
        initial[polarity] = [f for f in initial.get(polarity, ())
                             if not str(f[0]).startswith("joint_in_range:") and f[0] not in ("open", "closed")]
    for target in targets:
        value = None
        obj = task_entities.get(target.entity)
        matches = [j for j in getattr(obj, "joints", ()) if getattr(j, "name", None) == target.joint]
        if len(matches) == 1 and physics is not None:
            try:
                data = np.asarray(physics.bind(matches[0]).qpos).reshape(-1)
                if data.size == 1 and isfinite(float(data[0])):
                    value = float(data[0])
            except (AttributeError, KeyError, TypeError, ValueError):
                pass  # No usable measurement; do not recycle a prediction.
        measured.append({"entity": target.entity, "joint": target.joint, "value": value})
    raw["joint_observations"] = measured
    for name, obj in task_entities.items():
        checks = {p: getattr(obj, "is_" + p, None) for p in ("open", "closed")}
        if not any(callable(c) for c in checks.values()):
            continue
        for predicate, check in checks.items():
            if callable(check) and physics is not None:
                try:
                    observed = check(physics)
                    if not isinstance(observed, (bool, np.bool_)):
                        continue
                    truth = bool(observed)
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
                initial["positive" if truth else "negative"].append([predicate, name])
    return context
