"""Qualitative physics of the VLABench QA tasks as DomiKnowS logical constraints.

``knowledge.tables`` holds the property orders and the instruction phrases; this module turns them into
rules over one press (variables ``q`` the request, ``s`` the pressed option, ``o`` every candidate option):

* ``physics__press_option_membership``: the pressed entity is one of the candidate buttons;
* ``physics__<task>__<phrase>__max|min``: no known candidate is strictly above (below) the pressed one;
* ``physics__density_qa__float__below``: the pressed object's material is below water;
* ``physics__<task>__<phrase>__n<k>``: with k options, the pressed one has the requested rank.

Observations only describe the episode: the task, which phrases occur in the instruction, the option count,
each option's material, shape or measured value, and which option is pressed. An option whose attribute is
unknown takes part in no comparison, so it never causes a violation; ``PhysicsMonitor`` relays the verdicts
into the ``affordance`` requirement family through ``domiknows.generation.lc_relay``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from domiknows.generation.grounded_facts import Check, RequirementObservation
from domiknows.generation.dfa.predicate_monitor import PredicateEvent
from domiknows.generation.lc_relay import RelayedKnowledge

from .tables import (
    PHYSICS_MATERIALS, PHYSICS_MAX_OPTIONS, PHYSICS_PHRASES, PHYSICS_PROPERTIES, PHYSICS_REVERSED, PHYSICS_SHAPES,
    PHYSICS_TASK_PHRASES,
)

DOMAIN = "physics_selection"
MEMBERSHIP = "physics__press_option_membership"
_ROLES = count()


def criterion_chain(task):
    """``(label, condition_phrases, excluded_phrases, criterion)`` in the order the instruction is read.

    ``second_fastest`` is consumed only with at least two options; ``medium`` is always consumed (an even
    option count then has no middle, so no criterion applies).
    """
    reversed_scale = task in PHYSICS_REVERSED
    chain = [(phrase, criterion) for phrase, criterion in PHYSICS_TASK_PHRASES.get(task, ())]
    chain += [("second_fastest", "rank:second"), ("medium", "rank:middle"),
              ("small", "max" if reversed_scale else "min"), ("large", "min" if reversed_scale else "max")]
    return tuple(chain)


def rank_for(criterion, task, options):
    """Zero-based ascending rank requested by a rank criterion, or None when it does not apply."""
    if criterion == "rank:second":
        return (1 if task in PHYSICS_REVERSED else options - 2) if options >= 2 else None
    if criterion == "rank:middle":
        return options // 2 if options % 2 else None
    return None


def declare_physics(rules):
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import andL, atMostL, notL, orL

    role = next(_ROLES)
    request = Concept(name="physics_request")
    option = Concept(name="physics_option")
    greater = Concept(name="physics_measured_greater")
    greater.has_a(**{f"physics_left_{role}": option, f"physics_right_{role}": option})
    tasks = {task: request(name=f"task__{task}") for task in PHYSICS_PROPERTIES}
    phrases = {phrase: request(name=f"phrase__{phrase}") for phrase in PHYSICS_PHRASES}
    sizes = {k: request(name=f"option_count__{k}") for k in range(1, PHYSICS_MAX_OPTIONS + 1)}
    selected, candidate, measured = (option(name=name) for name in ("selected", "candidate", "measured"))
    materials = {m: option(name=f"material__{m}") for m in PHYSICS_MATERIALS}
    shapes = {sh: option(name=f"shape__{sh}") for sh in PHYSICS_SHAPES}

    def attribute(kind, values, x):
        table = materials if kind == "material" else shapes
        atoms = [table[value](x) for value in values]
        return atoms[0] if len(atoms) == 1 else orL(*atoms)

    def above(task, a, b):
        """``a`` is strictly above ``b`` on the task's scale."""
        kind, tiers = PHYSICS_PROPERTIES[task]
        if kind == "measurement":
            return greater(a, b)
        pairs = [andL(attribute(kind, high, a), attribute(kind, low, b))
                 for i, high in enumerate(tiers) for low in tiers[:i]]
        return pairs[0] if len(pairs) == 1 else orL(*pairs)

    def known(task, x):
        kind, tiers = PHYSICS_PROPERTIES[task]
        if kind == "measurement":
            return measured(x)
        return attribute(kind, [value for tier in tiers for value in tier], x)

    def condition(entry, chain_before, size=None):
        """The phrase of ``entry`` occurs and no phrase read before it was consumed."""
        terms = [tasks[entry[2]]("q")]
        for label, _ in chain_before:
            if label == "second_fastest":
                terms.append(notL(andL(phrases[label]("q"), notL(sizes[1]("q")))))
            elif label in ("small", "large"):
                other = "large" if label == "small" else "small"
                terms.append(notL(andL(phrases[label]("q"), notL(phrases[other]("q")))))
            else:
                terms.append(notL(phrases[label]("q")))
        label = entry[0]
        terms.append(phrases[label]("q"))
        if label in ("small", "large"):
            terms.append(notL(phrases["large" if label == "small" else "small"]("q")))
        if size is not None:
            terms.append(sizes[size]("q"))
        return terms

    rules.rule(MEMBERSHIP, "affordance", lambda: selected("s"), lambda: candidate("s"))
    for task in PHYSICS_PROPERTIES:
        chain = criterion_chain(task)
        for index, (label, criterion) in enumerate(chain):
            entry, before = (label, criterion, task), chain[:index]
            base = f"physics__{task}__{label}"
            if criterion in ("max", "min"):
                def violation(t=task, c=criterion):
                    return above(t, "o", "s") if c == "max" else above(t, "s", "o")
                rules.rule(f"{base}__{criterion}", "affordance",
                           lambda e=entry, b=before: andL(*condition(e, b), selected("s"), candidate("o")),
                           lambda v=violation: notL(v()),
                           lambda t=task, v=violation: andL(known(t, "s"), known(t, "o"), notL(v())))
            elif criterion.startswith("below:"):
                kind, tiers = PHYSICS_PROPERTIES[task]
                limit = next(i for i, tier in enumerate(tiers) if criterion[6:] in tier)
                lower = [value for tier in tiers[:limit] for value in tier]
                rules.rule(f"{base}__below", "affordance",
                           lambda e=entry, b=before: andL(*condition(e, b), selected("s")),
                           lambda t=task, k=kind, lo=tuple(lower): orL(notL(known(t, "s")), attribute(k, lo, "s")),
                           lambda t=task, k=kind, lo=tuple(lower): andL(known(t, "s"), attribute(k, lo, "s")))
            else:
                for size in range(1, PHYSICS_MAX_OPTIONS + 1):
                    rank = rank_for(criterion, task, size)
                    if rank is None:
                        continue
                    def counts(t=task, r=rank, n=size):
                        return [atMostL(andL(candidate("o"), above(t, "s", "o")), limit=r),
                                atMostL(andL(candidate("o"), above(t, "o", "s")), limit=n - r - 1)]
                    rules.rule(f"{base}__n{size}", "affordance",
                               lambda e=entry, b=before, n=size: andL(*condition(e, b, n), selected("s")),
                               lambda c=counts: andL(*c()),
                               lambda t=task, c=counts: andL(known(t, "s"),
                                                            atMostL(andL(candidate("o"), notL(known(t, "o"))), limit=0),
                                                            *c()))
    concepts = [(tasks[t], f"task__{t}") for t in tasks] + [(phrases[p], f"phrase__{p}") for p in phrases]
    concepts += [(sizes[k], f"option_count__{k}") for k in sizes]
    concepts += [(selected, "selected"), (candidate, "candidate"), (measured, "measured"), (greater, "measured_greater")]
    concepts += [(materials[m], f"material__{m}") for m in materials] + [(shapes[sh], f"shape__{sh}") for sh in shapes]
    return [(concept, name, DOMAIN, "immediate") for concept, name in concepts]


def build_physics_knowledge(active=True):
    return RelayedKnowledge("vlabench_physics_knowledge", declare_physics, active=active)


def instruction_phrases(instruction):
    text = " ".join(str(instruction).lower().split())
    return tuple(phrase for phrase, words in PHYSICS_PHRASES.items() if any(word in text for word in words))


def physics_event(task, block, chosen):
    """One complete valuation of a press of ``chosen`` (no key the rules can read is left out)."""
    candidates = tuple(block["candidates"])
    attributes = block.get("attributes") or {}
    measurements = block.get("measurements") or {}
    entities = tuple(dict.fromkeys((*candidates, chosen)))
    phrases = set(block.get("phrases", ()))
    values = {(f"task__{t}", "request"): t == task for t in PHYSICS_PROPERTIES}
    values.update({(f"phrase__{p}", "request"): p in phrases for p in PHYSICS_PHRASES})
    values.update({(f"option_count__{k}", "request"): k == len(candidates) for k in range(1, PHYSICS_MAX_OPTIONS + 1)})
    for entity in entities:
        attribute = attributes.get(entity) or {}
        values[("selected", entity)] = entity == chosen
        values[("candidate", entity)] = entity in candidates
        values[("measured", entity)] = measurements.get(entity) is not None
        values.update({(f"material__{m}", entity): attribute.get("material") == m for m in PHYSICS_MATERIALS})
        values.update({(f"shape__{sh}", entity): attribute.get("shape") == sh for sh in PHYSICS_SHAPES})
    for a in entities:
        for b in entities:
            left, right = measurements.get(a), measurements.get(b)
            values[("measured_greater", a, b)] = left is not None and right is not None and left > right
    return PredicateEvent(DOMAIN, values, variable_domains={"q": ("request",), "s": (chosen,), "o": candidates})


@dataclass
class PhysicsMonitor:
    """Requirement monitor for one QA episode: relays the physics rules on every press."""

    knowledge: RelayedKnowledge
    task: str
    block: dict
    _cache: dict = field(default_factory=dict, repr=False)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def action(self):
        return self.block.get("action", "press")

    def operation(self, state, op):
        if op.action != self.action:
            return ()
        chosen = getattr(op, self.block.get("argument", "subject"))
        if chosen not in self._cache:
            self._cache[chosen] = self._checks(chosen)
        return self._cache[chosen]

    def completion(self, state):
        return ()

    def _checks(self, chosen):
        if chosen is None:
            requirements = (RequirementObservation(MEMBERSHIP, None),)
            missing = ("selection subject",)
            return (Check("affordance", True, None, missing, requirements), Check("selection", True, None, missing, requirements))
        # Only the membership rule and this task's rules can fire; the others are guarded by another task.
        names = {name for name in self.knowledge.enforce_constraints
                 if name == MEMBERSHIP or name.startswith(f"physics__{self.task}__")}
        relayed = [observation for _, observation in
                   self.knowledge.relay(physics_event(self.task, self.block, chosen), names)]
        criteria = [o for o in relayed if o.name != MEMBERSHIP]
        truths = [o.truth for o in relayed]
        satisfied = False if False in truths else (True if criteria and all(truths) else None)
        candidates = tuple(self.block["candidates"])
        missing = [] if criteria else ["selection criterion"]
        kind = PHYSICS_PROPERTIES.get(self.task, ("material",))[0]

        def unknown(entity):
            if kind == "measurement":
                return (self.block.get("measurements") or {}).get(entity) is None
            return ((self.block.get("attributes") or {}).get(entity) or {}).get(kind) is None
        needed = (chosen,) if any(o.name.endswith("__below") for o in criteria) else (chosen, *candidates)
        missing += [f"{self.block.get('property', 'value')} of {entity}" for entity in dict.fromkeys(needed) if unknown(entity)]
        requirements, missing = tuple(relayed), tuple(missing)
        return (Check("affordance", True, satisfied, missing, requirements), Check("selection", True, satisfied, missing, requirements))


_STANDALONE = []


def physics_requirement(raw, task, knowledge=None):
    """The press requirement of ``raw`` planning facts, judged by the compiled physics LCs.

    ``check(Operation("press", button))`` returns the ``selection`` diagnostic; it is meant for adapters and
    tests that examine one episode outside a decoder.
    """
    if knowledge is None:
        if not _STANDALONE:
            _STANDALONE.append(build_physics_knowledge())
        knowledge = _STANDALONE[0]
    monitor = PhysicsMonitor(knowledge, task, raw["physics_selection"])

    class _Requirement:
        @staticmethod
        def check(op):
            return monitor.operation(None, op)[-1]
    return _Requirement()
