"""Relay compiled logical constraints into grounded requirement observations.

A knowledge module declares each rule once, as ``antecedent -> consequent`` with an optional strict
consequent in which unknown observations count as failure. The relay builds three graphs from the same
declaration: ``enforce`` (the rule itself; its ``active`` flag is the knowledge switch), ``applies``
(``not antecedent``, which fails exactly when the rule fires) and ``decided`` (``antecedent -> strict``).
For every active rule that fires on an event the relay yields one ``RequirementObservation``: False when
the rule fails, True when its strict form holds, otherwise None (unknown, so not enforced). Callers feed
these to a declared family (``grounded__<family>``), so every decoder and verifier that already consumes
grounded requirements enforces the knowledge without change.
"""
from __future__ import annotations

from itertools import count

from .dfa.predicate_monitor import PredicateBinding, PredicateConstraintProgram, compile_predicate_constraints
from .grounded_facts import RequirementObservation

MODES = ("enforce", "applies", "decided")
_BUILDS = count()


class _Declarations:
    def __init__(self, mode):
        self.mode = mode
        self.families = {}
        self.constraints = {}

    def rule(self, name, family, antecedent, consequent, strict=None, *, active=True):
        """Declare one rule. ``antecedent``/``consequent``/``strict`` are callables building LC expressions."""
        from domiknows.graph.logicalConstrain import ifL, notL
        if name in self.families:
            raise ValueError(f"duplicate relayed rule {name!r}")
        self.families[name] = family
        if self.mode == "enforce":
            lc = ifL(antecedent(), consequent(), name=name, active=active)
        elif self.mode == "applies":
            lc = notL(antecedent(), name=name)
        else:
            lc = ifL(antecedent(), (strict or consequent)(), name=name)
        self.constraints[name] = lc
        return lc


class RelayedKnowledge:
    """Three compiled views of one declared rule set, evaluated per event.

    ``declare(declarations)`` runs inside each graph: it creates the concepts, calls
    ``declarations.rule(...)`` and returns ``[(concept, predicate_name, domain, phase), ...]``.
    """

    def __init__(self, name, declare, *, active=True, max_groundings=100000):
        from domiknows.graph import Graph
        build = next(_BUILDS)
        self.name = name
        self.max_groundings = max_groundings
        self.graphs, self.declarations, self.bindings = {}, {}, {}
        for mode in MODES:
            declarations = _Declarations(mode)
            with Graph(f"{name}__{mode}__{build}") as graph:
                rows = declare(declarations)
            self.graphs[mode] = graph
            self.declarations[mode] = declarations
            self.bindings[mode] = [PredicateBinding(concept, predicate, domain, phase)
                                   for concept, predicate, domain, phase in rows]
        self.families = self.declarations["enforce"].families
        for lc in self.enforce_constraints.values():
            lc.active = active and lc.active
        self._programs = {}

    @property
    def enforce_constraints(self):
        return self.declarations["enforce"].constraints

    def set_active(self, active, names=None):
        for name, lc in self.enforce_constraints.items():
            if names is None or name in names:
                lc.active = active

    def _compiled(self):
        key = tuple(lc.active for lc in self.enforce_constraints.values())
        if key not in self._programs:
            per_mode = {}
            for mode in MODES:
                program = compile_predicate_constraints(self.graphs[mode], self.bindings[mode],
                                                        max_groundings=self.max_groundings)
                per_mode[mode] = {rule.name: PredicateConstraintProgram((rule,), max_groundings=self.max_groundings)
                                  for rule in program.rules}
            self._programs = {key: per_mode}  # keep only the current activation
        return self._programs[key]

    def relay(self, event, names=None):
        """``(family, RequirementObservation)`` for every active rule that fires on ``event``."""
        programs = self._compiled()
        for name, enforce in programs["enforce"].items():
            if names is not None and name not in names:
                continue
            if programs["applies"][name].check(event):
                continue  # the antecedent holds for no grounding: the rule does not fire
            if not enforce.check(event):
                truth = False
            elif programs["decided"][name].check(event):
                truth = True
            else:
                truth = None
            yield self.families[name], RequirementObservation(name, truth)


class LazyKnowledge:
    """Declares a ``RelayedKnowledge`` on first use; its activation is kept until then."""

    def __init__(self, factory, active=True):
        self._factory, self._active, self._knowledge = factory, active, None

    @property
    def active(self):
        return self._active

    def set_active(self, active):
        self._active = active
        if self._knowledge is not None:
            self._knowledge.set_active(active)

    def get(self):
        if self._knowledge is None:
            self._knowledge = self._factory(self._active)
        return self._knowledge
