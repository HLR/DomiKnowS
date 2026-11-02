"""The native plan grammar, declared as logical constraints.

``NativePlanDFA`` (``native_planning.py``) is only a JSON lexer: it spells every skill name
of the profile with each argument shape (no argument, an object, a destination) and integer
ids, and reports what it read. These formulas decide what a plan may be:

* per action, once for the whole lexicon: an operation names exactly one skill, each skill has
  its declared argument roles (``skill_arguments__<skill>``), and object ids stay below the
  episode's id bound (``target_within_id_bound``). The admitted actions are the DFA's action
  inventory;
* per operation, against the plan prefix: the plan has room for it (``operation_within_budget``)
  and it continues a declared subtask pattern that can still be finished within the budget
  (``operation_continues_subtask_pattern``);
* per plan, when it closes: it has an operation (``plan_has_an_operation``) and it ends on a
  complete pattern (``plan_completes_subtask_pattern``).

The skill-argument table and the pattern list of a profile generate the per-skill and pattern
formulas at build time. The open pattern prefixes are observations of the plan prefix, as the
causal rules observe earlier ids. Deactivating or editing a formula changes the language after
``GrammarGraph.compiled()`` is taken again (a new DFA does so).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import count as _count

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import andL, exactL, ifL, notL, orL
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding,
    PredicateEvent,
    compile_predicate_constraints,
)

ENTITY_ROLE = "target_entity_name"
CONTAINER_ROLE = "target_container_name"
# Argument shapes the lexer can spell; the per-skill formulas decide which one each skill takes.
ARGUMENT_SHAPES = ((), (ENTITY_ROLE,), (CONTAINER_ROLE,))
LEXICAL_IDS = 100  # the lexer spells ids 0..99 (or up to the bound); the id formula keeps them below the bound
_BUILDS = _count()


def prefix_name(prefix) -> str:
    return "open__subtask_start" if not prefix else "open__after_" + "_".join(prefix)


@dataclass(frozen=True)
class GrammarGraph:
    graph: Graph
    skills: tuple
    patterns: tuple
    action_bindings: tuple
    step_bindings: tuple
    plan_bindings: tuple
    open_prefixes: tuple
    max_remaining: int
    _programs: dict = field(default_factory=dict, compare=False, repr=False)

    def compiled(self):
        """``(action, step, plan)`` programs for the currently active formulas (cached per activation)."""
        key = tuple(bool(getattr(rule, "active", True)) for rule in self.graph.logicalConstrains.values())
        if key not in self._programs:
            bindings = list(self.action_bindings) + list(self.step_bindings) + list(self.plan_bindings)
            program = compile_predicate_constraints(self.graph, bindings)
            self._programs[key] = program
        return self._programs[key]

    # Observations. Each is a complete valuation of its domain; the formulas decide legality.
    def action_event(self, skill, roles, pointer, id_bound):
        values = {"action": True, **{"action_" + name: name == skill for name in self.skills}}
        values.update(has_entity_argument=ENTITY_ROLE in roles, has_container_argument=CONTAINER_ROLE in roles,
                      target_within_id_bound=pointer is not None and 0 <= pointer < id_bound)
        return PredicateEvent("grammar_action", values)

    def step_event(self, skill, prefixes, count, max_operations):
        remaining = max_operations - count - 1
        values = {"operation": True, **{"skill_" + name: name == skill for name in self.skills}}
        values.update({prefix_name(prefix): prefix in prefixes for prefix in self.open_prefixes})
        values.update({f"remaining_at_least_{k}": remaining >= k for k in range(self.max_remaining + 1)})
        return PredicateEvent("grammar_step", values)

    @staticmethod
    def plan_event(prefixes, count):
        return PredicateEvent("grammar_plan", {"plan": True, "subtask_boundary": () in prefixes,
                                               "has_operations": count > 0})

    def admitted_actions(self, id_bound, lexical_ids=None):
        """``(skill, roles, pointer)`` of every lexical action the action formulas admit, in lexicon order."""
        program = self.compiled()
        ids = range(lexical_ids if lexical_ids is not None else max(LEXICAL_IDS, id_bound + 1))
        key = ("actions", id(program), id_bound, len(ids))
        if key in self._programs:
            return self._programs[key]
        result = []
        for skill in self.skills:
            for roles in ARGUMENT_SHAPES:
                for pointer in (ids if roles else (None,)):
                    if program.check(self.action_event(skill, roles, pointer, id_bound)):
                        result.append((skill, roles, pointer))
        self._programs[key] = tuple(result)
        return self._programs[key]


@lru_cache(maxsize=None)
def grammar_graph(skill_arguments: tuple, patterns: tuple) -> GrammarGraph:
    """The grammar graph of one profile; ``skill_arguments`` is a tuple of ``(skill, roles)``."""
    return build_grammar_graph(dict(skill_arguments), patterns)


def build_grammar_graph(skill_arguments, patterns, name=None) -> GrammarGraph:
    patterns = tuple(tuple(pattern) for pattern in patterns)
    skills = tuple(skill_arguments)
    open_prefixes = tuple(sorted({tuple(pattern[:i]) for pattern in patterns for i in range(len(pattern))},
                                 key=lambda prefix: (len(prefix), prefix)))
    max_remaining = max((len(pattern) - 1 for pattern in patterns), default=0)
    with Graph(name or f"vlabench_native_grammar_{next(_BUILDS)}") as graph:
        action = Concept(name="grammar_action")
        action_is = action(name="action")
        action_skill = {skill: action(name="action_" + skill) for skill in skills}
        has_entity = action(name="has_entity_argument")
        has_container = action(name="has_container_argument")
        within_bound = action(name="target_within_id_bound")

        step = Concept(name="grammar_step")
        step_is = step(name="operation")
        step_skill = {skill: step(name="skill_" + skill) for skill in skills}
        open_prefix = {prefix: step(name=prefix_name(prefix)) for prefix in open_prefixes}
        remaining = {k: step(name=f"remaining_at_least_{k}") for k in range(max_remaining + 1)}

        plan = Concept(name="grammar_plan")
        plan_is = plan(name="plan")
        boundary = plan(name="subtask_boundary")
        has_operations = plan(name="has_operations")

        # An operation names exactly one skill.
        ifL(action_is("a"), exactL(*(concept("a") for concept in action_skill.values()), 1),
            name="operation_exactly_one_skill")
        # Each skill takes exactly its declared argument roles.
        for skill, roles in skill_arguments.items():
            shape = [has_entity("a") if ENTITY_ROLE in roles else notL(has_entity("a")),
                     has_container("a") if CONTAINER_ROLE in roles else notL(has_container("a"))]
            ifL(action_skill[skill]("a"), andL(*shape), name=f"skill_arguments__{skill}")
        # Object ids are below the episode's id bound.
        ifL(orL(has_entity("a"), has_container("a")), within_bound("a"), name="target_within_id_bound")

        # The plan has room for one more operation.
        ifL(step_is("s"), remaining[0]("s"), name="operation_within_budget")
        # The operation extends an open prefix of a declared pattern that can still finish in the budget.
        terms = {}
        for pattern in patterns:
            for i, skill in enumerate(pattern):
                key = (pattern[:i], skill, len(pattern) - i - 1)
                if key in terms:
                    continue  # a formula built inside the graph is declared; build each term once
                terms[key] = andL(open_prefix[pattern[:i]]("s"), step_skill[skill]("s"), remaining[key[2]]("s"))
        ifL(step_is("s"), orL(*terms.values()) if len(terms) > 1 else next(iter(terms.values())),
            name="operation_continues_subtask_pattern")

        # A plan has an operation and ends on a complete pattern.
        ifL(plan_is("p"), has_operations("p"), name="plan_has_an_operation")
        ifL(plan_is("p"), boundary("p"), name="plan_completes_subtask_pattern")

    bind = lambda concepts, domain: tuple(PredicateBinding(concept, concept.name, domain) for concept in concepts)
    return GrammarGraph(
        graph, skills, patterns,
        bind((action_is, *action_skill.values(), has_entity, has_container, within_bound), "grammar_action"),
        bind((step_is, *step_skill.values(), *open_prefix.values(), *remaining.values()), "grammar_step"),
        bind((plan_is, boundary, has_operations), "grammar_plan"),
        open_prefixes, max_remaining)
