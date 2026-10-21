"""Task tier of the VLABench world graph: concepts, logical constraints and the observations that feed them.

For every task in a :class:`TaskKnowledge` the builder declares a ``task__<name>`` concept on the operation
concept and these constraints (``op`` is one operation, ``position>=k`` says its index is at least ``k``):

* ``task__T__skill_sequence``: ``task_T(op) and at_skill_choice(op) -> or_i(position==i(op) and skill_i(op))``
* ``task__T__completion``:     ``task_T(op) and is_final(op) -> position==n-1(op)``
* ``task__T__obligations``:    ``task_T(op) and is_final(op) -> position>=min-1(op) and history_s(op) ...`` (complex tasks)

When two plan interfaces declare different sequences for one task (``compact`` and ``full``) the constraint also
carries an ``interface__<name>`` atom. The truth of ``task__T`` is an observation: the episode's task name at the
``task``/``metadata`` levels, a predicted task at ``learned``, nothing below. The constraints are ordinary LCs;
the generic compiler and the verifier enforce them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

MAX_THRESHOLD = 16  # position>=k exists for k = 1..16; plans have at most 16 operations
SKILL_CHOICE_PHASE = "task_skill"
COMPLETION_PHASE = "completion"
INTERFACES = ("compact", "full")


@dataclass(frozen=True)
class TaskConcepts:
    tasks: Mapping[str, Any]
    interfaces: Mapping[str, Any]
    positions: Mapping[int, Any]
    at_skill_choice: Any
    is_final: Any
    history: Mapping[str, Any]
    explicit_sequence: Any = None
    constraints: tuple = ()

    def bindings(self):
        """``(concept, predicate name, phase)`` for every concept, for the generic compiler."""
        rows = [(concept, f"task:{name}", "immediate") for name, concept in self.tasks.items()]
        rows += [(concept, f"interface:{name}", "immediate") for name, concept in self.interfaces.items()]
        rows += [(concept, f"position>={k}", "immediate") for k, concept in self.positions.items()]
        rows.append((self.explicit_sequence, "explicit_sequence", "immediate"))
        rows.append((self.at_skill_choice, "at_skill_choice", SKILL_CHOICE_PHASE))
        rows.append((self.is_final, "is_final", COMPLETION_PHASE))
        rows += [(concept, f"history:{skill}", COMPLETION_PHASE) for skill, concept in self.history.items()]
        return rows


def declare_task_concepts(operation, skills: Mapping[str, Any], knowledge, *, active: bool,
                          constraints: bool = True) -> TaskConcepts:
    """Declare the task concepts and constraints on the operation concept (inside the graph context)."""
    from domiknows.graph.logicalConstrain import andL, ifL, notL, orL

    tasks = {name: operation(name=f"task__{name}") for name in knowledge.tasks()}
    interfaces = {name: operation(name=f"interface__{name}") for name in INTERFACES}
    positions = {k: operation(name=f"position__ge_{k}") for k in range(1, MAX_THRESHOLD + 1)}
    explicit_sequence = operation(name="fact__explicit_sequence")
    at_skill_choice = operation(name="operation__at_skill_choice")
    is_final = operation(name="operation__is_final")
    history = {skill: operation(name=f"history__has_skill__{skill}") for skill in skills}
    declared = []

    def index_is(i):
        if not 0 <= i < MAX_THRESHOLD:
            raise ValueError(f"operation index {i} is outside the declared positions")
        return notL(positions[1]("op")) if i == 0 else andL(positions[i]("op"), notL(positions[i + 1]("op")))

    def add(antecedent, consequent, name):
        declared.append(ifL(antecedent, consequent, name=name, active=active))

    for task in (knowledge.tasks() if constraints else ()):
        for interface, expected in knowledge.variants(task):
            carry = (interfaces[interface]("op"),) if interface else ()
            suffix = f"__{interface}" if interface else ""
            branches = [andL(index_is(i), skills[skill]("op")) for i, skill in enumerate(expected)]
            # An explicitly supplied sequence (an observation) takes precedence over the declared default.
            add(andL(tasks[task]("op"), *carry, notL(explicit_sequence("op")), at_skill_choice("op")),
                branches[0] if len(branches) == 1 else orL(*branches), f"task__{task}__skill_sequence{suffix}")
            add(andL(tasks[task]("op"), *carry, notL(explicit_sequence("op")), is_final("op")), index_is(len(expected) - 1),
                f"task__{task}__completion{suffix}")
        schema = knowledge.schema(task)
        if schema is not None:
            terms = ([positions[schema.min_operations - 1]("op")] if schema.min_operations > 1 else [])
            terms += [history[skill]("op") for skill in schema.required_skills]
            add(andL(tasks[task]("op"), is_final("op")), terms[0] if len(terms) == 1 else andL(*terms),
                f"task__{task}__obligations")
    return TaskConcepts(tasks, interfaces, positions, at_skill_choice, is_final, history, explicit_sequence,
                        tuple(declared))


def knowledge_values(concepts: TaskConcepts, task: str | None, interface: str, index: int, *,
                     explicit: bool = False, choice: bool = False, final: bool = False, history=()) -> dict[str, bool]:
    """Observation values for the task concepts at operation ``index`` (counting any completed prefix)."""
    values = {f"task:{name}": task == name for name in concepts.tasks}
    values.update({f"interface:{name}": interface == name for name in concepts.interfaces})
    values.update({f"position>={k}": index >= k for k in concepts.positions})
    values["explicit_sequence"] = bool(explicit)
    values["at_skill_choice"] = bool(choice)
    values["is_final"] = bool(final)
    seen = set(history)
    values.update({f"history:{skill}": skill in seen for skill in concepts.history})
    return values
