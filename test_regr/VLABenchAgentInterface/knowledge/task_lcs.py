"""Task knowledge as named concepts and ordinary DomiKnowS logical constraints.

Read the declarations in three groups: requested facts, declared task sequences,
and final obligations. Observations at the bottom of the file bind task identity,
operation index and history; they never choose which logical rule to enforce.

For a task T with sequence [pick, place], the graph says:
  T AND choosing_skill AND position=0 -> pick
  T AND choosing_skill AND position=1 -> place
  T AND final_operation -> position=1
Complex tasks add minimum length and required skills in the completed history.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping

MAX_THRESHOLD = 16
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
    requests: Mapping[str, Any] = field(default_factory=dict)

    def bindings(self):
        """Explicit concept identities and evaluation phases for the compiler."""
        rows = [(concept, f"task:{name}", "immediate") for name, concept in self.tasks.items()]
        rows += [(concept, f"interface:{name}", "immediate") for name, concept in self.interfaces.items()]
        rows += [(concept, f"position>={k}", "immediate") for k, concept in self.positions.items()]
        rows += [(self.explicit_sequence, "explicit_sequence", "immediate"),
                 (self.at_skill_choice, "at_skill_choice", SKILL_CHOICE_PHASE),
                 (self.is_final, "is_final", COMPLETION_PHASE)]
        rows += [(concept, f"history:{skill}", COMPLETION_PHASE) for skill, concept in self.history.items()]
        for key, concept in getattr(self, "requests", {}).items():
            phase = (SKILL_CHOICE_PHASE if key.startswith(("sequence_skill:", "cycle_skill:")) else
                     COMPLETION_PHASE if key.endswith("complete") or key.startswith("required_skill") else
                     "task_target" if key == "target_matches" else "immediate")
            rows.append((concept, "request:" + key, phase))
        return rows


def _any(calls):
    from domiknows.graph.logicalConstrain import orL
    calls = tuple(calls)
    return calls[0] if len(calls) == 1 else orL(*calls)


def _position_is(concepts, index):
    """Position i means >=i and not >=i+1; positions are zero-based."""
    from domiknows.graph.logicalConstrain import andL, notL
    if not 0 <= index < MAX_THRESHOLD:
        raise ValueError(f"operation index {index} is outside the declared positions")
    if index == 0:
        return notL(concepts.positions[1]("op"))
    return andL(concepts.positions[index]("op"), notL(concepts.positions[index + 1]("op")))


def _declare_requested_rules(concepts, operation, skills, add):
    """Request data selects skills and targets; LCs combine those observations."""
    from domiknows.graph.logicalConstrain import andL
    request = concepts.requests
    # Both an explicit sequence and a repeating cycle must permit the choice.
    add(andL(operation("op"), concepts.at_skill_choice("op")), _any(
        andL(skill("op"), request["sequence_skill:" + name]("op"), request["cycle_skill:" + name]("op"))
        for name, skill in skills.items()
    ), "task_skill_sequence")
    add(andL(operation("op"), request["initial_target_known"]("op"), request["first_operation"]("op")),
        request["target_matches"]("op"), "task_initial_target")
    add(andL(operation("op"), concepts.is_final("op")),
        andL(request["sequence_complete"]("op"), request["cycle_complete"]("op")), "task_sequence_completion")
    # Explicit caller obligations have the same logical structure as task
    # defaults: a minimum observed length and every requested history skill.
    from domiknows.graph.logicalConstrain import ifL
    obligations = [request["minimum_complete"]("op"), request["required_skills_supported"]("op")]
    obligations.extend(ifL(request["required_skill:" + skill]("op"), concepts.history[skill]("op")) for skill in skills)
    add(andL(operation("op"), concepts.is_final("op")), andL(*obligations), "composite_subgoal_completion")


def _declare_task_sequence(concepts, skills, task, interface, expected, add):
    """An observed task and interface imply one skill at each declared index."""
    from domiknows.graph.logicalConstrain import andL, notL
    if not expected:
        raise ValueError(f"declared task {task!r} needs a nonempty skill sequence")
    interface_guard = (concepts.interfaces[interface]("op"),) if interface else ()
    suffix = f"__{interface}" if interface else ""

    def task_guard(phase):
        # An explicit request overrides the task's default sequence, while its
        # separate requested-fact rules above still apply.
        return andL(concepts.tasks[task]("op"), *interface_guard,
                    notL(concepts.explicit_sequence("op")), phase("op"))

    add(task_guard(concepts.at_skill_choice), _any(
        andL(_position_is(concepts, index), skills[skill]("op")) for index, skill in enumerate(expected)
    ), f"task__{task}__skill_sequence{suffix}")
    add(task_guard(concepts.is_final), _position_is(concepts, len(expected) - 1),
        f"task__{task}__completion{suffix}")


def _declare_obligations(concepts, task, schema, add):
    """A final complex-task operation requires enough operations and every skill."""
    from domiknows.graph.logicalConstrain import andL
    terms = []
    if schema.min_operations > 1:
        threshold = schema.min_operations - 1
        if threshold not in concepts.positions:
            raise ValueError(f"minimum length of {task!r} exceeds declared positions")
        terms.append(concepts.positions[threshold]("op"))
    terms.extend(concepts.history[skill]("op") for skill in schema.required_skills)
    if not terms:
        return  # An empty obligation adds no restriction.
    conclusion = terms[0] if len(terms) == 1 else andL(*terms)
    add(andL(concepts.tasks[task]("op"), concepts.is_final("op")), conclusion, f"task__{task}__obligations")


def declare_task_concepts(operation, skills: Mapping[str, Any], knowledge, *, active: bool,
                          constraints: bool = True) -> TaskConcepts:
    """Declare all task concepts; tier activation only changes LC.active."""
    from domiknows.graph.logicalConstrain import ifL
    request_keys = ["initial_target_known", "first_operation", "target_matches", "sequence_complete", "cycle_complete",
                    "minimum_complete", "required_skills_supported"]
    request_keys += [prefix + skill for prefix in ("sequence_skill:", "cycle_skill:", "required_skill:") for skill in skills]
    concepts = TaskConcepts(
        tasks={name: operation(name=f"task__{name}") for name in knowledge.tasks()},
        interfaces={name: operation(name=f"interface__{name}") for name in INTERFACES},
        positions={k: operation(name=f"position__ge_{k}") for k in range(1, MAX_THRESHOLD + 1)},
        explicit_sequence=operation(name="fact__explicit_sequence"),
        at_skill_choice=operation(name="operation__at_skill_choice"),
        is_final=operation(name="operation__is_final"),
        history={skill: operation(name=f"history__has_skill__{skill}") for skill in skills},
        requests={key: operation(name="request__" + key.replace(":", "__")) for key in request_keys},
    )
    declared = []
    def add(antecedent, consequent, name):
        declared.append(ifL(antecedent, consequent, name=name, active=active))
    if constraints:
        _declare_requested_rules(concepts, operation, skills, add)
        for task in knowledge.tasks():
            for interface, expected in knowledge.variants(task):
                _declare_task_sequence(concepts, skills, task, interface, expected, add)
            schema = knowledge.schema(task)
            if schema is not None:
                _declare_obligations(concepts, task, schema, add)
    from dataclasses import replace
    return replace(concepts, constraints=tuple(declared))


# Observation adapters -------------------------------------------------------

def knowledge_values(concepts: TaskConcepts, task: str | None, interface: str, index: int, *,
                     explicit: bool = False, choice: bool = False, final: bool = False, history=(),
                     requirements=None, params=None, entities=None,
                     min_operations=None, required_skills=()) -> dict[str, bool]:
    """Bind task identity, indices, comparisons and history from supplied facts.

    No task table is interpreted here. Declared sequences/obligations are in
    the formulas above. Request arrays provide one observed skill per index.
    """
    values = {f"task:{name}": task == name for name in concepts.tasks}
    values.update({f"interface:{name}": interface == name for name in concepts.interfaces})
    values.update({f"position>={k}": index >= k for k in concepts.positions})
    values.update(explicit_sequence=bool(explicit), at_skill_choice=bool(choice), is_final=bool(final))
    seen = set(history)
    values.update({f"history:{skill}": skill in seen for skill in concepts.history})
    expected = requirements.expected_skills if requirements is not None and requirements.explicit_sequence else None
    cycle = requirements.skill_cycle if requirements is not None else ()
    for skill in concepts.history:
        values["request:sequence_skill:" + skill] = expected is None or (0 <= index < len(expected) and expected[index] == skill)
        values["request:cycle_skill:" + skill] = not cycle or cycle[index % len(cycle)] == skill
        values["request:required_skill:" + skill] = skill in (required_skills or ())
    target = requirements.initial_target if requirements is not None else None
    values.update({"request:sequence_complete": expected is None or index + 1 == len(expected),
                   "request:cycle_complete": not cycle or (index + 1) % len(cycle) == 0,
                   "request:minimum_complete": min_operations is None or index + 1 >= min_operations,
                   "request:required_skills_supported": set(required_skills or ()) <= set(concepts.history),
                   "request:initial_target_known": target is not None,
                   "request:first_operation": index == 0,
                   "request:target_matches": False})
    if target is not None and params is not None:
        try:
            from ..semantics import entity_mapping, resolve_pointer
        except ImportError:
            from semantics import entity_mapping, resolve_pointer
        held = dict(requirements.planning.initial.holdings).get("right")
        if "target_entity_name" not in params and held is not None:
            values["request:target_matches"] = held == target
        else:
            name, valid = resolve_pointer(params.get("target_entity_name"), entity_mapping(entities or ()))
            values["request:target_matches"] = valid and name == target
    return values
