"""Observation grounding and symbolic state updates for the VLABench graph.

Evaluate an operation against the state *before* it executes, then advance
the immutable state. Partial decoders can use the same predicates once the
relevant arguments are available; missing arguments are handled by schema
validation, not interpreted as causal violations. ``primitive_observations``
exposes scene facts, equality comparisons and state membership to world_graph.
The older composite predicates are retained for public/custom-rule compatibility;
the default graph combines the primitive observations in logical constraints.
"""

from dataclasses import dataclass, replace
from domiknows.generation.grounded import GroundedState, PlanningFacts, Operation
from typing import Any, Mapping, Sequence
try:
    from .visibility import visible_entity_ids, reference_is_visible
except ImportError:
    from visibility import visible_entity_ids, reference_is_visible


# Every benchmark subtask pattern begins with a pick or a press; its destinations are tracked from there.
SUBTASK_STARTS = frozenset({"pick", "press"})
DESTINATION_SKILLS = frozenset({"place", "insert", "pour", "push"})

ENCLOSED_CONTAINER_KEYWORDS = (
    "cooler", "fridge", "refrigerator", "drawer", "microwave", "blender", "oven", "cabinet",
)


def is_enclosed_container(name: Any) -> bool:
    return name is not None and any(word in str(name).strip().lower() for word in ENCLOSED_CONTAINER_KEYWORDS)


def is_container_opened(container_name: Any, opened_entities) -> bool:
    """Match a receptacle or its named door/lid, never every object of its type."""
    name = str(container_name).strip().lower()
    aliases = {name, *(name + suffix for suffix in ("_door", " door", "_lid", " lid"))}
    return any(str(opened).strip().lower() in aliases for opened in opened_entities)


def is_static_fixture(name: Any) -> bool:
    lower = str(name).strip().lower()
    return any(word in lower for word in ("table", "counter", "room_", "lab_", "wall", "floor"))


def entity_mapping(entity_table: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    items = entity_table.items() if isinstance(entity_table, Mapping) else enumerate(entity_table)
    return {int(index): str(name) for index, name in items}


def resolve_pointer(value: Any, entities: Mapping[int, str]) -> tuple[str, bool]:
    """Resolve an explicit scene pointer; an empty scene has no valid pointers."""
    if isinstance(value, bool) or value is None:
        return str(value), False
    if isinstance(value, int) or str(value).isdigit():
        index = int(value)
        return entities.get(index, str(value)), index in entities
    name = str(value)
    return name, name in entities.values()


def operation_schema_errors(skill, params, skill_arguments, entity_table=None, *, index=0):
    """Shared schema and grounding checks; None means the scene is unspecified."""
    if skill not in skill_arguments:
        return (f"operation {index} uses unknown skill {skill!r}",)
    errors = []
    required = set(skill_arguments[skill])
    entities = None if entity_table is None else entity_mapping(entity_table)
    for key in sorted(required):
        if key not in params:
            errors.append(f"operation {index} ({skill}) requires {key}")
        else:
            value = params[key]
            valid = type(value) is int or (isinstance(value, str) and bool(value.strip()))
            if valid and entities is not None:
                valid = resolve_pointer(value, entities)[1]
            if not valid:
                errors.append(f"operation {index} ({skill}) has invalid {key}={value!r}")
    try:
        from .world_graph import OPTIONAL_SKILL_ARGUMENTS
    except ImportError:
        from world_graph import OPTIONAL_SKILL_ARGUMENTS
    optional = set(OPTIONAL_SKILL_ARGUMENTS.get(skill, ()))
    for key in optional & params.keys():
        value = params[key]
        if not (type(value) is int or (isinstance(value, str) and bool(value.strip()))) or (entities is not None and not resolve_pointer(value, entities)[1]):
            errors.append(f"operation {index} ({skill}) has invalid {key}={value!r}")
    for key in sorted({"target_entity_name", "target_container_name"} - required - optional):
        if key in params:
            errors.append(f"operation {index} ({skill}) forbids {key}")
    for key in params.keys() - required - optional - {"target_entity_name", "target_container_name"}:
        errors.append(f"operation {index} ({skill}) has undeclared argument {key!r}")
    return tuple(errors)


@dataclass(frozen=True)
class PlanSemanticState:
    grounded: GroundedState = GroundedState()
    held_entity: str | None = None
    opened_entities: frozenset[str] = frozenset()
    destination: str | None = None
    previous_skill: str | None = None
    previous_entity: str | None = None
    completed_operations: int = 0
    completed_skills: frozenset[str] = frozenset()
    visited_picks: frozenset[str] = frozenset()
    subtask_destinations: tuple[str, ...] = ()  # destinations since the current subtask's pick or press
    former_destinations: frozenset[str] = frozenset()  # every destination so far


@dataclass(frozen=True)
class OperationPredicates:
    valid_skill_type: bool
    valid_operation: bool
    valid_pointer: bool
    has_target_entity: bool
    has_target_container: bool
    targets_button: bool
    is_self_placement: bool
    targets_static_fixture: bool
    repeated_pick_entity: bool
    inconsistent_container: bool
    unopened_container_placement: bool


class PlanSemantics:
    """Scene-bound predicate meanings reusable by verification and decoding."""

    def __init__(self, entity_table, skill_arguments, planning_facts=None, *, visible_entities=None):
        self.entities = entity_mapping(entity_table)
        self.skill_arguments = skill_arguments
        self.planning = planning_facts or PlanningFacts()
        self.visible_entities = visible_entity_ids(visible_entities, self.entities)

    def initial_state(self):
        initial = self.planning.initial
        from domiknows.generation.grounded import FactState
        yes, no = set(initial.facts.positive), set(initial.facts.negative)
        for fact in tuple(yes):
            if len(fact) == 2 and fact[0] in ("open", "closed"):
                yes.update((fact[0], name) for name in self.entities.values()
                           if is_container_opened(name, (fact[1],)))
        for fact in tuple(yes):
            if fact[0] == "open": no.add(("closed", *fact[1:]))
            if fact[0] == "closed": no.add(("open", *fact[1:]))
        for hand, obj in initial.holdings:
            if obj is not None:
                yes.add(("held", hand, obj))
            no.update(("held", hand, entity) for entity in self.entities.values() if entity != obj)
        initial = replace(initial, facts=FactState(frozenset(yes), frozenset(no)))
        opened = frozenset(f[1] for f in initial.facts.positive if len(f) == 2 and f[0] == "open")
        prefix = self.planning.completed_skill_prefix
        return PlanSemanticState(grounded=initial, held_entity=dict(initial.holdings).get("right"),
                                 opened_entities=opened, completed_operations=len(prefix), completed_skills=frozenset(prefix))

    def grounded_transition(self, state, skill, params):
        entity, container = self._arguments(params)
        subject = entity if entity is not None else state.held_entity
        places = skill in ("place", "insert", "pour")
        capabilities = []
        if skill == "pick":
            capabilities.append((entity, "graspable"))
        if skill == "open_door":
            capabilities.append((entity, "openable"))
        if skill == "pour":
            capabilities.append((subject, "pourable"))
        op = Operation(skill, subject, container, "right", grasp=skill == "pick",
                       requires_held=skill in ("place", "pour", "insert", "lift") or (skill == "push" and container is not None), release=skill in ("place", "insert"),
                       container_required=skill in ("insert", "pour"), affordances=tuple(capabilities), writes=tuple(x for x in (subject, container) if x is not None),
                       preconditions=(("open", container),) if places and is_enclosed_container(container) else ())
        yes, no = [], []
        clear = []
        if skill == "pick" and subject is not None:
            clear = [(r, subject) for r in ("at", "inside", "ontop", "inserted")]
        if places and subject is not None and container is not None:
            relation = "poured_into" if skill == "pour" else "inside" if skill == "insert" else "at"
            yes.append((relation, subject, container))
            if skill != "pour":
                clear = [(r, subject) for r in ("at", "inside", "ontop", "inserted")]
        if skill == "open_door" and entity is not None:
            yes.extend(("open", name) for name in self.entities.values()
                       if is_container_opened(name, (entity,)))
            no.extend(("closed", fact[1]) for fact in yes if fact[0] == "open")
        if skill == "push" and subject is not None and container is not None:
            yes.append(("pushed_to", subject, container))
        return self.planning.evaluate(state.grounded, op, positive=yes, negative=no, clear_relations=clear,
                                      holding_fact=lambda hand, obj: ("held", hand, obj))


    def _arguments(self, params):
        entity = params.get("target_entity_name")
        container = params.get("target_container_name")
        return (
            resolve_pointer(entity, self.entities)[0] if entity is not None else None,
            resolve_pointer(container, self.entities)[0] if container is not None else None,
        )

    def evaluate_operation(self, state: PlanSemanticState, skill: str, params: Mapping[str, Any]) -> OperationPredicates:
        entity, container = self._arguments(params)
        placement = skill in ("place", "pour", "insert")
        subject = entity if entity is not None else state.held_entity
        return OperationPredicates(
            valid_skill_type=skill in self.skill_arguments,
            valid_operation=not operation_schema_errors(skill, params, self.skill_arguments, self.entities),
            valid_pointer=all(resolve_pointer(params[key], self.entities)[1]
                              for key in ("target_entity_name", "target_container_name") if key in params),
            has_target_entity="target_entity_name" in params,
            has_target_container="target_container_name" in params,
            targets_button=any(value is not None and "button" in value.lower() for value in (entity, container)),
            is_self_placement=container is not None and subject == container and (placement or skill == "push" or entity is not None),
            targets_static_fixture=skill == "pick" and entity is not None and is_static_fixture(entity),
            repeated_pick_entity=skill == "pick" and entity is not None and entity in state.visited_picks and entity not in self.planning.reusable_entities,
            inconsistent_container=container is not None and
                (expected := self.planning.destination_for(skill, subject)) is not None and container != expected,
            unopened_container_placement=placement and container is not None and is_enclosed_container(container)
                and not is_container_opened(container, state.opened_entities)
                and (not self.planning.supplied or state.grounded.facts.truth(("open", container)) is False
                     or state.grounded.facts.truth(("closed", container)) is True),
        )

    def primitive_observations(self, state, skill, params):
        """Ground atomic comparisons; policy combinations live in world_graph.

        Membership/identity and measured attributes are observations. For
        example, picking a static fixture is forbidden by an LC combining
        ``pick`` with ``entity_is_static_fixture``, not by this adapter.
        """
        entity, container = self._arguments(params)
        subject = entity if entity is not None else state.held_entity
        expected = self.planning.destination_for(skill, subject)
        try:
            from .world_domain import OPTIONAL_SKILL_ARGUMENTS
        except ImportError:
            from world_domain import OPTIONAL_SKILL_ARGUMENTS
        allowed = set(self.skill_arguments.get(skill, ())) | set(OPTIONAL_SKILL_ARGUMENTS.get(skill, ()))

        def usable(key):
            value = params.get(key)
            return value is not None and value != "" and not isinstance(value, bool) and resolve_pointer(value, self.entities)[1]

        return {
            "has_undeclared_argument": bool(params.keys() - allowed),
            "visibility_supplied": self.visible_entities is not None,
            "entity_seen_initially": reference_is_visible(params.get("target_entity_name"), self.entities, self.visible_entities),
            "container_seen_initially": reference_is_visible(params.get("target_container_name"), self.entities, self.visible_entities),
            "entity_value_usable": usable("target_entity_name"),
            "container_value_usable": usable("target_container_name"),
            "entity_resolves": resolve_pointer(params.get("target_entity_name"), self.entities)[1],
            "container_resolves": resolve_pointer(params.get("target_container_name"), self.entities)[1],
            "entity_is_button": entity is not None and "button" in entity.lower(),
            "container_is_button": container is not None and "button" in container.lower(),
            "subject_equals_destination": container is not None and subject == container,
            "entity_is_static_fixture": entity is not None and is_static_fixture(entity),
            "next_is_pick": skill == "pick",
            "next_is_placement": skill in ("place", "pour", "insert"),
            "next_has_entity": entity is not None,
            "next_has_destination": container is not None,
            "target_previously_picked": entity in state.visited_picks,
            "target_reusable": entity in self.planning.reusable_entities,
            "expected_destination_known": expected is not None,
            "destination_matches_expected": container == expected,
            "destination_is_enclosed": is_enclosed_container(container),
            "destination_was_opened": is_container_opened(container, state.opened_entities),
            "planning_facts_supplied": self.planning.supplied,
            "open_observed_false": state.grounded.facts.truth(("open", container)) is False,
            "closed_observed_true": state.grounded.facts.truth(("closed", container)) is True,
            "destination_used_in_subtask": container is not None and container in state.subtask_destinations,
            "target_was_destination": entity is not None and entity in state.former_destinations,
        }

    def advance(self, state: PlanSemanticState, skill: str, params: Mapping[str, Any]) -> PlanSemanticState:
        grounded, _checks = self.grounded_transition(state, skill, params)
        entity, container = self._arguments(params)
        held = dict(grounded.holdings).get("right", state.held_entity)
        if skill == "pick":
            held = entity
        elif skill in ("place", "insert"):
            held = None
        # Pouring retains the held vessel. Opening a door does not release it.
        opened = state.opened_entities
        if skill == "open_door" and entity is not None:
            opened = opened | {entity}
        # Explicit release effects override the legacy held-object mirror.
        held = dict(grounded.holdings).get("right", held)
        return replace(state, grounded=grounded, held_entity=held, opened_entities=frozenset(opened),
                       destination=state.destination if state.destination is not None else container,
                       previous_skill=skill, previous_entity=entity,
                       completed_operations=state.completed_operations + 1,
                       completed_skills=state.completed_skills | {skill},
                       visited_picks=state.visited_picks | ({entity} if skill == "pick" and entity is not None else set()),
                       subtask_destinations=(() if skill in SUBTASK_STARTS else state.subtask_destinations)
                       + ((container,) if skill in DESTINATION_SKILLS and container is not None else ()),
                       former_destinations=state.former_destinations
                       | ({container} if skill in DESTINATION_SKILLS and container is not None else set()))

    @staticmethod
    def premature_composite_eos(state: PlanSemanticState, *, min_operations=None, required_skills=None) -> bool:
        return ((min_operations is not None and state.completed_operations < min_operations)
                or (required_skills is not None and not set(required_skills) <= state.completed_skills))
