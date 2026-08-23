"""DomiKnowS schema and deterministic trajectory materialization for EAI.

This graph is deliberately separate from the token-generation graph.  It is a
semantic view of a simulated world trajectory and is only used to score
constraints declared by callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


Fact = tuple[str, ...]
ABSENT_ENTITY = "__absent_action_argument__"
_VERIFIER_GRAPH: Any = None
_WORLD_GRAPH_BUILD_IDS = count()


@dataclass(frozen=True)
class ActionSpec:
    name: str
    min_args: int
    max_args: int
    is_goal_action: bool = False
    requires_task_entity: bool = False


@dataclass(frozen=True)
class StatePredicateSpec:
    name: str
    arity: int
    aliases: tuple[str, ...] = ()
    positive_counterpart: str | None = None


@dataclass(frozen=True)
class ActionPreconditionSpec:
    name: str
    action: str
    kind: str


_ACTION_NAMES = frozenset({
    "clean", "close", "cook", "drink", "drop", "find", "freeze", "grab",
    "left_grasp", "left_place_inside", "left_place_nextto",
    "left_place_nextto_ontop", "left_place_on_top", "left_place_ontop",
    "left_place_under", "left_release", "lie", "lookat", "open", "plugin",
    "plugout", "pointat", "pour", "pull", "push", "put", "putback", "putin",
    "putobjback", "putoff", "puton", "read", "right_grasp",
    "right_place_inside", "right_place_nextto", "right_place_nextto_ontop",
    "right_place_on_top", "right_place_ontop", "right_place_under",
    "right_realease", "right_release", "right_transfer_contents_inside",
    "rinse", "run", "scrub", "sit", "sleep", "slice", "soak", "squeeze",
    "standup", "switch_off", "switch_on", "switchoff", "switchon",
    "toggle_off", "toggle_on", "touch", "turn_off", "turn_on", "turnto",
    "type", "unfreeze", "walk", "wash", "watch", "wipe",
})

_TASK_ENTITY_ACTION_NAMES = frozenset({
    "clean", "close", "cook", "freeze", "left_place_inside",
    "left_place_nextto", "left_place_nextto_ontop",
    "left_place_nextto_on_top", "left_place_on_top", "left_place_ontop",
    "left_place_under", "open", "right_place_inside",
    "right_place_nextto", "right_place_nextto_ontop",
    "right_place_nextto_on_top", "right_place_on_top",
    "right_place_ontop", "right_place_under", "slice", "soak",
    "switch_off", "switch_on", "toggle_off", "toggle_on", "turn_off",
    "turn_on", "unfreeze",
})

ACTION_GOAL_NAMES = frozenset({
    "drink", "grab", "lookat", "pour", "push", "read", "rinse", "scrub",
    "sleep", "switchoff", "touch", "type", "wash", "watch", "wipe",
})

_UNARY_STATE_NAMES = frozenset({
    "clean", "closed", "cooked", "dusty", "frozen", "lying", "not_closed",
    "not_dusty", "not_frozen", "not_on", "not_open", "not_plugged_in",
    "not_stained", "off", "on", "open", "plugged_in", "released", "rinsed",
    "sitting", "sliced", "soaked", "stained", "touch", "washed",
})
_BINARY_STATE_NAMES = frozenset({
    "facing", "holds_lh", "holds_rh", "inside", "near", "nextto",
    "not_inside", "not_ontop", "ontop", "onfloor", "touching", "under",
})
STATE_PREDICATES = _UNARY_STATE_NAMES | _BINARY_STATE_NAMES

PREDICATE_ALIASES: Mapping[str, str] = MappingProxyType({
    "obj_inside": "inside", "obj_next_to": "nextto", "obj_ontop": "ontop",
    "next_to": "nextto", "on_top": "ontop", "switch_on": "on",
    "switchon": "on", "toggle_on": "on", "turn_on": "on", "turnon": "on",
    "toggled_on": "on", "switch_off": "off", "toggle_off": "off",
    "turn_off": "off", "turnoff": "off", "toggled_off": "off",
    "toggledon": "on", "unfrozen": "not_frozen",
})

NEGATIVE_TO_POSITIVE: Mapping[str, str] = MappingProxyType({
    "not_closed": "closed", "not_dusty": "dusty", "not_frozen": "frozen",
    "not_inside": "inside", "not_on": "on", "not_open": "open",
    "not_ontop": "ontop", "not_plugged_in": "plugged_in",
    "not_stained": "stained",
})


def _aliases_for(name: str) -> tuple[str, ...]:
    return tuple(sorted(alias for alias, canonical in PREDICATE_ALIASES.items() if canonical == name))


ACTION_SPECS: Mapping[str, ActionSpec] = MappingProxyType({
    name: ActionSpec(
        name=name,
        min_args=0 if name in {"sleep", "standup"} else 1,
        max_args=0 if name in {"sleep", "standup"} else 2,
        is_goal_action=name in ACTION_GOAL_NAMES,
        requires_task_entity=name in _TASK_ENTITY_ACTION_NAMES,
    )
    for name in sorted(_ACTION_NAMES)
})
TASK_ENTITY_ACTION_NAMES = frozenset(
    name for name, spec in ACTION_SPECS.items() if spec.requires_task_entity
)
STATE_SPECS: Mapping[str, StatePredicateSpec] = MappingProxyType({
    name: StatePredicateSpec(
        name=name,
        arity=1 if name in _UNARY_STATE_NAMES else 2,
        aliases=_aliases_for(name),
        positive_counterpart=NEGATIVE_TO_POSITIVE.get(name),
    )
    for name in sorted(STATE_PREDICATES)
})


@dataclass(frozen=True)
class EAIWorldGraphBundle:
    graph: Any
    trajectory: Any
    step: Any
    entity: Any
    state: Any
    action: Any
    next_step: Any
    contains_step: Any
    contains_entity: Any
    contains_next_step: Any
    contains_state: Any
    contains_action: Any
    step_roles: Mapping[str, Any]
    state_roles: Mapping[str, Any]
    action_roles: Mapping[str, Any]
    actions: Mapping[str, Any]
    states: Mapping[str, Any]
    aliases: Mapping[str, Any]
    negative_to_positive: Mapping[Any, Any]
    goal_actions: frozenset[Any]
    precondition_concepts: Mapping[str, Any]
    default_preconditions: tuple[ActionPreconditionSpec, ...]
    default_constraint_names: frozenset[str]

    @property
    def has_constraints(self) -> bool:
        return any(
            getattr(constraint, "headLC", True)
            for constraint in getattr(self.graph, "logicalConstrains", {}).values()
        )

    @property
    def has_custom_constraints(self) -> bool:
        return any(
            (getattr(constraint, "name", None) or key)
            not in self.default_constraint_names
            for key, constraint in getattr(self.graph, "logicalConstrains", {}).items()
            if getattr(constraint, "headLC", True)
        )

    def canonical_state_name(self, name: str) -> str:
        """Resolve an alias through the graph's canonical state concepts."""
        concept = self.aliases.get(name)
        if concept is None:
            concept = self.states.get(name)
        if concept is None:
            return name
        return concept.name.removeprefix("state__")

    def is_action(self, name: str) -> bool:
        return name in self.actions

    def is_goal_action(self, name: str) -> bool:
        concept = self.actions.get(name)
        return concept is not None and concept in self.goal_actions

    def is_state_predicate(self, name: str) -> bool:
        return self.canonical_state_name(name) in self.states

    def positive_state_name(self, name: str) -> str | None:
        concept = self.states.get(self.canonical_state_name(name))
        positive = self.negative_to_positive.get(concept)
        return positive.name.removeprefix("state__") if positive is not None else None


def canonical_state_name(name: str, bundle: EAIWorldGraphBundle | None = None) -> str:
    return bundle.canonical_state_name(name) if bundle is not None else PREDICATE_ALIASES.get(name, name)


def is_known_action(name: str, bundle: EAIWorldGraphBundle | None = None) -> bool:
    return bundle.is_action(name) if bundle is not None else name in ACTION_SPECS


def is_goal_action(name: str, bundle: EAIWorldGraphBundle | None = None) -> bool:
    return bundle.is_goal_action(name) if bundle is not None else name in ACTION_GOAL_NAMES


def is_state_predicate(name: str, bundle: EAIWorldGraphBundle | None = None) -> bool:
    if bundle is not None:
        return bundle.is_state_predicate(name)
    return canonical_state_name(name) in STATE_SPECS


def positive_state_name(name: str, bundle: EAIWorldGraphBundle | None = None) -> str | None:
    if bundle is not None:
        return bundle.positive_state_name(name)
    return NEGATIVE_TO_POSITIVE.get(canonical_state_name(name))


@dataclass(frozen=True)
class WorldConstraintEvaluation:
    score: float
    # Number of constraints that contributed to this trajectory's aggregate.
    constraint_count: int
    results: Mapping[str, Any]
    # Total number declared on the graph, including inactive constraints.
    declared_constraint_count: int


class WorldGraphConfigurationError(RuntimeError):
    """A declared world constraint could not be materialized or verified."""


def _action_side(action: str) -> str | None:
    if action.startswith("left_"):
        return "left"
    if action.startswith("right_"):
        return "right"
    return None


def _placement_relation_name(action: str) -> str | None:
    if "inside" in action or action in {"put", "putin"}:
        return "inside"
    if "ontop" in action or "on_top" in action or action in {"putback", "puton"}:
        return "ontop"
    if "nextto" in action or "next_to" in action:
        return "nextto"
    if "under" in action:
        return "under"
    return None


def _held_by_side(state: set[Fact]) -> dict[str, set[str]]:
    return {
        "left": {
            fact[2] for fact in state
            if len(fact) == 3 and fact[0] == "holds_lh" and fact[1] == "character"
        },
        "right": {
            fact[2] for fact in state
            if len(fact) == 3 and fact[0] == "holds_rh" and fact[1] == "character"
        },
    }


def _precondition_status(
    kind: str,
    action: str,
    args: tuple[str, ...],
    source_state: set[Fact],
    task_entity_types: Iterable[str] = (),
) -> tuple[bool, bool]:
    """Return ``(applicable, satisfied)`` for one source-state condition."""
    held = _held_by_side(source_state)
    side = _action_side(action)
    held_for_action = held[side] if side is not None else held["left"] | held["right"]

    if kind == "argument_available_in_task":
        available = frozenset(str(value) for value in task_entity_types)
        if not available:
            return False, True
        if not args:
            return True, False
        argument_types = set()
        for argument in args:
            entity_type = argument
            previous = None
            while entity_type != previous:
                previous = entity_type
                entity_type = re.sub(r"_(?:part|n)_\d+$", "", entity_type)
                entity_type = re.sub(r"_\d+$", "", entity_type)
            argument_types.add(entity_type)
        return True, argument_types <= available

    if kind == "placement_source_ready":
        if held_for_action:
            return True, True
        relation = _placement_relation_name(action)
        destination = args[0] if args else None
        # Flat EAI placement events name only the destination. Repeated
        # placement demonstrations are valid when a matching object is already
        # at that destination even though no hand remains occupied.
        already_placed = bool(
            relation and destination and any(
                len(fact) == 3
                and fact[0] == relation
                and fact[2] == destination
                for fact in source_state
            )
        )
        return True, already_placed

    if kind == "release_source_ready":
        if held_for_action:
            return True, True
        obj = args[0] if args else None
        already_placed = bool(
            obj and any(
                len(fact) == 3
                and fact[0] in {"inside", "nextto", "ontop", "onfloor", "under"}
                and fact[1] == obj
                for fact in source_state
            )
        )
        return True, already_placed

    if kind == "pour_source_ready":
        return True, bool(held["left"] or held["right"])

    if kind == "destination_open_if_known":
        destination = args[0] if args else None
        if destination is None:
            return False, True
        facts = {(fact[0], fact[1]) for fact in source_state if len(fact) == 2}
        known = any(
            (predicate, destination) in facts
            for predicate in ("open", "closed", "not_open", "not_closed")
        )
        if not known:
            return False, True
        satisfied = (
            ("open", destination) in facts
            or ("not_closed", destination) in facts
        ) and (
            ("closed", destination) not in facts
            and ("not_open", destination) not in facts
        )
        return True, satisfied

    raise ValueError(f"Unknown EAI action precondition kind: {kind!r}")


def build_eai_world_graph(
    graph_name: str = "eai_world",
    constraint_builders: Iterable[Callable[[EAIWorldGraphBundle], None]] = (),
    include_default_constraints: bool = False,
) -> EAIWorldGraphBundle:
    """Build the independent EAI trajectory graph and apply caller constraints."""
    from domiknows.graph import Concept, Graph

    # DomiKnowS relation names are process-global even though the relation
    # objects belong to a particular graph. Give structural roles a private,
    # per-build name so constructing a second world graph cannot turn `step`
    # into `step-1` and corrupt path-based constraint candidate selection.
    build_id = next(_WORLD_GRAPH_BUILD_IDS)
    role_prefix = f"__eai_world_{build_id}__"

    def role(name: str) -> str:
        return f"{role_prefix}{name}"

    with Graph(graph_name) as graph:
        # These base concepts describe the nodes that make up one materialized
        # world trajectory. State, action, and next_step are relation-like
        # nodes whose required links are declared below.
        trajectory = Concept(name="world_trajectory")
        step = Concept(name="world_step")
        entity = Concept(name="world_entity")
        state = Concept(name="world_state")
        action = Concept(name="world_action")
        next_step = Concept(name="next_step")

        # A trajectory owns its timeline, entity universe, and temporal-link
        # nodes. Each step in turn owns the state groundings and actions that
        # originate at that point in the timeline.
        contains_step = trajectory.contains(step)[0]
        contains_entity = trajectory.contains(entity)[0]
        contains_next_step = trajectory.contains(next_step)[0]
        contains_state = step.contains(state)[0]
        contains_action = step.contains(action)[0]

        # A next_step node explicitly connects adjacent timeline nodes. This
        # preserves transition direction instead of relying on node order.
        current_step, following_step = next_step.has_a(**{
            role("current"): step,
            role("following"): step,
        })

        # Every state grounding is scoped to one step and a subject/object
        # pair. Unary predicates use the reserved absent-entity object, so
        # unary and binary state facts share a single relation schema.
        state_step, state_subject, state_object = state.has_a(**{
            role("state_step"): step,
            role("state_subject"): entity,
            role("state_object"): entity,
        })

        # Every action grounding records the state transition it represents,
        # its actor, and up to two action arguments. Missing arguments are
        # materialized later with the same reserved absent-entity node.
        source_step, result_step, actor, arg1, arg2 = action.has_a(**{
            role("action_source_step"): step,
            role("action_result_step"): step,
            role("action_actor"): entity,
            role("action_arg1"): entity,
            role("action_arg2"): entity,
        })

        # Named predicate concepts specialize the shared state/action schemas.
        # Materialized relation nodes receive truth logits for these concepts,
        # allowing constraints to target semantic predicates such as `open`
        # or `grab` while retaining their common structural roles.
        states = {
            name: state(name=f"state__{name}")
            for name in STATE_SPECS
        }
        actions = {
            name: action(name=f"action__{name}")
            for name in ACTION_SPECS
        }
        precondition_concepts = {
            kind: action(name=f"precondition__{kind}")
            for kind in (
                "placement_source_ready",
                "release_source_ready",
                "pour_source_ready",
                "destination_open_if_known",
                "argument_available_in_task",
            )
        }
        default_preconditions: list[ActionPreconditionSpec] = []
        default_constraint_names: set[str] = set()

        if include_default_constraints:
            from domiknows.graph.logicalConstrain import ifL

            placement_actions = tuple(
                name for name in actions if _placement_relation_name(name) is not None
            )
            release_actions = tuple(
                name for name in actions
                if name == "drop" or "release" in name or "realease" in name
                or name == "putobjback"
            )
            specifications = [
                *(
                    ActionPreconditionSpec(
                        name=f"action_precondition__{name}__argument_available",
                        action=name,
                        kind="argument_available_in_task",
                    )
                    for name, spec in ACTION_SPECS.items()
                    if spec.min_args > 0 and spec.requires_task_entity
                ),
                *(
                    ActionPreconditionSpec(
                        name=f"action_precondition__{name}__source_holding",
                        action=name,
                        kind="placement_source_ready",
                    )
                    for name in placement_actions
                ),
                *(
                    ActionPreconditionSpec(
                        name=f"action_precondition__{name}__source_holding",
                        action=name,
                        kind="release_source_ready",
                    )
                    for name in release_actions
                ),
                ActionPreconditionSpec(
                    name="action_precondition__pour__source_holding",
                    action="pour",
                    kind="pour_source_ready",
                ),
                *(
                    ActionPreconditionSpec(
                        name=f"action_precondition__{name}__destination_open_if_known",
                        action=name,
                        kind="destination_open_if_known",
                    )
                    for name in placement_actions
                    if _placement_relation_name(name) == "inside"
                ),
            ]
            for specification in specifications:
                ifL(
                    actions[specification.action]("event"),
                    precondition_concepts[specification.kind]("event"),
                    name=specification.name,
                )
                default_preconditions.append(specification)
                default_constraint_names.add(specification.name)

        bundle = EAIWorldGraphBundle(
            graph=graph,
            trajectory=trajectory,
            step=step,
            entity=entity,
            state=state,
            action=action,
            next_step=next_step,
            contains_step=contains_step,
            contains_entity=contains_entity,
            contains_next_step=contains_next_step,
            contains_state=contains_state,
            contains_action=contains_action,
            step_roles=MappingProxyType({"current": current_step, "following": following_step}),
            state_roles=MappingProxyType({"step": state_step, "subject": state_subject, "object": state_object}),
            action_roles=MappingProxyType({
                "source_step": source_step, "result_step": result_step,
                "actor": actor, "arg1": arg1, "arg2": arg2,
            }),
            actions=MappingProxyType(actions),
            states=MappingProxyType(states),
            aliases=MappingProxyType({alias: states[canonical] for alias, canonical in PREDICATE_ALIASES.items()}),
            negative_to_positive=MappingProxyType({states[negative]: states[positive] for negative, positive in NEGATIVE_TO_POSITIVE.items()}),
            goal_actions=frozenset(actions[name] for name in ACTION_GOAL_NAMES),
            precondition_concepts=MappingProxyType(precondition_concepts),
            default_preconditions=tuple(default_preconditions),
            default_constraint_names=frozenset(default_constraint_names),
        )
        for builder in tuple(constraint_builders):
            builder(bundle)
    return bundle


def _node(concept: Any, instance_id: Any, value: Any = None, attributes: dict | None = None):
    from domiknows.graph import DataNode
    return DataNode(
        instanceID=instance_id,
        # DataNode.__str__ returns instanceValue directly, so non-string
        # values fail whenever verification logging formats a node.
        instanceValue=None if value is None else str(value),
        ontologyNode=concept,
        attributes=dict(attributes or {}),
    )


def _set_truth(node: Any, concept: Any, value: bool) -> None:
    logits = torch.tensor([-30.0, 30.0] if value else [30.0, -30.0])
    node.attributes[f"<{concept.name}>"] = logits


def _link(node: Any, role: Any, target: Any) -> None:
    node.addRelationLink(role.name, target)


def _canonical_fact(fact: Sequence[Any]) -> Fact:
    if not fact:
        return ()
    name = str(fact[0]).strip().lower().replace("-", "_").replace(".", "_")
    name = PREDICATE_ALIASES.get(name, name)
    return (name, *(str(arg).strip().lower().replace("-", "_").replace(".", "_") for arg in fact[1:]))


def materialize_world_trajectory(
    prepared_goal: Any,
    states: Sequence[set[Fact]],
    events: Sequence[Any],
    world_bundle: EAIWorldGraphBundle,
):
    """Create deterministic DataNodes for one simulated trajectory.

    Unary groundings are complete over the entity universe. Binary groundings
    are sparse: only pairs known to the prepared goal or encountered at runtime
    are represented.
    """
    task_id = str(getattr(prepared_goal, "task_id", "") or "<unknown task>")
    try:
        normalized_states = [set(filter(None, (_canonical_fact(f) for f in state))) for state in states]
        entities = set(getattr(prepared_goal, "entity_universe", ()) or ())
        tracked_pairs = set(getattr(prepared_goal, "tracked_binary_pairs", ()) or ())
        for state in normalized_states:
            for fact in state:
                entities.update(fact[1:])
                if len(fact) == 3:
                    tracked_pairs.add((fact[1], fact[2]))
        for event in events:
            args = tuple(str(arg).strip().lower().replace("-", "_").replace(".", "_") for arg in getattr(event, "args", ()))
            entities.update(args)
            if len(args) >= 2:
                tracked_pairs.add((args[0], args[1]))
        entities.update({"character", ABSENT_ENTITY})
        entities = tuple(sorted(e for e in entities if e))
        tracked_pairs = tuple(sorted((a, b) for a, b in tracked_pairs if a and b))

        root = _node(world_bundle.trajectory, task_id, value=task_id, attributes={
            "task_id": task_id,
            "entity_count": len(entities),
            "tracked_binary_pair_count": len(tracked_pairs),
            "action_event_count": len(events),
            "action_names": tuple(sorted({
                str(getattr(event, "name", "")).strip().lower().replace("-", "_").replace(".", "_")
                for event in events
            })),
            "true_state_predicates": tuple(sorted({
                fact[0] for state in normalized_states for fact in state if fact
            })),
        })
        _set_truth(root, world_bundle.trajectory, True)
        entity_nodes = {name: _node(world_bundle.entity, name, value=name) for name in entities}
        for entity_node in entity_nodes.values():
            _set_truth(entity_node, world_bundle.entity, True)
            root.addChildDataNode(entity_node)

        step_nodes = [_node(world_bundle.step, index, value=index) for index in range(len(normalized_states))]
        for step_node in step_nodes:
            _set_truth(step_node, world_bundle.step, True)
            root.addChildDataNode(step_node)

        for index in range(max(0, len(step_nodes) - 1)):
            edge = _node(world_bundle.next_step, index, value=index)
            _link(edge, world_bundle.step_roles["current"], step_nodes[index])
            _link(edge, world_bundle.step_roles["following"], step_nodes[index + 1])
            _set_truth(edge, world_bundle.next_step, True)
            root.addChildDataNode(edge)

        unary_specs = tuple((name, spec, world_bundle.states[name]) for name, spec in STATE_SPECS.items() if spec.arity == 1)
        binary_specs = tuple((name, spec, world_bundle.states[name]) for name, spec in STATE_SPECS.items() if spec.arity == 2)
        for index, (step_node, state) in enumerate(zip(step_nodes, normalized_states)):
            unary_true = {(fact[0], fact[1]) for fact in state if len(fact) == 2}
            binary_true = {(fact[0], fact[1], fact[2]) for fact in state if len(fact) == 3}
            for subject in entities:
                grounding = _node(
                    world_bundle.state,
                    f"unary:{index}:{subject}",
                    attributes={"grounding_arity": 1},
                )
                _link(grounding, world_bundle.state_roles["step"], step_node)
                _link(grounding, world_bundle.state_roles["subject"], entity_nodes[subject])
                _link(grounding, world_bundle.state_roles["object"], entity_nodes[ABSENT_ENTITY])
                _set_truth(grounding, world_bundle.state, True)
                for name, _spec, concept in unary_specs:
                    _set_truth(grounding, concept, (name, subject) in unary_true)
                step_node.addChildDataNode(grounding)
            for subject, obj in tracked_pairs:
                if subject not in entity_nodes or obj not in entity_nodes:
                    continue
                grounding = _node(
                    world_bundle.state,
                    f"binary:{index}:{subject}:{obj}",
                    attributes={"grounding_arity": 2},
                )
                _link(grounding, world_bundle.state_roles["step"], step_node)
                _link(grounding, world_bundle.state_roles["subject"], entity_nodes[subject])
                _link(grounding, world_bundle.state_roles["object"], entity_nodes[obj])
                _set_truth(grounding, world_bundle.state, True)
                for name, _spec, concept in binary_specs:
                    _set_truth(grounding, concept, (name, subject, obj) in binary_true)
                step_node.addChildDataNode(grounding)

        for index, event in enumerate(events):
            if not step_nodes:
                break
            name = str(getattr(event, "name", "")).strip().lower().replace("-", "_").replace(".", "_")
            args = tuple(str(arg).strip().lower().replace("-", "_").replace(".", "_") for arg in getattr(event, "args", ()))
            source_index = min(index, len(step_nodes) - 1)
            result_index = min(index + 1, len(step_nodes) - 1)
            event_node = _node(world_bundle.action, index, value=name)
            _link(event_node, world_bundle.action_roles["source_step"], step_nodes[source_index])
            _link(event_node, world_bundle.action_roles["result_step"], step_nodes[result_index])
            _link(event_node, world_bundle.action_roles["actor"], entity_nodes["character"])
            _link(event_node, world_bundle.action_roles["arg1"], entity_nodes[args[0] if args else ABSENT_ENTITY])
            _link(event_node, world_bundle.action_roles["arg2"], entity_nodes[args[1] if len(args) > 1 else ABSENT_ENTITY])
            _set_truth(event_node, world_bundle.action, True)
            for action_name, concept in world_bundle.actions.items():
                _set_truth(event_node, concept, action_name == name)
            source_state = normalized_states[source_index]
            for kind, concept in world_bundle.precondition_concepts.items():
                _applicable, satisfied = _precondition_status(
                    kind,
                    name,
                    args,
                    source_state,
                    getattr(prepared_goal, "task_entity_types", ()),
                )
                _set_truth(event_node, concept, satisfied)
            step_nodes[source_index].addChildDataNode(event_node)
        return root
    except Exception as exc:
        raise WorldGraphConfigurationError(f"{task_id}: failed to materialize EAI world trajectory: {exc}") from exc


def _aggregate_constraint_scores(scores: Sequence[float], aggregate: str) -> float:
    if aggregate == "mean":
        return sum(scores) / len(scores)
    if aggregate == "min":
        return min(scores)
    score = 1.0
    for value in scores:
        score *= value
    return score


def evaluate_default_world_constraints(
    states: Sequence[set[Fact]],
    events: Sequence[Any],
    world_bundle: EAIWorldGraphBundle,
    aggregate: str = "mean",
    task_entity_types: Iterable[str] = (),
) -> WorldConstraintEvaluation | None:
    """Evaluate built-in source-state preconditions without the general solver.

    The trajectory is already deterministic simulator output, so the built-in
    constraints have direct equivalents. Custom constraint builders continue
    through :func:`materialize_world_trajectory` and
    :func:`verify_world_constraints`.
    """
    if aggregate not in {"mean", "min", "prod"}:
        raise ValueError(f"Unsupported world-constraint aggregation: {aggregate!r}")
    if not world_bundle.default_constraint_names:
        return None

    # Normalize simulator facts and events into the same canonical vocabulary
    # used while constructing the graph. This keeps aliases such as
    # ``switch_on`` and punctuation variants from changing the outcome.
    normalized_states = [
        set(filter(None, (_canonical_fact(fact) for fact in state)))
        for state in states
    ]
    normalized_events = [
        (
            str(getattr(event, "name", "")).strip().lower().replace("-", "_").replace(".", "_"),
            tuple(
                str(arg).strip().lower().replace("-", "_").replace(".", "_")
                for arg in getattr(event, "args", ())
            ),
        )
        for event in events
    ]
    results: dict[str, dict[str, Any]] = {}
    scores: list[float] = []

    def record(name: str, applicable: bool, score: float = 1.0) -> None:
        # Store percentages to match the LC verifier's public result shape,
        # while aggregating normalized values in the [0, 1] range below.
        bounded = max(0.0, min(1.0, float(score)))
        results[name] = {
            "satisfied": bounded * 100.0,
            "applicable": applicable,
            "evaluation": "deterministic",
        }
        if applicable:
            scores.append(bounded)

    for specification in world_bundle.default_preconditions:
        matching = [
            (index, args)
            for index, (name, args) in enumerate(normalized_events)
            if name == specification.action
        ]
        applicable_count = 0
        satisfied_count = 0
        for index, args in matching:
            source_state = (
                normalized_states[index]
                if index < len(normalized_states) else set()
            )
            applicable, satisfied = _precondition_status(
                specification.kind,
                specification.action,
                args,
                source_state,
                task_entity_types,
            )
            if applicable:
                applicable_count += 1
                satisfied_count += int(satisfied)
        score = (
            satisfied_count / applicable_count if applicable_count else 1.0
        )
        record(specification.name, bool(applicable_count), score)

    if not scores:
        # No applicable default precondition means callers should continue with
        # custom constraints, rather than treating the trajectory as perfect.
        return None
    return WorldConstraintEvaluation(
        score=_aggregate_constraint_scores(scores, aggregate),
        constraint_count=len(scores),
        results=MappingProxyType(results),
        declared_constraint_count=len(world_bundle.default_constraint_names),
    )


def verify_world_constraints(
    root: Any,
    world_bundle: EAIWorldGraphBundle,
    aggregate: str = "mean",
) -> WorldConstraintEvaluation | None:
    """Verify declared world constraints and aggregate their satisfaction."""
    constraints = {
        key: constraint
        for key, constraint in getattr(
            world_bundle.graph, "logicalConstrains", {}
        ).items()
        if getattr(constraint, "headLC", True)
    }
    if not constraints:
        return None
    if aggregate not in {"mean", "min", "prod"}:
        raise ValueError(f"Unsupported world-constraint aggregation: {aggregate!r}")
    task_id = str((root.getAttributes() or {}).get("task_id", "<unknown task>"))
    try:
        # The solver factory uses one shared cache entry for ontology-free
        # graphs. Switch it only when verification moves to another world
        # graph; repeated RL samples on the same bundle retain the cache.
        global _VERIFIER_GRAPH
        if _VERIFIER_GRAPH is not world_bundle.graph:
            from domiknows.solver import ilpOntSolverFactory

            ilpOntSolverFactory.clear()
            _VERIFIER_GRAPH = world_bundle.graph
        raw_results = root.verifyResultsLC(key="/local/argmax")
        named_results = {}
        scores = []
        root_attributes = root.getAttributes() or {}
        action_names = set(root_attributes.get("action_names", ()))
        state_predicates = set(root_attributes.get("true_state_predicates", ()))
        for key, constraint in constraints.items():
            value = raw_results.get(key, {})
            percent = value.get("satisfied", value.get("ifSatisfied"))
            if percent is None:
                raise KeyError(f"constraint {key!r} produced no satisfaction value")
            name = getattr(constraint, "name", None) or key
            if name in named_results:
                name = key

            # Logical verification correctly regards a false antecedent as
            # vacuously true. For reward shaping, however, inactive constraints
            # must not inflate the aggregate. Use referenced semantic concepts
            # when possible and fall back to the verifier's candidate list for
            # purely structural custom constraints.
            from domiknows.graph.lcUtils import getConceptsFromLogicalConstraint

            concept_names = set(getConceptsFromLogicalConstraint(constraint))
            referenced_actions = {
                concept.removeprefix("action__")
                for concept in concept_names if concept.startswith("action__")
            }
            referenced_states = {
                concept.removeprefix("state__")
                for concept in concept_names if concept.startswith("state__")
            }
            if referenced_actions or referenced_states:
                applicable = bool(
                    referenced_actions & action_names
                    or referenced_states & state_predicates
                )
            else:
                applicable = value.get("verifyList") not in (None, [])

            result_value = dict(value)
            result_value["applicable"] = applicable
            named_results[name] = result_value
            if not applicable:
                continue

            constraint_score = float(percent) / 100.0
            scores.append(max(0.0, min(1.0, constraint_score)))
        if not scores:
            return None
        score = _aggregate_constraint_scores(scores, aggregate)
        return WorldConstraintEvaluation(
            score=score,
            constraint_count=len(scores),
            results=MappingProxyType(named_results),
            declared_constraint_count=len(constraints),
        )
    except Exception as exc:
        raise WorldGraphConfigurationError(f"{task_id}: failed to verify EAI world constraints: {exc}") from exc
