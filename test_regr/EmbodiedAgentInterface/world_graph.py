"""DomiKnowS schema and deterministic trajectory materialization for EAI.

This graph is deliberately separate from the token-generation graph.  It is a
semantic view of a simulated world trajectory and is only used to score
constraints declared by callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


Fact = tuple[str, ...]
ABSENT_ENTITY = "__absent_action_argument__"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    min_args: int
    max_args: int
    is_goal_action: bool = False


@dataclass(frozen=True)
class StatePredicateSpec:
    name: str
    arity: int
    aliases: tuple[str, ...] = ()
    positive_counterpart: str | None = None


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
    )
    for name in sorted(_ACTION_NAMES)
})
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
    unary_grounding: Any
    binary_grounding: Any
    action_event: Any
    next_step: Any
    contains_step: Any
    contains_entity: Any
    contains_next_step: Any
    contains_unary: Any
    contains_binary: Any
    contains_action: Any
    step_roles: Mapping[str, Any]
    unary_roles: Mapping[str, Any]
    binary_roles: Mapping[str, Any]
    action_roles: Mapping[str, Any]
    action: Mapping[str, Any]
    state: Mapping[str, Any]
    aliases: Mapping[str, Any]
    negative_to_positive: Mapping[Any, Any]
    goal_actions: frozenset[Any]

    @property
    def has_constraints(self) -> bool:
        return bool(getattr(self.graph, "logicalConstrains", {}))

    def canonical_state_name(self, name: str) -> str:
        """Resolve an alias through the graph's canonical state concepts."""
        concept = self.aliases.get(name)
        if concept is None:
            concept = self.state.get(name)
        if concept is None:
            return name
        return concept.name.removeprefix("state__")

    def is_action(self, name: str) -> bool:
        return name in self.action

    def is_goal_action(self, name: str) -> bool:
        concept = self.action.get(name)
        return concept is not None and concept in self.goal_actions

    def is_state_predicate(self, name: str) -> bool:
        return self.canonical_state_name(name) in self.state

    def positive_state_name(self, name: str) -> str | None:
        concept = self.state.get(self.canonical_state_name(name))
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
    constraint_count: int
    results: Mapping[str, Any]


class WorldGraphConfigurationError(RuntimeError):
    """A declared world constraint could not be materialized or verified."""


def open_closed_exclusivity(bundle: EAIWorldGraphBundle) -> None:
    """Example invariant: an entity cannot be both open and closed."""
    from domiknows.graph.logicalConstrain import nandL

    nandL(bundle.state["open"], bundle.state["closed"])


def build_eai_world_graph(
    graph_name: str = "eai_world",
    constraint_builders: Iterable[Callable[[EAIWorldGraphBundle], None]] = (),
) -> EAIWorldGraphBundle:
    """Build the independent EAI trajectory graph and apply caller constraints."""
    from domiknows.graph import Concept, Graph

    with Graph(graph_name) as graph:
        trajectory = Concept(name="world_trajectory")
        step = Concept(name="world_step")
        entity = Concept(name="world_entity")
        unary_grounding = Concept(name="unary_state_grounding")
        binary_grounding = Concept(name="binary_state_grounding")
        action_event = Concept(name="world_action_event")
        next_step = Concept(name="next_step")

        contains_step = trajectory.contains(step)[0]
        contains_entity = trajectory.contains(entity)[0]
        contains_next_step = trajectory.contains(next_step)[0]
        contains_unary = step.contains(unary_grounding)[0]
        contains_binary = step.contains(binary_grounding)[0]
        contains_action = step.contains(action_event)[0]

        current_step, following_step = next_step.has_a(current=step, following=step)
        unary_step, unary_subject = unary_grounding.has_a(step=step, subject=entity)
        binary_step, binary_subject, binary_object = binary_grounding.has_a(
            step=step, subject=entity, object=entity,
        )
        source_step, result_step, actor, arg1, arg2 = action_event.has_a(
            source_step=step, result_step=step, actor=entity, arg1=entity, arg2=entity,
        )

        state = {
            name: (unary_grounding if spec.arity == 1 else binary_grounding)(
                name=f"state__{name}"
            )
            for name, spec in STATE_SPECS.items()
        }
        action = {
            name: action_event(name=f"action__{name}")
            for name in ACTION_SPECS
        }

        bundle = EAIWorldGraphBundle(
            graph=graph,
            trajectory=trajectory,
            step=step,
            entity=entity,
            unary_grounding=unary_grounding,
            binary_grounding=binary_grounding,
            action_event=action_event,
            next_step=next_step,
            contains_step=contains_step,
            contains_entity=contains_entity,
            contains_next_step=contains_next_step,
            contains_unary=contains_unary,
            contains_binary=contains_binary,
            contains_action=contains_action,
            step_roles=MappingProxyType({"current": current_step, "following": following_step}),
            unary_roles=MappingProxyType({"step": unary_step, "subject": unary_subject}),
            binary_roles=MappingProxyType({"step": binary_step, "subject": binary_subject, "object": binary_object}),
            action_roles=MappingProxyType({
                "source_step": source_step, "result_step": result_step,
                "actor": actor, "arg1": arg1, "arg2": arg2,
            }),
            action=MappingProxyType(action),
            state=MappingProxyType(state),
            aliases=MappingProxyType({alias: state[canonical] for alias, canonical in PREDICATE_ALIASES.items()}),
            negative_to_positive=MappingProxyType({state[negative]: state[positive] for negative, positive in NEGATIVE_TO_POSITIVE.items()}),
            goal_actions=frozenset(action[name] for name in ACTION_GOAL_NAMES),
        )
        for builder in tuple(constraint_builders):
            builder(bundle)
    return bundle


def _node(concept: Any, instance_id: Any, value: Any = None, attributes: dict | None = None):
    from domiknows.graph import DataNode
    return DataNode(
        instanceID=instance_id,
        instanceValue=value,
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
        })
        entity_nodes = {name: _node(world_bundle.entity, name, value=name) for name in entities}
        for entity_node in entity_nodes.values():
            root.addChildDataNode(entity_node)

        step_nodes = [_node(world_bundle.step, index, value=index) for index in range(len(normalized_states))]
        for step_node in step_nodes:
            root.addChildDataNode(step_node)

        for index in range(max(0, len(step_nodes) - 1)):
            edge = _node(world_bundle.next_step, index, value=index)
            _link(edge, world_bundle.step_roles["current"], step_nodes[index])
            _link(edge, world_bundle.step_roles["following"], step_nodes[index + 1])
            root.addChildDataNode(edge)

        unary_specs = tuple((name, spec, world_bundle.state[name]) for name, spec in STATE_SPECS.items() if spec.arity == 1)
        binary_specs = tuple((name, spec, world_bundle.state[name]) for name, spec in STATE_SPECS.items() if spec.arity == 2)
        for index, (step_node, state) in enumerate(zip(step_nodes, normalized_states)):
            unary_true = {(fact[0], fact[1]) for fact in state if len(fact) == 2}
            binary_true = {(fact[0], fact[1], fact[2]) for fact in state if len(fact) == 3}
            for subject in entities:
                grounding = _node(world_bundle.unary_grounding, f"{index}:{subject}")
                _link(grounding, world_bundle.unary_roles["step"], step_node)
                _link(grounding, world_bundle.unary_roles["subject"], entity_nodes[subject])
                for name, _spec, concept in unary_specs:
                    _set_truth(grounding, concept, (name, subject) in unary_true)
                step_node.addChildDataNode(grounding)
            for subject, obj in tracked_pairs:
                if subject not in entity_nodes or obj not in entity_nodes:
                    continue
                grounding = _node(world_bundle.binary_grounding, f"{index}:{subject}:{obj}")
                _link(grounding, world_bundle.binary_roles["step"], step_node)
                _link(grounding, world_bundle.binary_roles["subject"], entity_nodes[subject])
                _link(grounding, world_bundle.binary_roles["object"], entity_nodes[obj])
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
            event_node = _node(world_bundle.action_event, index, value=name)
            _link(event_node, world_bundle.action_roles["source_step"], step_nodes[source_index])
            _link(event_node, world_bundle.action_roles["result_step"], step_nodes[result_index])
            _link(event_node, world_bundle.action_roles["actor"], entity_nodes["character"])
            _link(event_node, world_bundle.action_roles["arg1"], entity_nodes[args[0] if args else ABSENT_ENTITY])
            _link(event_node, world_bundle.action_roles["arg2"], entity_nodes[args[1] if len(args) > 1 else ABSENT_ENTITY])
            for action_name, concept in world_bundle.action.items():
                _set_truth(event_node, concept, action_name == name)
            step_nodes[source_index].addChildDataNode(event_node)
        return root
    except Exception as exc:
        raise WorldGraphConfigurationError(f"{task_id}: failed to materialize EAI world trajectory: {exc}") from exc


def verify_world_constraints(
    root: Any,
    world_bundle: EAIWorldGraphBundle,
    aggregate: str = "mean",
) -> WorldConstraintEvaluation | None:
    """Verify declared world constraints and aggregate their satisfaction."""
    constraints = getattr(world_bundle.graph, "logicalConstrains", {})
    if not constraints:
        return None
    if aggregate not in {"mean", "min", "prod"}:
        raise ValueError(f"Unsupported world-constraint aggregation: {aggregate!r}")
    task_id = str((root.getAttributes() or {}).get("task_id", "<unknown task>"))
    try:
        results = root.verifyResultsLC(key="/local/argmax")
        scores = []
        for name in constraints:
            value = results.get(name, {})
            percent = value.get("satisfied", value.get("ifSatisfied"))
            if percent is None:
                raise KeyError(f"constraint {name!r} produced no satisfaction value")
            scores.append(max(0.0, min(1.0, float(percent) / 100.0)))
        if not scores:
            raise ValueError("constraints were declared but none were verified")
        if aggregate == "mean":
            score = sum(scores) / len(scores)
        elif aggregate == "min":
            score = min(scores)
        else:
            score = 1.0
            for value in scores:
                score *= value
        return WorldConstraintEvaluation(score=score, constraint_count=len(scores), results=MappingProxyType(dict(results)))
    except Exception as exc:
        raise WorldGraphConfigurationError(f"{task_id}: failed to verify EAI world constraints: {exc}") from exc
