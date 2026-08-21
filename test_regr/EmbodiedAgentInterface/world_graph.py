"""DomiKnowS schema and deterministic trajectory materialization for EAI.

This graph is deliberately separate from the token-generation graph.  It is a
semantic view of a simulated world trajectory and is only used to score
constraints declared by callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count
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
    state: Any
    action: Any
    adjacent_transition: Any
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
    default_action_effects: Mapping[str, str]
    default_state_mutex_pairs: tuple[tuple[str, str], ...]

    @property
    def has_constraints(self) -> bool:
        return any(
            getattr(constraint, "headLC", True)
            for constraint in getattr(self.graph, "logicalConstrains", {}).values()
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
        source_step, result_step, actor, arg1, arg2, action_result_state = action.has_a(**{
            role("action_source_step"): step,
            role("action_result_step"): step,
            role("action_actor"): entity,
            role("action_arg1"): entity,
            role("action_arg2"): entity,
            role("action_result_state"): state,
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
        adjacent_transition = action(name="action__valid_next_transition")
        default_action_effects: Mapping[str, str] = MappingProxyType({})
        default_state_mutex_pairs: tuple[tuple[str, str], ...] = ()

        if include_default_constraints:
            from domiknows.graph.logicalConstrain import atMostL, exactL, ifL, nandL

            DEFAULT_STATE_MUTEX_PAIRS = tuple(dict.fromkeys((
                *(tuple(sorted((negative, positive))) for negative, positive in NEGATIVE_TO_POSITIVE.items()),
                ("closed", "open"),
                ("off", "on"),
                ("clean", "dusty"),
                ("clean", "stained"),
                ("inside", "onfloor"),
                ("ontop", "under"),
            )))
            default_state_mutex_pairs = DEFAULT_STATE_MUTEX_PAIRS

            # Mutually exclusive state descriptions on the same grounding.
            # This includes every explicit positive/negative pair plus common
            # physical contradictions represented by the EAI vocabulary.
            for left, right in DEFAULT_STATE_MUTEX_PAIRS:
                nandL(
                    states[left],
                    states[right],
                    name=f"state_mutex__{left}__{right}",
                )

            # Every action event has exactly one semantic action type.
            exactL(
                *(concept("event") for concept in actions.values()),
                limit=1,
                name="action_exactly_one_type",
            )

            # An action's result must be the step immediately following its
            # source according to the explicit next_step relation.
            ifL(
                action("event"),
                adjacent_transition("event"),
                name="action_result_is_next_step",
            )

            # At a state step, each hand can hold at most one object.
            for predicate in ("holds_lh", "holds_rh"):
                ifL(
                    step("step"),
                    atMostL(
                        states[predicate](
                            "holding", path=("step", state_step.reversed)
                        ),
                        limit=1,
                    ),
                    name=f"hand_capacity__{predicate}",
                )

            DEFAULT_ACTION_EFFECTS: Mapping[str, str] = MappingProxyType({
                "clean": "clean", "wipe": "clean", "scrub": "clean",
                "wash": "washed", "rinse": "rinsed",
                "open": "open", "close": "closed",
                "toggle_on": "on", "switchon": "on", "switch_on": "on", "turn_on": "on",
                "toggle_off": "off", "switchoff": "off", "switch_off": "off", "turn_off": "off",
                "slice": "sliced", "soak": "soaked", "freeze": "frozen",
                "unfreeze": "not_frozen", "cook": "cooked",
                "plugin": "plugged_in", "plugout": "not_plugged_in", "touch": "touch",
            })
            default_action_effects = DEFAULT_ACTION_EFFECTS

            # Direct unary effects from the simulator must hold for the action
            # argument at the event's result step.
            for action_name, predicate in DEFAULT_ACTION_EFFECTS.items():
                ifL(
                    actions[action_name]("event"),
                    states[predicate](
                        "effect", path=("event", action_result_state)
                    ),
                    name=f"action_effect__{action_name}__{predicate}",
                )

        bundle = EAIWorldGraphBundle(
            graph=graph,
            trajectory=trajectory,
            step=step,
            entity=entity,
            state=state,
            action=action,
            adjacent_transition=adjacent_transition,
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
                "result_state": action_result_state,
            }),
            actions=MappingProxyType(actions),
            states=MappingProxyType(states),
            aliases=MappingProxyType({alias: states[canonical] for alias, canonical in PREDICATE_ALIASES.items()}),
            negative_to_positive=MappingProxyType({states[negative]: states[positive] for negative, positive in NEGATIVE_TO_POSITIVE.items()}),
            goal_actions=frozenset(actions[name] for name in ACTION_GOAL_NAMES),
            default_action_effects=default_action_effects,
            default_state_mutex_pairs=default_state_mutex_pairs,
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
        state_nodes: dict[tuple[int, int, str, str], Any] = {}
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
                state_nodes[(index, 1, subject, ABSENT_ENTITY)] = grounding
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
                state_nodes[(index, 2, subject, obj)] = grounding

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
            effect_predicate = world_bundle.default_action_effects.get(name)
            if effect_predicate is not None and args:
                effect_spec = STATE_SPECS[effect_predicate]
                effect_key = (
                    result_index,
                    effect_spec.arity,
                    args[0],
                    ABSENT_ENTITY if effect_spec.arity == 1 else (
                        args[1] if len(args) > 1 else ABSENT_ENTITY
                    ),
                )
                effect_node = state_nodes.get(effect_key)
                if effect_node is not None:
                    _link(
                        event_node,
                        world_bundle.action_roles["result_state"],
                        effect_node,
                    )
            _set_truth(event_node, world_bundle.action, True)
            _set_truth(
                event_node,
                world_bundle.adjacent_transition,
                result_index == source_index + 1,
            )
            for action_name, concept in world_bundle.actions.items():
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
            # must not inflate the aggregate. A built-in constraint is relevant
            # only when its state/action vocabulary occurs in this trajectory.
            if name == "action_exactly_one_type" or name == "action_result_is_next_step":
                applicable = bool(action_names)
            elif name.startswith("action_effect__"):
                action_name = name.split("__", 2)[1]
                applicable = action_name in action_names
            elif name.startswith("hand_capacity__"):
                predicate = name.split("__", 1)[1]
                applicable = predicate in state_predicates
            elif name.startswith("state_mutex__"):
                _prefix, left, right = name.split("__", 2)
                applicable = left in state_predicates or right in state_predicates
            else:
                # Custom constraints use their referenced semantic concepts
                # when possible. Fall back to the verifier's candidate list for
                # purely structural constraints.
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
        if aggregate == "mean":
            score = sum(scores) / len(scores)
        elif aggregate == "min":
            score = min(scores)
        else:
            score = 1.0
            for value in scores:
                score *= value
        return WorldConstraintEvaluation(
            score=score,
            constraint_count=len(scores),
            results=MappingProxyType(named_results),
            declared_constraint_count=len(constraints),
        )
    except Exception as exc:
        raise WorldGraphConfigurationError(f"{task_id}: failed to verify EAI world constraints: {exc}") from exc
