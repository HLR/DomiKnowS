"""EAI world concepts and readable DomiKnowS logical constraints.

Vocabulary records live in utils/world_domain.py. Runtime observations and
verification live in utils/world_runtime.py; neither chooses policy semantics.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping
from test_regr.EmbodiedAgentInterface.utils.world_domain import (
    ActionSpec,
    StatePredicateSpec,
    ActionPreconditionSpec,
    ACTION_SPECS,
    ACTION_GOAL_NAMES,
    TASK_ENTITY_ACTION_NAMES,
    STATE_SPECS,
    STATE_PREDICATES,
    PREDICATE_ALIASES,
    NEGATIVE_TO_POSITIVE,
    _placement_relation_name,
)

_WORLD_GRAPH_BUILD_IDS = count()


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
    grounded_constraints: Any = None
    facts: Any = None
    default_constraints: tuple = ()

    @property
    def has_constraints(self) -> bool:
        return any(
            getattr(constraint, "headLC", True)
            for constraint in getattr(self.graph, "logicalConstrains", {}).values()
        )

    @property
    def has_custom_constraints(self) -> bool:
        return any(
            not any(constraint is default for default in self.default_constraints)
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


def _default_preconditions(actions):
    """Expand action declarations into named rule instances for diagnostics."""
    placement_actions = tuple(
        name for name in actions if _placement_relation_name(name) is not None
    )
    release_actions = tuple(
        name
        for name in actions
        if name == "drop" or "release" in name or "realease" in name or name == "putobjback"
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
    return tuple(specifications)


def _declare_precondition_constraints(actions, facts, specifications):
    """Task arguments, occupied hands, idempotence and known container access."""
    from domiknows.graph.logicalConstrain import andL, ifL, notL, orL

    def observed(name):
        return facts[name]("operation")

    for specification in specifications:
        action = actions[specification.action]("operation")
        kind = specification.kind
        if kind == "argument_available_in_task":
            # Every supplied argument must belong to the task's entity types.
            antecedent = andL(action, observed("task_entities_supplied"))
            conclusion = andL(
                observed("first_argument_supplied"),
                observed("first_argument_in_task"),
                ifL(observed("second_argument_supplied"), observed("second_argument_in_task")),
            )
        elif kind == "placement_source_ready":
            # A known source must be held, or already placed at this destination.
            antecedent = andL(action, observed("source_state_known"))
            conclusion = orL(
                observed("selected_hand_occupied"), observed("matching_placement_exists")
            )
        elif kind == "release_source_ready":
            # Releasing an already placed object is idempotent.
            antecedent = andL(action, observed("source_state_known"))
            conclusion = orL(
                observed("selected_hand_occupied"), observed("matching_release_relation_exists")
            )
        elif kind == "pour_source_ready":
            antecedent = andL(action, observed("source_state_known"))
            conclusion = orL(observed("left_hand_occupied"), observed("right_hand_occupied"))
        elif kind == "destination_open_if_known":
            # Unknown access is not invented. Known contradictory access fails.
            antecedent = andL(
                action,
                observed("destination_supplied"),
                orL(
                    observed("destination_observed_open"),
                    observed("destination_observed_not_closed"),
                    observed("destination_observed_closed"),
                    observed("destination_observed_not_open"),
                ),
            )
            conclusion = andL(
                orL(
                    observed("destination_observed_open"),
                    observed("destination_observed_not_closed"),
                ),
                notL(observed("destination_observed_closed")),
                notL(observed("destination_observed_not_open")),
            )
        else:
            raise ValueError(f"Unsupported declared precondition: {kind}")
        ifL(antecedent, conclusion, name=specification.name)


def build_eai_world_graph(
    graph_name: str = "eai_world",
    constraint_builders: Iterable[Callable[[EAIWorldGraphBundle], None]] = (),
    include_default_constraints: bool = False,
    semantic_parents: Mapping[str, Any] | None = None,
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

    semantic_parents = dict(semantic_parents or {})
    with Graph(graph_name) as graph:
        # These base concepts describe the nodes that make up one materialized
        # world trajectory. State, action, and next_step are relation-like
        # nodes whose required links are declared below.
        trajectory_parent = semantic_parents.get("episode")
        entity_parent = semantic_parents.get("entity")
        operation_parent = semantic_parents.get("operation")
        trajectory = (
            trajectory_parent(name="world_trajectory")
            if trajectory_parent is not None
            else Concept(name="world_trajectory")
        )
        step = Concept(name="world_step")
        entity = (
            entity_parent(name="world_entity")
            if entity_parent is not None
            else Concept(name="world_entity")
        )
        state = Concept(name="world_state")
        action = (
            operation_parent(name="world_action")
            if operation_parent is not None
            else Concept(name="world_action")
        )
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
        current_step, following_step = next_step.has_a(
            **{
                role("current"): step,
                role("following"): step,
            }
        )

        # Every state grounding is scoped to one step and a subject/object
        # pair. Unary predicates use the reserved absent-entity object, so
        # unary and binary state facts share a single relation schema.
        state_step, state_subject, state_object = state.has_a(
            **{
                role("state_step"): step,
                role("state_subject"): entity,
                role("state_object"): entity,
            }
        )

        # Every action grounding records the state transition it represents,
        # its actor, and up to two action arguments. Missing arguments are
        # materialized later with the same reserved absent-entity node.
        source_step, result_step, actor, arg1, arg2 = action.has_a(
            **{
                role("action_source_step"): step,
                role("action_result_step"): step,
                role("action_actor"): entity,
                role("action_arg1"): entity,
                role("action_arg2"): entity,
            }
        )

        # Named predicate concepts specialize the shared state/action schemas.
        # Materialized relation nodes receive truth logits for these concepts,
        # allowing constraints to target semantic predicates such as `open`
        # or `grab` while retaining their common structural roles.
        states = {name: state(name=f"state__{name}") for name in STATE_SPECS}
        actions = {name: action(name=f"action__{name}") for name in ACTION_SPECS}
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

        from test_regr.EmbodiedAgentInterface.utils.world_facts import PRECONDITION_FACTS

        facts = {name: action(name="observed__" + name) for name in PRECONDITION_FACTS}
        if include_default_constraints:
            default_preconditions.extend(_default_preconditions(actions))
            _declare_precondition_constraints(actions, facts, default_preconditions)
            default_constraint_names.update(spec.name for spec in default_preconditions)

        from domiknows.generation.grounded import declare_predicates, FAMILIES

        grounded_constraints = declare_predicates(
            action,
            trajectory,
            active=include_default_constraints,
            constraints=include_default_constraints,
            detailed=True,
        )
        if include_default_constraints:
            default_constraint_names.update("grounded__" + name for name in FAMILIES)

        bundle = EAIWorldGraphBundle(
            default_constraints=tuple(
                lc for lc in graph.logicalConstrains.values() if getattr(lc, "headLC", True)
            ),
            grounded_constraints=grounded_constraints,
            facts=MappingProxyType(facts),
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
            state_roles=MappingProxyType(
                {"step": state_step, "subject": state_subject, "object": state_object}
            ),
            action_roles=MappingProxyType(
                {
                    "source_step": source_step,
                    "result_step": result_step,
                    "actor": actor,
                    "arg1": arg1,
                    "arg2": arg2,
                }
            ),
            actions=MappingProxyType(actions),
            states=MappingProxyType(states),
            aliases=MappingProxyType(
                {alias: states[canonical] for alias, canonical in PREDICATE_ALIASES.items()}
            ),
            negative_to_positive=MappingProxyType(
                {
                    states[negative]: states[positive]
                    for negative, positive in NEGATIVE_TO_POSITIVE.items()
                }
            ),
            goal_actions=frozenset(actions[name] for name in ACTION_GOAL_NAMES),
            precondition_concepts=MappingProxyType(precondition_concepts),
            default_preconditions=tuple(default_preconditions),
            default_constraint_names=frozenset(default_constraint_names),
        )
        for builder in tuple(constraint_builders):
            builder(bundle)
    return bundle


def __getattr__(name):
    """Preserve existing imports while keeping runtime code out of this file."""
    if name.startswith("__"):
        raise AttributeError(name)
    import importlib

    for suffix in ("world_domain", "world_runtime"):
        module = importlib.import_module(f"test_regr.EmbodiedAgentInterface.utils.{suffix}")
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)
