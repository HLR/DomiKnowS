"""Materialize EAI observations and evaluate the declared logical constraints."""

from __future__ import annotations
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import torch
from test_regr.EmbodiedAgentInterface.utils.world_domain import (
    Fact,
    ABSENT_ENTITY,
    STATE_SPECS,
    PREDICATE_ALIASES,
    WorldConstraintEvaluation,
    WorldGraphConfigurationError,
)
from test_regr.EmbodiedAgentInterface.world_graph import EAIWorldGraphBundle

_VERIFIER_GRAPH = None


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
    return (
        name,
        *(str(arg).strip().lower().replace("-", "_").replace(".", "_") for arg in fact[1:]),
    )


def materialize_world_trajectory(
    prepared_goal: Any,
    states: Sequence[set[Fact]],
    events: Sequence[Any],
    world_bundle: EAIWorldGraphBundle,
    *,
    planning_facts=None,
):
    """Create deterministic DataNodes for one simulated trajectory.

    Unary groundings are complete over the entity universe. Binary groundings
    are sparse: only pairs known to the prepared goal or encountered at runtime
    are represented.
    """
    task_id = str(getattr(prepared_goal, "task_id", "") or "<unknown task>")
    try:
        normalized_states = [
            set(filter(None, (_canonical_fact(f) for f in state))) for state in states
        ]
        entities = set(getattr(prepared_goal, "entity_universe", ()) or ())
        tracked_pairs = set(getattr(prepared_goal, "tracked_binary_pairs", ()) or ())
        for state in normalized_states:
            for fact in state:
                entities.update(fact[1:])
                if len(fact) == 3:
                    tracked_pairs.add((fact[1], fact[2]))
        for event in events:
            args = tuple(
                str(arg).strip().lower().replace("-", "_").replace(".", "_")
                for arg in getattr(event, "args", ())
            )
            entities.update(args)
            if len(args) >= 2:
                tracked_pairs.add((args[0], args[1]))
        entities.update({"character", ABSENT_ENTITY})
        entities = tuple(sorted(e for e in entities if e))
        tracked_pairs = tuple(sorted((a, b) for a, b in tracked_pairs if a and b))

        root = _node(
            world_bundle.trajectory,
            task_id,
            value=task_id,
            attributes={
                "task_id": task_id,
                "entity_count": len(entities),
                "tracked_binary_pair_count": len(tracked_pairs),
                "action_event_count": len(events),
                "action_names": tuple(
                    sorted(
                        {
                            str(getattr(event, "name", ""))
                            .strip()
                            .lower()
                            .replace("-", "_")
                            .replace(".", "_")
                            for event in events
                        }
                    )
                ),
                "true_state_predicates": tuple(
                    sorted({fact[0] for state in normalized_states for fact in state if fact})
                ),
            },
        )
        _set_truth(root, world_bundle.trajectory, True)
        entity_nodes = {name: _node(world_bundle.entity, name, value=name) for name in entities}
        for entity_node in entity_nodes.values():
            _set_truth(entity_node, world_bundle.entity, True)
            root.addChildDataNode(entity_node)

        step_nodes = [
            _node(world_bundle.step, index, value=index) for index in range(len(normalized_states))
        ]
        for step_node in step_nodes:
            _set_truth(step_node, world_bundle.step, True)
            root.addChildDataNode(step_node)

        for index in range(max(0, len(step_nodes) - 1)):
            edge = _node(world_bundle.next_step, index, value=index)
            _link(edge, world_bundle.step_roles["current"], step_nodes[index])
            _link(edge, world_bundle.step_roles["following"], step_nodes[index + 1])
            _set_truth(edge, world_bundle.next_step, True)
            root.addChildDataNode(edge)

        unary_specs = tuple(
            (name, spec, world_bundle.states[name])
            for name, spec in STATE_SPECS.items()
            if spec.arity == 1
        )
        binary_specs = tuple(
            (name, spec, world_bundle.states[name])
            for name, spec in STATE_SPECS.items()
            if spec.arity == 2
        )
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

        from test_regr.EmbodiedAgentInterface.utils.constraint_compiler import EAIExecutor
        from domiknows.generation.grounded import observations

        explicit = (
            planning_facts
            if planning_facts is not None
            else getattr(prepared_goal, "planning_facts", None)
        )
        executor = EAIExecutor(
            world_bundle,
            {
                "planning_facts": explicit,
                "generation_entity_types": getattr(prepared_goal, "task_entity_types", ()),
            },
            entities,
        )
        grounded_values, final_values, grounding, operation_checks, completion = (
            executor.audit_details(events)
        )
        _materialize_requirements(root, world_bundle.grounded_constraints, (completion,), "episode")
        root.attributes["grounding_report"] = grounding
        for key in ("goal_completion__applicable", "goal_completion__satisfied"):
            _set_truth(root, world_bundle.grounded_constraints[key], final_values[key])
        for index, event in enumerate(events):
            if not step_nodes:
                break
            name = (
                str(getattr(event, "name", "")).strip().lower().replace("-", "_").replace(".", "_")
            )
            args = tuple(
                str(arg).strip().lower().replace("-", "_").replace(".", "_")
                for arg in getattr(event, "args", ())
            )
            source_index = min(index, len(step_nodes) - 1)
            result_index = min(index + 1, len(step_nodes) - 1)
            event_node = _node(world_bundle.action, index, value=name)
            _link(event_node, world_bundle.action_roles["source_step"], step_nodes[source_index])
            _link(event_node, world_bundle.action_roles["result_step"], step_nodes[result_index])
            _link(event_node, world_bundle.action_roles["actor"], entity_nodes["character"])
            _link(
                event_node,
                world_bundle.action_roles["arg1"],
                entity_nodes[args[0] if args else ABSENT_ENTITY],
            )
            _link(
                event_node,
                world_bundle.action_roles["arg2"],
                entity_nodes[args[1] if len(args) > 1 else ABSENT_ENTITY],
            )
            _set_truth(event_node, world_bundle.action, True)
            for key, concept in world_bundle.grounded_constraints.items():
                if not key.startswith("goal_completion__"):
                    _set_truth(event_node, concept, grounded_values[index][key])
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
                _set_truth(
                    event_node,
                    concept,
                    (
                        grounded_values[index]["precondition:" + kind]
                        if explicit is not None
                        else satisfied
                    ),
                )
            from .world_facts import precondition_observations

            primitive = precondition_observations(
                name, args, source_state, getattr(prepared_goal, "task_entity_types", ())
            )
            for key, concept in world_bundle.facts.items():
                _set_truth(
                    event_node,
                    concept,
                    (
                        grounded_values[index]["fact:" + key]
                        if explicit is not None
                        else primitive[key]
                    ),
                )
            _materialize_requirements(
                event_node,
                world_bundle.grounded_constraints,
                operation_checks[index],
                f"action:{index}",
            )
            step_nodes[source_index].addChildDataNode(event_node)
        return root
    except Exception as exc:
        raise WorldGraphConfigurationError(
            f"{task_id}: failed to materialize EAI world trajectory: {exc}"
        ) from exc


def _materialize_requirements(parent, concepts, checks, prefix):
    """One observation node per required fact, including unknown observations."""
    from domiknows.generation.grounded import requirement_observations

    for index, (_family, description, values) in enumerate(requirement_observations(checks)):
        node = _node(concepts.requirement, f"{prefix}:requirement:{index}", description)
        _set_truth(node, concepts.requirement, True)
        _set_truth(node, concepts.known, values["requirement_known"])
        _set_truth(node, concepts.holds, values["requirement_holds"])
        for family, concept in concepts.families.items():
            _set_truth(node, concept, values["requirement_family:" + family])
        for name, concept in concepts.inputs.items():
            _set_truth(node, concept, values["requirement_input:" + name])
        parent.addChildDataNode(node)


def _evaluate_events(world_bundle, events, aggregate, grounding):
    """Evaluate actual formulas and their guards; names are report labels only."""
    from .constraint_compiler import compile_world_constraints
    from domiknows.generation.dfa.predicate_monitor import _evaluate

    program = compile_world_constraints(world_bundle, defaults_only=True)
    results, scores = {}, []
    for rule in program.rules:
        applicable = []
        for event in events:
            if event.domain != rule.domain or event.phase != rule.phase:
                continue
            expression = rule.expression
            # A false implication guard is vacuous, not an applicable obligation.
            if expression[0] in {"ifL", "forAllL"} and not _evaluate(expression[1], event.values):
                continue
            applicable.append(_evaluate(expression, event.values))
        score = sum(applicable) / len(applicable) if applicable else 1.0
        results[rule.name] = {
            "satisfied": 100.0 * score,
            "applicable": bool(applicable),
            "evaluation": "declared_formula",
        }
        if applicable:
            scores.append(score)
    if not scores:
        return None
    return WorldConstraintEvaluation(
        score=_aggregate_constraint_scores(scores, aggregate),
        constraint_count=len(scores),
        results=MappingProxyType(results),
        declared_constraint_count=len(program.rules) + len(program.inactive_constraints),
        grounding_report=grounding,
    )


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
    planning_facts: Any = None,
    entity_universe: Sequence[str] = (),
) -> WorldConstraintEvaluation | None:
    """Ground primitive facts and evaluate the declared default formulas."""
    if aggregate not in {"mean", "min", "prod"}:
        raise ValueError(f"Unsupported world-constraint aggregation: {aggregate!r}")
    if not world_bundle.default_constraint_names:
        return None

    from .constraint_compiler import EAIExecutor
    from .world_facts import precondition_observations
    from domiknows.generation.grounded import requirement_events
    from domiknows.generation.dfa.predicate_monitor import PredicateEvent

    identities = set(entity_universe) | set(task_entity_types)
    identities.update(arg for event in events for arg in event.args)
    executor = EAIExecutor(
        world_bundle,
        {"planning_facts": planning_facts, "generation_entity_types": tuple(task_entity_types)},
        identities,
    )
    rows, final, grounding, checks, completion = executor.audit_details(events)
    predicate_events = []
    for index, (event, values) in enumerate(zip(events, rows)):
        if planning_facts is None:
            source = set(map(_canonical_fact, states[index])) if index < len(states) else set()
            values.update(
                {
                    "fact:" + name: truth
                    for name, truth in precondition_observations(
                        event.name, event.args, source, task_entity_types
                    ).items()
                }
            )
        predicate_events.append(PredicateEvent("operation", values))
        predicate_events.extend(
            requirement_events(world_bundle.grounded_constraints, checks[index])
        )
    predicate_events.append(PredicateEvent("episode", final, "completion"))
    predicate_events.extend(requirement_events(world_bundle.grounded_constraints, (completion,)))
    return _evaluate_events(world_bundle, predicate_events, aggregate, grounding)


def verify_world_constraints(
    root: Any,
    world_bundle: EAIWorldGraphBundle,
    aggregate: str = "mean",
) -> WorldConstraintEvaluation | None:
    """Verify declared world constraints and aggregate their satisfaction."""
    constraints = {
        key: constraint
        for key, constraint in getattr(world_bundle.graph, "logicalConstrains", {}).items()
        if getattr(constraint, "headLC", True) and getattr(constraint, "active", True)
    }
    if not constraints:
        return None
    if aggregate not in {"mean", "min", "prod"}:
        raise ValueError(f"Unsupported world-constraint aggregation: {aggregate!r}")
    from .constraint_compiler import world_predicate_bindings
    from domiknows.generation.dfa.predicate_monitor import PredicateEvent

    bindings = world_predicate_bindings(world_bundle)
    parents = {
        "operation": world_bundle.action,
        "episode": world_bundle.trajectory,
        "grounded_requirement": world_bundle.grounded_constraints.requirement,
    }
    nodes, pending = [], [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(node.getChildDataNodes() or ())
    events = []
    for domain, parent in parents.items():
        for node in nodes:
            if node.ontologyNode is not parent:
                continue
            values = {
                b.name: bool(torch.argmax(node.attributes[f"<{b.concept.name}>"]).item())
                for b in bindings
                if b.domain == domain
            }
            for phase in ("immediate", "completion"):
                events.append(PredicateEvent(domain, values, phase))
    defaults = _evaluate_events(
        world_bundle, events, aggregate, root.attributes.get("grounding_report")
    )
    if not world_bundle.has_custom_constraints:
        return defaults
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
        named_results = dict(defaults.results) if defaults else {}
        scores = [row["satisfied"] / 100.0 for row in named_results.values() if row["applicable"]]
        root_attributes = root.getAttributes() or {}
        action_names = set(root_attributes.get("action_names", ()))
        state_predicates = set(root_attributes.get("true_state_predicates", ()))
        for key, constraint in constraints.items():
            if any(constraint is default for default in world_bundle.default_constraints):
                continue
            value = raw_results.get(key, {})
            percent = value.get("satisfied", value.get("ifSatisfied"))
            name = getattr(constraint, "name", None) or key
            if percent is None:
                raise KeyError(f"constraint {key!r} produced no satisfaction value")
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
                for concept in concept_names
                if concept.startswith("action__")
            }
            referenced_states = {
                concept.removeprefix("state__")
                for concept in concept_names
                if concept.startswith("state__")
            }
            if referenced_actions or referenced_states:
                applicable = bool(
                    referenced_actions & action_names or referenced_states & state_predicates
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
            grounding_report=root_attributes.get("grounding_report"),
        )
    except Exception as exc:
        raise WorldGraphConfigurationError(
            f"{task_id}: failed to verify EAI world constraints: {exc}"
        ) from exc


from functools import lru_cache


@lru_cache(maxsize=1)
def _compatibility_precondition_program():
    """Legacy scalar fields are projections of the same five declared formulas."""
    from domiknows.graph import Concept, Graph
    from domiknows.generation.dfa.predicate_monitor import (
        PredicateBinding,
        compile_predicate_constraints,
    )
    from .world_facts import PRECONDITION_FACTS
    from .world_domain import ActionPreconditionSpec
    from ..world_graph import _declare_precondition_constraints

    kinds = (
        "argument_available_in_task",
        "placement_source_ready",
        "release_source_ready",
        "pour_source_ready",
        "destination_open_if_known",
    )
    with Graph("eai_compatibility_preconditions") as graph:
        action = Concept(name="operation")
        facts = {name: action(name="observed__" + name) for name in PRECONDITION_FACTS}
        _declare_precondition_constraints(
            {"operation": action},
            facts,
            tuple(ActionPreconditionSpec(kind, "operation", kind) for kind in kinds),
        )
    bindings = [PredicateBinding(action, "operation", "operation")]
    bindings.extend(PredicateBinding(concept, name, "operation") for name, concept in facts.items())
    return {rule.name: rule for rule in compile_predicate_constraints(graph, bindings).rules}


def _precondition_status(kind, action, args, source_state, task_entity_types=()):
    """Compatibility API; no second Python implementation of the policies."""
    from .world_facts import precondition_observations
    from domiknows.generation.dfa.predicate_monitor import _evaluate

    try:
        rule = _compatibility_precondition_program()[kind]
    except KeyError:
        raise ValueError(f"Unknown EAI action precondition kind: {kind!r}") from None
    values = {
        "operation": True,
        **precondition_observations(action, args, source_state, task_entity_types),
    }
    return _evaluate(rule.expression[1], values), _evaluate(rule.expression, values)
