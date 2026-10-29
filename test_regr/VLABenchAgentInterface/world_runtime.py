"""Materialize observations and verify the declared VLABench world graph.

This adapter owns DataNode construction and solver calls. Domain policies are
read from world_graph.py; this module does not decide which rules are active.
"""
from __future__ import annotations
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import torch
try:
    from .world_domain import (ABSENT_ENTITY, PlanGraphConfigurationError, PlanConstraintEvaluation,
        canonicalize_plan, validate_plan, PlanSemantics, _entity_mapping, _resolve_pointer)
    from .knowledge.task_lcs import knowledge_values
    from .world_graph import VLABenchWorldGraphBundle, WORLD_FACT_DOMAINS
    from .pattern_runtime import pattern_progress_for, trace_progress
except ImportError:
    from world_domain import (ABSENT_ENTITY, PlanGraphConfigurationError, PlanConstraintEvaluation,
        canonicalize_plan, validate_plan, PlanSemantics, _entity_mapping, _resolve_pointer)
    from knowledge.task_lcs import knowledge_values
    from world_graph import VLABenchWorldGraphBundle, WORLD_FACT_DOMAINS
    from pattern_runtime import pattern_progress_for, trace_progress
_VERIFIER_GRAPH = None


def _materialize_requirements(parent, concepts, checks, prefix):
    """Create one node per required fact; LCs decide their conjunction."""
    from domiknows.generation.grounded import requirement_observations
    if getattr(concepts, "requirement", None) is None:
        return
    for index, (family, description, values) in enumerate(requirement_observations(checks)):
        node = _node(concepts.requirement, f"{prefix}:requirement:{index}", value=description)
        _truth(node, concepts.requirement, True)
        _truth(node, concepts.known, values["requirement_known"])
        _truth(node, concepts.holds, values["requirement_holds"])
        for name, concept in concepts.families.items():
            _truth(node, concept, name == family)
        for name, concept in concepts.inputs.items():
            _truth(node, concept, values["requirement_input:" + name])
        parent.addChildDataNode(node)


def _node(concept: Any, instance_id: Any, value: Any = None, attributes: dict | None = None):
    from domiknows.graph import DataNode
    return DataNode(
        instanceID=instance_id,
        instanceValue=None if value is None else str(value),
        ontologyNode=concept,
        attributes=dict(attributes or {}),
    )


def _truth(node: Any, concept: Any, value: bool) -> None:
    node.attributes[f"<{concept.name}>"] = torch.tensor(
        [-30.0, 30.0] if value else [30.0, -30.0], dtype=torch.float32,
    )


def _link(node: Any, role: Any, target: Any) -> None:
    node.addRelationLink(role.name, target)


def _task_concept_rows(concepts):
    return [(name, concept) for concept, name, _phase in concepts.bindings()]


def materialize_plan(
    value: Any,
    entity_table: Mapping[int, str] | Sequence[str],
    bundle: VLABenchWorldGraphBundle,
    *,
    task_name: str | None = None,
    min_operations: int | None = None,
    required_skills: Sequence[str] | None = None,
    task_context: Mapping[str, Any] | None = None,
    policy: Any = None,
):
    """Ground one canonical plan as a DomiKnowS ``DataNode`` hierarchy."""
    try:
        plan = canonicalize_plan(value)
        entities = _entity_mapping(entity_table)
        pointer_names = set(entities.values())
        for operation in plan:
            for key in ("target_entity_name", "target_container_name"):
                if key in operation["params"]:
                    pointer_names.add(_resolve_pointer(operation["params"][key], entities)[0])
        pointer_names.add(ABSENT_ENTITY)

        root = _node(bundle.plan, "plan", value="plan", attributes={"operation_count": len(plan)})
        _truth(root, bundle.plan, True)
        entity_nodes = {
            name: _node(bundle.entity, f"entity:{index}", value=name)
            for index, name in enumerate(sorted(pointer_names))
        }
        for node in entity_nodes.values():
            _truth(node, bundle.entity, True)
            root.addChildDataNode(node)

        try:
            from test_regr.VLABenchAgentInterface.task_policy import task_requirements
            from test_regr.VLABenchAgentInterface.constraint_compiler import requirements_for_policy
        except ImportError:
            from task_policy import task_requirements
            from constraint_compiler import requirements_for_policy
        requirements = requirements_for_policy(policy) if policy is not None else None
        if requirements is None:
            context = dict(task_context or {})
            if task_name is not None:
                context.setdefault("task", task_name)
            requirements = task_requirements(context, bundle, entities)
        # Declared obligations of a complex task are task LCs over the knowledge concepts; only explicit ones remain here.
        if requirements.allowed_sequences is None:
            from dataclasses import replace
            from test_regr.VLABenchAgentInterface.task_policy import request_sequences
            requirements = replace(requirements, allowed_sequences=request_sequences(bundle, max(8,len(plan)), requirements))
        from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
        monitor = find_predicate_monitor(policy) if policy is not None else None
        progress = getattr(getattr(monitor, "adapter", None), "pattern_dfa", None)
        if progress is None:
            progress = pattern_progress_for(bundle, requirements)
        prefix_viable, pattern_complete = trace_progress(progress, (operation["name"] for operation in plan))
        _truth(root, bundle.pattern_complete, pattern_complete)
        from domiknows.generation.grounded import observations, report
        semantics = PlanSemantics(entities, bundle.skill_arguments, requirements.planning,
                                   visible_entities=requirements.visible_entities)
        root.attributes["visibility_report"] = {
            "status": "unavailable" if requirements.visible_entities is None else "supplied",
            "visible_ids": None if requirements.visible_entities is None else sorted(requirements.visible_entities),
        }
        state = semantics.initial_state()
        grounded_operations, grounded_checks = [], []
        operation_predicates, primitive_observations = [], []
        for operation in plan:
            _successor, checks = semantics.grounded_transition(state, operation["name"], operation["params"])
            grounded_operations.append(checks)
            grounded_checks.extend(checks)
            operation_predicates.append(semantics.evaluate_operation(state, operation["name"], operation["params"]))
            primitive_observations.append(semantics.primitive_observations(state, operation["name"], operation["params"]))
            state = semantics.advance(state, operation["name"], operation["params"])
        is_premature_composite = semantics.premature_composite_eos(
            state, min_operations=min_operations, required_skills=required_skills,
        )

        final_check = requirements.planning.completion(state.grounded)
        grounded_checks.append(final_check)
        root.attributes["grounding_report"] = report(grounded_checks)
        for key, value in observations((final_check,)).items():
            _truth(root, bundle.grounded_constraints[key], value)
        _materialize_requirements(root, bundle.grounded_constraints, (final_check,), "plan")
        operation_nodes = []
        task_concepts = getattr(bundle, "task_concepts", None)
        offset = len(requirements.planning.completed_skill_prefix)
        history_skills = list(requirements.planning.completed_skill_prefix)
        for index, operation_value in enumerate(plan):
            name = operation_value["name"]
            params = operation_value["params"]
            operation_node = _node(bundle.operation, index, value=name)
            _truth(operation_node, bundle.operation, True)
            _truth(operation_node, bundle.pattern_first_operation, index == 0)
            _truth(operation_node, bundle.pattern_start_allowed, prefix_viable[index])
            _link(operation_node, bundle.operation_roles["plan"], root)

            for key, role_name in (
                ("target_entity_name", "target_entity"),
                ("target_container_name", "target_container"),
            ):
                if key in params:
                    pointer_name, _ = _resolve_pointer(params[key], entities)
                else:
                    pointer_name = ABSENT_ENTITY
                _link(operation_node, bundle.operation_roles[role_name], entity_nodes[pointer_name])

            for key, value in observations(grounded_operations[index]).items():
                _truth(operation_node, bundle.grounded_constraints[key], value)
            _materialize_requirements(operation_node, bundle.grounded_constraints, grounded_operations[index], f"op:{index}")
            predicates = operation_predicates[index]
            for key, concept in (bundle.facts or {}).items():
                if WORLD_FACT_DOMAINS[key] == "operation":
                    _truth(operation_node, concept, primitive_observations[index][key])

            for skill_name, concept in bundle.skills.items():
                _truth(operation_node, concept, skill_name == name)
            _truth(operation_node, bundle.valid_skill_type, predicates.valid_skill_type)
            _truth(operation_node, bundle.has_target_entity, predicates.has_target_entity)
            _truth(operation_node, bundle.has_target_container, predicates.has_target_container)
            if getattr(bundle, "targets_button", None) is not None:
                _truth(operation_node, bundle.targets_button, predicates.targets_button)
            if getattr(bundle, "is_self_placement", None) is not None:
                _truth(operation_node, bundle.is_self_placement, predicates.is_self_placement)
            if getattr(bundle, "targets_static_fixture", None) is not None:
                _truth(operation_node, bundle.targets_static_fixture, predicates.targets_static_fixture)

            is_premature_at_op = is_premature_composite and (index == len(plan) - 1)
            if getattr(bundle, "premature_composite_eos", None) is not None:
                _truth(operation_node, bundle.premature_composite_eos, is_premature_at_op)

            _truth(operation_node, bundle.task_skill_allowed, requirements.skill_allowed(index + len(requirements.planning.completed_skill_prefix), name))
            _truth(operation_node, bundle.task_target_allowed, requirements.target_allowed(index + len(requirements.planning.completed_skill_prefix), params, entities))
            _truth(operation_node, bundle.task_sequence_complete,
                   index != len(plan) - 1 or requirements.sequence_complete(len(plan) + len(requirements.planning.completed_skill_prefix)))
            history_skills.append(name)
            if task_concepts is not None:
                observed = knowledge_values(
                    task_concepts, requirements.task, requirements.interface, index + offset,
                    explicit=requirements.explicit_sequence, choice=True, final=index == len(plan) - 1,
                    history=history_skills, requirements=requirements, params=params, entities=entities,
                    min_operations=min_operations, required_skills=required_skills)
                for predicate, concept in _task_concept_rows(task_concepts):
                    _truth(operation_node, concept, observed[predicate])
            _truth(operation_node, bundle.valid_pointer, predicates.valid_pointer)
            _truth(operation_node, bundle.valid_operation, predicates.valid_operation)
            root.addChildDataNode(operation_node)
            operation_nodes.append(operation_node)

        for index in range(max(0, len(operation_nodes) - 1)):
            transition = _node(bundle.transition, index, value=index)
            _truth(transition, bundle.transition, True)
            _truth(transition, bundle.allowed_transition, prefix_viable[index + 1])

            predicates = operation_predicates[index + 1]
            for key, concept in (bundle.facts or {}).items():
                if WORLD_FACT_DOMAINS[key] == "transition":
                    _truth(transition, concept, primitive_observations[index + 1][key])
            for predicate_name in (
                "repeated_pick_entity", "inconsistent_container", "unopened_container_placement",
            ):
                concept = getattr(bundle, predicate_name, None)
                if concept is not None:
                    _truth(transition, concept, getattr(predicates, predicate_name))

            _link(transition, bundle.transition_roles["current"], operation_nodes[index])
            _link(transition, bundle.transition_roles["following"], operation_nodes[index + 1])
            root.addChildDataNode(transition)
        return root
    except Exception as exc:
        raise PlanGraphConfigurationError(f"failed to materialize VLABench plan: {exc}") from exc


def verify_plan_constraints(
    root: Any,
    bundle: VLABenchWorldGraphBundle,
    aggregate: str = "min",
) -> PlanConstraintEvaluation | None:
    """Verify all hard plan constraints and aggregate normalized satisfaction."""
    constraints = {
        key: constraint
        for key, constraint in getattr(bundle.graph, "logicalConstrains", {}).items()
        if getattr(constraint, "headLC", True) and getattr(constraint, "active", True)
    }
    if not constraints:
        return None
    if aggregate not in {"mean", "min", "prod"}:
        raise ValueError(f"unsupported constraint aggregation {aggregate!r}")
    try:
        global _VERIFIER_GRAPH
        if _VERIFIER_GRAPH is not bundle.graph:
            from domiknows.solver import ilpOntSolverFactory
            ilpOntSolverFactory.clear()
            _VERIFIER_GRAPH = bundle.graph
        raw = root.verifyResultsLC(key="/local/argmax")
        scores: list[float] = []
        named: dict[str, Any] = {}
        operation_count = int((root.getAttributes() or {}).get("operation_count", 0))
        for key, constraint in constraints.items():
            value = raw.get(key, {})
            percent = value.get("satisfied", value.get("ifSatisfied"))
            name = getattr(constraint, "name", None) or key
            vacuous = value.get("verifyList") == [] or (name in ("transition_matches_subtask_pattern", "no_redundant_repeated_picks", "container_destination_consistency", "open_before_enclosed_placement") and operation_count < 2)
            if percent is None or (isinstance(percent, float) and math.isnan(percent)):
                if vacuous:
                    percent = 100.0
                else:
                    raise KeyError(f"constraint {key!r} produced no satisfaction value")
            named[name] = value
            scores.append(1.0 if vacuous else max(0.0, min(1.0, float(percent) / 100.0)))
        if aggregate == "mean":
            score = sum(scores) / len(scores)
        elif aggregate == "min":
            score = min(scores)
        else:
            score = 1.0
            for item in scores:
                score *= item
        return PlanConstraintEvaluation(score, len(scores), MappingProxyType(named), (root.getAttributes() or {}).get("grounding_report"))
    except Exception as exc:
        raise PlanGraphConfigurationError(f"failed to verify VLABench plan constraints: {exc}") from exc
