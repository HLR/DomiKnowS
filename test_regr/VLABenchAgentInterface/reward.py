"""Planner and simulator rewards for the hierarchical VLABench agent.

The planner score follows VLABench's four published planning components but
recomputes their weighted total.  The upstream helper currently omits the
skill-with-entity and exact-graph terms from ``total_score`` even though it
declares weights for them, so consuming that field directly caps a perfect
plan at 0.8.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import networkx as nx
import torch

try:
    from domiknows.reinforcement.rewards import flatten_generator_output
except ImportError:  # pragma: no cover - only used outside the repository
    def flatten_generator_output(value):
        return value

try:
    from .graph import tokens_to_plan
    from .world_graph import (
        SKILL_ARGUMENTS,
        PlanValidation,
        SUBTASK_PATTERNS,
        canonicalize_plan,
        validate_plan,
    )
    from .planning_metrics import _graph as _official_graph, _ground_plan_entities, _precise_match
except ImportError:
    from graph import tokens_to_plan
    from world_graph import (
        SKILL_ARGUMENTS,
        PlanValidation,
        SUBTASK_PATTERNS,
        canonicalize_plan,
        validate_plan,
    )
    from planning_metrics import _graph as _official_graph, _ground_plan_entities, _precise_match


PLANNER_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "skill_match": 0.40,
    "entity_match": 0.40,
    "skill_with_entity_match": 0.10,
    "exact_graph_match": 0.10,
})


@dataclass(frozen=True)
class PlanRewardBreakdown:
    total: float
    skill_match: float
    entity_match: float
    skill_with_entity_match: float
    exact_graph_match: float
    constraint_score: float | None
    valid: bool
    errors: tuple[str, ...] = ()
    grounding_report: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    success: float
    progress: float
    intention: float
    efficiency: float
    valid: bool
    steps: int
    max_steps: int


def _clamp(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(1.0, value))


def _subtasks(plan: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    result: list[list[Mapping[str, Any]]] = []
    names = [str(operation["name"]) for operation in plan]
    index = 0
    while index < len(plan):
        matched = False
        for pattern in SUBTASK_PATTERNS:
            if tuple(names[index : index + len(pattern)]) == tuple(pattern):
                result.append(list(plan[index : index + len(pattern)]))
                index += len(pattern)
                matched = True
                break
        if not matched:
            index += 1
    return result


def _subtask_signature(subtask: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    name = "-".join(str(operation["name"]) for operation in subtask)
    entities = [
        operation.get("params", {}).get("target_entity_name")
        for operation in subtask
        if operation.get("params", {}).get("target_entity_name") is not None
    ]
    containers = [
        operation.get("params", {}).get("target_container_name")
        for operation in subtask
        if operation.get("params", {}).get("target_container_name") is not None
    ]
    return name, entities[-1] if entities else None, containers[-1] if containers else None


def _build_graph(plan: Sequence[Mapping[str, Any]], dependency: Any) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("START", signature=("START", None, None))
    nodes: list[str] = []
    for index, subtask in enumerate(_subtasks(plan), start=1):
        node = f"subtask_{index}"
        graph.add_node(node, signature=_subtask_signature(subtask))
        nodes.append(node)
    if dependency == "Sequential" or dependency is None:
        previous = "START"
        for node in nodes:
            graph.add_edge(previous, node)
            previous = node
    elif dependency == "Seq-independent":
        for node in nodes:
            graph.add_edge("START", node)
    elif isinstance(dependency, Mapping):
        for source, destinations in dependency.items():
            source_index = int(source) - 1
            for destination in destinations:
                destination_index = int(destination) - 1
                if 0 <= source_index < len(nodes) and 0 <= destination_index < len(nodes):
                    graph.add_edge(nodes[source_index], nodes[destination_index])
        for node in nodes:
            if graph.in_degree(node) == 0:
                graph.add_edge("START", node)
    else:
        raise ValueError(f"unsupported VLABench dependency {dependency!r}")
    return graph


def _layers(graph: nx.DiGraph) -> dict[str, int]:
    result = {"START": 0}
    for node in nx.topological_sort(graph):
        if node == "START":
            continue
        parents = [result[parent] for parent in graph.predecessors(node) if parent in result]
        result[node] = (max(parents) + 1) if parents else 1
    return result


def _exact_graph_match(reference: Sequence[Mapping[str, Any]], predicted: Sequence[Mapping[str, Any]], dependency: Any) -> float:
    """Dependency-graph match in [0, 1], using the official PM matcher.

    This equals ``planning_metrics`` PM / 100 whenever the reference has
    subtasks, so reward-side ``exact_graph_match`` and reported PM agree. A
    prediction whose dependency graph cannot be built scores 0. Empty references
    and unusable reference dependencies keep the historical tolerant behaviour.
    """
    dependency = "Sequential" if dependency is None else dependency
    try:
        reference_graph = _official_graph(reference, dependency)
    except ValueError:
        return _legacy_exact_graph_match(reference, predicted, dependency)
    if len(reference_graph) == 1:
        return _legacy_exact_graph_match(reference, predicted, dependency)
    try:
        predicted_graph = _official_graph(predicted, dependency)
    except ValueError:
        return 0.0
    return _precise_match(reference_graph, predicted_graph) / 100.0


def _legacy_exact_graph_match(reference: Sequence[Mapping[str, Any]], predicted: Sequence[Mapping[str, Any]], dependency: Any) -> float:
    reference_graph = _build_graph(reference, dependency)
    predicted_graph = _build_graph(predicted, dependency)
    total = max(0, len(reference_graph) - 1)
    if total == 0:
        return 1.0 if len(predicted_graph) == 1 else 0.0
    ref_layers = _layers(reference_graph)
    pred_layers = _layers(predicted_graph)
    matched = 0
    used: set[str] = set()
    for ref_node in (node for node in nx.topological_sort(reference_graph) if node != "START"):
        layer = ref_layers[ref_node]
        signature = reference_graph.nodes[ref_node]["signature"]
        ref_parent_signatures = Counter(
            reference_graph.nodes[parent]["signature"]
            for parent in reference_graph.predecessors(ref_node)
        )
        for pred_node in predicted_graph:
            if pred_node == "START" or pred_node in used or pred_layers.get(pred_node) != layer:
                continue
            if predicted_graph.nodes[pred_node]["signature"] != signature:
                continue
            pred_parent_signatures = Counter(
                predicted_graph.nodes[parent]["signature"]
                for parent in predicted_graph.predecessors(pred_node)
            )
            if pred_parent_signatures == ref_parent_signatures:
                used.add(pred_node)
                matched += 1
                break
    return matched / total


def _component_scores(reference: Sequence[Mapping[str, Any]], predicted: Sequence[Mapping[str, Any]], dependency: Any) -> dict[str, float]:
    reference_skills = Counter(operation["name"] for operation in reference)
    predicted_skills = Counter(operation["name"] for operation in predicted)
    skill_match = sum((reference_skills & predicted_skills).values()) / len(reference) if reference else 0.0

    def entities(plan):
        result = []
        for operation in plan:
            params = operation.get("params", {})
            for role in ("target_entity_name", "target_container_name"):
                if params.get(role) is not None:
                    result.append((role, params[role]))
        return result

    reference_entities = Counter(entities(reference))
    predicted_entities = Counter(entities(predicted))
    entity_match = (
        sum((reference_entities & predicted_entities).values()) / sum(reference_entities.values())
        if reference_entities else 0.0
    )
    reference_joint = Counter(
        (
            operation["name"],
            operation.get("params", {}).get("target_entity_name"),
            operation.get("params", {}).get("target_container_name"),
        )
        for operation in reference
    )
    predicted_joint = Counter(
        (
            operation["name"],
            operation.get("params", {}).get("target_entity_name"),
            operation.get("params", {}).get("target_container_name"),
        )
        for operation in predicted
    )
    joint_match = sum((reference_joint & predicted_joint).values()) / len(reference) if reference else 0.0
    return {
        "skill_match": _clamp(skill_match),
        "entity_match": _clamp(entity_match),
        "skill_with_entity_match": _clamp(joint_match),
        "exact_graph_match": _clamp(_exact_graph_match(reference, predicted, dependency)),
    }


def _decode_output(generator_output: Any, vocabulary: Any = None) -> Any:
    if isinstance(generator_output, (str, Mapping)):
        return generator_output
    if isinstance(generator_output, Sequence) and generator_output and isinstance(generator_output[0], Mapping):
        return generator_output
    flattened = flatten_generator_output(generator_output)
    if torch.is_tensor(flattened):
        flattened = flattened.detach().cpu().reshape(-1).tolist()
    if not isinstance(flattened, Sequence) or isinstance(flattened, (str, bytes, bytearray)):
        return flattened
    tokens: list[str] = []
    for value in flattened:
        if isinstance(value, str):
            tokens.append(value)
        elif vocabulary is not None:
            tokens.append(vocabulary.token_for_label(int(value)))
        else:
            tokens.append(str(value))
    return tokens_to_plan(tokens)


def score_vlabench_plan(
    prediction: Any,
    reference: Any,
    dependency: Any = "Sequential",
    *,
    entity_table: Mapping[int, str] | Sequence[str] | None = None,
    mode: str = "dense",
    world_bundle: Any = None,
    task_name: str | None = None,
    task_context: Mapping[str, Any] | None = None,
    policy: Any = None,
) -> PlanRewardBreakdown:
    """Score a plan and hard-gate it through schema and graph constraints."""
    if mode not in {"dense", "binary"}:
        raise ValueError("planner reward mode must be 'dense' or 'binary'")
    try:
        reference_plan = canonicalize_plan(reference)
    except Exception as exc:
        raise ValueError(f"invalid reference VLABench plan: {exc}") from exc
    skill_arguments = getattr(world_bundle, "skill_arguments", SKILL_ARGUMENTS)
    requirements = None
    if policy is None and world_bundle is not None and task_context is not None:
        from dataclasses import replace
        from test_regr.VLABenchAgentInterface.task_policy import task_requirements, request_sequences
        requirements = task_requirements(task_context, world_bundle, entity_table or ())
        requirements = replace(requirements, allowed_sequences=request_sequences(world_bundle, max(8, len(reference_plan)), requirements))
    validation: PlanValidation = validate_plan(
        prediction,
        entity_table=entity_table,
        skill_arguments=skill_arguments,
        policy=policy,
        requirements=requirements,
    )
    predicted_plan = list(validation.canonical_plan)
    if entity_table:
        grounded_reference = _ground_plan_entities(reference_plan, entity_table)
        grounded_predicted = _ground_plan_entities(predicted_plan, entity_table)
    else:
        grounded_reference = reference_plan
        grounded_predicted = predicted_plan
    scores = _component_scores(grounded_reference, grounded_predicted, dependency) if predicted_plan else {
        key: 0.0 for key in PLANNER_WEIGHTS
    }

    constraint_score: float | None = None
    grounding = None
    errors = list(validation.errors)
    if world_bundle is not None and validation.valid:
        try:
            try:
                from .world_graph import materialize_plan, verify_plan_constraints
            except ImportError:
                from world_graph import materialize_plan, verify_plan_constraints
            root = materialize_plan(
                predicted_plan,
                entity_table or (),
                world_bundle,
                task_name=task_name,
                task_context=task_context,
                policy=policy,
            )
            evaluation = verify_plan_constraints(root, world_bundle)
            constraint_score = None if evaluation is None else evaluation.score
            grounding = root.attributes.get("grounding_report")
            if constraint_score is not None and constraint_score < 1.0:
                errors.append(f"DomiKnowS plan constraints scored {constraint_score:.6f}")
        except Exception as exc:
            constraint_score = 0.0
            errors.append(f"DomiKnowS constraint verification failed: {exc}")

    valid = validation.valid and (constraint_score is None or constraint_score >= 1.0)
    dense = sum(PLANNER_WEIGHTS[key] * scores[key] for key in PLANNER_WEIGHTS)
    total = float(all(value >= 1.0 for value in scores.values())) if mode == "binary" else dense
    if not valid:
        total = 0.0
    return PlanRewardBreakdown(
        total=_clamp(total),
        constraint_score=constraint_score,
        valid=valid,
        errors=tuple(errors),
        grounding_report=grounding,
        **scores,
    )


def make_vlabench_reward_function(
    sample: Mapping[str, Any],
    *,
    vocabulary: Any = None,
    mode: str = "dense",
    world_bundle: Any = None,
):
    """Create the per-item closure expected by DomiKnowS ReinforcementProgram."""
    reference = sample.get("operation_sequence", sample.get("skill_sequence", sample.get("plan", sample.get("output"))))
    dependency = sample.get("dependency", "Sequential")
    entity_table = sample.get("entity_table", sample.get("entities"))

    def reward(generator_output: Any, **context) -> torch.Tensor:
        item = context.get("data_item") or sample
        current_reference = item.get("operation_sequence", item.get("skill_sequence", item.get("plan", reference)))
        decoded = _decode_output(generator_output, vocabulary)
        task_name = item.get("task") or item.get("task_name") or sample.get("task") or sample.get("task_name")
        breakdown = score_vlabench_plan(
            decoded,
            current_reference,
            item.get("dependency", dependency),
            entity_table=item.get("entity_table", item.get("entities", entity_table)),
            mode=mode,
            world_bundle=world_bundle,
            task_name=task_name,
            task_context=item,
            policy=context.get("policy"),
        )
        reward.last_breakdown = breakdown
        return torch.tensor([breakdown.total], dtype=torch.float32)

    reward.mode = mode
    reward.task_id = str(sample.get("task_id", sample.get("episode_id", "")))
    reward.last_breakdown = None
    return reward
