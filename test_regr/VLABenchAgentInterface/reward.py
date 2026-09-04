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
except ImportError:
    from graph import tokens_to_plan
    from world_graph import (
        SKILL_ARGUMENTS,
        PlanValidation,
        SUBTASK_PATTERNS,
        canonicalize_plan,
        validate_plan,
    )


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
) -> PlanRewardBreakdown:
    """Score a plan and hard-gate it through schema and graph constraints."""
    if mode not in {"dense", "binary"}:
        raise ValueError("planner reward mode must be 'dense' or 'binary'")
    try:
        reference_plan = canonicalize_plan(reference)
    except Exception as exc:
        raise ValueError(f"invalid reference VLABench plan: {exc}") from exc
    skill_arguments = getattr(world_bundle, "skill_arguments", SKILL_ARGUMENTS)
    validation: PlanValidation = validate_plan(
        prediction,
        entity_table=entity_table,
        skill_arguments=skill_arguments,
    )
    predicted_plan = list(validation.canonical_plan)
    scores = _component_scores(reference_plan, predicted_plan, dependency) if predicted_plan else {
        key: 0.0 for key in PLANNER_WEIGHTS
    }

    constraint_score: float | None = None
    errors = list(validation.errors)
    if world_bundle is not None and predicted_plan:
        try:
            try:
                from .world_graph import materialize_plan, verify_plan_constraints
            except ImportError:
                from world_graph import materialize_plan, verify_plan_constraints
            root = materialize_plan(predicted_plan, entity_table or (), world_bundle)
            evaluation = verify_plan_constraints(root, world_bundle)
            constraint_score = None if evaluation is None else evaluation.score
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
        breakdown = score_vlabench_plan(
            decoded,
            current_reference,
            item.get("dependency", dependency),
            entity_table=item.get("entity_table", item.get("entities", entity_table)),
            mode=mode,
            world_bundle=world_bundle,
        )
        reward.last_breakdown = breakdown
        return torch.tensor([breakdown.total], dtype=torch.float32)

    reward.mode = mode
    reward.task_id = str(sample.get("task_id", sample.get("episode_id", "")))
    reward.last_breakdown = None
    return reward


class RolloutRewardAccumulator:
    """Compute the bounded terminal reward from VLABench environment signals."""

    def __init__(self, max_steps: int):
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.progress = 0.0
        self.intention = 0.0
        self.valid = True

    def update(self, *, progress: Any = None, intention: Any = None, valid: bool = True, steps: int = 1) -> None:
        self.steps = min(self.max_steps, self.steps + max(0, int(steps)))
        if progress is not None:
            self.progress = _clamp(progress)
        if intention is not None:
            self.intention = _clamp(intention)
        self.valid = self.valid and bool(valid)

    def finalize(
        self,
        success: Any,
        *,
        progress: Any = None,
        intention: Any = None,
        steps: int | None = None,
        valid: bool | None = None,
        mode: str = "composite",
    ) -> RewardBreakdown:
        if mode not in {"composite", "success"}:
            raise ValueError("rollout reward mode must be 'composite' or 'success'")
        final_steps = self.steps if steps is None else max(0, min(self.max_steps, int(steps)))
        final_progress = self.progress if progress is None else _clamp(progress)
        final_intention = self.intention if intention is None else _clamp(intention)
        final_valid = self.valid if valid is None else self.valid and bool(valid)
        final_success = _clamp(success)
        efficiency = final_success * _clamp(1.0 - final_steps / self.max_steps)
        if mode == "success":
            total = final_success
        else:
            total = (
                0.60 * final_success
                + 0.25 * final_progress
                + 0.10 * final_intention
                + 0.05 * efficiency
            )
        if not final_valid:
            total = 0.0
        return RewardBreakdown(
            total=_clamp(total),
            success=final_success,
            progress=final_progress,
            intention=final_intention,
            efficiency=efficiency,
            valid=final_valid,
            steps=final_steps,
            max_steps=self.max_steps,
        )
