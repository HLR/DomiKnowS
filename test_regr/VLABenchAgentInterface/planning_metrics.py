"""Official VLABench planning metrics, independent of training reward gates.

Benchmark patterns/matching follow OpenMOSS/VLABench evaluation/utils.py at
cf588fe60c0c7282174fe979f5913170cfe69017. The full score fixes that helper's
omission of SPR and PM; legacy_score retains its two-term total.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx

try:
    from .world_graph import canonicalize_plan
except ImportError:
    from world_graph import canonicalize_plan


DIMENSIONS = ("M&T", "Spatial", "ComSense", "Semantic", "PhysLaw", "Complex")
COMPONENTS = ("SR", "PR", "SPR", "PM", "score", "legacy_score")
BENCHMARK_PATTERNS = (
    ("pick", "place"), ("pick", "insert"), ("pick", "pour", "place"),
    ("pick", "pour"), ("pick", "pull"), ("pick", "lift"),
    ("pick", "push", "pull"), ("pick", "push", "place"),
    ("pick", "push"), ("pick", "open_door"), ("press",),
)
_CONFIG = json.loads(Path(__file__).with_name("planning_benchmark.json").read_text(encoding="utf-8"))
_LABELS = {"CommonSense": "ComSense", "PhysicsLaw": "PhysLaw"}
TASK_DIMENSIONS = {
    task: _LABELS.get(dimension, dimension)
    for mapping in (_CONFIG["dim2task"], _CONFIG.get("dim2task_extensions", {}))
    for dimension, tasks in mapping.items() for task in tasks
}
INDEPENDENT_TASKS = frozenset(_CONFIG["seq_independent_task"])


def planning_metadata(episode_id: str) -> tuple[str, str | None]:
    """Resolve a task directory, allowing optional dimension directories."""
    parts = episode_id.replace("\\", "/").split("/")[:-1]
    task = next((part for part in reversed(parts) if part in TASK_DIMENSIONS), parts[-1] if parts else "")
    return task, TASK_DIMENSIONS.get(task)


def default_dependency(task: str) -> str:
    return "Seq-independent" if task in INDEPENDENT_TASKS else "Sequential"


def _read_plan(value: Any) -> list[dict[str, Any]]:
    plan = canonicalize_plan(value)
    for operation in plan:
        for role in ("target_entity_name", "target_container_name"):
            argument = operation["params"].get(role)
            if argument is not None and (isinstance(argument, bool) or not isinstance(argument, (str, int))):
                raise ValueError(f"{role} must be a string or integer")
    return plan


def _graph(plan, dependency) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, signature=("START", None, None))
    names = [operation["name"] for operation in plan]
    index = 0
    while index < len(plan):
        for pattern in BENCHMARK_PATTERNS:
            if tuple(names[index:index + len(pattern)]) == pattern:
                subtask = plan[index:index + len(pattern)]
                arguments = [
                    [operation["params"].get(role) for operation in subtask if operation["params"].get(role) is not None]
                    for role in ("target_entity_name", "target_container_name")
                ]
                graph.add_node(len(graph), signature=("-".join(pattern), *(values[-1] if values else None for values in arguments)))
                index += len(pattern)
                break
        else:
            index += 1
    nodes = list(graph)[1:]
    if dependency == "Sequential":
        graph.add_edges_from(zip([0] + nodes, nodes))
    elif dependency == "Seq-independent":
        graph.add_edges_from((0, node) for node in nodes)
    elif isinstance(dependency, Mapping):
        for source, destinations in dependency.items():
            try:
                source_index = int(source)
                if str(source_index) != str(source) or isinstance(source, bool):
                    raise ValueError("dependency indices must be integers")
                if not isinstance(destinations, (list, tuple)):
                    raise ValueError("dependency destinations must be an array")
                for destination in destinations:
                    destination_index = int(destination)
                    if str(destination_index) != str(destination) or isinstance(destination, bool):
                        raise ValueError("dependency indices must be integers")
                    if not (1 <= source_index <= len(nodes) and 1 <= destination_index <= len(nodes)):
                        raise ValueError("dependency index is out of range")
                    graph.add_edge(source_index, destination_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid dependency graph: {exc}") from exc
        graph.add_edges_from((0, node) for node in nodes if graph.in_degree(node) == 0)
    else:
        raise ValueError(f"unsupported dependency {dependency!r}")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("invalid dependency graph: cycle detected")
    return graph


def _layers(graph):
    layers = {0: 0}
    for node in nx.topological_sort(graph):
        if node:
            layers[node] = max(layers[parent] for parent in graph.predecessors(node)) + 1
    return layers


def _precise_match(reference_graph, predicted_graph) -> float:
    """Mirror upstream layer and predecessor-signature matching with node reuse forbidden."""
    if len(reference_graph) == 1:
        return 0.0
    reference_layers, predicted_layers = _layers(reference_graph), _layers(predicted_graph)
    available = set(predicted_graph) - {0}
    matched = 0
    for node in nx.topological_sort(reference_graph):
        if not node:
            continue
        for candidate in sorted(available):
            if reference_layers[node] != predicted_layers[candidate]:
                continue
            if reference_graph.nodes[node]["signature"] != predicted_graph.nodes[candidate]["signature"]:
                continue
            parents = list(reference_graph.predecessors(node))
            candidates = list(predicted_graph.predecessors(candidate))
            if len(parents) == len(candidates) and all(
                any(reference_graph.nodes[parent]["signature"] == predicted_graph.nodes[other]["signature"] for other in candidates)
                for parent in parents
            ):
                matched += 1
                available.remove(candidate)
                break
    return 100.0 * matched / (len(reference_graph) - 1)


def _recall(reference, prediction) -> float:
    reference_counts = Counter(reference)
    denominator = sum(reference_counts.values())
    return 100.0 * sum((reference_counts & Counter(prediction)).values()) / denominator if denominator else 0.0


def _ground_plan_entities(plan: Sequence[Mapping[str, Any]], entity_table: Any) -> list[dict[str, Any]]:
    if not entity_table:
        return [dict(op) for op in plan]
    mapping: dict[int, str] = {}
    if isinstance(entity_table, Mapping):
        mapping = {int(k): str(v) for k, v in entity_table.items() if str(k).isdigit() or isinstance(k, int)}
    elif isinstance(entity_table, Sequence):
        mapping = {i: str(v) for i, v in enumerate(entity_table)}

    grounded_plan = []
    for op in plan:
        params = dict(op.get("params", {}))
        for role in ("target_entity_name", "target_container_name"):
            val = params.get(role)
            if val is not None:
                val_str = str(val).strip()
                if val_str.isdigit() and int(val_str) in mapping:
                    params[role] = mapping[int(val_str)]
                elif isinstance(val, int) and val in mapping:
                    params[role] = mapping[val]
        grounded_plan.append({"name": op.get("name"), "params": params})
    return grounded_plan


def score_planning_metrics(
    prediction: Any,
    reference: Any,
    dependency: Any = "Sequential",
    entity_table: Any = None,
) -> dict[str, float]:
    """Score readable plans without applying schema, grounding, or world validity gates."""
    try:
        reference_plan = _read_plan(reference)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid reference VLABench plan: {exc}") from exc
    if entity_table:
        reference_plan = _ground_plan_entities(reference_plan, entity_table)
    reference_graph = _graph(reference_plan, dependency)

    try:
        predicted_plan = _read_plan(prediction)
    except (TypeError, ValueError):
        return dict.fromkeys(COMPONENTS, 0.0)
    if not predicted_plan:
        return dict.fromkeys(COMPONENTS, 0.0)

    if entity_table:
        predicted_plan = _ground_plan_entities(predicted_plan, entity_table)

    def skills(plan):
        return [operation["name"] for operation in plan]

    def parameters(plan):
        return [(role, operation["params"][role]) for operation in plan
                for role in ("target_entity_name", "target_container_name")
                if operation["params"].get(role) is not None]

    def pairs(plan):
        return [(operation["name"], operation["params"].get("target_entity_name"), operation["params"].get("target_container_name")) for operation in plan]

    sr = _recall(skills(reference_plan), skills(predicted_plan))
    pr = _recall(parameters(reference_plan), parameters(predicted_plan))
    spr = _recall(pairs(reference_plan), pairs(predicted_plan))
    try:
        pm = _precise_match(reference_graph, _graph(predicted_plan, dependency))
    except ValueError:
        # A short prediction cannot materialize the reference's indexed dependencies.
        pm = 0.0
    legacy = 0.4 * sr + 0.4 * pr
    return {"SR": sr, "PR": pr, "SPR": spr, "PM": pm,
            "score": legacy + 0.1 * spr + 0.1 * pm, "legacy_score": legacy}


class PlanningMetrics:
    """Accumulate prompt means, dimension means, and six-dimension macro means."""

    def __init__(self):
        self.count = 0
        self.unknown_tasks = 0
        self.totals = dict.fromkeys(COMPONENTS, 0.0)
        self.dimensions = {dimension: {"examples": 0, **dict.fromkeys(COMPONENTS, 0.0)} for dimension in DIMENSIONS}

    def update(self, scores: Mapping[str, float], *, task: str = "", dimension: str | None = None):
        dimension = _LABELS.get(dimension, dimension) if dimension else TASK_DIMENSIONS.get(task)
        self.count += 1
        if task not in TASK_DIMENSIONS:
            self.unknown_tasks += 1
        for key in COMPONENTS:
            self.totals[key] += scores[key]
        # Unknown tasks never enter dimension summaries, even with a dimension hint.
        if task in TASK_DIMENSIONS and dimension in self.dimensions:
            item = self.dimensions[dimension]
            item["examples"] += 1
            for key in COMPONENTS:
                item[key] += scores[key]

    def result(self) -> dict[str, Any]:
        dimensions = {
            dimension: {"examples": item["examples"], **{
                key: item[key] / item["examples"] if item["examples"] else None for key in COMPONENTS
            }} for dimension, item in self.dimensions.items()
        }
        complete = all(item["examples"] for item in dimensions.values())
        return {
            "examples": self.count, "unknown_task_count": self.unknown_tasks,
            "overall": {key: value / self.count if self.count else None for key, value in self.totals.items()},
            "dimensions": dimensions,
            "Macro": sum(item["score"] for item in dimensions.values()) / 6 if complete else None,
            "legacy_Macro": sum(item["legacy_score"] for item in dimensions.values()) / 6 if complete else None,
            "source_revision": _CONFIG["source_revision"],
            "dimension_mapping_revision": _CONFIG["dimension_mapping_revision"],
        }
