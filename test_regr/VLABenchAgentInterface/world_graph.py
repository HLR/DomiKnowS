"""DomiKnowS semantic graph for grounded VLABench operation plans.

As in ``EmbodiedAgentInterface/world_graph.py``, this graph is independent of
the token-generation graph.  Planner JSON is materialized as deterministic
``DataNode`` instances, then hard schema/grounding constraints are verified by
the ordinary DomiKnowS logical-constraint machinery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

EOS_TOKEN = "<eos>"
ABSENT_ENTITY = "__absent_plan_argument__"
_BUILD_IDS = count()
_VERIFIER_GRAPH: Any = None

# Domain declarations live here.  All codecs, generation vocabularies, data
# validation, rewards, and policies derive from the world graph bundle.
SUBTASK_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("pick", "pour", "place"),
    ("pick", "push", "place"),
    ("pick", "push", "pull"),
    ("pick", "place"),
    ("pick", "insert"),
    ("pick", "pour"),
    ("pick", "pull"),
    ("pick", "lift"),
    ("pick", "push"),
    ("pick", "open_door"),
    ("press",),
)

PRIMITIVE_TASK_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "add_condiment": ("pick", "pour"),
    "insert_flower": ("pick", "insert"),
    "select_book": ("pick", "place"),
    "select_chemistry_tube": ("pick", "place"),
    "select_drink": ("pick", "place"),
    "select_fruit": ("pick", "place"),
    "select_mahjong": ("pick", "place"),
    "select_painting": ("pick", "place"),
    "select_poker": ("pick", "lift"),
    "select_toy": ("pick", "place"),
})

SKILL_ARGUMENTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "pick": ("target_entity_name",),
    "place": ("target_container_name",),
    "insert": ("target_container_name",),
    "pour": ("target_container_name",),
    "push": (),
    "pull": (),
    "lift": (),
    "open_door": ("target_entity_name",),
    "press": ("target_entity_name",),
})

# Index zero is deliberately reserved for "no active graph operation".  The
# controller vocabulary is graph-owned so dataset loading and online rollout
# cannot silently assign different meanings to the same embedding row.
CONTROLLER_SKILLS: tuple[str, ...] = ("<none>", *SKILL_ARGUMENTS)


def controller_skill_index(skill: str | None) -> int:
    """Return the stable graph-derived controller index for ``skill``."""

    if skill is None:
        return 0
    try:
        return CONTROLLER_SKILLS.index(str(skill))
    except ValueError:
        return 0


def controller_plan_context(
    plan: Sequence[Mapping[str, Any]],
    operation_index: int,
    entities: Sequence[Any] = (),
) -> tuple[int, int, int]:
    """Encode the active graph operation as skill/entity/position indices.

    Entity index zero means absent or unresolved; graph pointers are shifted by
    one so the embedding has an explicit padding row.  The final coordinate is
    likewise one-based, leaving the all-zero tuple as the backwards-compatible
    context for controller-only datasets without a known operation.
    """

    if not plan:
        return (0, 0, 0)
    index = min(max(0, int(operation_index)), len(plan) - 1)
    operation = plan[index]
    entity_value = None
    parameters = operation.get("parameters", operation.get("params", {}))
    for role in ("target_entity_name", "target_container_name"):
        if role in parameters:
            entity_value = parameters[role]
            break
    entity_index = 0
    if isinstance(entity_value, int) and 0 <= entity_value < len(entities):
        entity_index = entity_value + 1
    elif entity_value is not None:
        for pointer, entity in enumerate(entities):
            if isinstance(entity, str):
                name = entity
            else:
                name = entity.get("name") if isinstance(entity, Mapping) else getattr(entity, "name", None)
            if str(name) == str(entity_value):
                entity_index = pointer + 1
                break
    return (
        controller_skill_index(operation.get("name")),
        entity_index,
        index + 1,
    )


class PlanSchemaError(ValueError):
    """A planner output cannot be converted to the graph-owned plan domain."""


@dataclass(frozen=True)
class PlanValidation:
    valid: bool
    errors: tuple[str, ...]
    canonical_plan: tuple[dict[str, Any], ...]


def condition_index_for_pattern(pattern: Sequence[str]) -> int:
    try:
        return SUBTASK_PATTERNS.index(tuple(pattern))
    except ValueError:
        return 0


def condition_index_for_task(task: str) -> int:
    normalized = str(task).strip().lower().replace("-", "_")
    for name, pattern in PRIMITIVE_TASK_PATTERNS.items():
        if normalized == name or normalized.startswith(name + "_"):
            return condition_index_for_pattern(pattern)
    return 0


def _json_payload(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("[")
    end = text.rfind("]")
    return text[start : end + 1] if start >= 0 and end >= start else text


def canonicalize_plan(value: Any) -> list[dict[str, Any]]:
    """Parse wire JSON and normalize it into the graph-owned plan shape."""
    if isinstance(value, str):
        try:
            value = json.loads(_json_payload(value))
        except json.JSONDecodeError as exc:
            raise PlanSchemaError(f"invalid plan JSON: {exc.msg}") from exc
    if isinstance(value, Mapping):
        if "operation_sequence" in value:
            value = value["operation_sequence"]
        elif "skill_sequence" in value:
            value = value["skill_sequence"]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise PlanSchemaError("plan must be a JSON array of operations")
    plan = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise PlanSchemaError(f"operation {index} must be an object")
        name = str(raw.get("name", "")).strip().lower().replace("-", "_")
        params = raw.get("params", {})
        if not name:
            raise PlanSchemaError(f"operation {index} has no name")
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise PlanSchemaError(f"operation {index} params must be an object")
        plan.append({"name": name, "params": dict(params)})
    return plan


def split_subtasks(
    skills: Sequence[str],
    patterns: Sequence[Sequence[str]] = SUBTASK_PATTERNS,
) -> tuple[tuple[str, ...], ...] | None:
    """Return a complete decomposition into graph-declared task patterns."""
    memo: dict[int, tuple[tuple[str, ...], ...] | None] = {}

    def visit(index: int):
        if index == len(skills):
            return ()
        if index in memo:
            return memo[index]
        for raw_pattern in patterns:
            pattern = tuple(raw_pattern)
            if tuple(skills[index : index + len(pattern)]) == pattern:
                rest = visit(index + len(pattern))
                if rest is not None:
                    memo[index] = (pattern, *rest)
                    return memo[index]
        memo[index] = None
        return None

    return visit(0)


def _known_entity(value: Any, entity_table: Mapping[int, str] | Sequence[str] | None) -> bool:
    if value is None or value == "":
        return False
    if entity_table is None:
        return True
    if isinstance(entity_table, Mapping):
        ids = set(entity_table)
        names = {str(name) for name in entity_table.values()}
    else:
        ids = set(range(len(entity_table)))
        names = {str(name) for name in entity_table}
    if isinstance(value, int):
        return value in ids
    text = str(value)
    return text in names or (text.isdigit() and int(text) in ids)


def validate_plan(
    value: Any,
    *,
    entity_table: Mapping[int, str] | Sequence[str] | None = None,
    skill_arguments: Mapping[str, Sequence[str]] = SKILL_ARGUMENTS,
    patterns: Sequence[Sequence[str]] = SUBTASK_PATTERNS,
    require_pattern: bool = True,
) -> PlanValidation:
    errors: list[str] = []
    try:
        plan = canonicalize_plan(value)
    except PlanSchemaError as exc:
        return PlanValidation(False, (str(exc),), ())
    if not plan:
        errors.append("plan is empty")
    for index, operation in enumerate(plan):
        name = operation["name"]
        params = operation["params"]
        if name not in skill_arguments:
            errors.append(f"operation {index} uses unknown skill {name!r}")
            continue
        required = set(skill_arguments[name])
        semantic_keys = {"target_entity_name", "target_container_name"}
        for key in required:
            if key not in params:
                errors.append(f"operation {index} ({name}) requires {key}")
            elif not _known_entity(params[key], entity_table):
                errors.append(f"operation {index} ({name}) has invalid {key}={params[key]!r}")
        for key in semantic_keys - required:
            if key in params:
                errors.append(f"operation {index} ({name}) forbids {key}")
    if require_pattern and plan and split_subtasks([op["name"] for op in plan], patterns) is None:
        errors.append("skill sequence is not a complete VLABench subtask pattern")
    return PlanValidation(not errors, tuple(errors), tuple(plan))


def _domain_checksum() -> str:
    payload = {
        "version": 2,
        "skills": {name: list(arguments) for name, arguments in SKILL_ARGUMENTS.items()},
        "patterns": [list(pattern) for pattern in SUBTASK_PATTERNS],
        "tasks": {name: list(pattern) for name, pattern in PRIMITIVE_TASK_PATTERNS.items()},
        "roles": ["target_entity_name", "target_container_name"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


VLABENCH_DOMAIN_CHECKSUM = _domain_checksum()


@dataclass(frozen=True)
class VLABenchWorldGraphBundle:
    graph: Any
    plan: Any
    operation: Any
    entity: Any
    transition: Any
    contains_operation: Any
    contains_entity: Any
    contains_transition: Any
    operation_roles: Mapping[str, Any]
    transition_roles: Mapping[str, Any]
    skills: Mapping[str, Any]
    skill_arguments: Mapping[str, tuple[str, ...]]
    subtask_patterns: tuple[tuple[str, ...], ...]
    primitive_task_patterns: Mapping[str, tuple[str, ...]]
    domain_checksum: str
    has_target_entity: Any
    has_target_container: Any
    valid_skill_type: Any
    valid_pointer: Any
    valid_operation: Any
    allowed_transition: Any

    @property
    def has_constraints(self) -> bool:
        return any(
            getattr(constraint, "headLC", True)
            for constraint in getattr(self.graph, "logicalConstrains", {}).values()
        )


@dataclass(frozen=True)
class PlanConstraintEvaluation:
    score: float
    constraint_count: int
    results: Mapping[str, Any]


class PlanGraphConfigurationError(RuntimeError):
    """A VLABench plan graph could not be built or verified."""


def build_vlabench_world_graph(
    graph_name: str = "vlabench_plan_world",
    constraint_builders: Iterable[Callable[[VLABenchWorldGraphBundle], None]] = (),
    *,
    include_default_constraints: bool = True,
    semantic_parents: Mapping[str, Any] | None = None,
) -> VLABenchWorldGraphBundle:
    """Define the DomiKnowS plan ontology and its hard logical constraints."""
    from domiknows.graph import Concept, Graph

    build_id = next(_BUILD_IDS)
    role_prefix = f"__vlabench_plan_{build_id}__"

    def role(name: str) -> str:
        return f"{role_prefix}{name}"

    semantic_parents = dict(semantic_parents or {})
    with Graph(graph_name) as graph:
        episode_parent = semantic_parents.get("episode")
        entity_parent = semantic_parents.get("entity")
        operation_parent = semantic_parents.get("operation")
        plan = (
            episode_parent(name="vlabench_plan")
            if episode_parent is not None else Concept(name="vlabench_plan")
        )
        operation = (
            operation_parent(name="vlabench_operation")
            if operation_parent is not None else Concept(name="vlabench_operation")
        )
        entity = (
            entity_parent(name="vlabench_entity")
            if entity_parent is not None else Concept(name="vlabench_entity")
        )
        transition = Concept(name="vlabench_operation_transition")

        contains_operation = plan.contains(operation)[0]
        contains_entity = plan.contains(entity)[0]
        contains_transition = plan.contains(transition)[0]

        operation_plan, target_entity, target_container = operation.has_a(**{
            role("operation_plan"): plan,
            role("target_entity"): entity,
            role("target_container"): entity,
        })
        current_operation, following_operation = transition.has_a(**{
            role("current_operation"): operation,
            role("following_operation"): operation,
        })

        skills = {
            name: operation(name=f"skill__{name}")
            for name in sorted(SKILL_ARGUMENTS)
        }
        has_target_entity = operation(name="operation__has_target_entity")
        has_target_container = operation(name="operation__has_target_container")
        valid_skill_type = operation(name="operation__valid_skill_type")
        valid_pointer = operation(name="operation__valid_pointer")
        valid_operation = operation(name="operation__valid_schema")
        allowed_transition = transition(name="transition__allowed_pattern")

        bundle = VLABenchWorldGraphBundle(
            graph=graph,
            plan=plan,
            operation=operation,
            entity=entity,
            transition=transition,
            contains_operation=contains_operation,
            contains_entity=contains_entity,
            contains_transition=contains_transition,
            operation_roles=MappingProxyType({
                "plan": operation_plan,
                "target_entity": target_entity,
                "target_container": target_container,
            }),
            transition_roles=MappingProxyType({
                "current": current_operation,
                "following": following_operation,
            }),
            skills=MappingProxyType(skills),
            skill_arguments=SKILL_ARGUMENTS,
            subtask_patterns=SUBTASK_PATTERNS,
            primitive_task_patterns=PRIMITIVE_TASK_PATTERNS,
            domain_checksum=VLABENCH_DOMAIN_CHECKSUM,
            has_target_entity=has_target_entity,
            has_target_container=has_target_container,
            valid_skill_type=valid_skill_type,
            valid_pointer=valid_pointer,
            valid_operation=valid_operation,
            allowed_transition=allowed_transition,
        )

        if include_default_constraints:
            from domiknows.graph.logicalConstrain import ifL, notL

            ifL(operation("op"), valid_skill_type("op"), name="operation_exactly_one_skill")
            ifL(operation("op"), valid_operation("op"), name="operation_schema_is_valid")
            ifL(operation("op"), valid_pointer("op"), name="operation_pointer_is_valid")
            ifL(transition("edge"), allowed_transition("edge"), name="transition_matches_subtask_pattern")

            for name, required in SKILL_ARGUMENTS.items():
                required = set(required)
                for key, flag in (
                    ("target_entity_name", has_target_entity),
                    ("target_container_name", has_target_container),
                ):
                    consequent = flag("op") if key in required else notL(flag("op"))
                    ifL(
                        skills[name]("op"),
                        consequent,
                        name=f"skill_argument__{name}__{key}",
                    )

        for builder in tuple(constraint_builders):
            builder(bundle)
    return bundle


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


def _entity_mapping(entity_table: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    if isinstance(entity_table, Mapping):
        return {int(index): str(name) for index, name in entity_table.items()}
    return {index: str(name) for index, name in enumerate(entity_table)}


def _resolve_pointer(value: Any, entities: Mapping[int, str]) -> tuple[str, bool]:
    if isinstance(value, int) or str(value).isdigit():
        index = int(value)
        return entities.get(index, str(value)), (not entities or index in entities)
    name = str(value)
    return name, (not entities or name in entities.values())


def materialize_plan(
    value: Any,
    entity_table: Mapping[int, str] | Sequence[str],
    bundle: VLABenchWorldGraphBundle,
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

        full_validation = validate_plan(
            plan,
            entity_table=entities,
            skill_arguments=bundle.skill_arguments,
        )
        operation_nodes = []
        for index, operation_value in enumerate(plan):
            name = operation_value["name"]
            params = operation_value["params"]
            operation_node = _node(bundle.operation, index, value=name)
            _truth(operation_node, bundle.operation, True)
            _link(operation_node, bundle.operation_roles["plan"], root)

            pointer_valid = True
            for key, role_name in (
                ("target_entity_name", "target_entity"),
                ("target_container_name", "target_container"),
            ):
                if key in params:
                    pointer_name, is_valid = _resolve_pointer(params[key], entities)
                    pointer_valid = pointer_valid and is_valid
                else:
                    pointer_name = ABSENT_ENTITY
                _link(operation_node, bundle.operation_roles[role_name], entity_nodes[pointer_name])

            local_validation = validate_plan(
                [operation_value],
                entity_table=entities,
                skill_arguments=bundle.skill_arguments,
                require_pattern=False,
            )
            for skill_name, concept in bundle.skills.items():
                _truth(operation_node, concept, skill_name == name)
            _truth(operation_node, bundle.valid_skill_type, name in bundle.skills)
            _truth(operation_node, bundle.has_target_entity, "target_entity_name" in params)
            _truth(operation_node, bundle.has_target_container, "target_container_name" in params)
            _truth(operation_node, bundle.valid_pointer, pointer_valid)
            _truth(operation_node, bundle.valid_operation, local_validation.valid)
            root.addChildDataNode(operation_node)
            operation_nodes.append(operation_node)

        for index in range(max(0, len(operation_nodes) - 1)):
            transition = _node(bundle.transition, index, value=index)
            _truth(transition, bundle.transition, True)
            _truth(transition, bundle.allowed_transition, full_validation.valid)
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
        if getattr(constraint, "headLC", True)
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
            if percent is None:
                raise KeyError(f"constraint {key!r} produced no satisfaction value")
            name = getattr(constraint, "name", None) or key
            named[name] = value
            vacuous = value.get("verifyList") == []
            if name == "transition_matches_subtask_pattern" and operation_count < 2:
                vacuous = True
            scores.append(1.0 if vacuous else max(0.0, min(1.0, float(percent) / 100.0)))
        if aggregate == "mean":
            score = sum(scores) / len(scores)
        elif aggregate == "min":
            score = min(scores)
        else:
            score = 1.0
            for item in scores:
                score *= item
        return PlanConstraintEvaluation(score, len(scores), MappingProxyType(named))
    except Exception as exc:
        raise PlanGraphConfigurationError(f"failed to verify VLABench plan constraints: {exc}") from exc
