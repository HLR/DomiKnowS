"""VLABench domain vocabulary, task contracts, and wire-plan normalization.

Logical policy belongs to world_graph and knowledge/task_lcs. These helpers
interpret input records and expose the stable public vocabulary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


EOS_TOKEN = "<eos>"
ABSENT_ENTITY = "__absent_plan_argument__"

# Domain declarations live here.  All codecs, generation vocabularies, data
# validation, rewards, and policies derive from the world graph bundle.
SUBTASK_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("pick", "lift", "pull"),
    ("pick", "lift", "place"),
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

# The task tables are declarations. world_graph builds their task concepts
# and knowledge.task_lcs declares the associated sequence/obligation formulas.
try:
    from .knowledge import (
        DEFAULT_TASK_KNOWLEDGE, KnowledgeState, active_task_knowledge, check_level, default_level, group_active,
        PLANNING_TASK_PATTERNS, PRIMITIVE_TASK_PATTERNS, QA_TASK_PATTERNS,
    )
    from .knowledge.task_lcs import declare_task_concepts, knowledge_values
except ImportError:  # pragma: no cover - flat-module execution
    from knowledge import (
        DEFAULT_TASK_KNOWLEDGE, KnowledgeState, active_task_knowledge, check_level, default_level, group_active,
        PLANNING_TASK_PATTERNS, PRIMITIVE_TASK_PATTERNS, QA_TASK_PATTERNS,
    )
    from knowledge.task_lcs import declare_task_concepts, knowledge_values


@dataclass(frozen=True)
class TaskContract:
    """Executable contract shared by planning, rollout diagnostics, and QA."""

    task: str
    family: str
    skill_sequence: tuple[str, ...]
    target_attributes: tuple[str, ...]
    success_predicate: str
    progress_signals: tuple[str, ...]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "skill_sequence": list(self.skill_sequence),
            "target_attributes": list(self.target_attributes),
            "success_predicate": self.success_predicate,
            "progress_signals": list(self.progress_signals),
            "notes": self.notes,
        }


_TASK_PROGRESS_SIGNALS = ("get_task_progress", "get_intention_score", "target_distance")
_TASK_SUCCESS_PREDICATE = "task.should_terminate_episode(physics)"

# Contracts cover the ten primitives currently represented by the LeRobot
# control adapter. Upstream VLABench has additional categories; those must be
# added only after their task objects and success conditions are audited.
TASK_CONTRACTS: Mapping[str, TaskContract] = MappingProxyType({
    "add_condiment": TaskContract("add_condiment", "primitive", ("pick", "pour"),
        ("target_entity", "target_container"), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick the condiment and pour it into the target container."),
    "insert_flower": TaskContract("insert_flower", "primitive", ("pick", "insert"),
        ("target_entity", "target_container"), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick the flower and insert it into the target container."),
    "select_book": TaskContract("select_book", "primitive", ("pick", "pull"),
        ("target_entity",), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick the book and pull it clear of the shelf."),
    "select_chemistry_tube": TaskContract("select_chemistry_tube", "primitive", ("pick", "lift"),
        ("target_entity",), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick the requested tube and lift it clear of the rack."),
    "select_drink": TaskContract("select_drink", "primitive", ("pick", "lift", "pull"),
        ("target_entity",), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick, lift, and pull the requested drink clear of the cooler."),
    "select_fruit": TaskContract("select_fruit", "primitive", ("pick", "place"),
        ("target_entity", "target_container"), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick the requested fruit and place it in the target."),
    "select_mahjong": TaskContract("select_mahjong", "primitive", ("pick", "place"),
        ("target_entity", "target_container"), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick the requested mahjong tile and place it in the target."),
    "select_painting": TaskContract("select_painting", "primitive", ("press",),
        ("target_button", "target_entity"), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Press the button associated with the requested painting style."),
    "select_poker": TaskContract("select_poker", "primitive", ("pick", "lift"),
        ("target_entity",), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick the requested poker item and lift it."),
    "select_toy": TaskContract("select_toy", "primitive", ("pick", "lift", "place"),
        ("target_entity", "target_container"), _TASK_SUCCESS_PREDICATE,
        _TASK_PROGRESS_SIGNALS, "Pick, lift, and place the requested toy in the target."),
})


def task_contract_for(task: str) -> TaskContract | None:
    """Return the audited adapter contract for a task identifier."""

    normalized = str(task).strip().lower().replace("-", "_")
    return TASK_CONTRACTS.get(normalized)


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

# Optional symbolic roles are independent of simulator motion parameters.
OPTIONAL_SKILL_ARGUMENTS = MappingProxyType({"push": ("target_container_name",)})

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


# Preserve the public world-graph helper names while sharing their semantics.
try:
    from .semantics import (
        ENCLOSED_CONTAINER_KEYWORDS, PlanSemantics, PlanSemanticState,
        entity_mapping as _entity_mapping, resolve_pointer as _resolve_pointer,
        is_enclosed_container, is_container_opened, operation_schema_errors,
    )
except ImportError:
    from semantics import (
        ENCLOSED_CONTAINER_KEYWORDS, PlanSemantics, PlanSemanticState,
        entity_mapping as _entity_mapping, resolve_pointer as _resolve_pointer,
        is_enclosed_container, is_container_opened, operation_schema_errors,
    )


def validate_plan(
    value: Any,
    *,
    entity_table: Mapping[int, str] | Sequence[str] | None = None,
    skill_arguments: Mapping[str, Sequence[str]] = SKILL_ARGUMENTS,
    patterns: Sequence[Sequence[str]] = SUBTASK_PATTERNS,
    require_pattern: bool = True,
    policy: Any = None,
    requirements: Any = None,
) -> PlanValidation:
    errors: list[str] = []
    try:
        plan = canonicalize_plan(value)
    except PlanSchemaError as exc:
        return PlanValidation(False, (str(exc),), ())
    if requirements is None and policy is not None:
        from test_regr.VLABenchAgentInterface.constraint_compiler import requirements_for_policy
        requirements = requirements_for_policy(policy)
    allowed = getattr(requirements, "allowed_sequences", None)
    if not plan and (allowed is None or () not in allowed):
        errors.append("plan is empty")
    for index, operation in enumerate(plan):
        errors.extend(operation_schema_errors(
            operation["name"], operation["params"], skill_arguments, entity_table, index=index,
        ))
    if require_pattern and plan and ((tuple(op["name"] for op in plan) not in allowed) if allowed is not None else split_subtasks([op["name"] for op in plan], patterns) is None):
        errors.append("skill sequence is not a complete VLABench subtask pattern")
    return PlanValidation(not errors, tuple(errors), tuple(plan))


def _domain_checksum(*, legacy=False) -> str:
    payload = {
        "version": 3,
        "optional_arguments": dict(OPTIONAL_SKILL_ARGUMENTS),
        "planning_tasks": dict(PLANNING_TASK_PATTERNS),
        "skills": {name: list(arguments) for name, arguments in SKILL_ARGUMENTS.items()},
        "patterns": [list(pattern) for pattern in SUBTASK_PATTERNS],
        "tasks": {name: list(pattern) for name, pattern in PRIMITIVE_TASK_PATTERNS.items()},
        "roles": ["target_entity_name", "target_container_name"],
    }
    if legacy:
        payload["version"] = 2
        payload.pop("optional_arguments")
        payload.pop("planning_tasks")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


VLABENCH_DOMAIN_CHECKSUM = _domain_checksum()
LEGACY_VLABENCH_DOMAIN_CHECKSUM = _domain_checksum(legacy=True)


@dataclass(frozen=True)
class PlanConstraintEvaluation:
    score: float
    constraint_count: int
    results: Mapping[str, Any]
    grounding_report: Any = None


class PlanGraphConfigurationError(RuntimeError):
    """A VLABench plan graph could not be built or verified."""
