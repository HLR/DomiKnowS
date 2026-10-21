"""DomiKnowS semantic graph for grounded VLABench operation plans.

As in ``EmbodiedAgentInterface/world_graph.py``, this graph is independent of
the token-generation graph.  Planner JSON is materialized as deterministic
``DataNode`` instances, then hard schema/grounding constraints are verified by
the ordinary DomiKnowS logical-constraint machinery.
"""

from __future__ import annotations

import hashlib
import json
import math
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

# The task tables are declarative data in ``knowledge.tables``; the builder below turns them into
# ``task__<name>`` concepts and logical constraints (``knowledge.task_lcs``).
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
    targets_button: Any = None
    is_self_placement: Any = None
    targets_static_fixture: Any = None
    repeated_pick_entity: Any = None
    inconsistent_container: Any = None
    unopened_container_placement: Any = None
    premature_composite_eos: Any = None
    task_skill_allowed: Any = None
    task_target_allowed: Any = None
    task_sequence_complete: Any = None
    grounded_constraints: Any = None
    task_concepts: Any = None
    task_knowledge: Any = None
    knowledge_state: Any = None
    knowledge_groups: Any = None

    @property
    def optional_skill_arguments(self):
        return OPTIONAL_SKILL_ARGUMENTS

    @property
    def has_constraints(self) -> bool:
        return any(
            getattr(constraint, "headLC", True) and getattr(constraint, "active", True)
            for constraint in getattr(self.graph, "logicalConstrains", {}).values()
        )


@dataclass(frozen=True)
class PlanConstraintEvaluation:
    score: float
    constraint_count: int
    results: Mapping[str, Any]
    grounding_report: Any = None


class PlanGraphConfigurationError(RuntimeError):
    """A VLABench plan graph could not be built or verified."""


def build_vlabench_world_graph(
    graph_name: str = "vlabench_plan_world",
    constraint_builders: Iterable[Callable[[VLABenchWorldGraphBundle], None]] = (),
    *,
    include_default_constraints: bool = True,
    semantic_parents: Mapping[str, Any] | None = None,
    knowledge_level: str | None = None,
    task_knowledge: Any = None,
) -> VLABenchWorldGraphBundle:
    """Define the DomiKnowS plan ontology and its hard logical constraints.

    Every rule is a concept plus logical constraints, in three groups: ``generic`` (schema, patterns, causal
    rules), ``scene`` (rules that need an entity table) and ``task`` (task facts and the ``task__<name>``
    rules of ``knowledge.task_lcs``). ``knowledge_level`` (default: ``knowledge.default_level()``) decides which
    groups are active; :func:`set_knowledge_level` changes it later.
    """
    from domiknows.graph import Concept, Graph

    level = check_level(knowledge_level or default_level())
    state = KnowledgeState(level)
    task_knowledge = task_knowledge or active_task_knowledge()

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
        targets_button = operation(name="operation__targets_button")
        is_self_placement = operation(name="operation__is_self_placement")
        targets_static_fixture = operation(name="operation__targets_static_fixture")
        valid_skill_type = operation(name="operation__valid_skill_type")
        valid_pointer = operation(name="operation__valid_pointer")
        valid_operation = operation(name="operation__valid_schema")
        allowed_transition = transition(name="transition__allowed_pattern")
        repeated_pick_entity = transition(name="transition__repeated_pick_entity")
        inconsistent_container = transition(name="transition__inconsistent_container")
        unopened_container_placement = transition(name="transition__unopened_container_placement")
        premature_composite_eos = operation(name="operation__premature_composite_eos")
        task_skill_allowed = operation(name="operation__task_skill_allowed")
        task_target_allowed = operation(name="operation__task_target_allowed")
        task_sequence_complete = operation(name="operation__task_sequence_complete")

        from domiknows.generation.grounded import declare_predicates
        grounded_constraints = declare_predicates(
            operation, plan, active=include_default_constraints and group_active(level, "task"))
        task_concepts = declare_task_concepts(
            operation, skills, task_knowledge, active=group_active(level, "task"),
            constraints=include_default_constraints)
        groups = {"generic": [], "scene": [], "task": list(task_concepts.constraints)}
        groups["task"].extend(lc for lc in graph.logicalConstrains.values()
                              if str(getattr(lc, "name", "")).startswith("grounded__"))

        bundle = VLABenchWorldGraphBundle(
            grounded_constraints=grounded_constraints,
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
            targets_button=targets_button,
            is_self_placement=is_self_placement,
            targets_static_fixture=targets_static_fixture,
            repeated_pick_entity=repeated_pick_entity,
            inconsistent_container=inconsistent_container,
            unopened_container_placement=unopened_container_placement,
            premature_composite_eos=premature_composite_eos,
            task_skill_allowed=task_skill_allowed,
            task_target_allowed=task_target_allowed,
            task_sequence_complete=task_sequence_complete,
            task_concepts=task_concepts,
            task_knowledge=task_knowledge,
            knowledge_state=state,
            knowledge_groups=groups,
        )

        def declare(group, antecedent, consequent, name):
            groups[group].append(ifL(antecedent, consequent, name=name, active=group_active(level, group)))

        if include_default_constraints:
            from domiknows.graph.logicalConstrain import ifL, notL

            declare("generic", operation("op"), valid_skill_type("op"), "operation_exactly_one_skill")
            declare("generic", operation("op"), valid_operation("op"), "operation_schema_is_valid")
            declare("scene", operation("op"), valid_pointer("op"), "operation_pointer_is_valid")
            declare("generic", transition("edge"), allowed_transition("edge"), "transition_matches_subtask_pattern")
            declare("scene", targets_button("op"), skills["press"]("op"), "button_entity_requires_press_skill")

            declare("scene", skills["press"]("op"), targets_button("op"), "press_requires_button_target")
            # Explicit per-request facts (a supplied sequence, initial target, completion) are observations.
            declare("task", operation("op"), task_skill_allowed("op"), "task_skill_sequence")
            declare("task", operation("op"), task_target_allowed("op"), "task_initial_target")
            declare("task", operation("op"), task_sequence_complete("op"), "task_sequence_completion")

            # Causal Precondition Constraints (First-Order Logic)
            declare("generic", operation("op"), notL(is_self_placement("op")), "anti_self_placement")
            declare("scene", operation("op"), notL(targets_static_fixture("op")), "pick_excludes_static_fixtures")
            declare("generic", transition("edge"), notL(repeated_pick_entity("edge")), "no_redundant_repeated_picks")
            declare("scene", transition("edge"), notL(inconsistent_container("edge")), "container_destination_consistency")
            declare("scene", transition("edge"), notL(unopened_container_placement("edge")), "open_before_enclosed_placement")
            declare("task", operation("op"), notL(premature_composite_eos("op")), "composite_subgoal_completion")

            for name, required in SKILL_ARGUMENTS.items():
                required = set(required)
                for key, flag in (
                    ("target_entity_name", has_target_entity),
                    ("target_container_name", has_target_container),
                ):
                    if key in OPTIONAL_SKILL_ARGUMENTS.get(name, ()):
                        continue
                    consequent = flag("op") if key in required else notL(flag("op"))
                    declare("generic", skills[name]("op"), consequent, f"skill_argument__{name}__{key}")

        for builder in tuple(constraint_builders):
            builder(bundle)
    return bundle


def set_knowledge_level(bundle: VLABenchWorldGraphBundle, level: str) -> None:
    """Activate the constraint groups of ``level`` in place. Recompile any DFA built from the graph afterwards."""
    state = bundle.knowledge_state
    state.level = check_level(level)
    for group, constraints in (bundle.knowledge_groups or {}).items():
        for constraint in constraints:
            constraint.active = group_active(state.level, group)


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
        full_validation = validate_plan(plan, entity_table=entities, skill_arguments=bundle.skill_arguments, requirements=requirements)
        from domiknows.generation.grounded import observations, report
        semantics = PlanSemantics(entities, bundle.skill_arguments, requirements.planning)
        state = semantics.initial_state()
        grounded_operations, grounded_checks = [], []
        operation_predicates = []
        for operation in plan:
            _successor, checks = semantics.grounded_transition(state, operation["name"], operation["params"])
            grounded_operations.append(checks)
            grounded_checks.extend(checks)
            operation_predicates.append(semantics.evaluate_operation(state, operation["name"], operation["params"]))
            state = semantics.advance(state, operation["name"], operation["params"])
        is_premature_composite = semantics.premature_composite_eos(
            state, min_operations=min_operations, required_skills=required_skills,
        )

        final_check = requirements.planning.completion(state.grounded)
        grounded_checks.append(final_check)
        root.attributes["grounding_report"] = report(grounded_checks)
        for key, value in observations((final_check,)).items():
            _truth(root, bundle.grounded_constraints[key], value)
        operation_nodes = []
        task_concepts = getattr(bundle, "task_concepts", None)
        offset = len(requirements.planning.completed_skill_prefix)
        history_skills = list(requirements.planning.completed_skill_prefix)
        for index, operation_value in enumerate(plan):
            name = operation_value["name"]
            params = operation_value["params"]
            operation_node = _node(bundle.operation, index, value=name)
            _truth(operation_node, bundle.operation, True)
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
            predicates = operation_predicates[index]

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
                    history=history_skills)
                for predicate, concept in _task_concept_rows(task_concepts):
                    _truth(operation_node, concept, observed[predicate])
            _truth(operation_node, bundle.valid_pointer, predicates.valid_pointer)
            _truth(operation_node, bundle.valid_operation, predicates.valid_operation)
            root.addChildDataNode(operation_node)
            operation_nodes.append(operation_node)

        for index in range(max(0, len(operation_nodes) - 1)):
            transition = _node(bundle.transition, index, value=index)
            _truth(transition, bundle.transition, True)
            _truth(transition, bundle.allowed_transition, full_validation.valid)

            predicates = operation_predicates[index + 1]
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
