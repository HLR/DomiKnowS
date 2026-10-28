"""Knowledge levels and the declarative task knowledge of the VLABench world graph.

Every rule is a graph concept plus logical constraints. Three independent settings decide what a planner knows:

``--knowledge`` (cumulative levels) decides which constraint groups are active:

==========  ==================================================================
level       active constraint groups
==========  ==================================================================
grammar     none (the plan grammar DFA alone)
causal      causal rules
generic     causal + generic rules
scene       causal + generic + scene-grounded rules
task        + task rules: sequences, obligations, grounded families, complex tasks, physics
==========  ==================================================================

``--task-identity`` decides where the ``task__*`` observation comes from once the task group is active:
``oracle`` (the episode's task name; privileged) or ``predicted`` (``context['predicted_task']`` from a classifier
fitted on the training split). ``--observation-mode`` (visual or metadata-assisted) decides whether property
observations come from vision or from simulator metadata; metadata-assisted needs the task level.

The former single list survives as aliases: ``learned`` is ``task`` with a predicted identity and ``metadata`` is
``task`` with the oracle identity (run with metadata-assisted observations).

``causal`` uses only the ids the plan writes: a destination is never the subtask's picked object, a subtask
never repeats a destination, and an earlier destination is never picked. It is the highest level the standard
protocol can ground (no entity table, no task); every level above keeps the causal group active.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .tables import (
    COMPLEX_TASK_SCHEMAS, PLANNING_TASK_PATTERNS, PRIMITIVE_TASK_PATTERNS, QA_TASK_PATTERNS, ComplexTaskSchema,
)

LEVELS = ("grammar", "causal", "generic", "scene", "task")
GROUPS = ("causal", "generic", "scene", "task")
TASK_IDENTITIES = ("oracle", "predicted")
# The former combined levels: (knowledge level, task identity).
LEGACY_LEVELS = {"learned": ("task", "predicted"), "metadata": ("task", "oracle")}
_GROUP_FROM = {"causal": "causal", "generic": "generic", "scene": "scene", "task": "task"}


def split_level(level: str, task_identity: str | None = None) -> tuple[str, str | None]:
    """``(level, task identity)`` for a level or a legacy alias; a conflicting explicit identity is an error."""
    if level in LEGACY_LEVELS:
        level, implied = LEGACY_LEVELS[level]
        if task_identity is not None and task_identity != implied:
            raise ValueError(f"--knowledge alias implies --task-identity {implied}, got {task_identity}")
        task_identity = implied
    if level not in LEVELS:
        raise ValueError(f"knowledge level must be one of {LEVELS} (or {tuple(LEGACY_LEVELS)}), got {level!r}")
    if task_identity is not None and task_identity not in TASK_IDENTITIES:
        raise ValueError(f"task identity must be one of {TASK_IDENTITIES}, got {task_identity!r}")
    return level, task_identity


def check_level(level: str) -> str:
    return split_level(level)[0]


def check_task_identity(task_identity: str) -> str:
    if task_identity not in TASK_IDENTITIES:
        raise ValueError(f"task identity must be one of {TASK_IDENTITIES}, got {task_identity!r}")
    return task_identity


_default_level = [split_level(os.environ.get("VLABENCH_KNOWLEDGE", "task"))[0]]
_default_task_identity = [split_level(os.environ.get("VLABENCH_KNOWLEDGE", "task"),
                                      os.environ.get("VLABENCH_TASK_IDENTITY"))[1] or "oracle"]


def default_level() -> str:
    """The level used by graphs built without an explicit one (set by the ``--knowledge`` flags)."""
    return _default_level[0]


def default_task_identity() -> str:
    """The task identity used by graphs built without an explicit one (set by ``--task-identity``)."""
    return _default_task_identity[0]


def set_default_level(level: str) -> None:
    level, identity = split_level(level)
    _default_level[0] = level
    if identity is not None:
        _default_task_identity[0] = identity


def set_default_task_identity(task_identity: str) -> None:
    _default_task_identity[0] = check_task_identity(task_identity)


def group_active(level: str, group: str) -> bool:
    """Whether a constraint group is enabled at ``level``."""
    return LEVELS.index(check_level(level)) >= LEVELS.index(_GROUP_FROM[group])


def task_facts_available(level: str) -> bool:
    return group_active(level, "task")


_excluded_tasks: set = set()


def set_excluded_tasks(tasks) -> None:
    """Tasks removed from the default declared knowledge (the unseen-task test)."""
    _excluded_tasks.clear()
    _excluded_tasks.update(tasks)


def add_knowledge_argument(parser) -> None:
    parser.add_argument("--knowledge", choices=(*LEVELS, *LEGACY_LEVELS), default=None,
                        help="active constraint groups: grammar < causal < generic < scene < task (default: task, or "
                             "$VLABENCH_KNOWLEDGE); the aliases learned (= task + --task-identity predicted) and metadata "
                             "(= task + --task-identity oracle) are kept for existing commands")
    parser.add_argument("--task-identity", choices=TASK_IDENTITIES, default=None,
                        help="source of the task identity at the task level: oracle (the episode's task name, privileged; "
                             "default) or predicted (a classifier fitted on the training split; needs --task-classifier)")
    parser.add_argument("--task-classifier", default=None,
                        help="classifier file from fit-task-classifier; required by --task-identity predicted")


def apply_knowledge_argument(args) -> str:
    """Make ``--knowledge``/``--task-identity`` the defaults of every graph built afterwards; return the level.

    The effective task identity is ``default_task_identity()`` afterwards.
    """
    level = getattr(args, "knowledge", None)
    identity = getattr(args, "task_identity", None)
    if level is not None:
        level, identity = split_level(level, identity)
    elif identity is not None:
        check_task_identity(identity)
    # Validate before changing any default, so a rejected request leaves the defaults as they were.
    if getattr(args, "observation_mode", "visual") == "metadata-assisted" and not group_active(level or default_level(), "task"):
        raise ValueError("--observation-mode metadata-assisted needs the task level (--knowledge task)")
    if level is not None:
        set_default_level(level)
    if identity is not None:
        set_default_task_identity(identity)
    return default_level()


def uses_predicted_task() -> bool:
    """Whether task identity is predicted (after ``apply_knowledge_argument``)."""
    return group_active(default_level(), "task") and default_task_identity() == "predicted"


class KnowledgeState:
    """Mutable level and task identity shared by a bundle and its constraint groups."""

    def __init__(self, level: str, task_identity: str | None = None):
        self.level, identity = split_level(level, task_identity)
        self.task_identity = identity or default_task_identity()


@dataclass(frozen=True)
class TaskKnowledge:
    """Declarations the builder turns into ``task__<name>`` concepts and constraints."""

    primitive: Mapping[str, tuple[str, ...]] = PRIMITIVE_TASK_PATTERNS
    qa: Mapping[str, tuple[str, ...]] = QA_TASK_PATTERNS
    planning: Mapping[str, tuple[str, ...]] = PLANNING_TASK_PATTERNS
    complex: Mapping[str, ComplexTaskSchema] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.complex is None:
            object.__setattr__(self, "complex", COMPLEX_TASK_SCHEMAS)

    def without(self, tasks) -> "TaskKnowledge":
        """The same declarations minus ``tasks`` (their concepts and constraints are then never built)."""
        drop = set(tasks)
        keep = lambda table: {name: value for name, value in table.items() if name not in drop}
        return TaskKnowledge(keep(self.primitive), keep(self.qa), keep(self.planning), keep(self.complex))

    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted({*self.primitive, *self.qa, *self.planning, *self.complex}))

    def expected(self, task: str, interface: str) -> tuple[str, ...] | None:
        """Declared skill sequence of ``task`` under the ``compact`` or ``full`` plan interface."""
        if interface == "compact":
            expected = self.planning.get(task, self.primitive.get(task))
        else:
            expected = self.primitive.get(task)
        return expected if expected is not None else self.qa.get(task)

    def variants(self, task: str) -> tuple[tuple[str | None, tuple[str, ...]], ...]:
        """Distinct sequences of a task: ``(None, skills)`` when both interfaces agree, else one per interface."""
        compact, full = self.expected(task, "compact"), self.expected(task, "full")
        if compact is None and full is None:
            return ()
        if compact == full:
            return ((None, compact),)
        return tuple((name, skills) for name, skills in (("compact", compact), ("full", full)) if skills is not None)

    def schema(self, task: str) -> ComplexTaskSchema | None:
        return self.complex.get(task)

    def is_qa(self, task: str) -> bool:
        return task in self.qa


DEFAULT_TASK_KNOWLEDGE = TaskKnowledge()

__all__ = [
    "COMPLEX_TASK_SCHEMAS", "ComplexTaskSchema", "DEFAULT_TASK_KNOWLEDGE", "GROUPS", "KnowledgeState", "LEVELS",
    "PLANNING_TASK_PATTERNS", "PRIMITIVE_TASK_PATTERNS", "QA_TASK_PATTERNS", "TaskKnowledge", "check_level",
    "default_level", "group_active", "set_default_level", "task_facts_available", "TASK_IDENTITIES", "LEGACY_LEVELS",
    "split_level", "check_task_identity", "default_task_identity", "set_default_task_identity", "uses_predicted_task",
]


def active_task_knowledge() -> TaskKnowledge:
    """The declared knowledge minus the tasks excluded by ``--holdout-tasks``."""
    return DEFAULT_TASK_KNOWLEDGE.without(_excluded_tasks) if _excluded_tasks else DEFAULT_TASK_KNOWLEDGE
