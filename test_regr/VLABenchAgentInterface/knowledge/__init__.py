"""Knowledge tiers and the declarative task knowledge of the VLABench world graph.

Every rule is a graph concept plus logical constraints. The tiers only decide which constraint groups are active
and where task identity comes from:

==========  ==============================================  =========================================
level       active constraint groups                          source of the ``task__*`` observation
==========  ==============================================  =========================================
generic     generic rules                                     none
scene       generic + scene-grounded rules                    none
learned     + task rules and facts                            a predicted task (``context['predicted_task']``)
task        + task rules and facts                            the episode's task name (privileged)
metadata    + property observations / simulator assistance    the episode's task name
==========  ==============================================  =========================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .tables import (
    COMPLEX_TASK_SCHEMAS, PLANNING_TASK_PATTERNS, PRIMITIVE_TASK_PATTERNS, QA_TASK_PATTERNS, ComplexTaskSchema,
)

LEVELS = ("generic", "scene", "learned", "task", "metadata")
GROUPS = ("generic", "scene", "task")
_GROUP_FROM = {"generic": "generic", "scene": "scene", "task": "learned"}
_default_level = [os.environ.get("VLABENCH_KNOWLEDGE", "task")]


def check_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"knowledge level must be one of {LEVELS}, got {level!r}")
    return level


def default_level() -> str:
    """The level used by graphs built without an explicit one (set by the ``--knowledge`` flags)."""
    return check_level(_default_level[0])


def set_default_level(level: str) -> None:
    _default_level[0] = check_level(level)


def group_active(level: str, group: str) -> bool:
    """Whether a constraint group is enabled at ``level``."""
    return LEVELS.index(check_level(level)) >= LEVELS.index(_GROUP_FROM[group])


def task_facts_available(level: str) -> bool:
    return group_active(level, "task")


def add_knowledge_argument(parser) -> None:
    parser.add_argument("--knowledge", choices=LEVELS, default=None,
                        help="knowledge tier of the world graph: generic < scene < learned < task < metadata "
                             "(default: task, or $VLABENCH_KNOWLEDGE)")


def apply_knowledge_argument(args) -> str:
    """Make ``--knowledge`` the default level of every graph built afterwards and return the effective level."""
    level = getattr(args, "knowledge", None)
    if level is not None:
        if getattr(args, "observation_mode", "visual") == "metadata-assisted" and level != "metadata":
            raise ValueError("--observation-mode metadata-assisted needs --knowledge metadata")
        set_default_level(level)
    return default_level()


class KnowledgeState:
    """Mutable level shared by a bundle and its constraint groups."""

    def __init__(self, level: str):
        self.level = check_level(level)


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
    "default_level", "group_active", "set_default_level", "task_facts_available",
]
