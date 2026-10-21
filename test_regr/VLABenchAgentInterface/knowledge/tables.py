"""Declarative task data: the single source the graph builder turns into concepts and logical constraints.

Nothing here is consulted at decoding time. ``world_graph.build_vlabench_world_graph`` loops over these records
and declares the ``task__<name>`` concepts and their LCs (see ``knowledge.task_lcs``); the compiler and the
verifier then enforce them like every other LC. The names are re-exported by ``world_graph`` and ``task_policy``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

PRIMITIVE_TASK_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "add_condiment": ("pick", "pour"),
    "insert_flower": ("pick", "insert"),
    # SelectBookTask removes the book from the shelf with pick followed by pull.
    "select_book": ("pick", "pull"),
    "select_chemistry_tube": ("pick", "lift"),
    # SelectDrinkTask officially lifts the drink before pulling it clear.
    "select_drink": ("pick", "lift", "pull"),
    "select_fruit": ("pick", "place"),
    "select_mahjong": ("pick", "place"),
    # VLABench SelectPaintingTask inherits PressButtonTask: the goal is to
    # press the button associated with the requested painting style.
    "select_painting": ("press",),
    "select_poker": ("pick", "lift"),
    # SelectToyTask lifts the toy before placing it in the target.
    "select_toy": ("pick", "lift", "place"),
})

# Planning-only QA aliases do not add controller primitives or change token IDs.
# Keep their task requirements separate from the checkpoint's controller schema.
QA_TASK_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "density_qa": ("press",),
    "friction_qa": ("press",),
    "magnetism_qa": ("press",),
    "reflection_qa": ("press",),
    "speed_of_sound_qa": ("press",),
    "thermal_expansion_qa": ("press",),
})

# Compact non-interactive labels omit these simulator lift movements.
PLANNING_TASK_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    **QA_TASK_PATTERNS,
    "select_drink": ("pick", "pull"),
    "select_toy": ("pick", "place"),
})


@dataclass(frozen=True)
class ComplexTaskSchema:
    task_name: str
    required_skills: tuple[str, ...]
    min_operations: int
    phase_transitions: Mapping[tuple[str, str], str] | None = None
    accepting_phases: frozenset[str] | None = None


COMPLEX_TASK_SCHEMAS: Mapping[str, ComplexTaskSchema] = {
    "cook_dishes": ComplexTaskSchema(task_name="cook_dishes", required_skills=("pick", "place"), min_operations=4),
    "cool_drink": ComplexTaskSchema(task_name="cool_drink", required_skills=("pick", "place"), min_operations=2),
    "make_juice": ComplexTaskSchema(task_name="make_juice", required_skills=("pick", "place"), min_operations=2),
    "hammer_nail_and_hang_picture": ComplexTaskSchema(
        task_name="hammer_nail_and_hang_picture", required_skills=("pick", "place", "push"), min_operations=3),
    "take_chemistry_experiment": ComplexTaskSchema(
        task_name="take_chemistry_experiment", required_skills=("pick", "place", "pour"), min_operations=3),
    "book_rearrange": ComplexTaskSchema(
        task_name="book_rearrange", required_skills=("pick", "push", "pull"), min_operations=3),
    "texas_holdem": ComplexTaskSchema(task_name="texas_holdem", required_skills=("pick", "place"), min_operations=2),
    "manage_study_table": ComplexTaskSchema(
        task_name="manage_study_table", required_skills=("pick", "place"), min_operations=3),
}
