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


# Qualitative physics of the QA tasks (VLABench/tasks/hierarchical_tasks/primitive/physical_series). Each property
# is an ascending order of tiers over one attribute of the object bound to a button; objects in one tier tie.
# physics_lcs.py turns these tables into logical constraints; nothing here is consulted while decoding.
PHYSICS_PROPERTIES: Mapping[str, tuple[str, tuple[tuple[str, ...], ...]]] = MappingProxyType({
    "density_qa": ("material", (("wood",), ("rubber",), ("water",), ("glass",), ("stone",), ("metal",))),
    "speed_of_sound_qa": ("material", (("rubber",), ("water",), ("wood",), ("stone",), ("glass",), ("metal",))),
    "thermal_expansion_qa": ("material", (("glass",), ("wood",), ("stone",), ("metal",), ("rubber",))),
    "magnetism_qa": ("material", (("wood", "rubber", "water", "glass", "stone"), ("metal",))),
    "friction_qa": ("shape", (("sphere",), ("cylinder",), ("box",))),  # slide cost: a sphere is fastest
    "reflection_qa": ("measurement", ()),  # explicitly bound reflectance fractions
})
PHYSICS_MATERIALS = ("wood", "rubber", "water", "glass", "stone", "metal")
PHYSICS_SHAPES = ("sphere", "cylinder", "box")
# Instruction phrases (substrings of the normalized instruction) observed for every QA episode.
PHYSICS_PHRASES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "non_magnetic": ("non-magnetic", "non magnetic"),
    "magnetic": ("magnetic",),
    "float": ("float in the water",),
    "second_fastest": ("second fastest", "second-fastest"),
    "medium": ("medium", "moderately"),
    "small": ("smallest", "least", "slowest"),
    "large": ("largest", "most", "fastest"),
})
# Task-specific phrases are read before the shared ones, in this order.
PHYSICS_TASK_PHRASES: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType({
    "magnetism_qa": (("non_magnetic", "min"), ("magnetic", "max")),
    "density_qa": (("float", "below:water"),),
})
# On a slide-cost scale "fastest" is the smallest value and the second fastest is rank 1 from the bottom.
PHYSICS_REVERSED = frozenset({"friction_qa"})
PHYSICS_MAX_OPTIONS = 8
