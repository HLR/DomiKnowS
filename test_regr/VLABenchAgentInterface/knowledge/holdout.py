"""Hold whole tasks out of training and out of the declared task knowledge (the unseen-task test).

A held-out task has no training episodes and, unless ``--declare-heldout-knowledge`` is given, no task concepts or
constraints: its episodes carry no declared sequence, targets or name-keyed facts, so the model sees it exactly like
a task nobody wrote rules for.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from . import set_excluded_tasks

# Tasks with declared knowledge, spread over Complex, PhysLaw and M&T/Semantic/Spatial; whole families are held out
# (select_book and its variants) so no near-duplicate task remains in training.
HELD_OUT_TASKS = ("cook_dishes", "take_chemistry_experiment", "friction_qa", "thermal_expansion_qa",
                  "select_book", "select_book_semantic", "select_book_spatial")
HELDOUT_SPLIT = "heldout"


def add_holdout_arguments(parser) -> None:
    parser.add_argument("--holdout-tasks", default=None,
                        help="comma-separated task names (or 'default') removed from training and from the declared "
                             "task knowledge; their episodes form the 'heldout' split")
    parser.add_argument("--declare-heldout-knowledge", action="store_true",
                        help="keep task knowledge for the held-out tasks at evaluation (they stay out of training)")


def holdout_tasks(args) -> tuple[str, ...]:
    value = getattr(args, "holdout_tasks", None)
    if not value:
        return ()
    return HELD_OUT_TASKS if value == "default" else tuple(name.strip() for name in value.split(",") if name.strip())


def apply_holdout_arguments(args) -> tuple[str, ...]:
    """Remove the held-out tasks from the declared knowledge (before any graph is built); return them."""
    tasks = holdout_tasks(args)
    set_excluded_tasks(() if getattr(args, "declare_heldout_knowledge", False) else tasks)
    return tasks


def split_by_task(examples: Sequence, held_out: Sequence[str]) -> tuple[list, list]:
    """``(seen, held)``: episodes of tasks outside and inside ``held_out``."""
    held_out = set(held_out)
    seen = [e for e in examples if e.task not in held_out]
    held = [e for e in examples if e.task in held_out]
    return seen, held


def strip_task_facts(example):
    """The example as a task without declared knowledge sees it: no sequence, targets or name-keyed facts."""
    return replace(example, planning_facts=None, planning_observation_profile="undeclared", task_skill_sequence=None,
                   task_target_container=None, task_target_entity=None)


def partition(examples: Sequence, args) -> tuple[list, list]:
    """``(seen, held)`` for ``args``; held episodes lose their task facts unless their knowledge is declared."""
    tasks = holdout_tasks(args)
    if not tasks:
        return list(examples), []
    seen, held = split_by_task(examples, tasks)
    if not getattr(args, "declare_heldout_knowledge", False):
        held = [strip_task_facts(e) for e in held]
    return seen, held
