"""Held-out tasks leave both training and the declared task knowledge."""
import argparse

import pytest

from ..dataset import PlanningExample
from ..knowledge import (DEFAULT_TASK_KNOWLEDGE, active_task_knowledge, add_knowledge_argument,
                         set_excluded_tasks)
from ..knowledge.holdout import (HELD_OUT_TASKS, add_holdout_arguments, apply_holdout_arguments, partition,
                                 split_by_task, strip_task_facts)
from ..task_policy import task_requirements
from ..world_graph import build_vlabench_world_graph


@pytest.fixture(autouse=True)
def clean_exclusions():
    yield
    set_excluded_tasks(())


def example(task):
    return PlanningExample(episode_id=f"{task}/example0", instruction="x", operation_sequence=(), image_paths=(),
                           segmented_image_paths=(), env_config_path=None, task=task,
                           planning_facts={"required_skill_sequence": ["pick"]}, task_skill_sequence=("pick",),
                           task_target_entity="apple", task_target_container="plate")


def parse(*argv):
    parser = argparse.ArgumentParser()
    add_knowledge_argument(parser)
    add_holdout_arguments(parser)
    return parser.parse_args(argv)


def test_default_held_out_tasks_all_have_declared_knowledge():
    assert set(HELD_OUT_TASKS) - {"select_book_semantic", "select_book_spatial"} <= set(DEFAULT_TASK_KNOWLEDGE.tasks())


def test_split_by_task_is_disjoint_and_complete():
    examples = [example(t) for t in (*HELD_OUT_TASKS, "select_fruit", "friction_qa_other")]
    seen, held = split_by_task(examples, HELD_OUT_TASKS)
    assert {e.task for e in held} == set(HELD_OUT_TASKS)
    assert not {e.task for e in seen} & set(HELD_OUT_TASKS) and len(seen) + len(held) == len(examples)


def test_held_out_episodes_lose_their_task_facts_unless_declared():
    args = parse("--holdout-tasks", "default")
    seen, held = partition([example("cook_dishes"), example("select_fruit")], args)
    assert seen[0].planning_facts is not None and seen[0].task_skill_sequence == ("pick",)
    assert held[0].planning_facts is None and held[0].task_skill_sequence is None
    assert held[0].task_target_entity is None and held[0].task_target_container is None
    declared = parse("--holdout-tasks", "default", "--declare-heldout-knowledge")
    assert partition([example("cook_dishes")], declared)[1][0].planning_facts is not None
    assert strip_task_facts(example("cook_dishes")).planning_observation_profile == "undeclared"


def test_excluded_tasks_have_no_concepts_or_constraints():
    apply_holdout_arguments(parse("--holdout-tasks", "cook_dishes,select_book"))
    assert "cook_dishes" not in active_task_knowledge().tasks()
    world = build_vlabench_world_graph("holdout_world", knowledge_level="task")
    assert "cook_dishes" not in world.task_concepts.tasks and "select_fruit" in world.task_concepts.tasks
    names = {lc.name for lc in world.graph.logicalConstrains.values() if getattr(lc, "name", None)}
    assert not any(n.startswith(("task__cook_dishes", "task__select_book")) for n in names)
    assert any(n.startswith("task__select_fruit") for n in names)
    assert task_requirements({"task": "cook_dishes"}, world, ("apple", "bowl")).expected_skills is None


def test_declare_flag_keeps_the_knowledge():
    apply_holdout_arguments(parse("--holdout-tasks", "default", "--declare-heldout-knowledge"))
    assert "cook_dishes" in active_task_knowledge().tasks()
