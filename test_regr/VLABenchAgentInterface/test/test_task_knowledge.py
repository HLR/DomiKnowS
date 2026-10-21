"""Task knowledge is graph concepts plus logical constraints, switchable by tier, and never leaks below its tier."""
import pytest

from ..knowledge import DEFAULT_TASK_KNOWLEDGE, LEVELS, apply_knowledge_argument, group_active
from ..task_policy import TaskRequirements, task_requirements
from ..world_graph import build_vlabench_world_graph, materialize_plan, set_knowledge_level, verify_plan_constraints

ENTITIES = ("apple", "bowl", "button0")
TASKS = DEFAULT_TASK_KNOWLEDGE.tasks()


def step(skill):
    key = "target_container_name" if skill == "place" else "target_entity_name"
    return {"name": skill, "params": {key: 1 if skill == "place" else 0}}


def lc_satisfaction(world, plan, task, interface, prefixes):
    """Satisfaction of the task LCs named ``task__<task>__<kind>*`` for ``plan`` (None when none applies)."""
    root = materialize_plan(plan, ENTITIES, world, task_context={"task": task, "planning_interface": interface})
    result = verify_plan_constraints(root, world)
    rows = [value for name, value in result.results.items()
            if any(name.startswith(f"task__{task}__{prefix}") for prefix in prefixes)]
    return all(row.get("satisfied", 100.0 if row.get("verifyList") == [] else 0.0) == 100.0 for row in rows)


@pytest.fixture(scope="module")
def world():
    return build_vlabench_world_graph("task_knowledge_world", knowledge_level="task")


def candidates(expected, skills):
    other = lambda skill: next(s for s in skills if s != skill)
    yield tuple(expected)
    yield tuple(expected[:-1]) or (other(expected[0]),)
    yield tuple(expected) + (expected[0],)
    yield (other(expected[0]),) + tuple(expected[1:])
    yield tuple(expected[:-1]) + (other(expected[-1]),)


@pytest.mark.parametrize("task", [t for t in TASKS if DEFAULT_TASK_KNOWLEDGE.variants(t)])
def test_task_lcs_accept_exactly_the_declared_sequence(world, task):
    skills = [s for s in ("pick", "place", "press", "lift", "pull", "push", "pour", "open_door") if s in world.skills]
    for interface, expected in DEFAULT_TASK_KNOWLEDGE.variants(task):
        shown = interface or "full"
        for sequence in candidates(expected, skills):
            reference = TaskRequirements(expected_skills=tuple(expected), explicit_sequence=True)
            old = (len(sequence) == len(expected)
                   and all(reference.skill_allowed(i, skill) for i, skill in enumerate(sequence))
                   and reference.sequence_complete(len(sequence)))
            new = lc_satisfaction(world, [step(skill) for skill in sequence], task, shown,
                                  ("skill_sequence", "completion"))
            assert new == old, (task, shown, sequence, expected)


def test_complex_obligations_reject_short_or_incomplete_plans(world):
    for task, schema in DEFAULT_TASK_KNOWLEDGE.complex.items():
        full = [step(skill) for skill in schema.required_skills] * 2
        enough = full[:max(schema.min_operations, len(schema.required_skills))]
        assert lc_satisfaction(world, enough, task, "full", ("obligations",)) == (
            len(enough) >= schema.min_operations and {s["name"] for s in enough} >= set(schema.required_skills))
        assert not lc_satisfaction(world, [step("pick")], task, "full", ("obligations",)) or schema.min_operations <= 1 \
            and set(schema.required_skills) <= {"pick"}


def test_explicit_sequence_overrides_the_declared_one(world):
    task = "select_fruit"
    declared = DEFAULT_TASK_KNOWLEDGE.expected(task, "full")
    other = tuple(reversed(declared)) if len(declared) > 1 else declared + ("press",)
    requirements = task_requirements({"task": task, "task_skill_sequence": other}, world, ENTITIES)
    assert requirements.explicit_sequence and requirements.expected_skills == other


@pytest.mark.parametrize("level", ["generic", "scene"])
def test_below_learned_no_task_fact_is_consulted(level):
    world = build_vlabench_world_graph(f"leak_{level}", knowledge_level=level)
    context = {"task": "select_fruit", "task_name": "select_fruit", "episode_id": "select_fruit/example1",
               "task_skill_sequence": ("pick", "place"), "planning_facts": {"required_skill_sequence": ["pick"]},
               "task_target_entity": "apple", "predicted_task": "select_fruit"}
    assert task_requirements(context, world, ENTITIES) == TaskRequirements()


def test_learned_level_uses_only_the_predicted_task():
    world = build_vlabench_world_graph("leak_learned", knowledge_level="learned")
    context = {"task": "select_fruit", "task_skill_sequence": ("press",), "predicted_task": "select_toy"}
    requirements = task_requirements(context, world, ENTITIES)
    assert requirements.task == "select_toy" and not requirements.explicit_sequence
    assert task_requirements({"task": "select_fruit"}, world, ENTITIES).task is None


def test_task_rules_are_inactive_below_learned_and_follow_set_knowledge_level():
    world = build_vlabench_world_graph("tier_toggle", knowledge_level="generic")
    wrong = [step("press")]

    def names():
        root = materialize_plan(wrong, ENTITIES, world, task_context={"task": "select_fruit"})
        return set(verify_plan_constraints(root, world).results)

    assert not any(name.startswith("task__") for name in names())
    set_knowledge_level(world, "task")
    assert any(name.startswith("task__select_fruit") for name in names())
    set_knowledge_level(world, "generic")
    assert not any(name.startswith("task__") for name in names())


def test_levels_are_ordered_and_groups_nest():
    for index, level in enumerate(LEVELS):
        assert group_active(level, "generic")
        assert group_active(level, "scene") == (index >= 1)
        assert group_active(level, "task") == (index >= 2)


def test_metadata_observations_need_the_metadata_level():
    class Args:
        knowledge = "scene"
        observation_mode = "metadata-assisted"
    with pytest.raises(ValueError):
        apply_knowledge_argument(Args)
