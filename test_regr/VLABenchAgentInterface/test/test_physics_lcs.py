"""The physics knowledge as logical constraints decides exactly what the former Python model decided."""
import itertools
import random

import pytest

from domiknows.generation.grounded import Operation, SelectionRequirement
from ..knowledge.physics_lcs import MEMBERSHIP, build_physics_knowledge, instruction_phrases, physics_requirement
from ..knowledge.tables import PHYSICS_PROPERTIES
from ..world_graph import set_knowledge_level
from .legacy_physics_oracle import legacy_selection
from .test_physics_grounding import scene
from .test_qa_and_pick_history import accepted, runtime  # noqa: F401 (fixture)
from ..physics_grounding import physics_planning_facts

INSTRUCTIONS = (
    "", "Choose the largest one.", "Choose the smallest one.", "Choose the medium one.", "It moves moderately.",
    "The second fastest one.", "The fastest one.", "The slowest one.", "The one that changes the least.",
    "The most of all.", "Which can float in the water?", "Choose the magnetic object.",
    "Choose the non-magnetic object.", "Not the smallest but the largest.", "The second fastest, medium one.",
)
STRANGER = "apple"


@pytest.fixture(scope="module")
def knowledge():
    return build_physics_knowledge()


def attribute_values(task):
    kind, tiers = PHYSICS_PROPERTIES[task]
    if kind == "measurement":
        return (.1, .5, .9, None)
    return (tiers[0][0], tiers[len(tiers) // 2][0], tiers[-1][0], "unobserved", None)


def old_verdict(task, instruction, values, chosen):
    buttons = [f"button{i}" for i in range(len(values))]
    if PHYSICS_PROPERTIES[task][0] == "measurement":
        raw = legacy_selection(task, instruction, dict.fromkeys(buttons), scores=dict(zip(buttons, values)))
    else:
        key = "geom_type" if task == "friction_qa" else "material"
        options = {b: None if v is None else {key: v} for b, v in zip(buttons, values)}
        raw = legacy_selection(task, instruction, options)
    check = SelectionRequirement.parse(raw, lambda name: name).check(Operation("press", chosen))
    return check.satisfied, check.status


def new_verdict(knowledge, task, instruction, values, chosen):
    buttons = [f"button{i}" for i in range(len(values))]
    block = {"action": "press", "argument": "subject", "candidates": buttons,
             "phrases": list(instruction_phrases(instruction)), "property": "p"}
    if PHYSICS_PROPERTIES[task][0] == "measurement":
        block["measurements"] = dict(zip(buttons, values))
    else:
        key = "shape" if task == "friction_qa" else "material"
        block["attributes"] = {b: None if v is None else {key: v} for b, v in zip(buttons, values)}
    check = physics_requirement({"physics_selection": block}, task, knowledge).check(Operation("press", chosen))
    return check.satisfied, check.status


def scenes(task):
    values = attribute_values(task)
    for n in (1, 2, 3):
        yield from itertools.product(values, repeat=n)
    generator = random.Random(f"physics:{task}")
    for n in range(4, 9):
        for _ in range(4):
            yield tuple(generator.choice(values) for _ in range(n))


@pytest.mark.parametrize("task", sorted(PHYSICS_PROPERTIES))
def test_compiled_physics_rules_reproduce_the_former_verdicts(knowledge, task):
    compared = 0
    for values in scenes(task):
        for instruction in INSTRUCTIONS:
            for chosen in (*(f"button{i}" for i in range(len(values))), STRANGER):
                old = old_verdict(task, instruction, values, chosen)
                new = new_verdict(knowledge, task, instruction, values, chosen)
                assert new == old, (task, instruction, values, chosen)
                compared += 1
    assert compared > 1000


def test_rules_are_generated_from_the_tables(knowledge):
    names = set(knowledge.enforce_constraints)
    assert MEMBERSHIP in names
    assert {"physics__density_qa__float__below", "physics__magnetism_qa__non_magnetic__min",
            "physics__friction_qa__small__max", "physics__density_qa__medium__n3"} <= names
    assert not any(name.endswith("__n2") and "__medium__" in name for name in names)  # no middle of two
    assert all(knowledge.families[name] == "affordance" for name in names)


def medium_density_policy(runtime, materials=("stone", "metal", "wood")):
    entities = ("button2", "button0", "button1")
    raw, _ = physics_planning_facts("density_qa", "Choose the medium density.", scene(materials))
    return runtime.dfa_for("vlabench", {"task": "density_qa", "entities": entities, "planning_facts": raw}), entities


def presses(runtime, policy, entities):
    return {index for index in range(3)
            if accepted(runtime, policy, [{"name": "press", "params": {"target_entity_name": index}}], entities)}


def test_deactivating_a_physics_rule_relaxes_decoding_and_verification(runtime):
    knowledge = runtime.world_bundle.physics_knowledge.get()
    policy, entities = medium_density_policy(runtime)
    assert presses(runtime, policy, entities) == {1}  # stone is button0, obj:1
    knowledge.set_active(False, names={"physics__density_qa__medium__n3"})
    try:
        relaxed, _ = medium_density_policy(runtime)
        assert presses(runtime, relaxed, entities) == {0, 1, 2}
    finally:
        knowledge.set_active(True)
    restored, _ = medium_density_policy(runtime)
    assert presses(runtime, restored, entities) == {1}


def test_membership_rule_is_what_rejects_a_press_outside_the_buttons(runtime):
    knowledge = runtime.world_bundle.physics_knowledge.get()
    raw, _ = physics_planning_facts("density_qa", "Choose the object.", scene())
    # button3 is a button (so the scene rule allows pressing it) but not one of the answer options.
    entities = ("button0", "button1", "button2", "button3")
    context = {"task": "density_qa", "entities": entities, "planning_facts": raw}
    plan = [{"name": "press", "params": {"target_entity_name": 3}}]
    assert not accepted(runtime, runtime.dfa_for("vlabench", context), plan, entities)
    knowledge.set_active(False, names={MEMBERSHIP})
    try:
        assert accepted(runtime, runtime.dfa_for("vlabench", context), plan, entities)
    finally:
        knowledge.set_active(True)


def test_knowledge_level_switches_the_physics_rules(runtime):
    world = runtime.world_bundle
    previous = world.knowledge_state.level
    try:
        set_knowledge_level(world, "scene")
        assert not world.physics_knowledge.active
        assert not any(lc.active for lc in world.physics_knowledge.get().enforce_constraints.values())
        set_knowledge_level(world, "task")
        assert all(lc.active for lc in world.physics_knowledge.get().enforce_constraints.values())
        policy, entities = medium_density_policy(runtime)
        assert presses(runtime, policy, entities) == {1}
    finally:
        set_knowledge_level(world, previous)
