"""Physics grounding is input-only and enforced by the compiled world LCs."""
import itertools
import json

import pytest

from domiknows.generation.grounded import PlanningFacts, Operation
from .dataset import load_planning_examples, load_scene_planning_examples
from .physics_grounding import PHYSICS_PROFILE, physics_planning_facts
from .test_qa_and_pick_history import accepted, encode, runtime


def scene(materials=("stone", "metal", "wood"), shapes=("box", "sphere", "cylinder")):
    components = []
    for i, (material, shape) in enumerate(zip(materials, shapes)):
        components.extend([
            {"name": f"button{i}", "class": "Button", "position": [i, 0, 0]},
            {"name": f"{material}_{shape}", "class": "RandomGeom", "material": material,
             "position": [i, .1, 0]},
        ])
    return components


def eligible(task, instruction, components):
    raw, profile = physics_planning_facts(task, instruction, components)
    assert profile == PHYSICS_PROFILE
    candidates = raw["selections"][0]["candidates"]
    requirement = PlanningFacts.parse(raw, entities=candidates).selections[0]
    return {name for name in candidates if requirement.check(Operation("press", name)).satisfied is True}


@pytest.mark.parametrize("materials", list(itertools.permutations(("wood", "stone", "metal"))))
@pytest.mark.parametrize("task, instruction, material", [
    ("density_qa", "Choose the largest density.", "metal"),
    ("density_qa", "Choose the medium density.", "stone"),
    ("density_qa", "Choose the object that can float in the water.", "wood"),
    ("speed_of_sound_qa", "Sound traveling the fastest?", "metal"),
    ("speed_of_sound_qa", "Sound traveling moderately?", "stone"),
    ("thermal_expansion_qa", "Who shrinks the least?", "wood"),
    ("thermal_expansion_qa", "Who expands moderately?", "stone"),
    ("magnetism_qa", "Choose the magnetic object.", "metal"),
])
def test_public_material_models_are_invariant_to_button_shuffle(materials, task, instruction, material):
    components = scene(materials)
    assert eligible(task, instruction, components) == {f"button{materials.index(material)}"}
    assert eligible(task, instruction, list(reversed(components))) == {f"button{materials.index(material)}"}


def test_shape_model_uses_relative_order_and_ties():
    components = scene(shapes=("sphere", "sphere", "cylinder"))
    assert eligible("friction_qa", "Choose the slowest.", components) == {"button2"}
    assert eligible("friction_qa", "Choose the fastest.", components) == {"button0", "button1"}
    assert eligible("friction_qa", "Choose the second fastest.", scene()) == {"button2"}
    assert eligible("magnetism_qa", "Choose the non-magnetic object.", scene()) == {"button0", "button2"}


def test_unknown_property_and_ambiguous_associations_remain_unknown():
    components = scene()
    components[1]["material"] = "unobserved"
    raw, _ = physics_planning_facts("density_qa", "largest density", components)
    requirement = PlanningFacts.parse(raw, entities=("button0", "button1", "button2")).selections[0]
    assert requirement.check(Operation("press", "button0")).satisfied is None
    assert requirement.check(Operation("press", "button2")).satisfied is False
    components[3]["position"] = [0, .1, 0]
    raw, _ = physics_planning_facts("density_qa", "largest density", components)
    assert raw["selections"][0]["scores"]["button0"] is None


def test_reflectance_is_not_invented_and_explicit_measurements_take_precedence():
    components = scene()
    raw, _ = physics_planning_facts("reflection_qa", "medium reflection ratio", components)
    assert all(value is None for value in raw["selections"][0]["scores"].values())
    explicit = {"selections": [{"action": "press", "criterion": "min", "candidates": ["button0"],
                                "scores": {"button0": .2}}]}
    actual, profile = physics_planning_facts("reflection_qa", "medium reflection ratio", components, explicit)
    assert actual == explicit and profile == "declared_planning_facts"
    assert actual is not explicit


def test_loader_grounding_is_invariant_to_gold_conditions_and_output(tmp_path):
    episode = tmp_path / "PhysicsLaw" / "density_qa" / "example0"
    for name in ("input", "output", "env_config"):
        (episode / name).mkdir(parents=True)
    (episode / "input/instruction.txt").write_text("Choose the medium density.")
    config = {"task": {"components": scene(), "conditions": {"press_button": {"target_button": "button2"}}}}
    (episode / "env_config/env_config.json").write_text(json.dumps(config))
    output = episode / "output/operation_sequence.json"
    output.write_text(json.dumps({"skill_sequence": [{"name": "press", "params": {"target_entity_name": "button2"}}]}))
    first = load_planning_examples(tmp_path, observation_mode="metadata-assisted")[0]
    config["task"]["conditions"]["press_button"]["target_button"] = "button1"
    (episode / "env_config/env_config.json").write_text(json.dumps(config))
    output.write_text(json.dumps({"skill_sequence": [{"name": "pick", "params": {"target_entity_name": "gold_only"}}]}))
    second = load_planning_examples(tmp_path, observation_mode="metadata-assisted")[0]
    safe = load_scene_planning_examples(tmp_path, observation_mode="metadata-assisted")[0]
    assert first.planning_facts == second.planning_facts == safe.planning_facts
    assert second.planning_observation_profile == PHYSICS_PROFILE
    assert second.as_reward_item()["planning_facts"] == second.planning_facts
    assert second.as_reward_item()["planning_observation_profile"] == PHYSICS_PROFILE
    assert "gold_only" not in safe.entities


def test_grounded_rank_changes_compiled_token_masks_and_independent_plan_verification(runtime):
    entities = ("button2", "button0", "button1")
    raw, _ = physics_planning_facts("density_qa", "Choose the medium density.", scene())
    context = {"task": "density_qa", "entities": entities, "planning_facts": raw}
    policy = runtime.dfa_for("vlabench", context)
    # Stone is button0, which is obj:1 in this deliberately shuffled entity table.
    for index in range(3):
        plan = [{"name": "press", "params": {"target_entity_name": index}}]
        assert accepted(runtime, policy, plan, entities) == (index == 1)
    state = policy.start_state
    for token in ("skill:press", "arg:target_entity_name"):
        state = policy.step(state, runtime.vocabulary.label_for_token(token))
    assert policy.allowed_tokens(state, 1) == {runtime.vocabulary.label_for_token("obj:1")}
    # A fresh request must bind its new property observations.
    raw, _ = physics_planning_facts("density_qa", "Choose the medium density.", scene(("wood", "stone", "metal")))
    other = runtime.dfa_for("vlabench", {**context, "planning_facts": raw})
    assert other.accepts(encode(runtime, [{"name": "press", "params": {"target_entity_name": 2}}], entities))
    assert not policy.accepts(encode(runtime, [{"name": "press", "params": {"target_entity_name": 2}}], entities))
    from .constraint_compiler import WorldConstraintDFA, compile_world_constraints
    rule = next(c for c in runtime.world_bundle.graph.logicalConstrains.values() if c.name == "grounded__affordance")
    rule.active = False
    try:
        relaxed = WorldConstraintDFA(policy.base_dfa, runtime.vocabulary, runtime.world_bundle,
                                     compile_world_constraints(runtime.world_bundle), entities,
                                     requirements=policy.adapter.requirements)
        assert all(accepted(runtime, relaxed, [{"name": "press", "params": {"target_entity_name": i}}], entities)
                   for i in range(3))
    finally:
        rule.active = True
