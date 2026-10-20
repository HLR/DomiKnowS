"""QA shape, observed option eligibility, and whole-plan pick history are LCs."""
import pytest
import torch

from ..constraint_compiler import WorldConstraintDFA, compile_world_constraints
from ..graph import plan_to_tokens
from ..training import build_constraint_runtime, evaluate_planner
from ..world_graph import materialize_plan, verify_plan_constraints


@pytest.fixture(scope="module")
def runtime():
    return build_constraint_runtime(max_entities=4, max_operations=4, name_prefix="qa_pick_history")


def encode(runtime, plan, entities):
    return tuple(runtime.vocabulary.label_for_token(t) for t in
                 plan_to_tokens(plan, entities, world=runtime.world_bundle))


def accepted(runtime, policy, plan, entities):
    sequence = encode(runtime, plan, entities)
    actual = policy.accepts(sequence)
    result = verify_plan_constraints(materialize_plan(plan, entities, runtime.world_bundle, policy=policy),
                                     runtime.world_bundle)
    assert actual == (result is None or result.score == 1.0)
    return actual


@pytest.mark.parametrize("task", ["density_qa", "friction_qa", "magnetism_qa", "reflection_qa",
                                  "speed_of_sound_qa", "thermal_expansion_qa"])
def test_known_qa_metadata_requires_press_without_guessing_the_button(runtime, task):
    entities = ("button2", "button0", "button1", "apple")
    policy = runtime.dfa_for("vlabench", {"entities": entities, "episode_id": f"PhysLaw/{task}/example0"})
    assert policy.allowed_tokens(policy.start_state) == {runtime.vocabulary.label_for_token("skill:press")}
    for button in range(3):
        plan = [{"name": "press", "params": {"target_entity_name": button}}]
        assert accepted(runtime, policy, plan, entities)
        root = materialize_plan(plan, entities, runtime.world_bundle, policy=policy)
        assert any(c["family"] == "selection" and c["status"] == "unavailable"
                   for c in root.attributes["grounding_report"]["checks"])
    assert not policy.accepts(encode(runtime, [{"name": "pick", "params": {"target_entity_name": 3}},
                                             {"name": "lift", "params": {}}], entities))


def test_density_selection_uses_observed_scores_not_button_names_or_reference_answers(runtime):
    entities = ("button2", "button0", "button1")
    context = {"entities": entities, "task": "density_qa", "planning_facts": {"selections": [{
        "action": "press", "property": "density", "criterion": "max",
        "candidates": ["button2", "button0", "button1"],
        "scores": {"button2": 1, "button0": 9, "button1": 4},
    }]}}
    policy = runtime.dfa_for("vlabench", context)
    plans = [[{"name": "press", "params": {"target_entity_name": i}}] for i in range(3)]
    assert [accepted(runtime, policy, plan, entities) for plan in plans] == [False, True, False]
    state = policy.start_state
    for token in ("skill:press", "arg:target_entity_name"):
        state = policy.step(state, runtime.vocabulary.label_for_token(token))
    assert policy.allowed_tokens(state, 1) == {runtime.vocabulary.label_for_token("obj:1")}
    context.update(operation_sequence=plans[0], prepared_initial_state={"answer": "button2"})
    rebound = runtime.dfa_for("vlabench", context)
    assert [rebound.accepts(encode(runtime, plan, entities)) for plan in plans] == [False, True, False]
    rule = next(c for c in runtime.world_bundle.graph.logicalConstrains.values() if c.name == "grounded__affordance")
    rule.active = False
    try:
        relaxed = WorldConstraintDFA(policy.base_dfa, runtime.vocabulary, runtime.world_bundle,
                                     compile_world_constraints(runtime.world_bundle), entities,
                                     requirements=policy.adapter.requirements)
        assert all(accepted(runtime, relaxed, plan, entities) for plan in plans)
        assert not policy.accepts(encode(runtime, plans[0], entities))
    finally:
        rule.active = True


def test_friction_question_can_bind_exposed_motion_measurements(runtime):
    entities = ("button0", "button1", "button2")
    policy = runtime.dfa_for("vlabench", {"entities": entities, "task": "friction_qa",
        "planning_facts": {"selections": [{"action": "press", "property": "observed_slide_time",
                                           "candidates": ["button0", "button1", "button2"],
                                           "criterion": "min", "scores": {"button0": 8, "button1": 2, "button2": 5}}]}})
    for i in range(3):
        assert accepted(runtime, policy, [{"name": "press", "params": {"target_entity_name": i}}], entities) == (i == 1)


def test_evaluation_reports_qa_selection_grounding_coverage(runtime):
    plan = [{"name": "press", "params": {"target_entity_name": 0}}]
    class Planner(torch.nn.Module):
        def generate_plan(self, **kwargs):
            return plan
    example = {"entities": ("button0", "button1"), "task": "density_qa", "operation_sequence": plan}
    annotated = {**example, "planning_observation_profile": "declared_planning_facts", "planning_facts": {"selections": [{
        "action": "press", "criterion": "max", "candidates": ["button0", "button1"],
        "scores": {"button0": 2, "button1": 1},
    }]}}
    metrics = evaluate_planner(Planner(), [example, annotated], runtime)
    assert metrics["generation_failures"] == 0
    assert metrics["grounding_coverage"]["selection"] == {"unavailable": 1, "checked": 1}
    assert metrics["observation_profiles"] == {"unspecified": 1, "declared_planning_facts": 1}


def test_repeated_pick_after_other_operations_is_rejected_and_lc_can_be_disabled(runtime):
    entities = ("apple", "pear", "bowl")
    policy = runtime.dfa_for("vlabench", {"entities": entities})
    def pair(source):
        return [{"name": "pick", "params": {"target_entity_name": source}},
                {"name": "place", "params": {"target_container_name": 2}}]
    assert accepted(runtime, policy, pair(0) + pair("pear"), entities)
    repeated = pair(0) + pair("apple")
    assert not accepted(runtime, policy, repeated, entities)
    prefix = encode(runtime, pair(0), entities)[:-1]
    state = policy.start_state
    for label in (*prefix, runtime.vocabulary.label_for_token("skill:pick"),
                  runtime.vocabulary.label_for_token("arg:target_entity_name")):
        state = policy.step(state, label)
        assert state is not None
    assert runtime.vocabulary.label_for_token("obj:0") not in policy.allowed_tokens(state)
    assert runtime.vocabulary.label_for_token("obj:1") in policy.allowed_tokens(state)
    rule = next(c for c in runtime.world_bundle.graph.logicalConstrains.values() if c.name == "no_redundant_repeated_picks")
    rule.active = False
    try:
        relaxed = WorldConstraintDFA(policy.base_dfa, runtime.vocabulary, runtime.world_bundle,
                                     compile_world_constraints(runtime.world_bundle), entities,
                                     requirements=policy.adapter.requirements)
        assert accepted(runtime, relaxed, repeated, entities)
        assert not policy.accepts(encode(runtime, repeated, entities))
    finally:
        rule.active = True
