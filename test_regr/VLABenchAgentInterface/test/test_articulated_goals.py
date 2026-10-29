import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ..articulation import refreshed_articulation_context
from ..native_planning import NativePlanDFA, native_schema
from ..plan_codec import PlanVocabulary, plan_to_tokens
from ..reachability import ReachablePlanDFA
from ..semantics import PlanSemantics
from ..training import build_constraint_runtime
from ..world_domain import SKILL_ARGUMENTS, CONTROLLER_SKILLS, controller_plan_context
from ..world_graph import materialize_plan, verify_plan_constraints


@pytest.fixture(scope="module")
def runtime():
    return build_constraint_runtime(max_entities=5, max_operations=4, name_prefix="articulated")


def op(name, value):
    return {"name": name, "params": {"target_container_name" if name in ("place", "insert") else "target_entity_name": value}}


def encoded(runtime, plan):
    return [runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(plan, world=runtime.world_bundle)]


def cool_context():
    return {"entities": ("drink", "fridge", "fridge_door", "bowl", "other"), "task": "cool_drink",
            "task_target_entity": "drink", "task_target_container": "fridge", "planning_facts": {
                "initial_state": {"held": {"right": None}, "positive": [["open", "fridge"]],
                                  "negative": [["inside", "drink", "fridge"]]},
                "entity_properties": {"drink": {"graspable": True}, "fridge": {"openable": True, "closeable": True}},
                "capacities": {"right": 1}, "destinations": {"drink": "fridge"},
                "compatibility": [{"subject": "drink", "destination": "fridge", "allowed": True}]}}


def test_fridge_must_be_closed_and_goal_survives_native_projection(runtime):
    policy = runtime.dfa_for("vlabench", cool_context())
    placed = [op("pick", 0), op("place", 1)]
    complete = [*placed, op("close_door", 1)]
    for plan, expected in ((placed, False), (complete, True), ([*placed, op("close_door", 4)], False)):
        assert policy.accepts(encoded(runtime, plan)) == expected
        result = verify_plan_constraints(materialize_plan(plan, cool_context()["entities"], runtime.world_bundle, policy=policy), runtime.world_bundle)
        assert (result.score == 1) == expected
    assert policy.action_reachability().can_reach(remaining_actions=3) is True
    assert policy.action_reachability().can_reach(remaining_actions=2) is False
    tokens = {i: bytes([i]) for i in range(32,127)}
    native = ReachablePlanDFA(tokens, 200, entity_count=5, max_operations=3,
                              grammar_profile="world", policy=policy, causal=False)
    assert native.accepts(list(json.dumps(complete).encode()) + [200])
    assert not native.accepts(list(json.dumps(placed).encode()) + [200])


def test_closed_container_can_be_opened_before_the_requested_subject(runtime):
    context = cool_context()
    context["planning_facts"]["initial_state"]["positive"] = [["closed", "fridge"]]
    policy = runtime.dfa_for("vlabench", context)
    plan = [op("open_door", 1), op("pick", 0), op("place", 1), op("close_door", 1)]
    assert policy.accepts(encoded(runtime, plan))


def test_closing_door_alias_clears_mirrored_open_state():
    from domiknows.generation.grounded_facts import PlanningFacts
    facts = PlanningFacts.parse({"initial_state": {"positive": [["open", "fridge"]]}}, entities=("fridge", "fridge_door"))
    semantics = PlanSemantics(("fridge", "fridge_door"), SKILL_ARGUMENTS, facts)
    state = semantics.advance(semantics.initial_state(), "close_door", {"target_entity_name": 1})
    assert not state.opened_entities
    assert state.grounded.facts.truth(("closed", "fridge")) is True
    assert state.grounded.facts.truth(("open", "fridge")) is False


def test_already_satisfied_task_needs_no_extra_actions(runtime):
    context = cool_context()
    context["planning_facts"]["initial_state"] = {"positive":[["closed","fridge"],["inside","drink","fridge"]]}
    policy = runtime.dfa_for("vlabench", context)
    assert policy.accepts(encoded(runtime, []))
    assert policy.action_reachability().can_reach(remaining_actions=0) is True


@pytest.mark.parametrize("kind,desired", [("drink_inside",True),("fridge_closed",True),("book_outside",False),("joint_target",True)])
def test_each_articulation_formula_is_authoritative(kind, desired):
    from ..articulation import ArticulationMonitor, build_articulation_knowledge
    from domiknows.generation.grounded_facts import FactState, GroundedState
    knowledge = build_articulation_knowledge()
    fact = ("observed", "object")
    state = GroundedState(facts=FactState(frozenset([fact]) if not desired else frozenset(),
                                        frozenset([fact]) if desired else frozenset()))
    monitor = ArticulationMonitor(knowledge, (), ((kind, fact, desired),))
    assert monitor.completion(state)[0].satisfied is False
    knowledge.enforce_constraints["articulation__" + kind].active = False
    assert ArticulationMonitor(knowledge, (), monitor.requirements).completion(state)[0].satisfied is True


def study_context(value=None, task="set_study_table"):
    return {"entities": ("book", "shelf", "table", "laptop", "other"), "task": task,
            "task_target_entity": "book", "planning_facts": {
                "initial_state": {"held": {"right": None}, "positive": [["inside", "book", "shelf"]]},
                "articulation_task": {"source_container": "shelf", "laptop": "laptop"},
                "articulation_targets": [{"entity": "laptop", "joint": "hinge", "lower": 1.2, "upper": 1.5}],
                "joint_observations": [{"entity": "laptop", "joint": "hinge", "value": value}]}}


@pytest.mark.parametrize("value,expected", [(1.2,True),(1.5,True),(1.19,False),(1.51,False),(None,None)])
def test_joint_interval_boundaries_and_unknown_measurements(runtime, value, expected):
    policy = runtime.dfa_for("vlabench", study_context(value))
    state = policy.adapter.semantics.initial_state()
    for skill, params in (("pick", {"target_entity_name":0}), ("place", {"target_container_name":2})):
        state = policy.adapter.semantics.advance(state, skill, params)
    check = policy.adapter.requirements.planning.completion(state.grounded)
    assert (None if check.missing else check.satisfied) is expected


@pytest.mark.parametrize("task", ["set_study_table", "manage_study_table"])
def test_laptop_prediction_establishes_contract_but_wrong_target_does_not(runtime, task):
    policy = runtime.dfa_for("vlabench", study_context(0.2, task))
    plan = [op("pick",0), op("place",2), op("open_laptop",3)]
    assert policy.accepts(encoded(runtime, plan))
    assert not policy.accepts(encoded(runtime, [*plan[:2], op("open_laptop",4)]))


def test_replanning_discards_predicted_success_and_reads_actual_joint():
    target = SimpleNamespace(name="hinge")
    laptop = SimpleNamespace(joints=[target])
    env = SimpleNamespace(task=SimpleNamespace(entities={"laptop": laptop}),
                          physics=SimpleNamespace(bind=lambda j: SimpleNamespace(qpos=np.array([0.3]))))
    ctx = study_context(1.4)
    ctx["planning_facts"]["initial_state"]["positive"].append(["joint_in_range:hinge:1.2:1.5", "laptop"])
    fresh = refreshed_articulation_context(env, ctx, ctx["entities"])
    assert fresh["planning_facts"]["joint_observations"][0]["value"] == 0.3
    assert not any(f[0].startswith("joint_in_range:") for f in fresh["planning_facts"]["initial_state"]["positive"])
    assert ctx["planning_facts"]["joint_observations"][0]["value"] == 1.4
    assert refreshed_articulation_context(env, {"task":"select_fruit"}, ()) == {"task":"select_fruit"}


def test_expanded_vocab_roundtrip_and_fixed_official_languages(runtime, tmp_path):
    vocab = runtime.vocabulary
    assert len(vocab.skills) == 11
    path = vocab.save(tmp_path / "vocab.json")
    assert PlanVocabulary.load(path).checksum == vocab.checksum
    old = replace(vocab, domain_checksum="nine-skill-domain")
    old.save(path)
    with pytest.raises(ValueError, match="nine-skill checkpoints require retraining"):
        PlanVocabulary.load(path)
    for profile in ("legacy", "benchmark"):
        assert len(native_schema(profile)[0]) == 9
        assert "close_door" not in native_schema(profile)[0]
    assert len(native_schema("world")[0]) == 11
    for skill in ("close_door", "open_laptop"):
        assert controller_plan_context([op(skill, 0)], 0)[0] == CONTROLLER_SKILLS.index(skill)
    from ..simulation.controller import MultiViewController, TinyImageEncoder
    controller = MultiViewController(TinyImageEncoder(output_dim=8), hidden_dim=16)
    assert controller.skill_embedding.num_embeddings == len(CONTROLLER_SKILLS) == 12


def test_articulation_formula_edit_changes_both_verification_and_decoding(runtime):
    policy = runtime.dfa_for("vlabench", cool_context())
    plan = [op("pick",0), op("place",1)]
    assert not policy.accepts(encoded(runtime, plan))
    knowledge = runtime.world_bundle.articulation_knowledge.get()
    rule = knowledge.enforce_constraints["articulation__fridge_closed"]
    try:
        rule.active = False
        # Bind a fresh monitor to avoid carrying evaluations of an older formula snapshot.
        relaxed = runtime.dfa_for("vlabench", cool_context())
        assert relaxed.accepts(encoded(runtime, plan))
        root = materialize_plan(plan, cool_context()["entities"], runtime.world_bundle, policy=relaxed)
        assert verify_plan_constraints(root, runtime.world_bundle).score == 1
    finally:
        rule.active = True


def test_live_joint_adapter_against_optional_simulator():
    import os
    if os.environ.get("VLABENCH_ARTICULATION_SIM_TESTS") != "1":
        pytest.skip("set VLABENCH_ARTICULATION_SIM_TESTS=1 with VLABench assets installed")
    pytest.importorskip("VLABench")
    from ..simulation.environment import create_environment
    from ..world_graph import build_vlabench_world_graph
    from ..task_policy import task_requirements
    env = create_environment("set_study_table")
    try:
        env.reset()
        laptop = env.task.entities["laptop"]
        joint = laptop.joints[0]
        ctx = {"planning_facts":{"articulation_targets":[{"entity":"laptop", "joint":joint.name,"lower":1.2,"upper":1.5}]}}
        world = build_vlabench_world_graph("live_joint_contract")
        for value, expected in ((0.3,False),(1.3,True)):
            with env.physics.reset_context():
                env.physics.bind(joint).qpos = value
            fresh = refreshed_articulation_context(env, ctx, ("laptop",))
            requirements = task_requirements(fresh, world, ("laptop",))
            assert requirements.planning.completion(requirements.planning.initial).satisfied is expected
    finally:
        env.close()
