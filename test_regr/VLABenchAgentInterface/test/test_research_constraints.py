from dataclasses import replace
import pytest
from domiknows.generation.grounded_facts import GroundedState,Operation
from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
from ..research import ResearchConfig,EvidenceLedger
from ..research.monitor import ResearchMonitor
from ..research.events import parse_events,temporal_truth
from ..research.spatial import compare
from ..knowledge.research_lcs import build_research_knowledge


@pytest.fixture(scope="module")
def knowledge():return build_research_knowledge()


def monitor(knowledge,raw,mode="robust"):
    return ResearchMonitor(knowledge,raw,ResearchConfig("research_v1",mode))


def accepted(m,state=GroundedState(),op=Operation("pick","a")):
    return all(c.satisfied is True for c in m.operation(state,op))


def ambiguity():
    return {"grounding":{"complete":True,"roles":{"object":["a","b"]},"requests":[{"role":"object","action":"pick"}]}}


def test_grounding_quantifier_is_shared_concrete_plan(knowledge):
    assert not accepted(monitor(knowledge,ambiguity()))
    assert accepted(monitor(knowledge,ambiguity(),"joint"))
    assert not accepted(monitor(knowledge,ambiguity(),"abstain"))
    assert accepted(monitor(knowledge,ambiguity()),op=Operation("open_door","fridge"))


def test_joint_preserves_correlations_across_operations(knowledge):
    raw={"grounding":{"complete":True,"hypotheses":[{"id":"x","bindings":{"object":"a","destination":"left"}},
          {"id":"y","bindings":{"object":"b","destination":"right"}}],"requests":[
          {"role":"object","action":"pick"},{"role":"destination","action":"place","argument":"destination"}]}}
    m=monitor(knowledge,raw,"joint")
    state=m.transition(GroundedState(),GroundedState(),Operation("pick","a"))
    assert not accepted(m,state,Operation("place","a","right"))
    assert accepted(m,state,Operation("place","a","left"))
    raw.update(evidence={"episode":"e","step":1},events=[{"id":"grasp","name":"pick","step":1,"entities":["a"]}],
        executed_operations=[{"action":"pick","subject":"a","event_id":"grasp"}])
    fresh=monitor(knowledge,raw,"joint")
    assert not accepted(fresh,GroundedState(),Operation("place","a","right"))
    assert accepted(fresh,GroundedState(),Operation("place","a","left"))
    raw["events"][0]["kind"]="predicted"
    with pytest.raises(ValueError,match="measured"):monitor(knowledge,raw,"joint")


def test_incomplete_universe_is_not_robust_proof(knowledge):
    raw=ambiguity();raw["grounding"]["roles"]["object"]=["a"];raw["grounding"]["complete"]=False
    assert not accepted(monitor(knowledge,raw))
    assert accepted(monitor(knowledge,raw,"joint"))


def test_hypothesis_limit_does_not_truncate(knowledge):
    raw={"grounding":{"roles":{"x":list(map(str,range(12))),"y":list(map(str,range(12)))}}}
    with pytest.raises(SemanticSearchLimitExceeded):monitor(knowledge,raw)


def freshness(kind="observed",step=3):
    return {"evidence":{"episode":"e1","step":3,"revisions":{"object:a":0},"records":[
        {"id":"face","value":True,"kind":kind,"step":step,"dependencies":{"object:a":0}}]},
        "freshness":[{"action":"pick","evidence":"face"}]}


def test_predictions_and_old_observations_are_not_fresh(knowledge):
    assert accepted(monitor(knowledge,freshness()))
    assert not accepted(monitor(knowledge,freshness("predicted")))
    assert not accepted(monitor(knowledge,freshness(step=2)))


def test_dependency_change_reset_and_future_rejection():
    ledger=EvidenceLedger.parse(freshness()["evidence"])
    assert not ledger.advance(3,changed=["object:a"]).fresh("face")
    assert not ledger.advance(0,episode="e2").fresh("face")
    with pytest.raises(ValueError,match="future"):
        EvidenceLedger.parse({"episode":"e1","step":0,"records":[{"id":"x","step":1}]})


def spatial_raw(left,right,frame="world"):
    return {"evidence":{"episode":"e1","step":0,"records":[
        {"id":"a","value":left,"frame":"world","units":"m"},{"id":"b","value":right,"frame":frame,"units":"m"}]},
        "spatial":[{"relation":"left","left":"a","right":"b","frame":"world"}]}


def test_spatial_frame_and_uncertainty_are_not_guessed(knowledge):
    assert accepted(monitor(knowledge,spatial_raw([0,0,0],[1,0,0])))
    assert not accepted(monitor(knowledge,spatial_raw([0,0,0],[1,0,0],"camera")))
    raw=spatial_raw({"lower":[0,0,0],"upper":[2,0,0]},[1,0,0])
    assert compare(raw["spatial"][0],EvidenceLedger.parse(raw["evidence"])) is None
    assert not accepted(monitor(knowledge,raw))


def test_repeated_events_and_missing_interior_samples():
    ledger=EvidenceLedger.parse({"episode":"e1","step":3,"records":[
        {"id":"closed:0","step":0,"value":True},{"id":"closed:3","step":3,"value":True}]})
    events=parse_events([{"id":"1","name":"pick","step":1},{"id":"2","name":"pick","step":2}],ledger)
    assert temporal_truth({"kind":"count","event":"pick","min":2,"max":2},events,ledger) is True
    assert temporal_truth({"kind":"before","event":"pick","predecessor":"open"},events,ledger) is False
    assert temporal_truth({"kind":"until","event":"finished","start_step":0,"sample_prefix":"closed:"},events,ledger) is None
    triggers=parse_events([{"id":"1","name":"start","step":0},{"id":"2","name":"finish","step":1},
        {"id":"3","name":"start","step":2}],ledger)
    assert temporal_truth({"kind":"within","event":"finish","predecessor":"start","steps":0},triggers,ledger) is False
    assert temporal_truth({"kind":"within","event":"finish","predecessor":"start","steps":2},triggers,ledger) is None
    assert temporal_truth({"kind":"within","event":"finish","predecessor":"start","steps":2},triggers,ledger.advance(4)) is False


@pytest.mark.parametrize("name,raw",[
    ("research__grounding_robust",ambiguity()),
    ("research__fresh_observed",freshness("predicted")),
    ("research__fresh_current",freshness(step=2)),
    ("research__spatial_holds",spatial_raw([2,0,0],[1,0,0])),
    ("research__spatial_known",spatial_raw([0,0,0],[1,0,0],"camera")),
])
def test_formula_disable_changes_acceptance_after_recompile(name,raw):
    k=build_research_knowledge();m=monitor(k,raw)
    assert not accepted(m)
    k.set_active(False,names={name})
    assert accepted(m)
    k.set_active(True,names={name})
    assert not accepted(m)


@pytest.mark.parametrize("mode,expected",[("robust",False),("joint",True),("abstain",False)])
def test_datanode_and_decoding_enforce_same_research_rules(mode,expected):
    from ..world_graph import build_vlabench_world_graph,materialize_plan,verify_plan_constraints
    from ..training import build_constraint_runtime
    from ..plan_codec import plan_to_tokens
    world=replace(build_vlabench_world_graph("research_integration_"+mode),research_config=ResearchConfig("research_v1",mode))
    runtime=build_constraint_runtime(world,max_entities=3,max_operations=2)
    raw=ambiguity()
    context={"entities":("a","b","bowl"),"task_skill_sequence":("pick","place"),"planning_facts":{"research":raw}}
    plan=[{"name":"pick","params":{"target_entity_name":0}},{"name":"place","params":{"target_container_name":2}}]
    policy=runtime.dfa_for("vlabench",context)
    labels=[runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(plan,context["entities"],world=world)]
    assert policy.accepts(labels) is expected
    root=materialize_plan(plan,context["entities"],world,policy=policy)
    assert (verify_plan_constraints(root,world).score==1.) is expected


@pytest.mark.parametrize("mode,expected",[("robust",False),("joint",True),("abstain",False)])
def test_native_reachability_retains_interpretations_and_valid_witnesses(mode,expected):
    from ..world_graph import build_vlabench_world_graph
    from ..training import build_constraint_runtime
    from .test_action_reachability import context,native,labels,plan
    world=replace(build_vlabench_world_graph("research_native"),research_config=ResearchConfig("research_v1",mode))
    runtime=build_constraint_runtime(world,max_entities=3,max_operations=2)
    ctx=context();ctx["planning_facts"]["entity_properties"]["cup"]["graspable"]=True
    ctx["planning_facts"]["research"]={"grounding":{"complete":True,"roles":{"object":["apple","cup"]},
        "requests":[{"action":"pick","role":"object"}]}}
    policy=runtime.dfa_for("vlabench",ctx)
    assert policy.action_reachability().can_reach(remaining_actions=2) is expected
    decoded=native(policy)
    assert decoded.can_reach_goal() is expected
    if expected:
        state=decoded.start_state
        for token in labels(plan()):
            assert token in decoded.allowed_tokens(state)
            state=decoded.step(state,token)


@pytest.mark.parametrize("formula,raw",[
    ("research__fresh_observed",freshness("predicted")),
    ("research__fresh_current",freshness(step=2)),
    ("research__fresh_dependencies",{**freshness(),"evidence":{**freshness()["evidence"],"revisions":{"object:a":1}}}),
])
def test_freshness_ablation_agrees_in_datanode_tokens_and_strict_reachability(formula,raw):
    from ..world_graph import build_vlabench_world_graph,materialize_plan,verify_plan_constraints
    from ..training import build_constraint_runtime
    from ..plan_codec import plan_to_tokens
    from .test_action_reachability import context,plan
    world=replace(build_vlabench_world_graph("research_fresh_paths"),research_config=ResearchConfig("research_v1"))
    runtime=build_constraint_runtime(world,max_entities=3,max_operations=2)
    ctx=context();ctx["planning_facts"]["research"]=raw
    encoded=[runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(plan(),ctx["entities"],world=world)]
    first=runtime.dfa_for("vlabench",ctx)
    assert not first.accepts(encoded) and first.action_reachability().can_reach(remaining_actions=2) is False
    world.research_knowledge.get().set_active(False,names={formula})
    second=runtime.dfa_for("vlabench",ctx)
    assert second.accepts(encoded) and second.action_reachability().can_reach(remaining_actions=2) is True
    root=materialize_plan(plan(),ctx["entities"],world,policy=second)
    assert verify_plan_constraints(root,world).score==1.
