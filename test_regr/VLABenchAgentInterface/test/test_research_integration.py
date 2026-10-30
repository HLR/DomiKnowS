"""Observation/runtime and training lifecycle counterexamples without simulator privileges."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from ..research import ResearchConfig,EvidenceLedger
from ..research.runtime import reconcile,admit_with_refresh,record_measured_completion
from ..research.sensors import attach_sensor_adapter
from ..world_graph import build_vlabench_world_graph


def world():return replace(build_vlabench_world_graph("research_integration"),research_config=ResearchConfig("research_v1"))


def context(raw):
    return {"entities":("apple","bowl"),"research_config":ResearchConfig("research_v1").as_dict(),"planning_facts":{"research":raw}}


@pytest.mark.parametrize("matched",[False,True])
def test_legacy_intended_target_annotations_cannot_resolve_research_grounding(matched):
    from test_regr.JointEmbodiedAgentInterface.utils.planner_context import planner_context
    from ..task_policy import task_requirements
    config=ResearchConfig("baseline" if matched else "research_v1")
    source={"entities":("apple","bowl"),"research_config":config.as_dict(),"research_matched":matched,
        "task_target_entity":"apple","task_target_container":"bowl",
        "planner_context":{"reference_plan":["pick apple"],"intended_target":"apple"}}
    prepared=planner_context("vlabench",source)
    assert not set(prepared)&{"task_target_entity","task_target_container","reference_plan","intended_target"}
    requirements=task_requirements(source,replace(world(),research_config=config),source["entities"])
    assert requirements.initial_target is None
    # Existing profile-off and EAI input contracts stay intact.
    old={**source,"research_config":ResearchConfig().as_dict(),"research_matched":False}
    assert planner_context("vlabench",old)["task_target_entity"]=="apple"
    assert planner_context("eai",source)["task_target_entity"]=="apple"


def test_refresh_once_and_sensor_reset_does_not_invent_events():
    env=SimpleNamespace();raw={"evidence":{"episode":"e","step":0},
        "freshness":[{"action":"pick","evidence":"held"}],"sensors":[{"id":"held","field":"held",
        "events":[{"name":"pick","to_nonempty":True,"entities":["$value"]}]}]}
    ctx=context(raw);attach_sensor_adapter(env,ctx)
    ctx=reconcile(env,ctx,step=0,observation={"grasped_obj_name":None})
    assert not ctx["planning_facts"]["research"].get("events")
    ctx=reconcile(env,ctx,step=1,changed=("robot",),observation={"grasped_obj_name":"apple"})
    ctx,completed=record_measured_completion(ctx,{"name":"pick","params":{"target_entity_name":0}})
    assert completed and ctx["planning_facts"]["completed_skill_prefix"]==["pick"]
    assert not record_measured_completion(ctx,{"name":"pick","params":{"target_entity_name":0}})[1]
    reset=reconcile(env,ctx,step=0,episode="new",observation={"grasped_obj_name":None})
    assert not reset["planning_facts"]["research"].get("events")
    assert "completed_skill_prefix" not in reset["planning_facts"]
    calls=[]
    def unavailable(**kwargs):calls.append(kwargs);return {}
    env.read_research_observations=unavailable
    reset,decision=admit_with_refresh(env,reset,world(),step=1,operation={"name":"pick","params":{"target_entity_name":0}})
    assert decision.accepted # measured known empty hand remains a valid observation at its captured step
    stale=reconcile(env,reset,step=2)
    _,decision=admit_with_refresh(env,stale,world(),step=2,operation={"name":"pick","params":{"target_entity_name":0}})
    assert decision.status=="needs_observation" and decision.refreshed
    assert len(calls)==2


def test_motion_invalidates_object_dependencies_and_vectors_are_serializable():
    env=SimpleNamespace()
    ctx=context({"evidence":{"episode":"e","step":0},"sensors":[{"id":"pose","observation_key":"pose",
        "entity":"apple","max_age_steps":None,"frame":"world","units":"m"}]})
    attach_sensor_adapter(env,ctx)
    ctx=reconcile(env,ctx,step=0,observation={"pose":np.array([0.,1.,2.])})
    assert EvidenceLedger.parse(ctx["planning_facts"]["research"]["evidence"]).fresh("pose")
    ctx=reconcile(env,ctx,step=1,changed=("robot",),observation={})
    assert not EvidenceLedger.parse(ctx["planning_facts"]["research"]["evidence"]).fresh("pose")
    assert "completed_skill_prefix" not in ctx["planning_facts"]


def test_upstream_grasp_tuple_is_measured_without_serializing_live_entities():
    from ..research.sensors import held_name
    assert held_name((["apple"],[object()]))=="apple"
    assert held_name(([],[]))==""
    assert held_name((["apple","cup"],[object(),object()])) is None


def test_reset_and_camera_change_invalidate_supplied_evidence():
    ctx=context({"evidence":{"episode":"e","step":4,"revisions":{"camera:front":0},"records":[
        {"id":"pose","value":[0,0,0],"max_age_steps":None,"dependencies":{"camera:front":0}}]},
        "grounding":{"roles":{"object":["apple"]},"complete":True,"requests":[{"action":"pick","role":"object"}]}})
    env=SimpleNamespace()
    changed=reconcile(env,ctx,step=4,changed=("camera",))
    assert not EvidenceLedger.parse(changed["planning_facts"]["research"]["evidence"]).fresh("pose")
    reset=reconcile(env,ctx,step=0,changed=("reset",))
    assert not reset["planning_facts"]["research"]["evidence"]["records"]
    assert reset["planning_facts"]["research"]["grounding"]["hypotheses"]==[]
    assert reset["planning_facts"]["research"]["grounding"]["requests"]
    env.read_research_observations=lambda **kwargs:{"grounding":{"roles":{"object":["apple"]},"complete":True}}
    refreshed=reconcile(env,reset,step=0)
    assert refreshed["planning_facts"]["research"]["grounding"]["roles"]=={"object":["apple"]}
    from ..research.runtime import blocked_admission
    assert blocked_admission(reset,world()).status=="needs_clarification"


def test_unavailable_spatial_frames_request_observation():
    from ..research.runtime import blocked_admission
    from .test_research_constraints import spatial_raw
    raw=spatial_raw([0,0,0],[1,0,0],frame="camera")
    raw["spatial"][0].update(phase="operation",action="pick")
    assert blocked_admission(context(raw),world()).status=="needs_observation"


def test_sampled_geometry_is_isolated_and_revision_bound():
    from ..research.geometry import check_sampled_path
    ledger=EvidenceLedger.parse({"episode":"e","step":2,"revisions":{"held_geometry":1}})
    live=np.array([0.]);clones=[]
    class Clone:
        complete_pairs=True
        held_geometry_complete=True
        def __enter__(self):self.state=live.copy();clones.append(self);return self
        def __exit__(self,*_):pass
        def set_configuration(self,value):self.state[:]=value
        def signed_distances(self,margin):return [("held","wall",.1-float(self.state[0]))]
    proof=check_sampled_path([[0.],[.2]],ledger,Clone,resolution=.02)
    assert proof.status=="blocked" and live[0]==0 and proof.continuous_safety is False
    assert check_sampled_path([[0.],[.2]],ledger,Clone,allowed_contacts=[("held","wall")]).status=="clear"
    assert check_sampled_path([[0.]],ledger,None).status=="unavailable"
    ctx=context({"evidence":ledger.as_dict(),"geometry_certificate":proof.as_dict(),"proposed_path_digest":proof.path_digest})
    from ..research.runtime import admission
    ctx["research_config"]["motion_geometry"]="sampled"
    # A stale clear certificate must not bypass the geometry-known LC.
    ctx["planning_facts"]["research"]["geometry_certificate"]["status"]="clear"
    ctx["planning_facts"]["research"]["evidence"]["revisions"]["held_geometry"]=2
    assert admission(ctx,world(),operation={"name":"motion_chunk"},execution=True).status=="needs_observation"
    from ..research.runtime import geometry_admission
    selected=world()
    assert geometry_admission(SimpleNamespace(),ctx,selected,None)[1].status=="needs_observation"
    selected.research_knowledge.get().set_active(False,names={"research__geometry_known"})
    assert geometry_admission(SimpleNamespace(),ctx,selected,None)[1].accepted


def test_canonical_rollout_refreshes_once_then_stops_without_an_unobserved_action():
    from ..training import build_constraint_runtime
    from ..simulation.controller import MultiViewController
    from .test_pipeline import TinyCompactPlanner,TinyImageEncoder,FakeSimulator,_joint_program
    runtime=build_constraint_runtime(world(),max_entities=2,max_operations=2)
    planner=TinyCompactPlanner(runtime.vocabulary)
    controller=MultiViewController(TinyImageEncoder(8),hidden_dim=8,action_horizon=1,max_views=1)
    env=FakeSimulator(success=False);reads=[]
    def unavailable(**kwargs):reads.append(kwargs["step"]);return {}
    env.read_research_observations=unavailable
    program=_joint_program(runtime,planner,controller,lambda **kwargs:env,num_samples=1)
    episode=program.collect_episode({"task":"select_book","task_skill_sequence":("pick","place"),
        "planning_facts":{"research":{"freshness":[{"action":"pick","evidence":"face"}]}}})
    assert env.count==0 and not episode.success
    assert episode.termination_reason=="needs_observation"
    assert len(reads)==2 and planner.calls==2
    assert sum(d.get("refreshed",False) for d in episode.diagnostics["research"]["decisions"])==1


def test_groups_and_true_boundaries(tmp_path):
    from ..research.data import grouped_split,SegmentedControlDataset,collection_gate
    items=[SimpleNamespace(scene_group=str(i//2),research_split=None) for i in range(40)]
    splits=grouped_split(items)
    assert [len(splits[k]) for k in ("train","validation","calibration","test")]==[28,4,4,4]
    items[0].research_split="train";items[1].research_split="test"
    with pytest.raises(ValueError,match="crosses"):grouped_split(items)
    np.savez(tmp_path/"trace.npz",state=np.arange(35).reshape(5,7),actions=np.arange(35).reshape(5,7),images=np.zeros((5,1,4,4,3),dtype=np.uint8))
    row={"segment_id":"s","file":"trace.npz","start_frame":2,"end_frame":4,"skill":"pick","instruction":"pick apple",
         "operation_index":0,"episode_index":0,"measured_task_success":True,"postcondition_verified":True,"stage_success":True,
         "boundary_source":"expert_call","controller_conditioning_version":2,"trace_capture_version":2}
    ds=SegmentedControlDataset(tmp_path,[row],{0:"pick apple"},action_horizon=4)
    assert ds[0]["state"][:,0].tolist()==[14.,14.]
    assert ds[0]["actions"][:,0].tolist()==[14.,21.,21.,21.]
    assert not collection_gate([row],20)["passed"]
    row["boundary_source"]="equal_duration"
    with pytest.raises(ValueError):SegmentedControlDataset(tmp_path,[row],{0:"pick apple"})


def test_conditioning_uses_sorted_scene_pointer_and_unknown_compact_position():
    from ..research.runtime import controller_context
    from ..world_domain import controller_skill_index
    assert controller_context([{"name":"place","params":{"target_container_name":1}}],0,("apple","bowl"))==(controller_skill_index("place"),2,0)
    assert controller_context([{"name":"lift","params":{}}],0,("apple","bowl"))==(controller_skill_index("lift"),0,0)


def test_research_loss_ablations_leave_eai_configuration_unchanged():
    from test_regr.JointEmbodiedAgentInterface.main import _dfa_loss_config
    values=dict(stage1_dfa_loss="on",stage1_masked_ce_domains="vlabench",stage1_dfa_mass_weight=.3,
        stage1_pmd_weight=1.,stage1_dual_lr=.05,stage1_dfa_mass_steps=12,stage1_dfa_domains="eai,vlabench",
        stage1_dfa_lookahead_domains="vlabench",stage1_pmd_domains="eai,vlabench")
    base=_dfa_loss_config(SimpleNamespace(**values))
    for mode in ("full","masked-ce","ce"):
        config=_dfa_loss_config(SimpleNamespace(**values,constraint_profile="research_v1",research_supervised_mode=mode))
        assert ("eai" in config.domains)==("eai" in base.domains)
        assert ("eai" in config.pmd_domains)==("eai" in base.pmd_domains)
        assert ("eai" in config.masked_ce_domains)==("eai" in base.masked_ce_domains)
        assert config.mass_weight==base.mass_weight and config.pmd_weight==base.pmd_weight


def test_matched_baseline_and_research_get_identical_observation_text():
    from ..research import config
    from ..research.prompt import observation_prompt
    raw={"evidence":{"episode":"e","step":0,"records":[{"id":"pose","value":[0,0,0]}]}}
    baseline=context(raw);baseline["research_config"]=ResearchConfig().as_dict()
    token=config._COMPILED_LOSS.set(False)
    try:
        assert observation_prompt(baseline)==""
        config._COMPILED_LOSS.set(True)
        assert observation_prompt(baseline)==observation_prompt(context(raw))
    finally:config._COMPILED_LOSS.reset(token)


def test_joint_prior_enters_actual_decoding_and_replay_once():
    from ..training import build_constraint_runtime
    from ..models import QwenVLPlanner
    from .test_decoding_integration import TinyBackbone,TinyProcessor
    from domiknows.generation import ProductiveDFA
    runtime=build_constraint_runtime(replace(world(),research_config=ResearchConfig("research_v1","joint")),max_entities=3,max_operations=2)
    planner=QwenVLPlanner(TinyBackbone(),TinyProcessor(),runtime.vocabulary,decoder_hidden_size=4)
    with torch.no_grad():planner.output.weight.zero_();planner.output.bias.zero_()
    ctx={"instruction":"pick the object and place it","images":(),"entities":("a","b","bowl"),"entity_table":("a","b","bowl"),
        "research_config":runtime.world_bundle.research_config.as_dict(),
        "task_skill_sequence":("pick","place"),"planning_facts":{"research":{"grounding":{"complete":True,
            "hypotheses":[{"id":"a","bindings":{"object":"a"},"weight":1.},{"id":"b","bindings":{"object":"b"},"weight":3.}],
            "requests":[{"action":"pick","role":"object"}]}}}}
    dfa=ProductiveDFA(runtime.dfa_for("vlabench",ctx))
    labels,logprob=planner.sample_labels(ctx,dfa,max_steps=runtime.max_tokens,deterministic=True)
    tokens=[runtime.vocabulary.token_for_label(int(label)) for label in labels]
    assert tokens[2]=="obj:1"
    replay=planner.replay_labels_logprob(planner.prepare_replay_context(ctx),labels,dfa,max_steps=runtime.max_tokens)
    assert torch.allclose(logprob,replay,atol=1e-5)


def test_standalone_forward_update_resume_and_profile_fingerprint(tmp_path):
    from ..training import build_constraint_runtime,prepare_planner_program_examples
    from ..models import QwenVLPlanner
    from ..simulation.program import build_stage1_program
    from ..research import config
    from .test_decoding_integration import TinyBackbone,TinyProcessor
    runtime=build_constraint_runtime(world(),max_entities=2,max_operations=2)
    planner=QwenVLPlanner(TinyBackbone(),TinyProcessor(),runtime.vocabulary,decoder_hidden_size=4)
    example={"episode_id":"independent","instruction":"pick apple then place in bowl","entities":("apple","bowl"),
        "task_target_entity":"bowl","task_target_container":"apple",
        "operation_sequence":({"name":"pick","params":{"target_entity_name":0}},{"name":"place","params":{"target_container_name":1}}),
        "planning_facts":{"research":{"grounding":{"complete":True,"roles":{"object":["apple"]},"requests":[{"action":"pick","role":"object"}]}}}}
    options=config._TRAIN_OPTIONS.set({"output":str(tmp_path),"grad_accumulation":1})
    try:
        items=prepare_planner_program_examples([example],runtime)
        assert "task_target_entity" not in items[0]["planner_context"]
        bound=runtime.dfa_for("vlabench",example)
        assert "task_target_entity" not in bound.research_context
        program=build_stage1_program(runtime,planner)
        before=planner.output.bias.detach().clone()
        program.train(items,train_epoch_num=1,Optim=lambda params:torch.optim.SGD(params,lr=.1))
        assert not torch.equal(before,planner.output.bias)
        saved=planner.output.bias.detach().clone()
        with torch.no_grad():planner.output.bias.add_(1)
        program.train(items,train_epoch_num=1,Optim=lambda params:torch.optim.SGD(params,lr=.1))
        assert torch.equal(saved,planner.output.bias)
        from ..research.checkpoint import checkpoint_metadata,restore_metadata
        payload=checkpoint_metadata(runtime.world_bundle,planner)
        restore_metadata(payload,runtime.world_bundle,planner,training=True)
        with pytest.raises(ValueError,match="profile"):
            restore_metadata({},runtime.world_bundle,planner,training=True)
        off=replace(runtime.world_bundle,research_config=ResearchConfig())
        assert checkpoint_metadata(off,planner)=={}
    finally:config._TRAIN_OPTIONS.reset(options)
