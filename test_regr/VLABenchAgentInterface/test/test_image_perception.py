"""Semantic counterexamples for image-only evidence and the real pipeline adapters."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

from ..research.perception import (PerceptionConfig,PerceptionUnavailable,ImagePerceptionModel,
    compile_evidence,parse_response,perceive,enrich_examples,ImageObservationAdapter,merge_context)
from ..research.evidence import EvidenceLedger
from ..research.spatial import compare
from ..research import config


def payload():
    return {"objects":[{"id":i,"confidence":.95,"bbox":box,"description":description,
        "states":{"visible":True,"closed":True}} for i,box,description in
        ((0,[.1,.1,.3,.3],"red apple"),(1,[.6,.6,.9,.9],"blue bowl"))],
        "requests":[{"action":"pick","role":"object","argument":"subject"}],
        "hypotheses":[{"id":"h0","bindings":{"object":0},"weight":1.}],"complete":True}


def evidence(data=None,**kwargs):
    return compile_evidence(data or payload(),config=PerceptionConfig(),episode="e",step=0,
        entity_table=("SECRET_ACE","SECRET_KING"),**kwargs)


class Observer:
    config=PerceptionConfig()
    def __init__(self,rows=None):self.rows=iter(rows or [payload()]);self.calls=[]
    def infer(self,raw,numbered,instruction):
        self.calls.append((raw.size,numbered.size,instruction))
        row=next(self.rows)
        if isinstance(row,Exception):raise row
        return row


def test_numeric_bindings_never_recover_identity_from_metadata():
    result=evidence()
    assert "SECRET" not in json.dumps(result)
    ledger=EvidenceLedger.parse(result["evidence"])
    assert ledger.fresh("image_bbox:0")
    assert ledger.get("image_bbox:0").units=="normalized_image"
    assert ledger.get("image_bbox:0").value==pytest.approx((.09,.09,.31,.31))
    assert result["grounding"]["hypotheses"][0]["bindings"]["object"]==0
    assert result["perception"]["confidence_calibration"]=="unvalidated_model_report"


def test_boxes_on_the_qwen_thousand_grid_are_normalized():
    data=payload();data["objects"][0]["bbox"]=[100,100,300,300]
    assert EvidenceLedger.parse(evidence(data)["evidence"]).get("image_bbox:0").value==pytest.approx((.09,.09,.31,.31))
    for bad in ([100,100,1300,300],[300,100,100,300]):
        data=payload();data["objects"][0]["bbox"]=bad
        with pytest.raises(PerceptionUnavailable):evidence(data)


def test_missing_placement_destination_cannot_bypass_observation_gate():
    result=perceive(Observer(),Image.new("RGB",(16,16)),Image.new("RGB",(16,16)),
        "Put the apple into the bowl",episode="e",step=0)
    assert result["grounding"]["complete"] is False
    assert EvidenceLedger.parse(result["evidence"]).value("perception:ready") is None
    assert result["perception"]["request_coverage"]=="missing_destination"


@pytest.mark.parametrize("field,value",[("position_m",[1,2,3]),("joint_angle",1.3),("intended_target",0)])
def test_metric_and_supervision_claims_are_rejected(field,value):
    data=payload();data["objects"][0][field]=value
    with pytest.raises(PerceptionUnavailable):evidence(data)


def test_occluded_low_confidence_and_unreadable_faces_are_unavailable():
    data=payload();data["objects"][0]["confidence"]=.3
    result=evidence(data)
    assert not result["grounding"]["complete"] and not result["grounding"]["hypotheses"]
    assert EvidenceLedger.parse(result["evidence"]).get("image_bbox:0") is None
    data=payload();data["objects"][0]["states"]["grasped"]=True
    with pytest.raises(PerceptionUnavailable):evidence(data)


def test_duplicate_keys_pointers_unknown_bindings_and_budget():
    with pytest.raises(PerceptionUnavailable):parse_response('{"objects":[],"objects":[]}')
    data=payload();data["objects"].append(deepcopy(data["objects"][0]))
    with pytest.raises(PerceptionUnavailable):evidence(data)
    data=payload();data["objects"][0]["id"]=999
    with pytest.raises(PerceptionUnavailable):evidence(data)
    from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
    data=payload();data["hypotheses"]=[{"id":str(i),"bindings":{"object":0}} for i in range(129)]
    with pytest.raises(SemanticSearchLimitExceeded):evidence(data)


def test_pointer_request_arguments_come_from_the_declarative_skill_schema():
    data=payload();data["requests"]=[{"action":"pick","role":"object"},{"action":"place","role":"destination"}]
    data["hypotheses"]=[{"id":"h","bindings":{"object":"0","destination":"1"}}]
    result=evidence(data)
    assert [r["argument"] for r in result["grounding"]["requests"]]==["subject","destination"]
    assert result["grounding"]["hypotheses"][0]["bindings"]=={"object":0,"destination":1}


def test_image_relations_do_not_turn_into_metric_or_physical_facts():
    ledger=EvidenceLedger.parse(evidence()["evidence"])
    requirement={"left":"image_bbox:0","right":"image_bbox:1","frame":"image:0","geometry":"image_boxes"}
    assert compare({**requirement,"relation":"left"},ledger) is True
    assert compare({**requirement,"relation":"above"},ledger) is True
    assert compare({**requirement,"relation":"below"},ledger) is False
    for relation in ("inside","on","near","front"):
        assert compare({**requirement,"relation":relation,"threshold_m":1},ledger) is None
    assert compare({**requirement,"relation":"left","frame":"world"},ledger) is None
    assert compare({**requirement,"relation":"left","geometry":"boxes"},ledger) is None
    assert compare({**requirement,"relation":"left"},ledger.advance(1)) is None


@pytest.mark.parametrize("ambiguous",[False,True])
def test_perceived_groundings_enforce_the_existing_lcs_in_decoding_and_datanodes(ambiguous):
    from ..research import ResearchConfig
    from ..training import build_constraint_runtime
    from ..world_graph import build_vlabench_world_graph,materialize_plan,verify_plan_constraints
    from ..plan_codec import plan_to_tokens
    data=payload()
    for row in data["objects"]:row["states"]={"visible":True}
    if ambiguous:data["hypotheses"].append({"id":"h1","bindings":{"object":1},"weight":1})
    ctx=merge_context({"entities":("apple","bowl")},evidence(data))
    world=replace(build_vlabench_world_graph("perceived_grounding"),research_config=ResearchConfig("research_v1"))
    runtime=build_constraint_runtime(world,max_entities=2,max_operations=2)
    policy=runtime.dfa_for("vlabench",ctx)
    plan=[{"name":"pick","params":{"target_entity_name":0}},{"name":"place","params":{"target_container_name":1}}]
    labels=[runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(plan,ctx["entities"],world=world)]
    assert policy.accepts(labels) is (not ambiguous)
    assert (verify_plan_constraints(materialize_plan(plan,ctx["entities"],world,policy=policy),world).score==1.) is (not ambiguous)


def test_existing_visibility_lc_consumes_perception_support_without_grounding_rules():
    from ..training import build_constraint_runtime
    from ..world_graph import materialize_plan,verify_plan_constraints
    from ..plan_codec import plan_to_tokens
    runtime=build_constraint_runtime(max_entities=2,max_operations=2,name_prefix="perceived_visibility")
    plan=[{"name":"pick","params":{"target_entity_name":0}},{"name":"place","params":{"target_container_name":1}}]
    for confidence,expected in ((.95,True),(.2,False)):
        data=payload();data["objects"][1]["confidence"]=confidence
        for row in data["objects"]:row["states"]={"visible":True}
        ctx=merge_context({"entities":("apple","bowl")},evidence(data))
        policy=runtime.dfa_for("vlabench",ctx)
        labels=[runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(plan,ctx["entities"],world=runtime.world_bundle)]
        assert policy.accepts(labels) is expected
        assert (verify_plan_constraints(materialize_plan(plan,ctx["entities"],runtime.world_bundle,policy=policy),runtime.world_bundle).score==1.) is expected


def test_static_image_enrichment_discards_metadata_and_uses_only_paired_pixels(tmp_path):
    from ..dataset import PlanningExample
    raw=tmp_path/"raw.png";numbered=tmp_path/"numbered.png"
    Image.new("RGB",(32,32)).save(raw);Image.new("RGB",(32,32),"yellow").save(numbered)
    example=PlanningExample("e","pick red apple",(),(raw,),(numbered,),None,
        entities=("SECRET_ACE","SECRET_KING"),task_target_entity="SECRET_KING",
        planning_facts={"joint_observations":[{"value":1.3}],"research":{"events":[{"name":"secret"}]}})
    model=Observer();observed=enrich_examples([example],model)[0]
    assert model.calls==[((32,32),(32,32),"pick red apple")]
    assert observed.task_target_entity is None and set(observed.planning_facts)=={"research","initial_state"}
    assert observed.visible_entities==(0,1)
    assert observed.planning_facts["initial_state"]["positive"]==[["closed",0],["closed",1]]
    assert observed.planning_facts["research"]["events"]==[]
    assert observed.planning_facts["research"]["freshness"][-1]["evidence"]=="perception:ready"
    with pytest.raises(ValueError,match="unambiguous"):
        enrich_examples([replace(example,image_paths=(raw,raw))],Observer())


def test_visual_changes_require_successive_known_captures_and_reset_breaks_history():
    opened=payload();opened["objects"][0]["states"]["closed"]=False
    occluded=payload();occluded["objects"][0]["confidence"]=.1
    model=Observer([payload(),opened,occluded,payload(),opened])
    adapter=ImageObservationAdapter(SimpleNamespace(),{"instruction":"pick apple"},model)
    images=[Image.new("RGB",(16,16)),Image.new("RGB",(16,16))]
    def read(step,episode="e"):
        return adapter(step=step,episode=episode,revisions={},observation={"research_image_pair":images,"entity_table":("a","b")})
    assert not read(0)["events"]
    second=read(1);assert len(second["events"])==1
    assert second["events"][0]["name"]=="visual_closed:0:false"
    assert not any(e["name"]=="pick" for e in second["events"])
    read(2);assert len(read(3)["events"])==1 # reacquisition is not a new opening/closing event
    assert not read(0,"reset")["events"]


def test_actual_runtime_replaces_privileged_reader_and_bad_refresh_cannot_reuse_fact(monkeypatch):
    from ..research.runtime import reconcile,admission
    from ..research.sensors import attach_sensor_adapter
    from ..world_graph import build_vlabench_world_graph
    from ..research import ResearchConfig
    model=Observer([payload(),PerceptionUnavailable("occluded")])
    monkeypatch.setattr("test_regr.VLABenchAgentInterface.research.perception.configured_model",lambda:model)
    token=config._PERCEPTION_OPTIONS.set(asdict(model.config))
    try:
        def privileged(**kwargs):raise AssertionError("privileged reader must not run")
        env=SimpleNamespace(read_research_observations=privileged)
        ctx={"instruction":"pick apple","research_config":ResearchConfig("research_v1").as_dict(),
            "entities":("a","b"),"planning_facts":{"joint_observations":[{"value":1.3}]}}
        attach_sensor_adapter(env,ctx)
        observation={"research_image_pair":[Image.new("RGB",(16,16)),Image.new("RGB",(16,16))],"entity_table":("a","b")}
        ctx=reconcile(env,ctx,step=0,observation=observation)
        world=replace(build_vlabench_world_graph("image_runtime"),research_config=ResearchConfig("research_v1"))
        assert admission(ctx,world).accepted
        assert "joint_observations" not in ctx["planning_facts"]
        assert ["closed",0] in ctx["planning_facts"]["initial_state"]["positive"]
        assert ctx["visible_entities"]==[0,1]
        ctx=reconcile(env,ctx,step=0,observation=observation)
        assert admission(ctx,world).status=="needs_observation"
        assert not EvidenceLedger.parse(ctx["planning_facts"]["research"]["evidence"]).fresh("perception:ready")
    finally:config._PERCEPTION_OPTIONS.reset(token)


def test_frozen_backbone_runs_real_generation_interface_and_cache_without_metadata():
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=0
        def generate(self,**kwargs):
            self.calls+=1;assert not self.training and not torch.is_grad_enabled()
            assert kwargs["do_sample"] is False
            return torch.tensor([[1,2,3]])
    class Batch(dict):
        def to(self,device):return self
    class Processor:
        tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:json.dumps(payload()))
        def apply_chat_template(self,messages,**kwargs):
            self.messages=messages;return "encoded"
        def __call__(self,**kwargs):
            assert len(kwargs["images"])==2
            return Batch(input_ids=torch.tensor([[1,2]]))
    model=Model();processor=Processor();observer=ImagePerceptionModel(model,processor)
    raw=Image.new("RGB",(16,16));numbered=Image.new("RGB",(16,16),"yellow")
    for _ in range(2):perceive(observer,raw,numbered,"pick apple",episode="e",step=0)
    assert model.calls==1 and not model.weight.requires_grad
    assert len(processor.messages[0]["content"])==3
    assert "SECRET" not in processor.messages[0]["content"][-1]["text"]
    with pytest.raises(PerceptionUnavailable):observer.infer(raw,Image.new("RGB",(8,8)),"pick apple")


def test_image_mode_never_refreshes_numeric_joints_from_simulator():
    from ..articulation import refreshed_articulation_context
    from ..models import planner_prompt
    from ..training import build_constraint_runtime
    class Forbidden:
        def __getattr__(self,name):raise AssertionError("Simulator physics cannot supply visual evidence")
    token=config._PERCEPTION_OPTIONS.set(asdict(PerceptionConfig()))
    try:
        ctx={"task":"set_study_table","planning_facts":{"research":{"perception":{}}}}
        assert refreshed_articulation_context(Forbidden(),ctx,("SECRET",))==ctx
        runtime=build_constraint_runtime(max_entities=1,max_operations=1,name_prefix="visual_prompt")
        assert "SECRET" not in planner_prompt("pick object",("SECRET",),runtime.vocabulary)
    finally:config._PERCEPTION_OPTIONS.reset(token)


def test_perception_configuration_is_explicit_and_checkpointed():
    from ..main import build_parser
    from ..research.checkpoint import checkpoint_metadata,restore_metadata
    from ..world_graph import build_vlabench_world_graph
    args=build_parser().parse_args(["train-planner","--planning-dir","data","--output","out",
        "--constraint-profile","research_v1","--research-perception","image-vlm"])
    old=(config._DEFAULT.get(),config._PERCEPTION_OPTIONS.get(),config._COMPILED_LOSS.get(),config._TRAIN_OPTIONS.get())
    try:
        config.apply_research_arguments(args)
        world=build_vlabench_world_graph("image_checkpoint")
        planner=SimpleNamespace()
        saved=checkpoint_metadata(world,planner)
        assert saved["research_configuration"]["perception_configuration"]["confidence"]==.85
        restore_metadata(saved,world,planner,training=True)
        config._PERCEPTION_OPTIONS.set(None)
        with pytest.raises(ValueError,match="perception"):restore_metadata(saved,world,planner,training=True)
    finally:
        config._DEFAULT.set(old[0]);config._PERCEPTION_OPTIONS.set(old[1]);config._COMPILED_LOSS.set(old[2]);config._TRAIN_OPTIONS.set(old[3])


@pytest.mark.parametrize("joint",[False,True])
@pytest.mark.parametrize("profile,selection,expected",[
    ("baseline",None,"off"),("research_v1",None,"image-vlm"),
    ("research_v1","off","off"),("baseline","image-vlm","image-vlm")])
def test_profile_aware_perception_default_in_both_main_commands(joint,profile,selection,expected):
    from contextvars import copy_context
    from importlib import import_module
    module="test_regr.JointEmbodiedAgentInterface.main" if joint else "test_regr.VLABenchAgentInterface.main"
    base=["train-agent","--two-stage"] if joint else ["train-planner","--planning-dir","data","--output","out"]
    flags=["--constraint-profile",profile,"--planner-model","/original/backbone"]
    if selection is not None:flags.extend(["--research-perception",selection])
    if profile=="baseline" and expected=="image-vlm":flags.append("--use-compiled-planner-loss")
    def check():
        args=import_module(module).build_parser().parse_args([*base,*flags])
        config.apply_research_arguments(args)
        assert args.research_perception==expected
        options=config.default_perception_options()
        if expected=="off":assert options is None
        else:assert options["model_id"]=="/original/backbone"
    copy_context().run(check)
