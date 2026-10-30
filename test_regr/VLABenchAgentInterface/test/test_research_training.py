import torch
import pytest
from ..research.losses import ProjectedPrimalDual


def test_declared_projection_lc_has_model_gradient_and_disable_changes_it():
    projector=ProjectedPrimalDual()
    logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
    loss=projector(logits,[{0,1}])
    assert loss.requires_grad and loss.item()>0
    loss.backward()
    assert logits.grad[0,2]>0
    for lc in projector.graph.logicalConstrains.values():lc.active=False
    other=torch.zeros(1,3,requires_grad=True)
    disabled=projector(other,[{0,1}]);disabled.backward()
    assert disabled.item()==0 and other.grad.abs().sum()==0


def test_disabling_original_grounding_formula_changes_training_gradient():
    from dataclasses import replace
    from ..world_graph import build_vlabench_world_graph
    from ..training import build_constraint_runtime
    from ..research import ResearchConfig
    from ..plan_codec import plan_to_tokens
    from ..research.losses import research_training_loss
    world=replace(build_vlabench_world_graph("research_gradient_source"),research_config=ResearchConfig("research_v1"))
    runtime=build_constraint_runtime(world,max_entities=3,max_operations=2)
    context={"entities":("a","b","bowl"),"task_skill_sequence":("pick","place"),"planning_facts":{"research":{
        "grounding":{"roles":{"object":["a"]},"complete":True,"requests":[{"action":"pick","role":"object"}]}}}}
    plan=[{"name":"pick","params":{"target_entity_name":0}},{"name":"place","params":{"target_container_name":2}}]
    labels=torch.tensor([runtime.vocabulary.label_for_token(t) for t in plan_to_tokens(plan,context["entities"],world=world)])
    projector=ProjectedPrimalDual()
    def gradient():
        logits=torch.zeros(len(labels),runtime.vocabulary.label_count,requires_grad=True)
        loss,_=research_training_loss(logits,labels,runtime.dfa_for("vlabench",context),projector=projector,max_steps=runtime.max_tokens)
        loss.backward();return loss.detach(),logits.grad.clone()
    before,first=gradient()
    world.research_knowledge.get().set_active(False,names={"research__grounding_robust"})
    after,second=gradient()
    assert not torch.allclose(before,after)
    assert not torch.allclose(first,second)


@pytest.mark.parametrize("formula",[
    "fresh_observed","fresh_current","fresh_dependencies","spatial_known","spatial_holds",
    "temporal_known","temporal_holds","geometry_known","geometry_holds","grounding_complete",
    "grounding_robust","grounding_joint","grounding_unique","grounding_abstain_match","grounding_available"])
def test_every_original_formula_changes_admitted_mass_and_gradient(formula):
    from domiknows.generation.dfa.predicate_monitor import PredicateEvent
    from ..knowledge.research_lcs import build_research_knowledge
    knowledge=build_research_knowledge();name="research__"+formula
    kind="freshness" if formula.startswith("fresh_") else formula.split("_",1)[0]
    fields=("known","holds","freshness","observed","current","dependencies_valid","spatial","temporal","geometry","grounding","complete","robust","joint","abstain")
    values={(key,0):key in ("known","holds","observed","current","dependencies_valid","complete",kind) for key in fields}
    hypotheses=("h",)
    values["hypothesis","h"]=True;values["alive","h"]=True;values["compatible","h"]=True;values["different","h","h"]=False
    if kind=="freshness":
        key={"fresh_observed":"observed","fresh_current":"current","fresh_dependencies":"dependencies_valid"}[formula]
        values[key,0]=False
    elif kind in ("spatial","temporal","geometry"):
        values["known" if formula.endswith("known") else "holds",0]=False
    else:
        policy="joint" if formula in ("grounding_joint","grounding_available") else "abstain" if formula in ("grounding_unique","grounding_abstain_match") else "robust"
        values[policy,0]=True
        if formula=="grounding_complete":values["complete",0]=False
        elif formula in ("grounding_robust","grounding_joint","grounding_abstain_match"):values["compatible","h"]=False
        elif formula=="grounding_available":values["known",0]=False
        else:
            hypotheses=("h","j")
            values.update({("hypothesis","j"):True,("alive","j"):True,("compatible","j"):True,
                ("different","h","j"):True,("different","j","h"):True,("different","j","j"):False})
    event=PredicateEvent("research",values,"immediate",variable_domains={"r":(0,),"h":hypotheses,"j":hypotheses})
    def admitted():return all(observation.truth is True for _,observation in knowledge.relay(event))
    assert [observation.name for _,observation in knowledge.relay(event) if observation.truth is False]==[name]
    projector=ProjectedPrimalDual()
    def gradient():
        logits=torch.zeros(1,2,requires_grad=True)
        loss=projector(logits,{0:{0,1} if admitted() else {1}});loss.backward()
        return float(loss.detach()),logits.grad.clone()
    before,first=gradient()
    knowledge.set_active(False,names={name})
    assert admitted()
    after,second=gradient()
    assert before>after and not torch.equal(first,second)
