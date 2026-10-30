"""Model-dependent DomiKnowS LC loss over the compiled policy's admissible choices."""
from itertools import count
import torch
from torch import nn

_IDS=count()


class ProjectedPrimalDual(nn.Module):
    """The original research formulas determine admitted; this LC penalizes model mass violating them.

    Unlike a constant sensor-violation score, chosen probabilities retain their
    connection to planner logits. LossModel supplies the actual DomiKnowS duals.
    """
    def __init__(self,device="cpu",dual_lr=.05,dual_warmup=10,dual_every=10):
        super().__init__()
        from domiknows.graph import Graph,Concept
        from domiknows.graph.logicalConstrain import ifL
        from domiknows.program.model.lossModel import LossModel
        with Graph("research_probability_projection_"+str(next(_IDS))) as graph:
            episode=Concept(name="projection_episode")
            candidate=Concept(name="projection_candidate")
            self.chosen=candidate(name="model_chosen")
            self.admitted=candidate(name="compiled_policy_admitted")
            ifL(self.chosen("c"),self.admitted("c"),name="research__model_choice_admitted")
        self.graph,self.episode,self.candidate=graph,episode,candidate
        self.dual=LossModel(graph,device=device,tnorm="P",compile_lc=True)
        self.optimizer=torch.optim.Adam(self.dual.parameters(),lr=dual_lr)
        self.register_buffer("rounds",torch.zeros((),dtype=torch.long))
        self.dual_warmup,self.dual_every=dual_warmup,dual_every
        self.last_violation=None

    def forward(self,logits,sets):
        from domiknows.graph import DataNode,DataNodeBuilder
        probabilities=torch.softmax(logits.float(),dim=-1)
        builder=DataNodeBuilder();builder.skeletonDataNodeFull=False
        root=DataNode(myBuilder=builder,instanceID=0,ontologyNode=self.episode,attributes={})
        dict.__setitem__(builder,"dataNode",[root])
        root.current_device=logits.device
        serial=0
        pairs=sorted(sets.items()) if isinstance(sets,dict) else list(enumerate(sets))
        for t,allowed in pairs:
            for label in set(range(logits.shape[-1]))-set(allowed):
                p=probabilities[t,label]
                node=DataNode(myBuilder=builder,instanceID=serial,ontologyNode=self.candidate,attributes={})
                node.attributes["<model_chosen>"]=torch.stack((torch.log1p(-p.clamp(max=1-1e-7)),torch.log(p.clamp(min=1e-7))))
                node.attributes["<compiled_policy_admitted>"]=p.new_tensor([30.,-30.])
                node.attributes["<model_chosen>/local/softmax"]=torch.stack((1-p,p))
                node.attributes["<compiled_policy_admitted>/local/softmax"]=p.new_tensor([1.,0.])
                root.addChildDataNode(node);serial+=1
        if not serial or not any(lc.active for lc in self.graph.logicalConstrains.values()):
            self.last_violation=logits.sum()*0
            return self.last_violation
        results=root.calculateLcLoss(tnorm="P",compiled=True)
        terms=[]
        for key,lc in self.graph.logicalConstrains.items():
            if not lc.active:continue
            value=results.get(key,{})
            loss=value.get("lossTensor")
            if loss is not None:
                violation=torch.nan_to_num(loss,nan=0.).sum()/max(1,len(sets))
                terms.append(self.dual.get_lmbd(key).detach().clone()*violation)
                self.last_violation=violation
        if not terms:raise RuntimeError("research LC projection produced no differentiable loss")
        return torch.stack(terms).sum()

    def step_dual(self):
        if self.last_violation is None:return
        self.rounds+=1
        if self.rounds.item()<=self.dual_warmup or self.rounds.item()%self.dual_every:return
        self.optimizer.zero_grad(set_to_none=True)
        objective=sum(self.dual.get_lmbd(k)*self.last_violation.detach() for k,lc in self.graph.logicalConstrains.items() if lc.active)
        (-objective).backward();self.optimizer.step();self.dual.project_lmbd_()


def research_training_loss(logits,labels,dfa,*,mode="full",projector=None,max_steps=None,mass_weight=0.,pmd_weight=1.):
    from domiknows.generation.learners.common.losses import allowed_label_sets,allowed_log_mass,masked_cross_entropy
    from domiknows.generation.latent.losses import GenerationLossWeights,compute_generation_training_loss
    labels=labels.reshape(-1);logits=logits.reshape(-1,logits.shape[-1])
    zero=logits.sum()*0
    if mode=="ce":return torch.nn.functional.cross_entropy(logits,labels),{"supervised_only":True}
    sets=allowed_label_sets(dfa,labels,logits.shape[-1],lookahead=True,max_steps=max_steps or len(labels))
    if len(sets)!=len(labels):return zero,{"admission":"needs_observation","blocked_samples":1}
    ce,misses=masked_cross_entropy(logits,labels,sets)
    if misses:raise ValueError("supervision is not admitted by the research constraints; do not invent an intent target")
    if mode=="masked-ce":return ce,{"masked_ce":float(ce.detach())}
    mass=-allowed_log_mass(torch.log_softmax(logits.float(),dim=-1),sets).mean()
    pmd=projector(logits,sets) if projector is not None else zero
    breakdown=compute_generation_training_loss(supervised_loss=ce,pmd_loss=pmd,allowed_mass_loss_value=mass,
        weights=GenerationLossWeights(supervised=1.,pmd=pmd_weight,allowed_mass=mass_weight))
    return breakdown.total,{**breakdown.as_float_dict(),"blocked_samples":0}
