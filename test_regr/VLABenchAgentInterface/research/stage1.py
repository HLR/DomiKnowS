"""Standalone SolverPOI lifecycle with the same compiled loss components as joint training."""
from pathlib import Path
import hashlib,json,random
import torch
from domiknows.program import SolverPOIProgram
from .config import default_loss_mode,default_training_options
from .losses import ProjectedPrimalDual,research_training_loss
from .runtime import admission


class ResearchStage1Program(SolverPOIProgram):
    def __init__(self,runtime,planner,device="cpu"):
        from ..simulation.program import attach_planner_sensors,EOSMaskedCrossEntropyLoss
        from domiknows.program.metric import MacroAverageTracker
        super().__init__(runtime.generation_graph,poi=attach_planner_sensors(runtime,planner,device=device),
            inferTypes=["local/argmax"],loss=MacroAverageTracker(EOSMaskedCrossEntropyLoss(runtime.vocabulary.eos_label)),device=device,metric={})
        self.runtime,self.planner_head=runtime,planner
        planner._research_world=runtime.world_bundle
        options=default_training_options()
        self.projector=ProjectedPrimalDual(device,options.get("stage1_dual_lr",.05),options.get("stage1_dual_warmup",10),options.get("stage1_dual_every",10))
        planner._research_training_state_provider=lambda:{"projector":self.projector.state_dict(),
            "dual_optimizer":self.projector.optimizer.state_dict()}
        self.history=[]

    def loss_for(self,item):
        context=item["planner_context"]
        decision=admission(context,self.runtime.world_bundle)
        if not decision.accepted:
            return None,{"admission":decision.status,"blocked_samples":1}
        labels=item["target_plan_labels"].to(self.planner_head.device)
        labels=labels.reshape(-1)
        keep=(labels==self.runtime.vocabulary.eos_label).cumsum(-1)<=1
        from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
        try:
            policy=self.runtime.dfa_for("vlabench",context)
            if default_loss_mode()=="ce" and not policy.allowed_tokens(policy.start_state,remaining_steps=self.runtime.max_tokens):
                from .runtime import blocked_admission
                return None,{"admission":blocked_admission(context,self.runtime.world_bundle).status,"blocked_samples":1}
        except SemanticSearchLimitExceeded:return None,{"admission":"budget_exhausted","blocked_samples":1}
        logits=self.planner_head.sequence_logits(context,self.planner_head._shift_right(labels)).reshape(-1,self.runtime.vocabulary.label_count)
        options=default_training_options()
        try:
            return research_training_loss(logits[keep],labels[keep],policy,mode=default_loss_mode(),projector=self.projector,
                max_steps=self.runtime.max_tokens,mass_weight=options.get("stage1_dfa_mass_weight",0.),pmd_weight=options.get("stage1_pmd_weight",1.))
        except SemanticSearchLimitExceeded:return None,{"admission":"budget_exhausted","blocked_samples":1}

    def train(self,train_set,valid_set=None,*,train_epoch_num=3,Optim=None,**kwargs):
        from ..training import _planner_trainable_state,_load_planner_trainable_state
        options=default_training_options()
        optimizer=Optim(p for p in self.planner_head.parameters() if p.requires_grad) if Optim else torch.optim.AdamW(
            (p for p in self.planner_head.parameters() if p.requires_grad),lr=options.get("learning_rate",2e-4))
        items=list(train_set)
        dataset_id=hashlib.sha256(json.dumps([(i.get("episode_id"),i["planner_context"],i["target_plan_labels"].tolist()) for i in items],sort_keys=True,default=str).encode()).hexdigest()
        metadata={"research_config":self.runtime.world_bundle.research_config.as_dict(),
            "policy_fingerprint":self.runtime.world_bundle.research_config.fingerprint,"loss_mode":default_loss_mode(),
            "loss_options":{k:v for k,v in options.items() if k!="output"},"dataset_fingerprint":dataset_id,
            "domain_checksum":self.runtime.world_bundle.domain_checksum,"vocabulary_checksum":self.runtime.vocabulary.checksum}
        from .checkpoint import checkpoint_metadata
        metadata["research_configuration"]=checkpoint_metadata(self.runtime.world_bundle,self.planner_head).get("research_configuration")
        metadata["world_formulas"]={name:lc.active for name,lc in self.runtime.world_bundle.graph.logicalConstrains.items()}
        folder=Path(options["output"])/"research_stage1" if options.get("output") else None
        start=0
        if folder and (folder/"latest.pt").exists():
            saved=torch.load(folder/"latest.pt",map_location=self.planner_head.device,weights_only=False)
            if saved["metadata"]!=metadata:raise ValueError("research Stage 1 resume changes data, loss or policy configuration")
            _load_planner_trainable_state(self.planner_head,saved["planner"])
            optimizer.load_state_dict(saved["optimizer"]);self.projector.load_state_dict(saved["projector"])
            self.projector.optimizer.load_state_dict(saved["dual_optimizer"])
            torch.set_rng_state(saved["torch_rng"].cpu());random.setstate(saved["python_rng"])
            if torch.cuda.is_available() and saved.get("cuda_rng") is not None:torch.cuda.set_rng_state_all(saved["cuda_rng"])
            start=saved["epoch"]+1;self.history=saved["history"]
        for epoch in range(start,train_epoch_num):
            self.planner_head.train();self.projector.train()
            total=0.;used=0;blocked=0
            optimizer.zero_grad(set_to_none=True);accumulation=max(1,options.get("grad_accumulation",8));pending=0
            for item in items:
                loss,metrics=self.loss_for(item)
                if loss is None or metrics.get("blocked_samples"):
                    blocked+=1;continue
                (loss/accumulation).backward();total+=float(loss.detach());used+=1;pending+=1
                if pending==accumulation:
                    optimizer.step();optimizer.zero_grad(set_to_none=True);pending=0
                if default_loss_mode()=="full":self.projector.step_dual()
            if pending:
                for p in self.planner_head.parameters():
                    if p.grad is not None:p.grad.mul_(accumulation/pending)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            row={"epoch":epoch,"loss":total/max(1,used),"examples":used,"blocked_samples":blocked,"total_examples":len(items)}
            if not used:raise ValueError("Research training has no admitted supervision; refresh observations or fix the dataset")
            if valid_set:
                self.planner_head.eval();self.projector.eval();values=[];unavailable=0
                with torch.no_grad():
                    for item in valid_set:
                        loss,metrics=self.loss_for(item)
                        if loss is None or metrics.get("blocked_samples"):unavailable+=1
                        else:values.append(float(loss))
                row.update(validation_loss=sum(values)/max(1,len(values)),validation_blocked=unavailable)
            self.history.append(row);print("[research-stage1] "+json.dumps(row),flush=True)
            if folder:
                folder.mkdir(parents=True,exist_ok=True)
                payload={"epoch":epoch,"metadata":metadata,"planner":_planner_trainable_state(self.planner_head),
                    "optimizer":optimizer.state_dict(),"projector":self.projector.state_dict(),
                    "dual_optimizer":self.projector.optimizer.state_dict(),"history":self.history,
                    "torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),
                    "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
                torch.save(payload,folder/f"epoch_{epoch:03d}.pt")
                torch.save(payload,folder/"latest.tmp");(folder/"latest.tmp").replace(folder/"latest.pt")
                (folder/"configuration.json").write_text(json.dumps(metadata,indent=2)+"\n")
        return self.history
