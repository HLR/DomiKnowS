"""Additive policy lineage; does not change the eleven-skill weight schema."""
from .config import ResearchConfig,default_config,default_loss_mode,default_training_options,default_compiled_loss,default_perception_options


def data_fingerprints(args):
    import hashlib
    from pathlib import Path
    result={}
    if getattr(args,"planner_model",None):result["original_backbone"]=str(args.planner_model)
    for key,name in (("planning_dir","dataset_manifest.json"),("vlabench_planning_dir","dataset_manifest.json"),
                     ("control_source","data_hashes.json"),("control_source","research_control.json"),
                     ("research_rollout_manifest",None)):
        value=getattr(args,key,None)
        if value:
            path=Path(value)/name if name else Path(value)
            if path.is_file():result[key+":"+str(name)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def save_adapter_metadata(target,planner):
    import json
    from pathlib import Path
    world=getattr(planner,"_research_world",None)
    if world is None:return
    metadata=checkpoint_metadata(world,planner).get("research_configuration")
    if metadata is not None:(Path(target)/"research_configuration.json").write_text(json.dumps(metadata,indent=2)+"\n")


def load_adapter_metadata(path,planner,*,training=False):
    import json
    from pathlib import Path
    source=Path(path)/"research_configuration.json"
    if not source.is_file():return
    saved=json.loads(source.read_text());config=default_config()
    changed=(saved["fingerprint"]!=config.fingerprint or saved.get("observation_inputs")!=(config.enabled or default_compiled_loss())
        or saved.get("perception_configuration")!=default_perception_options())
    if training and changed:raise ValueError("adapter research profile differs from requested training configuration")
    planner._research_checkpoint_profile=saved
    planner.research_evaluation_override=changed


def clone_warmed_controller(path,controller,optimizer,runtime,control_source):
    import json,hashlib
    from pathlib import Path
    import torch
    from ..training import _controller_configuration
    path=Path(path)
    lineage=path.parent/"dataset_protocol.json"
    if not lineage.is_file() or json.loads(lineage.read_text()).get("official_evaluation_training_seen") is not False:
        raise ValueError("Warm controller lacks independent training-data lineage")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("stage")!="supervised":raise ValueError("Clone the BC controller before reinforcement/evaluation selection")
    if payload.get("domain_checksum")!=runtime.world_bundle.domain_checksum:raise ValueError("Warm controller skill schema differs")
    if payload.get("controller_configuration")!=_controller_configuration(controller):raise ValueError("Warm controller vocabulary/configuration differs")
    from ..dataset import load_control_task_instructions
    fingerprint=hashlib.sha256(json.dumps(load_control_task_instructions(control_source),sort_keys=True).encode()).hexdigest()
    if payload.get("controller_instruction_fingerprint")!=fingerprint:raise ValueError("Warm controller instruction vocabulary differs")
    controller.load_state_dict(payload["controller"])
    optimizer.load_state_dict(payload["controller_optimizer"])


def checkpoint_metadata(world,planner):
    config=ResearchConfig.parse(getattr(world,"research_config",None) or default_config())
    if not config.enabled and not default_compiled_loss():return {}
    knowledge=getattr(world,"research_knowledge",None)
    formulas={name:lc.active for name,lc in knowledge.get().enforce_constraints.items()} if knowledge and config.enabled else {}
    if set(config.disabled_formulas)-set(formulas):raise ValueError("Unknown disabled research formula")
    formulas.update({name:False for name in config.disabled_formulas})
    provider=getattr(planner,"_research_training_state_provider",None)
    options=default_training_options();full=default_loss_mode()=="full"
    return {"research_configuration":{"config":config.as_dict(),"fingerprint":config.fingerprint,
                "formulas":formulas,"supervised_mode":default_loss_mode(),
                "observation_inputs":config.enabled or default_compiled_loss(),
                "loss_options":{k:v for k,v in default_training_options().items() if k not in ("output",)},
                "effective_loss_weights":{"supervised":1.,"allowed_mass":options.get("stage1_dfa_mass_weight",0.) if full else 0.,
                    "primal_dual":options.get("stage1_pmd_weight",1.) if full else 0.},
                "evidence_schema_version":1,"controller_conditioning_version":2,
                "perception_configuration":default_perception_options(),
                "data_fingerprints":getattr(planner,"_research_data_fingerprints",{})},
            "research_training_state":provider() if callable(provider) else None}


def restore_metadata(payload,world,planner,*,training=False):
    saved=payload.get("research_configuration")
    config=ResearchConfig.parse(getattr(world,"research_config",None) or default_config())
    if saved is None and not config.enabled and not default_compiled_loss():return
    if training and (saved is None or saved["fingerprint"]!=config.fingerprint):
        raise ValueError("research resume changes policy/evidence profile; start a fresh matched run")
    if training and saved.get("supervised_mode")!=default_loss_mode():
        raise ValueError("research resume changes supervised-loss mode")
    if training:
        if saved.get("perception_configuration")!=default_perception_options():raise ValueError("research resume changes image-perception configuration")
        if saved.get("observation_inputs")!=(config.enabled or default_compiled_loss()):raise ValueError("research resume changes observation-input mode")
        options={k:v for k,v in default_training_options().items() if k!="output"}
        if saved.get("loss_options",{})!=options:raise ValueError("research resume changes loss weights or dual optimization settings")
        if saved.get("data_fingerprints",{})!=getattr(planner,"_research_data_fingerprints",{}):
            raise ValueError("research resume changes data/evidence fingerprints")
        knowledge=getattr(world,"research_knowledge",None)
        formulas={name:lc.active for name,lc in knowledge.get().enforce_constraints.items()} if knowledge and config.enabled else {}
        formulas.update({name:False for name in config.disabled_formulas})
        if saved.get("formulas")!=formulas:raise ValueError("research resume changes active logical constraints")
    # Evaluation may deliberately ablate a profile; expose the override, do not claim a matching run.
    payload["research_evaluation_override"]=(saved is None or saved["fingerprint"]!=config.fingerprint
        or saved.get("observation_inputs")!=(config.enabled or default_compiled_loss())
        or saved.get("perception_configuration")!=default_perception_options())
    planner.research_evaluation_override=payload["research_evaluation_override"]
    planner._research_checkpoint_profile=saved
    if payload.get("research_training_state") is not None:
        planner._research_restored_training_state=payload["research_training_state"]
