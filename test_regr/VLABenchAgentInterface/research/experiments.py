"""Validated GPU2 experiment manifests and per-process, resumable run records."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from .collection import gpu2_only,run_record
from .data import collection_gate,validate_control_hashes


def family_episodes(rows,count,seed=42):
    import random
    if count<1:raise ValueError("research family episode budget must be positive")
    families={}
    for row in rows:
        family=row.get("research_family",row["task"])
        families.setdefault(family,{}).setdefault(row["scene_group"],[]).append(row)
    selected=[];rng=random.Random(seed)
    for family,groups in sorted(families.items()):
        if len(groups)<count:raise ValueError(f"{family}: need {count} held-out scene groups, have {len(groups)}")
        for group in rng.sample(sorted(groups),count):selected.append(rng.choice(groups[group]))
    return selected


def resolve_backbone(manifest,root):
    from ..models import PLANNER_MODEL_ID
    from huggingface_hub import snapshot_download
    identifier=manifest.get("planner_model",PLANNER_MODEL_ID)
    if Path(identifier).exists():raise ValueError("Matched experiments require an original Hub backbone ID; the runner resolves its immutable snapshot")
    record=Path(root)/"original_backbone.json"
    if record.is_file():
        saved=json.loads(record.read_text())
        if saved["repo_id"]!=identifier or saved.get("requested_revision")!=manifest.get("backbone_revision"):
            raise ValueError("Experiment output belongs to a different original backbone")
        snapshot=Path(saved["snapshot"])
        for name,digest in saved["files"].items():
            if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:raise ValueError("Original backbone snapshot changed")
        return str(snapshot)
    snapshot=Path(snapshot_download(identifier,revision=manifest.get("backbone_revision")))
    files={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob("*")
           if p.is_file() and (p.suffix in (".safetensors",".bin") or p.name=="config.json")}
    record.write_text(json.dumps({"repo_id":identifier,"requested_revision":manifest.get("backbone_revision"),
        "resolved_revision":snapshot.name,"snapshot":str(snapshot),"files":files},indent=2)+"\n")
    return str(snapshot)


def rollout_descriptors(path,split):
    payload=json.loads(Path(path).read_text())
    groups={};result=[]
    for name,rows in payload.items():
        if name not in ("train","validation","calibration","test"):raise ValueError("unknown rollout split")
        for row in rows:
            group=row["scene_group"]
            if group in groups and groups[group]!=name:raise ValueError("rollout scene group crosses splits")
            groups[group]=name
            if name==split:
                if not row.get("planning_facts",{}).get("research"):raise ValueError("research rollout lacks observation/policy evidence")
                if set(row)&{"operation_sequence","target_plan_labels","reference_plan","winning_cards"}:
                    raise ValueError("rollout descriptor contains supervision")
                result.append({**row,"research_matched":True})
    if not result:raise ValueError(f"No research rollout scenes for {split}")
    return result


def validate_manifest(manifest,*,pilot=False):
    root=Path(manifest["control_source"])
    rows=[json.loads(line) for line in (root/"segments.jsonl").read_text().splitlines() if line]
    gate=collection_gate(rows,20 if pilot else 500)
    if not gate["passed"]:return {"status":"collection_gate_failed",**gate}
    validate_control_hashes(root)
    if not pilot:
        pilot_root=Path(manifest.get("pilot_output","__missing_pilot__"))
        for profile in ("baseline","research_v1"):
            record=pilot_root/"seed42"/profile/"run.json"
            if not record.is_file() or json.loads(record.read_text()).get("status")!="completed":
                raise ValueError("Full experiments require an accepted completed pilot in pilot_output")
    config=json.loads((root/"research_control.json").read_text())
    if not config.get("official_source") or not config.get("official_episode_splits"):
        raise ValueError("Matched warm-up requires official unsegmented data with reserved episode splits")
    from ..dataset_protocol import assert_training_dataset
    audit=assert_training_dataset(manifest["planning_dir"])
    from ..dataset import load_planning_examples
    examples=load_planning_examples(manifest["planning_dir"])
    supplied=[e.planning_facts.get("research",{}) for e in examples if isinstance(e.planning_facts,dict)]
    if not supplied or any(not any(block.get(k) for block in supplied) for k in ("grounding","freshness","spatial","temporal")):
        raise ValueError("Research planner corpus lacks grounding/freshness/spatial/temporal evidence coverage")
    for split in ("train","validation","calibration","test"):
        rows=rollout_descriptors(manifest["rollout_manifest"],split)
        if split in ("validation","test"):family_episodes(rows,2 if pilot else 10)
    if manifest.get("seeds",[42,43,44])!=[42,43,44]:raise ValueError("Matched full experiment requires seeds 42,43,44")
    if manifest.get("resume_adapter"):raise ValueError("Matched experiments start at the original pretrained backbone")
    return {"status":"ready","collection":gate,"planning":audit}


def command_matrix(manifest,*,pilot=False):
    """Each full condition gets its own original planner and the identical warmed controller."""
    root=Path(manifest["output"]);result=[]
    tasks={row["task"] for row in rollout_descriptors(manifest["rollout_manifest"],"train")}
    for seed in ([42] if pilot else [42,43,44]):
        for profile in ("baseline","research_v1"):
            out=root/f"seed{seed}"/profile
            args=[sys.executable,"-m","test_regr.VLABenchAgentInterface.main","train-agent","--two-stage",
                "--planning-dir",manifest["planning_dir"],"--control-source",manifest["control_source"],"--output",str(out),
                "--seed",str(seed),"--sft-epochs","1" if pilot else "3","--controller-warmup-steps","200" if pilot else "20000",
                "--rl-epochs","1" if pilot else "3","--constraint-profile",profile,"--grounding-policy","robust",
                "--use-compiled-planner-loss",
                "--controller-warmup-units","updates",
                "--execution-assistance","off","--research-rollout-manifest",manifest["rollout_manifest"],
                "--eval-rollouts-per-task","1","--research-eval-episodes-per-family","2" if pilot else "10",
                "--rl-preflight-min-successful-tasks",str(len(tasks)),"--rl-min-successful-tasks",str(min(3,len(tasks)))]
            if profile=="research_v1":args.extend(["--warm-controller-from",str(root/f"seed{seed}"/"baseline"/"agent_stage1.pt")])
            if manifest.get("resolved_backbone") or manifest.get("planner_model"):
                args.extend(["--planner-model",manifest.get("resolved_backbone",manifest.get("planner_model"))])
            args.extend(manifest.get("common_arguments",[]))
            forbidden=("--resume","--resume-adapter","--constraint-profile","--controller-warmup-steps","--controller-warmup-units","--sft-epochs","--rl-epochs","--seed","--planner-model")
            if any(arg.split("=",1)[0] in forbidden for arg in manifest.get("common_arguments",[])):
                raise ValueError("Common arguments cannot override matched initialization, profile or budgets")
            result.append({"seed":seed,"profile":profile,"output":str(out),"arguments":args})
    return result


def evaluation_matrix(row,manifest,*,pilot=False):
    """Frozen held-out evaluation and formula/policy ablations, with identical scene budgets."""
    from ..main import build_parser
    trained=build_parser().parse_args(row["arguments"][3:])
    checkpoint=Path(row["output"])/("agent_rl_epoch_000.pt" if pilot else "agent_rl_epoch_002.pt")
    base=[sys.executable,"-m","test_regr.VLABenchAgentInterface.main","rollout","--env-factory",trained.env_factory,
        "--instruction","","--agent-checkpoint",str(checkpoint),"--vocab",str(Path(row["output"])/"vocab.json"),
        "--control-source",manifest["control_source"],"--research-rollout-manifest",manifest["rollout_manifest"],
        "--research-rollout-split","test","--research-eval-episodes-per-family","2" if pilot else "10",
        "--constraint-profile",row["profile"],"--use-compiled-planner-loss","--seed",str(row["seed"])]
    for key in ("planner_model","planner_decoder_hidden_dim","vision_model","vision_dim","image_size","hidden_dim",
                "max_views","max_steps","max_operations","action_horizon","execute_horizon","device"):
        value=getattr(trained,key,None)
        if value is not None:base.extend(["--"+key.replace("_","-"),str(value)])
    base.append("--load-in-4bit" if trained.load_in_4bit else "--no-load-in-4bit")
    if trained.tiny_vision:base.append("--tiny-vision")
    result=[{"name":"heldout_robust","arguments":[*base,"--grounding-policy","robust"]}]
    if row["profile"]=="research_v1" and not pilot:
        result.extend({"name":"heldout_"+policy,"arguments":[*base,"--grounding-policy",policy]} for policy in ("joint","abstain"))
        from ..knowledge.research_lcs import build_research_knowledge
        result.extend({"name":"without_"+name,"arguments":[*base,"--grounding-policy","robust","--research-disable-formula",name]}
                      for name in build_research_knowledge().enforce_constraints)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True);parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--execute",action="store_true",help="execute only after measured collection and role gates pass")
    args=parser.parse_args();gpu2_only()
    source=Path(args.manifest);manifest=json.loads(source.read_text());root=Path(manifest["output"])
    root.mkdir(parents=True,exist_ok=True)
    status=validate_manifest(manifest,pilot=args.pilot)
    (root/"gate.json").write_text(json.dumps(status,indent=2)+"\n")
    if status["status"]!="ready":print(json.dumps(status),flush=True);return
    manifest["resolved_backbone"]=resolve_backbone(manifest,root)
    matrix=command_matrix(manifest,pilot=args.pilot)
    (root/"commands.json").write_text(json.dumps(matrix,indent=2)+"\n")
    if not args.execute:return
    for row in matrix:
        folder=Path(row["output"]);folder.mkdir(parents=True,exist_ok=True)
        record_path=folder/"run.json"
        prior=json.loads(record_path.read_text()) if record_path.exists() else {}
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if prior and (prior.get("manifest_sha256")!=digest or prior.get("arguments")!=row["arguments"]):
            raise ValueError("Experiment output belongs to different data/configuration; choose a new output directory")
        if prior.get("status")=="completed":continue
        record=prior or run_record("pilot" if args.pilot else "matched_full",folder)
        code=0
        if prior.get("status")!="trained":
            record.update(row,manifest_sha256=digest,status="running",log=str(folder/"run.log"))
            with (folder/"run.log").open("a") as log:
                process=subprocess.Popen(row["arguments"],stdout=log,stderr=subprocess.STDOUT)
                record["pid"]=process.pid;record_path.write_text(json.dumps(record,indent=2)+"\n")
                code=process.wait()
        expected=folder/("agent_rl_epoch_000.pt" if args.pilot else "agent_rl_epoch_002.pt")
        complete=code==0 and expected.is_file()
        if complete:
            import torch
            payload=torch.load(expected,map_location="cpu",weights_only=False)
            complete=payload.get("stage")=="reinforcement" and payload.get("metrics",{}).get("retention_eligible") is not False
        record.update(status="trained" if complete else "failed_or_gated",returncode=code,finished_at=time.time())
        record_path.write_text(json.dumps(record,indent=2)+"\n")
        if not complete:raise RuntimeError(f"Training did not complete its accepted RL budget; inspect {folder/'run.log'}")
        for evaluation in evaluation_matrix(row,manifest,pilot=args.pilot):
            log_path=folder/(evaluation["name"]+".log")
            report=folder/(evaluation["name"]+".run.json")
            if report.is_file() and json.loads(report.read_text()).get("status")=="completed":continue
            with log_path.open("w") as log:
                process=subprocess.Popen(evaluation["arguments"],stdout=log,stderr=subprocess.STDOUT)
                evidence={**record,**evaluation,"stage":"heldout_evaluation","pid":process.pid,"log":str(log_path),"status":"running"}
                report.write_text(json.dumps(evidence,indent=2))
                code=process.wait()
            evidence.update(status="completed" if code==0 else "failed",returncode=code,finished_at=time.time())
            report.write_text(json.dumps(evidence,indent=2))
            if code:raise RuntimeError(f"Held-out evaluation failed: {log_path}")
        record.update(status="completed",finished_at=time.time());record_path.write_text(json.dumps(record,indent=2))


if __name__=="__main__":main()
