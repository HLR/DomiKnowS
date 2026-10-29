"""Finalize independent planning data and pinned download, with an optional bounded-poll wait."""
from pathlib import Path
import sys,json,subprocess,time,hashlib,traceback,argparse
ROOT=Path("/home/auszok/experiments/vlabench_dataset_fix_20261029")
DATA=Path("/workspace/test_regr/VLABenchAgentInterface/data")
sys.path.insert(0,"/workspace")
from test_regr.VLABenchAgentInterface.dataset_protocol import dataset_episodes,episode_fingerprint,assert_training_dataset
from test_regr.VLABenchAgentInterface.world_domain import SKILL_ARGUMENTS
def dump(p,v):
    q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n");q.replace(p)
def workers():
    rows=subprocess.run(["ps","-eo","pid,args"],capture_output=True,text=True,check=True).stdout.splitlines()
    return [r.strip() for r in rows if str(ROOT)+"/generate_" in r and "/python" in r]
def freeze():
    protected=set(json.loads((ROOT/"official_evaluation_manifest.json").read_text())["scene_fingerprints"].values())
    results={}
    for folder in ["planning_train","planning_extended"]:
        root=DATA/folder;eps=dataset_episodes(root);seen=set();files={};skills=set()
        for ep in eps:
            fp=episode_fingerprint(ep)
            if fp in protected:raise ValueError(f"Official evaluation overlap: {ep}")
            if fp in seen:raise ValueError(f"Duplicate scene: {ep}")
            seen.add(fp)
            for p in ep.rglob("*"):
                if p.is_file():files[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
            seq=json.loads((ep/"output/operation_sequence.json").read_text())["skill_sequence"]
            skills.update(s["name"] for s in seq)
            if folder=="planning_train":
                if len(seq)>12:raise ValueError(f"Operation budget exceeded: {ep}")
                for step in seq:
                    if step["name"] not in SKILL_ARGUMENTS:raise ValueError(f"Unsupported skill: {ep}")
                    for key in SKILL_ARGUMENTS[step["name"]]:
                        if key not in step["params"]:raise ValueError(f"Missing role {key}: {ep}")
        if not eps:raise ValueError(f"Empty corpus: {root}")
        manifest={"episodes":len(eps),"files":files,"scene_fingerprints":sorted(seen),"official_evaluation_overlap":0,"skill_labels":sorted(skills),"expert_execution_validated":False}
        dump(root/"dataset_manifest.json",manifest)
        role=json.loads((root/"dataset_role.json").read_text())
        role.update(ready=False,episode_count=len(eps),manifest_sha256=hashlib.sha256((root/"dataset_manifest.json").read_bytes()).hexdigest(),required_skills=sorted(skills),provenance={"kind":"independent_simulator_resets_and_expert_plan_labels","upstream_revision":"cf588fe60c0c7282174fe979f5913170cfe69017","seed_bases":[214700000,314700000],"source_hashes":"per-collection generation_audit.json","expert_execution_validated":False})
        if folder=="planning_extended":role["reason_not_ready"]="Unsupported articulation, unresolved symbolic roles, or operation budgets; retained without truncation outside default training."
        dump(root/"dataset_role.json",role);results[folder]={"episodes":len(eps),"ready":role["ready"],"official_evaluation_overlap":0}
    # Exercise the actual training label encoder after all source workers finish.
    from test_regr.VLABenchAgentInterface.dataset import load_planning_examples,deterministic_split
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    from test_regr.VLABenchAgentInterface.training import build_constraint_runtime,prepare_planner_program_examples
    root=DATA/"planning_train"
    try:
        examples=load_planning_examples(root)
        runtime=build_constraint_runtime(build_vlabench_world_graph("gpu2_dataset_encoding_audit"),max_entities=64,max_operations=12,name_prefix="gpu2_dataset_encoding_audit")
        encoded=prepare_planner_program_examples(examples,runtime)
        if len(encoded)!=len(examples):raise ValueError("Training label encoder dropped examples")
        from test_regr.VLABenchAgentInterface.world_runtime import materialize_plan,verify_plan_constraints
        verifier=build_vlabench_world_graph("gpu2_dataset_task_lc_audit",knowledge_level="task")
        lc_rows=[]
        for example in examples:
            node=materialize_plan(example.operation_sequence,example.entities,verifier,task_name=example.task,task_context=example.as_reward_item())
            evaluation=verify_plan_constraints(node,verifier)
            lc_rows.append({"episode_id":example.episode_id,"score":evaluation.score})
            if evaluation.score<1.0:
                dump(ROOT/"training_lc_validation.json",lc_rows)
                raise ValueError(f"Generated training plan violates current LCs: {example.episode_id}")
        dump(ROOT/"training_lc_validation.json",lc_rows)
        role=json.loads((root/"dataset_role.json").read_text());role["ready"]=True;dump(root/"dataset_role.json",role)
        report=assert_training_dataset(root)
        results["planning_train"]["ready"]=True
        results["training_encoding"]={"passed":True,"examples":len(encoded),"lc_verified_examples":len(lc_rows),"splits":{k:len(v) for k,v in deterministic_split(examples,seed=42).items()},"scene_manifest_sha256":report["scene_manifest_sha256"]}
    except BaseException:
        role=json.loads((root/"dataset_role.json").read_text());role["ready"]=False;role["reason_not_ready"]="Actual training encoder failed; see finalization log";dump(root/"dataset_role.json",role);raise
    dump(ROOT/"planning_finalization.json",results)
    return results
args=argparse.ArgumentParser();args.add_argument("--wait",action="store_true");opts=args.parse_args()
try:
    while workers():
        if not opts.wait:raise RuntimeError("Scene workers are still running. Re-run with --wait or after they finish.")
        dump(ROOT/"finalization_status.json",{"stage":"waiting_for_scene_generation","workers":workers()})
        time.sleep(10)
    dump(ROOT/"finalization_status.json",{"stage":"validating_compact_planning_corpus"})
    result=freeze()
    subprocess.run([sys.executable,str(ROOT/"refresh_dataset_report.py")],check=True)
    while True:
        state=json.loads((ROOT/"download_status.json").read_text())
        if state["stage"]=="failed":raise RuntimeError("Composite acquisition failed; inspect download_composite.log and resume the pinned downloader")
        if state["stage"]=="complete":
            subprocess.run([sys.executable,str(ROOT/"refresh_dataset_report.py")],check=True)
            dump(ROOT/"finalization_status.json",{"stage":"complete","planning":result,"composite_revision":state["revision"],"verified_files":state["verified_files"]})
            print("DATASET_FINALIZATION_COMPLETE",flush=True);break
        dump(ROOT/"finalization_status.json",{"stage":"planning_ready_composite_videos_pending","planning":result})
        if not opts.wait:print("PLANNING_READY_COMPOSITE_DOWNLOAD_PENDING",flush=True);break
        time.sleep(10)
except BaseException as e:
    dump(ROOT/"finalization_status.json",{"stage":"failed","error":str(e)});traceback.print_exc();raise
