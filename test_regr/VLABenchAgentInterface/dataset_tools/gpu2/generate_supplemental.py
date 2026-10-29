"""Fresh simulator reset scenes and expert plan labels; no physical execution claim."""
import os
os.environ.update(MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID="0",CUDA_VISIBLE_DEVICES="")
from pathlib import Path
import json, random, time, traceback, subprocess, hashlib, sys
import numpy as np
from PIL import Image
import VLABench.robots, VLABench.tasks
from VLABench.envs import load_env
ROOT=Path("/home/auszok/experiments/vlabench_dataset_fix_20261029/supplemental")
ROOT.mkdir(parents=True,exist_ok=True)
DATA=Path("/workspace/test_regr/VLABenchAgentInterface/data")
sys.path.insert(0,"/workspace")
from test_regr.VLABenchAgentInterface.dataset_protocol import episode_fingerprint
from test_regr.VLABenchAgentInterface.world_domain import SKILL_ARGUMENTS, PLANNING_TASK_PATTERNS, PRIMITIVE_TASK_PATTERNS
UPSTREAM=Path("/opt/dominows/VLABench")
TASKS=["book_rearrange","get_coffee","get_coffee_with_sugar","heat_food","take_chemistry_experiment"]
COUNT=10
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
rev=subprocess.run(["git","rev-parse","HEAD"],cwd=UPSTREAM,capture_output=True,text=True).stdout.strip()
source_hashes={str(p.relative_to(UPSTREAM)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [UPSTREAM/"VLABench/utils/skill_lib.py",*sorted((UPSTREAM/"VLABench/tasks/hierarchical_tasks").rglob("*.py"))]}
provenance={"kind":"independent_simulator_resets_and_expert_plan_labels","seed_base":314700000,"upstream_git_revision":rev,"upstream_source_sha256":source_hashes,"expert_execution_validated":False,"note":"Training labels, never planner input. Images and initial scenes are independently sampled. Helper calls retained in native expert record, omitted only from graph abstraction."}
for folder in ["planning_train","planning_extended"]:
    p=DATA/folder;p.mkdir(parents=True,exist_ok=True)
report={"provenance":provenance,"completed":[],"unavailable":[],"unsupported_schema":[],"errors":[]}
helpers={"moveto","move_offset","reset","wait","close_gripper","open_gripper","step_trajectory"}
for ti,task in enumerate(TASKS):
    for i in range(COUNT):
        seed=314700000+ti*10000+i
        if any((DATA/f/"Generated"/task/f"seed{seed}"/"output/operation_sequence.json").exists() for f in ["planning_train","planning_extended"]): continue
        random.seed(seed);np.random.seed(seed);env=None
        try:
            env=load_env(task,random_init=True)
            steps=env.get_expert_skill_sequence()
            if not steps:
                report["unavailable"].append({"task":task,"reason":"upstream has no expert sequence"})
                break
            native=[{"name":s.func.__name__,"params":serial(s.keywords),"bound_args":serial(s.args)} for s in steps]
            unsupported=sorted({s["name"] for s in native if s["name"] not in SKILL_ARGUMENTS and s["name"] not in helpers})
            graph=[]
            for s in native:
                name=s["name"];params=s["params"]
                if name in helpers:continue
                if name not in SKILL_ARGUMENTS:continue
                if task=="insert_flower" and (name=="place" or (name=="lift" and float(params.get("lift_height",0))<0)):name="insert"
                mapped={}
                for key in SKILL_ARGUMENTS[name]:
                    alias="target_container_name" if name=="open_door" and key=="target_entity_name" else key
                    if alias in params and params[alias] is not None:
                        mapped[key]=params[alias]
                    elif name=="press" and key=="target_entity_name" and getattr(env.task,"target_button",None):
                        mapped[key]=env.task.target_button
                    elif (name=="insert" or (name=="pour" and task=="add_condiment")) and key=="target_container_name" and getattr(env.task,"target_container",None):
                        mapped[key]=env.task.target_container
                    elif name=="press" and key=="target_entity_name" and "target_pos" in params:
                        matches=[]
                        for entity_name,entity in env.task.entities.items():
                            if hasattr(entity,"get_start_button_pos"):
                                try:
                                    if np.allclose(entity.get_start_button_pos(env.physics),params["target_pos"],atol=1e-6):matches.append(entity_name)
                                except (TypeError,ValueError,AttributeError):pass
                        if len(matches)==1:mapped[key]=matches[0]
                        else:unsupported.append(f"unresolved_symbolic_role:{name}.{key}")
                    else:unsupported.append(f"unresolved_symbolic_role:{name}.{key}")
                graph.append({"name":name,"params":mapped})
            if len(graph)>12: unsupported.append("action_budget>12")
            expected=PLANNING_TASK_PATTERNS.get(task,PRIMITIVE_TASK_PATTERNS.get(task))
            if expected and not unsupported:
                selected=[];position=0
                for expected_name in expected:
                    while position<len(graph) and graph[position]["name"]!=expected_name:position+=1
                    if position==len(graph):raise ValueError(f"Expert does not support compact contract {expected}")
                    selected.append(graph[position]);position+=1
                graph=selected
            if task in {"book_rearrange","get_coffee"}:unsupported.append("task_projection_requires_review")
            folder="planning_extended" if unsupported else "planning_train"
            episode=DATA/folder/"Generated"/task/f"seed{seed}"
            if (episode/"output/operation_sequence.json").exists():continue
            for sub in ["env_config","input","output","provenance"]:(episode/sub).mkdir(parents=True,exist_ok=True)
            config=serial(env.save());instruction=str(env.task.get_instruction())
            (episode/"env_config/env_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n")
            (episode/"input/instruction.txt").write_text(instruction+"\n")
            for cam in range(min(4,env.physics.model.ncam)):
                image=env.render(camera_id=cam,height=224,width=224)
                Image.fromarray(image).save(episode/f"input/view_{cam}.png")
            # For unsupported schemas, preserve the complete manipulation sequence, never silently truncate it.
            plan=([s for s in native if s["name"] not in helpers] if unsupported else graph)
            (episode/"output/operation_sequence.json").write_text(json.dumps({"skill_sequence":plan},indent=2,ensure_ascii=False)+"\n")
            (episode/"provenance/expert_sequence.json").write_text(json.dumps({"seed":seed,"task":task,"native_sequence":native,"unsupported_graph_skills":unsupported,"expert_execution_validated":False},indent=2,ensure_ascii=False)+"\n")
            row={"task":task,"seed":seed,"episode":str(episode),"graph_skills":[s["name"] for s in plan],"native_skills":[s["name"] for s in native],"fingerprint":episode_fingerprint(episode)}
            report["completed"].append(row)
            if unsupported:report["unsupported_schema"].append({"task":task,"skills":unsupported})
        except Exception as e:
            report["errors"].append({"task":task,"seed":seed,"error":str(e)})
            print("GENERATION_ERROR",task,seed,repr(e),flush=True)
            continue
        finally:
            if env is not None:env.close()
        tmp=ROOT/"generation_audit.tmp";tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n");tmp.replace(ROOT/"generation_audit.json")
        print("GENERATED",task,seed,folder,flush=True)
    (ROOT/"generation_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
(ROOT/"generation_complete.json").write_text(json.dumps({"complete":True,"episodes":len(report["completed"]),"unavailable":report["unavailable"],"errors":report["errors"]},indent=2)+"\n")
print("GENERATION_COMPLETE",len(report["completed"]),flush=True)
