"""GPU2-only measured expert collection, with boundaries at actual macro calls.

Expert target attributes are labels/diagnostics here. They are never copied into
research observations. Unavailable postconditions cannot count toward the gate.
"""
import argparse,hashlib,json,os,random,socket,subprocess,time,signal
from pathlib import Path
from functools import partial
import numpy as np
from .data import collection_gate,group_assignments

TASKS=("select_fruit","select_toy","select_book","select_drink","select_painting",
       "insert_flower","add_condiment","set_study_table","take_out_cool_drink","cool_drink","book_rearrange")


class ExecutedStepRecorder:
    """Capture pre-control observations and the commanded EE target via isolated FK.

    Upstream waypoint returns omit terminal controls and collapse substeps. This
    recorder counts actual env.step calls, including the last successful control.
    The physics clone supplies only a joint-command -> EE-label conversion.
    """
    def __init__(self,env,*,camera_slots=None):
        self.env=env;self.original_step=env.step;self.original_observe=env.get_observation
        self.clone=env.physics.copy(share_model=True)
        self.frames=[];self.states=[];self.actions=[];self.times=[];self.steps=0;self.observed_step=-1
        self.camera_slots=camera_slots
        env.get_observation=self.observe;env.step=self.step

    def observe(self,require_pcd=True):
        self.observation=self.original_observe(require_pcd=require_pcd)
        self.observed_step=self.steps
        return self.observation

    def step(self,command):
        from ..simulation.environment import world_to_robot_ee_state,quaternion_to_euler,robot_frame_position
        from ..observations import camera_indices,resolve_controller_camera_names,DEFAULT_CONTROLLER_CAMERA_KEYS
        from PIL import Image
        command=np.asarray(command,dtype=float).reshape(-1)
        count=self.env.robot.n_dof
        if len(command)!=count+2 or not np.isfinite(command).all():raise ValueError("Unrecognized expert joint/gripper action contract")
        if self.observed_step!=self.steps:self.observe(False)
        observation=self.observation
        measured=np.asarray(observation["ee_state"]).reshape(-1)
        if measured.size!=8:raise ValueError("Unknown measured EE-state contract")
        state=np.r_[measured[:3],quaternion_to_euler(measured[3:7]),measured[7]]
        for joint,value in zip(self.env.robot.joints[:count],command[:count]):self.clone.bind(joint).qpos=[value]
        self.clone.forward()
        target=np.r_[self.env.robot.get_end_effector_pos(self.clone),
                     quaternion_to_euler(self.env.robot.get_end_effector_quat(self.clone)),float(command[-2]>.03)]
        frame=robot_frame_position(self.env)
        state=world_to_robot_ee_state(state,frame);target=world_to_robot_ee_state(target,frame)
        if self.camera_slots is None:
            names,_=resolve_controller_camera_names(self.env,observation,dataset_keys=DEFAULT_CONTROLLER_CAMERA_KEYS)
            self.camera_slots=camera_indices(self.env,observation,names)
        image=np.stack([np.asarray(Image.fromarray(np.asarray(observation["rgb"])[i]).resize((224,224))) for i in self.camera_slots])
        captured=float(self.env.physics.data.time)
        result=self.original_step(command)
        self.states.append(state);self.actions.append(target);self.frames.append(image);self.times.append(captured)
        self.steps+=1
        return result

    def close(self):
        self.env.step=self.original_step;self.env.get_observation=self.original_observe
        self.clone=None


def serial(value):
    if hasattr(value,"tolist"):return value.tolist()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def gpu2_only():
    host=socket.gethostname()
    # Container hostnames differ; the launcher must explicitly attest its SSH destination.
    if "gpu2" not in host.lower() and os.environ.get("DOMINOWS_COLLECTION_HOST")!="gpu2":
        raise RuntimeError("Research datasets/artifacts must be created on GPU2")


def run_record(stage,root):
    command=["nvidia-smi","--query-gpu=index,uuid,name","--format=csv,noheader"]
    result=subprocess.run(command,capture_output=True,text=True,check=False)
    revision=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip() or os.environ.get("DOMINOWS_SOURCE_REVISION")
    source=Path(__file__).resolve().parents[3]/"research_source_manifest.json"
    return {"stage":stage,"pid":os.getpid(),"host":socket.gethostname(),"source_revision":revision,
        "physical_gpus":result.stdout.strip().splitlines(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_manifest_sha256":hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
        "started_at":time.time(),"artifact_root":str(root),"log":str(root/"collection.log")}


def cool_drink_expert(env):
    """Missing upstream put-in-fridge expert; execution still must pass measured gates."""
    from VLABench.utils.skill_lib import SkillLib
    task=env.task;fridge=task.init_container
    container=task.entities[fridge]
    corners=np.asarray([env.physics.bind(site).xpos for site in container.key_sites(env.physics)],dtype=float)
    if corners.ndim!=2 or corners.shape[1]!=3 or len(corners)<2:raise ValueError("Cool-drink expert lacks observed fridge interior bounds")
    placement=(corners.min(axis=0)+corners.max(axis=0))*.5
    return [partial(SkillLib.pick,target_entity_name=fridge,prior_eulers=[[-np.pi,0,-np.pi/2]]),
        partial(SkillLib.open_door,target_container_name=fridge),
        partial(SkillLib.lift,lift_height=.1,gripper_state=np.ones(2)*.04),
        partial(SkillLib.move_offset,offset=[.35,0,0],gripper_state=np.ones(2)*.04),
        partial(SkillLib.pick,target_entity_name=task.target_entity),
        partial(SkillLib.lift,lift_height=.1),
        partial(SkillLib.place,target_container_name=fridge,target_pos=placement),
        partial(SkillLib.lift,lift_height=.2,gripper_state=np.ones(2)*.04),
        partial(SkillLib.pick,target_entity_name=fridge,prior_eulers=[[-np.pi,0,np.pi]]),
        partial(SkillLib.close_door,target_container_name=fridge)]


def measured_support(obj,container,physics):
    """Upward physical contact for flat surfaces without usable containment sites."""
    try:
        subject={int(physics.bind(g).element_id) for g in obj.geoms}
        support={int(physics.bind(g).element_id) for g in container.geoms}
        if not subject or not support:return None
        height=float(np.asarray(obj.get_xpos(physics)).reshape(3)[2])
        for contact in physics.data.contact:
            first,second=int(contact.geom1),int(contact.geom2)
            sign=1 if first in support and second in subject else -1 if second in support and first in subject else 0
            normal=np.asarray(contact.frame).reshape(3,3)[0]
            if sign and float(contact.dist)<=.003 and sign*normal[2]>.5 and height>contact.pos[2]:return True
        return False
    except (AttributeError,TypeError,ValueError,IndexError):return None


def postcondition(env,skill,params,before):
    """Independent physical tests. Unsupported readers return None, never True."""
    entities=env.task.entities;physics=env.physics
    subject=params.get("target_entity_name") or getattr(env.task,"target_entity",None)
    destination=params.get("target_container_name") or getattr(env.task,"target_container",None)
    obj=entities.get(subject);container=entities.get(destination)
    try:
        if skill=="pick":return bool(obj.is_grasped(physics,env.robot))
        if skill in ("open_door","close_door"):
            obj=entities.get(params.get("target_container_name",subject))
            return bool(getattr(obj,"is_open" if skill=="open_door" else "is_closed")(physics))
        if skill=="open_laptop":
            joints=list(obj.joints)
            if len(joints)!=1:return None
            value=float(np.asarray(physics.bind(joints[0]).qpos).item())
            return 1.2<=value<=1.5
        if skill in ("place","insert"):
            if callable(getattr(obj,"is_grasped",None)) and obj.is_grasped(physics,env.robot):return False
            try:return bool(container.contain(np.asarray(obj.get_xpos(physics)),physics))
            except IndexError:
                return measured_support(obj,container,physics) if skill=="place" and hasattr(container,"z_threshold") else None
        if skill=="press":return bool(entities[getattr(env.task,"target_button",subject)].is_pressed())
        if skill=="pour":
            from VLABench.tasks.condition import PourCondition
            return bool(PourCondition(obj).is_met(physics))
        # Require measured object motion while held for carry skills.
        if skill=="lift":
            old=before.get(subject)
            return None if old is None else bool(obj.is_grasped(physics,env.robot) and obj.get_xpos(physics)[2]-old[2]>.01)
        if skill=="pull":
            old=before.get(subject)
            return None if old is None else bool(obj.is_grasped(physics,env.robot) and np.linalg.norm(np.asarray(obj.get_xpos(physics))[:2]-old[:2])>.01)
        if skill=="push":
            old=before.get(subject)
            return None if old is None else bool(np.linalg.norm(np.asarray(obj.get_xpos(physics))[:2]-old[:2])>.01)
    except (AttributeError,TypeError,KeyError,ValueError,IndexError):return None
    return None


def collect(args):
    gpu2_only()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    run=run_record("measured_expert_collection",root)
    (root/"run.json").write_text(json.dumps(run,indent=2)+"\n")
    import VLABench.robots,VLABench.tasks
    from VLABench.envs import load_env
    from ..simulation.agent import _task_success
    from ..world_domain import SKILL_ARGUMENTS
    from ..dataset_protocol import protected_scene_fingerprints
    reserved=protected_scene_fingerprints()
    rows=[];errors=[];instructions={};groups=[]
    if getattr(args,"official_source",None):
        from ..dataset import load_control_task_instructions
        instructions={text:i for i,text in load_control_task_instructions(args.official_source).items()}
    manifest=root/"segments.jsonl"
    if manifest.exists():
        rows=[json.loads(line) for line in manifest.read_text().splitlines() if line]
        from ..dataset import load_control_task_instructions
        instructions={text:i for i,text in load_control_task_instructions(root).items()}
        groups=list({r["scene_group"] for r in rows})
    for task_index,task in enumerate(args.tasks):
        for attempt in range(args.scenes_per_task):
            seed=args.seed_base+100000*task_index+attempt
            episode=f"{task}:{seed}"
            if any(r.get("episode_id")==episode for r in rows):continue
            random.seed(seed);np.random.seed(seed);env=None;recorder=None;native=None;operation_index=-1;params={}
            try:
                def timeout(*_):raise TimeoutError("Measured scene collection exceeded its bounded time budget")
                signal.signal(signal.SIGALRM,timeout);signal.alarm(args.scene_timeout)
                env=load_env(task,random_init=True)
                instruction=str(env.task.get_instruction());instructions.setdefault(instruction,max(instructions.values(),default=-1)+1)
                scene=serial(env.save())
                from ..dataset_protocol import config_scene_fingerprint
                group=config_scene_fingerprint(scene);groups.append(group)
                if group in reserved:raise ValueError("Scene is reserved for official evaluation")
                experts=cool_drink_expert(env) if task=="cool_drink" else env.get_expert_skill_sequence()
                if not experts:raise ValueError("No collection expert exists")
                recorder=ExecutedStepRecorder(env)
                frames=recorder.frames;states=recorder.states;actions=recorder.actions;segments=[];last_subject=None
                for operation_index,expert in enumerate(experts):
                    native=expert.func.__name__;params=dict(expert.keywords or {})
                    before={name:np.asarray(entity.get_xpos(env.physics)).copy() for name,entity in env.task.entities.items()}
                    start=len(states);result=expert(env)
                    if result is None or len(result)!=4:raise ValueError("Expert returned no measured trace")
                    observations,waypoints,stage_success,_=result
                    skill="insert" if task=="insert_flower" and (native=="place" or native=="lift" and params.get("lift_height",0)<0) else native
                    if params.get("target_entity_name"):last_subject=params["target_entity_name"]
                    if skill not in SKILL_ARGUMENTS:continue
                    if last_subject and "target_entity_name" not in params:params["target_entity_name"]=last_subject
                    entities=sorted(env.task.entities)
                    subject=params.get("target_entity_name") or last_subject
                    pointer=None
                    if skill in ("place","insert","pour"):pointer=params.get("target_container_name") or getattr(env.task,"target_container",None)
                    elif skill in ("open_door","close_door"):pointer=params.get("target_container_name",subject)
                    elif skill in ("pick","open_laptop"):pointer=subject
                    elif skill=="press":pointer=getattr(env.task,"target_button",subject)
                    verified=postcondition(env,skill,params,before) is True
                    segments.append({"segment_id":f"{episode}:{operation_index}","episode_id":episode,"scene_group":group,
                        "episode_index":seed,"skill":skill,"native_skill":native,"operation_index":len(segments),
                        "entity_index":entities.index(pointer)+1 if pointer in entities else 0,"instruction":instruction,
                        "entity_vocabulary":entities,"pointer_entity":pointer,"controller_conditioning_version":2,
                        "trace_capture_version":2,"native_stage_success":bool(stage_success),
                        "start_frame":start,"end_frame":len(states),"boundary_source":"expert_call",
                        "stage_success":verified,"postcondition_verified":verified})
                success=_task_success(env)
                folder=root/"episodes"/task/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
                (folder/"scene.json").write_text(json.dumps(scene,indent=2)+"\n")
                archive=folder/"control.npz"
                if states:np.savez_compressed(archive,state=np.asarray(states),actions=np.asarray(actions),images=np.asarray(frames),
                    simulator_time=np.asarray(recorder.times),simulator_step=np.arange(len(states)))
                for row in segments:
                    row.update(measured_task_success=bool(success),file=str(archive.relative_to(root)))
                    if success and row["stage_success"] and row["postcondition_verified"]:rows.append(row)
                (folder/"episode.json").write_text(json.dumps({"task":task,"seed":seed,"instruction":instruction,
                    "scene_group":group,"measured_task_success":bool(success),"segments":segments},indent=2)+"\n")
                print(json.dumps({"task":task,"seed":seed,"measured_success":success,"segments":len(segments)}),flush=True)
            except Exception as error:
                import traceback
                errors.append({"task":task,"seed":seed,"macro":native,"operation_index":operation_index,
                    "parameters":serial(params),"error":repr(error),"traceback":traceback.format_exc()})
                print(json.dumps(errors[-1]),flush=True)
            finally:
                signal.alarm(0)
                if recorder is not None:recorder.close()
                if env is not None:env.close()
            split=group_assignments(groups,args.split_seed)
            for row in rows:row["split"]=split[row["scene_group"]]
            manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
            (root/"collection_gate.json").write_text(json.dumps(collection_gate(rows,args.minimum_per_skill),indent=2)+"\n")
            (root/"collection_errors.json").write_text(json.dumps(errors,indent=2)+"\n")
            meta=root/"meta";meta.mkdir(exist_ok=True)
            (meta/"tasks.jsonl").write_text("".join(json.dumps({"task_index":i,"task":text})+"\n" for text,i in instructions.items()))
    gate=collection_gate(rows,args.minimum_per_skill)
    (root/"collection_gate.json").write_text(json.dumps(gate,indent=2)+"\n")
    config={"version":2,"trace_capture_version":2,"measured":True,"split_seed":args.split_seed}
    if getattr(args,"official_source",None):config["official_source"]=args.official_source
    if getattr(args,"official_splits",None):config["official_episode_splits"]=json.loads(Path(args.official_splits).read_text())
    (root/"research_control.json").write_text(json.dumps(config,indent=2)+"\n")
    meta=root/"meta";meta.mkdir(exist_ok=True)
    (meta/"tasks.jsonl").write_text("".join(json.dumps({"task_index":i,"task":text})+"\n" for text,i in instructions.items()))
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.npz")}
    files.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (root/"episodes").rglob("*.json")})
    for name in ("segments.jsonl","meta/tasks.jsonl","research_control.json","collection_gate.json"):
        if (root/name).is_file():files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    (root/"data_hashes.json").write_text(json.dumps(files,indent=2)+"\n")
    run.update(status="coverage_ready" if gate["passed"] else "coverage_insufficient",finished_at=time.time(),gate=gate)
    (root/"run.json").write_text(json.dumps(run,indent=2)+"\n")
    print(json.dumps(gate),flush=True)
    return gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True);parser.add_argument("--tasks",nargs="+",default=TASKS)
    parser.add_argument("--scenes-per-task",type=int,default=30)
    parser.add_argument("--minimum-per-skill",type=int,choices=(20,500),default=20)
    parser.add_argument("--seed-base",type=int,default=319700000);parser.add_argument("--split-seed",type=int,default=42)
    parser.add_argument("--scene-timeout",type=int,default=180)
    parser.add_argument("--official-source");parser.add_argument("--official-splits")
    args=parser.parse_args();collect(args)


if __name__=="__main__":main()
