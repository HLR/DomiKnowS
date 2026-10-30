"""Adapters for supplied observations and explicitly privileged simulator diagnostics."""
import numpy as np


def held_name(value):
    # Upstream returns (measured name list, live entity objects); never serialize the objects.
    if isinstance(value,tuple) and len(value)==2 and isinstance(value[0],list) and isinstance(value[1],list):value=value[0]
    if value is None:return ""
    if isinstance(value,(list,tuple)):
        return value[0] if len(value)==1 and isinstance(value[0],str) else "" if not value else None
    return value if isinstance(value,str) else None


def attach_sensor_adapter(env,context):
    if callable(getattr(env,"read_research_observations",None)):return
    from .runtime import research_block
    raw=research_block(context)
    specs=raw.get("sensors",())
    diagnostic=raw.get("sensor_profile")=="simulator_diagnostic"
    previous={}
    clock=[None,-1,None]
    def read(*,step,episode,revisions,observation=None):
        if clock[0]!=episode or step<clock[1] or clock[2]!=revisions.get("reset"):previous.clear()
        clock[:]=[episode,step,revisions.get("reset")]
        observation=observation or {}
        records=list(observation.get("research_records",()))
        measured={};positive=[];negative=[];held={};events=[];joints=[]
        captured_revisions=dict(revisions)
        for spec in specs:
            entity_name=spec.get("entity");value=None
            field=spec.get("field","observation")
            if field=="observation":
                key=spec.get("observation_key",spec["id"])
                if any(s in key.lower() for s in ("success","winning","target","progress","condition")):
                    raise ValueError("evaluator annotations cannot be research sensor observations")
                value=observation.get(key)
            elif field=="held" and "grasped_obj_name" in observation:
                value=held_name(observation["grasped_obj_name"])
            elif diagnostic:
                obj=getattr(getattr(env,"task",None),"entities",{}).get(entity_name)
                physics=getattr(env,"physics",None)
                if obj is not None and physics is not None:
                    if field=="position":value=np.asarray(obj.get_xpos(physics),dtype=float).reshape(3).tolist()
                    elif field in ("open","closed"):
                        reader=getattr(obj,"is_"+field,None)
                        if callable(reader):
                            value=bool(reader(physics));(positive if value else negative).append([field,entity_name])
                    elif field=="joint":
                        matches=[j for j in getattr(obj,"joints",()) if getattr(j,"name",None)==spec.get("joint")]
                        if len(matches)==1:
                            values=np.asarray(physics.bind(matches[0]).qpos).reshape(-1)
                            if values.size==1:value=float(values[0])
            deps={name:revisions.get(name,0) for name in spec.get("dependencies",(["object:"+entity_name] if entity_name else []))}
            captured_revisions.update(deps)
            if value is not None:
                if isinstance(value,np.generic):value=value.item()
                elif isinstance(value,np.ndarray):value=value.tolist()
                record={"id":spec["id"],"value":value,"step":step,"episode":episode,
                    "source":"simulator_diagnostic" if diagnostic and field!="observation" else "perception",
                    "kind":"observed","frame":spec.get("frame"),"units":spec.get("units"),
                    "max_age_steps":spec.get("max_age_steps",0),"dependencies":deps}
                records.append(record)
                if spec.get("retain_samples"):records.append({**record,"id":spec["id"]+":"+str(step)})
                measured[spec["id"]]=value
                if spec.get("joint") and entity_name and type(value) in (int,float):
                    joints.append({"entity":entity_name,"joint":spec["joint"],"value":float(value)})
                if spec["id"] in previous and value!=previous[spec["id"]]:
                    for binding in spec.get("events",()):
                        if "from" in binding and previous[spec["id"]]!=binding["from"]:continue
                        if "to" in binding and value!=binding["to"]:continue
                        if binding.get("to_nonempty") and not (isinstance(value,str) and value):continue
                        entities=[value if x=="$value" else x for x in binding.get("entities",())]
                        events.append({"id":f"{episode}:{step}:{spec['id']}:{binding['name']}","name":binding["name"],
                            "step":step,"episode":episode,"kind":"observed","entities":entities})
        if "grasped_obj_name" in observation:
            name=held_name(observation["grasped_obj_name"])
            if name is not None:held["right"]=name or None
        for key,value in measured.items():
            old=previous.get(key)
            if type(value) is bool and old is not None and value!=old:
                events.append({"id":f"{episode}:{step}:{key}","name":key+(":true" if value else ":false"),
                    "step":step,"episode":episode,"kind":"observed"})
            previous[key]=value
        sample={"records":records,"events":events,"revisions":captured_revisions,
                "joint_observations":joints,
                "initial_state":{"held":held,"positive":positive,"negative":negative}}
        if "research_grounding" in observation:sample["grounding"]=observation["research_grounding"]
        return sample
    env.read_research_observations=read
