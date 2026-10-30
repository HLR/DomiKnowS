"""Group-disjoint research splits and measured, explicitly segmented control data."""
from pathlib import Path
from collections import OrderedDict,Counter
import json,random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Subset,WeightedRandomSampler

SPLITS=("train","validation","calibration","test")


def validate_control_hashes(root):
    import hashlib
    root=Path(root).resolve()
    if json.loads((root/"run.json").read_text()).get("status")!="coverage_ready":raise ValueError("Measured collection has not completed its coverage gate")
    expected=json.loads((root/"data_hashes.json").read_text())
    for relative,digest in expected.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Measured control data changed: {relative}")
    return expected


def group_assignments(groups,seed=42):
    groups=sorted(set(groups));random.Random(seed).shuffle(groups)
    n=len(groups);ends=[int(.7*n),int(.8*n),int(.9*n),n];result={};start=0
    for name,end in zip(SPLITS,ends):
        result.update((g,name) for g in groups[start:end]);start=end
    return result


def grouped_split(items,seed=42):
    items=list(items);assignments={}
    for item in items:
        group=getattr(item,"scene_group",None);split=getattr(item,"research_split",None)
        if not group:raise ValueError("research examples need a stable scene group")
        if split is not None:
            if split not in SPLITS:raise ValueError("unknown research split")
            if group in assignments and assignments[group]!=split:raise ValueError("scene group crosses research splits")
            assignments[group]=split
    generated=group_assignments((x.scene_group for x in items),seed);generated.update(assignments)
    return {name:[x for x in items if generated[x.scene_group]==name] for name in SPLITS}


def count_successful_segments(rows):
    counts=Counter();seen=set()
    for row in rows:
        identifier=row.get("segment_id")
        if not identifier or identifier in seen:raise ValueError("measured segments need unique identifiers")
        seen.add(identifier)
        if (row.get("measured_task_success") is True and row.get("stage_success") is True
                and row.get("boundary_source")=="expert_call" and row.get("end_frame",0)>row.get("start_frame",0)
                and row.get("controller_conditioning_version")==2
                and row.get("trace_capture_version")==2
                and row.get("postcondition_verified") is True):counts[row["skill"]]+=1
    return counts


def collection_gate(rows,minimum):
    from ..world_domain import SKILL_ARGUMENTS
    counts=count_successful_segments(rows)
    return {"passed":all(counts[s]>=minimum for s in SKILL_ARGUMENTS),"minimum_per_skill":minimum,
            "counts":{s:counts[s] for s in SKILL_ARGUMENTS},"missing":{s:minimum-counts[s] for s in SKILL_ARGUMENTS if counts[s]<minimum}}


class SegmentedControlDataset(Dataset):
    """Windows never cross measured macro boundaries or scene splits."""
    def __init__(self,root,segments,instructions,*,action_horizon=16,observation_horizon=2):
        self.root=Path(root);self.rows=list(segments);self.instructions={v:k for k,v in instructions.items()}
        self.action_horizon=action_horizon;self.observation_horizon=observation_horizon;self.cache=OrderedDict()
        self.index=[(i,frame) for i,r in enumerate(self.rows) for frame in range(r["start_frame"],r["end_frame"])]
        from ..world_domain import SKILL_ARGUMENTS
        count_successful_segments(self.rows)
        if any(r["skill"] not in SKILL_ARGUMENTS or r.get("measured_task_success") is not True
               or r.get("postcondition_verified") is not True or r.get("stage_success") is not True
               or r.get("boundary_source")!="expert_call" or type(r.get("start_frame")) is not int
               or r.get("controller_conditioning_version")!=2
               or r.get("trace_capture_version")!=2
               or type(r.get("end_frame")) is not int or r["start_frame"]<0 or r["end_frame"]<=r["start_frame"] for r in self.rows):
            raise ValueError("unverified robot segments cannot train the controller")

    def __len__(self):return len(self.index)

    def __getitem__(self,index):
        from ..world_domain import controller_skill_index
        from ..observations import image_tensor
        number,frame=self.index[index];row=self.rows[number];path=(self.root/row["file"]).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError("segment file escapes its dataset")
        if str(path) not in self.cache:
            with np.load(path,allow_pickle=False) as archive:self.cache[str(path)]={k:archive[k] for k in ("state","actions","images")}
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(str(path));data=self.cache[str(path)]
        if any(len(v)<row["end_frame"] for v in data.values()):raise ValueError("segment bounds exceed the measured trace")
        before=[max(row["start_frame"],frame-self.observation_horizon+1+i) for i in range(self.observation_horizon)]
        after=[min(row["end_frame"]-1,frame+i) for i in range(self.action_horizon)]
        return {"state":torch.as_tensor(data["state"][before],dtype=torch.float32),
            "images":image_tensor(data["images"][before],channels_last=True),
            "actions":torch.as_tensor(data["actions"][after],dtype=torch.float32),
            "task_index":torch.tensor(self.instructions[row["instruction"]],dtype=torch.long),
            "plan_context":torch.tensor([controller_skill_index(row["skill"]),row.get("entity_index",0),0]),
            "episode_index":torch.tensor(row["episode_index"])}


def build_control_loaders(args):
    from ..dataset import load_control_task_instructions,LeRobotWindowDataset
    from datasets import load_dataset
    root=Path(args.control_source);config=json.loads((root/"research_control.json").read_text())
    hashes=validate_control_hashes(root)
    instructions=load_control_task_instructions(root)
    rows=[json.loads(line) for line in (root/"segments.jsonl").read_text().splitlines() if line.strip()]
    gate=json.loads((root/"collection_gate.json").read_text())
    actual=collection_gate(rows,max(20,int(gate.get("minimum_per_skill",20))))
    if not gate.get("passed") or not actual["passed"] or gate.get("counts")!=actual["counts"]:
        raise ValueError("segmented control corpus has not passed its measured collection gate")
    groups={}
    for row in rows:
        if row["file"] not in hashes:raise ValueError("Measured segment has no validated trace hash")
        group=row["scene_group"];split=row["split"]
        if split not in SPLITS or group in groups and groups[group]!=split:raise ValueError("control scene group crosses splits")
        groups[group]=split
    result={}
    for split in SPLITS:
        selected=[r for r in rows if r["split"]==split]
        ds=SegmentedControlDataset(root,selected,instructions,action_horizon=args.action_horizon)
        sampler=None
        if split=="train":
            from ..world_domain import SKILL_ARGUMENTS
            totals=Counter(row["skill"] for row in selected)
            if any(totals[s]==0 for s in SKILL_ARGUMENTS):raise ValueError("training split lacks measured examples for a controller skill")
            frames=Counter(ds.rows[i]["skill"] for i,_ in ds.index)
            sampler=WeightedRandomSampler([1./frames[ds.rows[i]["skill"]] for i,_ in ds.index],len(ds),replacement=True)
        result[split]=DataLoader(ds,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers)
    if config.get("official_source"):
        source=Path(config["official_source"]);original=load_control_task_instructions(source)
        mapping={i:next(j for j,text in instructions.items() if text==value) for i,value in original.items()}
        records=load_dataset("parquet",data_files={"train":[str(p) for p in sorted((source/"data").rglob("*.parquet"))]},split="train")
        class MappedRecords:
            column_names=records.column_names
            def __len__(self):return len(records)
            def __getitem__(self,i):
                if isinstance(i,str):return records[i]
                row=dict(records[i]);row["task_index"]=mapping[row["task_index"]];return row
        windows=LeRobotWindowDataset(MappedRecords(),video_root=source,action_horizon=args.action_horizon,plan_pattern=())
        assignment=config["official_episode_splits"]
        indices=[i for i,(ep,_) in enumerate(windows.index) if assignment[str(ep)]=="train"]
        result["warmup_official"]=DataLoader(Subset(windows,indices),batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    result["warmup_segmented"]=result["train"]
    return result
