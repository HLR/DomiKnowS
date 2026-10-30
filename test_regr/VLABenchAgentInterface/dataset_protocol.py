"""GPU2 dataset-role guard: official VLM evaluation episodes cannot train a model."""
from pathlib import Path
import hashlib
import json
import os

ROLE_FILE = "dataset_role.json"
SIMULATOR_TRACKS = Path("/opt/dominows/VLABench/VLABench/configs/evaluation/tracks")
OFFICIAL_EVALUATION = Path(os.environ.get("VLABENCH_OFFICIAL_EVALUATION_ROOT",Path(__file__).resolve().parent / "data" / "planning"))

def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def episode_fingerprint(episode):
    episode=Path(episode)
    configs=sorted((episode/"env_config").glob("*.json"))
    if not configs:
        raise ValueError(f"Missing scene configuration: {episode}")
    instruction=""
    for p in sorted((episode/"input").glob("*instruction*")):
        if p.suffix.lower()==".json":
            value=json.loads(p.read_text())
            instruction=str(value.get("instruction",value.get("text",""))) if isinstance(value,dict) else str(value)
        else: instruction=p.read_text()
        break
    value={"scene":json.loads(configs[0].read_text()),"instruction":instruction.strip()}
    research=episode/"input/research_observations.json"
    if research.is_file():value["research_observations"]=json.loads(research.read_text())
    return _json_digest(value)

def scene_fingerprint(episode):
    configs=sorted((Path(episode)/"env_config").glob("*.json"))
    if not configs:raise ValueError(f"Missing scene configuration: {episode}")
    return config_scene_fingerprint(json.loads(configs[0].read_text()))

def config_scene_fingerprint(config):
    task=config.get("task",config)
    # Initial scene identity must not change merely because a reference instruction or goal was edited.
    scene={k:task[k] for k in ("components","scene","entities") if k in task}
    if not scene:scene={k:v for k,v in task.items() if k not in {"instructions","instruction","conditions","target_entity","target_container"}}
    return _json_digest(scene)

def dataset_episodes(root):
    return sorted(p.parent for p in Path(root).rglob("output") if p.is_dir() and list(p.glob("*.json")))


def protected_scene_fingerprints():
    if os.environ.get("VLABENCH_OFFICIAL_EVALUATION_ROOT") and not OFFICIAL_EVALUATION.is_dir():
        raise ValueError("Configured official evaluation reservation is unavailable")
    scenes={scene_fingerprint(p) for p in dataset_episodes(OFFICIAL_EVALUATION)} if OFFICIAL_EVALUATION.exists() else set()
    if SIMULATOR_TRACKS.exists():
        for path in sorted(SIMULATOR_TRACKS.glob("*.json")):
            for configs in json.loads(path.read_text()).values():
                scenes.update(config_scene_fingerprint(config) for config in configs)
    return scenes

def assert_training_dataset(root, *, protected_root=None, simulator_root=None):
    root=Path(root).resolve()
    protected=Path(protected_root) if protected_root is not None else OFFICIAL_EVALUATION
    if not root.is_dir():
        raise ValueError(f"Training dataset missing: {root}. Generate independent training scenes; do not use data/planning (official evaluation).")
    if protected.exists() and os.path.samefile(root, protected):
        raise ValueError("Official VLABench VLM evaluation data cannot be used for training. Use independently generated planning_train data and fresh weights.")
    marker=root/ROLE_FILE
    if not marker.is_file():
        raise ValueError(f"Unclassified planning dataset: {root}. A {ROLE_FILE} with role=training and provenance is required.")
    role=json.loads(marker.read_text())
    if role.get("role")!="training" or not role.get("provenance"):
        raise ValueError(f"Dataset role is not independently sourced training: {root}")
    if "vlm_evaluation" in json.dumps(role).lower():
        raise ValueError("Training provenance refers to the official VLM evaluation collection.")
    if role.get("ready") is not True:
        raise ValueError("Training scene generation/validation is incomplete; dataset_role.json must be marked ready after final audit")
    if role.get("manifest_sha256"):
        manifest_path=root/"dataset_manifest.json"
        if not manifest_path.is_file() or hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=role["manifest_sha256"]:
            raise ValueError("Training dataset manifest changed after validation")
        for relative,expected in json.loads(manifest_path.read_text()).get("files",{}).items():
            source=(root/relative).resolve()
            if not source.is_relative_to(root) or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError(f"Training file changed after validation: {relative}")
    episodes=dataset_episodes(root)
    if role.get("episode_count") is not None and role["episode_count"] != len(episodes):
        raise ValueError("Training corpus changed after validation; rebuild its manifest")
    if not episodes:raise ValueError(f"No complete training episodes: {root}")
    protected_episodes=dataset_episodes(protected) if protected.exists() else []
    protected_fingerprints={episode_fingerprint(p) for p in protected_episodes}
    protected_scenes={scene_fingerprint(p) for p in protected_episodes}
    tracks=Path(simulator_root) if simulator_root is not None else SIMULATOR_TRACKS if protected_root is None else None
    if tracks is not None and tracks.exists():
        for path in sorted(tracks.glob("*.json")):
            payload=json.loads(path.read_text())
            for track_configs in payload.values():
                for config in track_configs:
                    protected_scenes.add(config_scene_fingerprint(config))
    overlap=[];seen={};digests=[]
    for episode in episodes:
        fingerprint=episode_fingerprint(episode);digests.append(fingerprint)
        if fingerprint in protected_fingerprints or scene_fingerprint(episode) in protected_scenes:overlap.append(str(episode.relative_to(root)))
        if fingerprint in seen:raise ValueError(f"Duplicate training scenes: {seen[fingerprint]} and {episode}")
        seen[fingerprint]=str(episode)
        if not any(p.suffix.lower() in {".png",".jpg",".jpeg",".webp"} for p in (episode/"input").glob("*")):
            raise ValueError(f"No independent scene images: {episode}")
    if overlap:raise ValueError(f"Official evaluation scene overlap: {overlap[:5]}")
    return {"role":"training","root":str(root),"episodes":len(episodes),"scene_manifest_sha256":_json_digest(sorted(digests)),"official_evaluation_overlap":0,"provenance":role["provenance"]}

def guard_training_dataset(root, output=None, *, resume=None):
    report=assert_training_dataset(root)
    if resume:
        checkpoint=Path(resume).resolve()
        candidate=checkpoint.parent/"dataset_protocol.json"
        if not candidate.is_file():
            raise ValueError("Cannot resume under the corrected dataset protocol: checkpoint has no verified training-data lineage. Start from the original pretrained backbone.")
        prior=json.loads(candidate.read_text())
        if prior.get("official_evaluation_training_seen") is not False:
            raise ValueError("Cannot resume a checkpoint exposed to official evaluation training data.")
    report["official_evaluation_training_seen"]=False
    report["checkpoint_lineage_note"]="Fresh pretrained initialization required; untracked or evaluation-exposed resumes are rejected. This is dataset provenance, not a benchmark performance claim."
    if output:
        out=Path(output)
        if out.suffix.lower() in {".pt",".pth",".bin",".json",".pkl"}:out=out.parent/(out.name+".dataset_protocol.json")
        else:out=out/"dataset_protocol.json"
        out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists() and json.loads(out.read_text()).get("official_evaluation_training_seen") is True:
            raise ValueError("Choose a new output directory; this directory belongs to an evaluation-exposed historical run")
        out.write_text(json.dumps(report,indent=2)+"\n")
    return report
