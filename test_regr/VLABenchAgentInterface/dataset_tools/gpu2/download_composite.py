"""Pinned, resumable official composite acquisition; no training is launched."""
from pathlib import Path
import json, time, traceback, os
from huggingface_hub import HfApi, snapshot_download
ROOT = Path("/home/auszok/experiments/vlabench_dataset_fix_20261029")
DEST = Path("/home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data/control_composite")
RID = "VLABench/vlabench_composite_ft_lerobot_video"
REV = "8ceba7df14015ba00c4572f6df648750364caf18"
def status(stage, **kw):
    data = dict(stage=stage, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pid=os.getpid(), repo_id=RID, revision=REV, destination=str(DEST), **kw)
    temp=ROOT/"download_status.tmp"; temp.write_text(json.dumps(data, indent=2)+"\n"); temp.replace(ROOT/"download_status.json")
try:
    status("downloading_metadata_and_actions")
    snapshot_download(RID, repo_type="dataset", revision=REV, local_dir=DEST, allow_patterns=["meta/**", "data/**", "*.md", "*.yaml"], max_workers=4)
    import pyarrow.parquet as pq
    from collections import Counter
    info=json.loads((DEST/"meta/info.json").read_text())
    files=sorted((DEST/"data").rglob("*.parquet"))
    frames=0; episodes=set(); paths=Counter(); columns=set(); task_ids=set()
    for f in files:
        pf=pq.ParquetFile(f); cols=pf.schema_arrow.names; columns.update(cols)
        t=pf.read(columns=[c for c in ["episode_index","task_index","metadata.filepath"] if c in cols])
        frames+=t.num_rows
        episodes.update(t.column("episode_index").to_pylist())
        if "task_index" in cols:task_ids.update(t.column("task_index").to_pylist())
        if "metadata.filepath" in cols:
            paths.update(set(str(x) for x in t.column("metadata.filepath").to_pylist()))
    tasks=pq.read_table(DEST/"meta/tasks.parquet").to_pylist()
    audit=dict(repo_id=RID, revision=REV, declared_info=info, parquet_files=len(files), actual_frames=frames, actual_episodes=len(episodes), observed_task_indices=len(task_ids), columns=sorted(columns), episode_source_paths=dict(paths), instructions=tasks, skill_labels_available=False)
    (ROOT/"composite_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if len(episodes)!=info["total_episodes"]:raise ValueError("episode count mismatch")
    # Upstream metadata currently understates its own frame count; retain both values.
    status("downloading_videos", actual_frames=frames, actual_episodes=len(episodes))
    snapshot_download(RID, repo_type="dataset", revision=REV, local_dir=DEST, max_workers=4)
    remote=HfApi().dataset_info(RID, revision=REV, files_metadata=True)
    missing=[];wrong=[]
    for f in remote.siblings:
        p=DEST/f.rfilename
        if not p.is_file():missing.append(f.rfilename)
        elif f.size is not None and p.stat().st_size!=f.size:wrong.append(f.rfilename)
    audit["download_validation"]=dict(missing=missing,wrong_size=wrong,remote_files=len(remote.siblings),remote_bytes=sum(f.size or 0 for f in remote.siblings))
    (ROOT/"composite_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if missing or wrong:raise ValueError("incomplete download; rerun resumable script")
    status("complete", actual_frames=frames,actual_episodes=len(episodes),verified_files=len(remote.siblings))
except BaseException as e:
    status("failed", error=str(e)); traceback.print_exc(); raise
