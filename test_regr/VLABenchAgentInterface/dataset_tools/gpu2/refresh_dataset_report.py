from pathlib import Path
from collections import Counter
import json,hashlib,time,sys,ast
ROOT=Path("/home/auszok/experiments/vlabench_dataset_fix_20261029")
REPO=Path("/workspace");DATA=REPO/"test_regr/VLABenchAgentInterface/data"
sys.path.insert(0,str(REPO))
from test_regr.VLABenchAgentInterface.dataset_protocol import episode_fingerprint,dataset_episodes
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
official=DATA/"planning";frozen=ROOT/"official_evaluation_manifest.json"
if not frozen.exists():
    files=[p for ep in dataset_episodes(official) for base in ["env_config","input","output"] for p in sorted((ep/base).rglob("*")) if p.is_file()]
    dump(frozen,{"role":"evaluation","repo_id":"VLABench/vlm_evaluation_v1.0","files":{str(p.relative_to(official)):digest(p) for p in files},"scene_fingerprints":{str(e.relative_to(official)):episode_fingerprint(e) for e in dataset_episodes(official)}})
old=Counter()
for e in dataset_episodes(official):old.update(s["name"] for s in json.loads((e/"output/operation_sequence.json").read_text())["skill_sequence"])
coverage={}
for folder in ["planning_train","planning_extended"]:
    root=DATA/folder;graph=Counter();native=Counter();tasks=Counter();fps=[]
    for ep in dataset_episodes(root):
        graph.update(s["name"] for s in json.loads((ep/"output/operation_sequence.json").read_text())["skill_sequence"])
        r=json.loads((ep/"provenance/expert_sequence.json").read_text())
        native.update(s["name"] for s in r["native_sequence"]);tasks[r["task"]]+=1;fps.append(episode_fingerprint(ep))
    coverage[folder]={"episodes":sum(tasks.values()),"tasks":dict(sorted(tasks.items())),"graph_skill_labels":dict(sorted(graph.items())),"native_expert_labels":dict(sorted(native.items())),"expert_execution_validated":False,"duplicate_scenes":len(fps)-len(set(fps))}
status=json.loads((ROOT/"download_status.json").read_text())
composite=json.loads((ROOT/"composite_audit.json").read_text()) if (ROOT/"composite_audit.json").exists() else {}
families=Counter(p.split("/")[-2] for p in composite.get("episode_source_paths",{}))
now=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
inventory={"generated_at_utc":now,"official_evaluation":{"episodes":4500,"labels":dict(sorted(old.items()))},"generated":coverage,"composite_download":status,"composite_task_families":dict(sorted(families.items()))}
dump(ROOT/"dataset_inventory.json",inventory)
if status["stage"]=="complete":
    dump(DATA/"control_composite/download_complete.json",{"repo_id":status["repo_id"],"revision":status["revision"],"verified_files":status["verified_files"],"validated_at":now})
    dump(DATA/"control_composite/dataset_role.json",{"role":"training","ready":True,"provenance":{"repo_id":status["repo_id"],"revision":status["revision"],"skill_labels":"not provided"}})
tree=ast.parse(Path("/opt/dominows/VLABench/VLABench/utils/skill_lib.py").read_text())
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SkillLib")
methods=[n.name for n in klass.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith("_")]
native_counts=Counter()
for v in coverage.values():native_counts.update(v["native_expert_labels"])
train_counts=coverage["planning_train"]["graph_skill_labels"]
method_table="\n".join(f"| {name} | {old.get(name,0)} | {train_counts.get(name,0)} | {native_counts.get(name,0)} |" for name in methods)
values={"METHOD_COVERAGE":method_table,"TIME":now,"DOWNLOAD":status["stage"],"ORIGINAL_COUNTS":"\n".join(f"| {n} | {c} |" for n,c in sorted(old.items())),"COMPOSITE_FAMILIES":"\n".join(f"| {n} | {c} |" for n,c in sorted(families.items())),"GENERATED_COUNTS":"\n".join(f"| {n} | {v['episodes']} | {', '.join(v['graph_skill_labels'])} |" for n,v in coverage.items())}
doc=(ROOT/"document_template.md").read_text()
for name,value in values.items():doc=doc.replace("{{"+name+"}}",str(value))
p=REPO/"test_regr/VLABenchAgentInterface/docs/VLABENCH_DATASETS_GPU2.md";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(doc);(ROOT/"VLABENCH_DATASETS_GPU2.md").write_text(doc)
print(json.dumps({"document":str(p),"download":status["stage"],"generated":{k:v["episodes"] for k,v in coverage.items()}}))
