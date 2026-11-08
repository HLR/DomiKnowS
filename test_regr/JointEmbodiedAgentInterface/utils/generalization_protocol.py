"""Frozen training identities, reference repairs and generation-input provenance."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_json(path, value):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, default=str) + "\n"
    if path.exists() and json.loads(path.read_text()) != json.loads(text):
        raise ValueError(f"frozen experiment manifest differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return file_hash(path)


def corrected_references(examples, output):
    from test_regr.VLABenchAgentInterface.repair_density_labels import updated_density_examples, REPAIR_REVISION
    updated = updated_density_examples(examples)
    changes = []
    for before, after in zip(examples, updated):
        if before.operation_sequence != after.operation_sequence:
            changes.append({"episode_id": before.episode_id, "original": before.operation_sequence,
                            "scene_verified": after.operation_sequence,
                            "scene_sha256": file_hash(before.env_config_path)})
    freeze_json(Path(output) / "training_reference_policy.json", {
        "version": REPAIR_REVISION, "mode": "in-memory scene verification, original files preserved",
        "episode_ids": [e.episode_id for e in examples], "changes": changes})
    freeze_json(Path(output) / "training_reference_hashes.json", {
        before.episode_id: {
            "original": hashlib.sha256(json.dumps(before.operation_sequence, sort_keys=True).encode()).hexdigest(),
            "scene_verified": hashlib.sha256(json.dumps(after.operation_sequence, sort_keys=True).encode()).hexdigest()
        } for before, after in zip(examples, updated)})
    return updated


def freeze_model_source(output, model_path, vision_model):
    path = Path(model_path)
    files = sorted({p for pattern in ('*.json', '*.safetensors', 'tokenizer*', '*.model')
                    for p in path.glob(pattern) if p.is_file()}) if path.is_dir() else []
    vision_path = Path(vision_model)
    if not vision_path.is_dir():
        from huggingface_hub import snapshot_download
        vision_path = Path(snapshot_download(vision_model, local_files_only=True))
    vision_files = sorted(p for p in vision_path.rglob('*')
                          if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin'))
    return freeze_json(Path(output) / "model_source.json", {
        "planner_model": str(model_path), "vision_model": vision_model,
        "planner_files": {p.name: file_hash(p) for p in files},
        "vision_snapshot": str(vision_path),
        "vision_files": {str(p.relative_to(vision_path)): file_hash(p) for p in vision_files}})


def freeze_inputs(output, planner, eai_train, eai_valid, splits, runtime):
    from .planner_context import planner_context
    output = Path(output)
    rows = []
    for domain, groups in (("eai", {"train": eai_train, "validation": eai_valid}), ("vlabench", splits)):
        for split, examples in groups.items():
            for index, example in enumerate(examples):
                context = planner_context(domain, example)
                images = list(context.get("images", ()))
                prompt = planner._chat(planner._prompt_base(domain, context), len(images))
                get = example.get if isinstance(example, dict) else lambda key, default=None: getattr(example, key, default)
                rows.append({"domain": domain, "split": split,
                    "episode_id": str(get("episode_id") or get("id") or f"{domain}:{split}:{index}"),
                    "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "images": [{"path": str(p), "sha256": file_hash(p)} for p in images],
                    "observation_fields": sorted(context),
                    "context": {k: v for k, v in context.items() if k != "images"}, "task": get("task")})
    return freeze_json(output / "generation_inputs.json", {"version": 1,
        "knowledge_manifest": runtime.knowledge_manifest, "inputs": rows})


def audit_training_paths(runtime, prepared, output):
    """Walk teacher targets as a diagnostic, never use them to bind a DFA."""
    from .planner_context import planner_context
    conflicts = []
    count = 0
    for split, items in prepared.items():
        for item in items:
            context = planner_context("vlabench", item)
            dfa = runtime.dfa_for("vlabench", context)
            state = dfa.start_state
            labels = item["target_plan_labels"].tolist()
            eos = runtime.vlabench_vocabulary.eos_label
            labels = labels[:labels.index(eos) + 1] if eos in labels else labels
            error = None
            for index, label in enumerate(labels):
                state = dfa.step(state, int(label))
                if state is None:
                    error = {"step": index, "label": int(label)}
                    break
            if error is not None or not dfa.is_accepting(state):
                conflicts.append({"split": split, "episode_id": context.get("episode_id"),
                                  "task": context.get("task"), "error": error or "nonaccepting"})
            count += 1
            if count % 100 == 0:
                print(f"[metadata-audit] paths={count} conflicts={len(conflicts)}", flush=True)
    freeze_json(Path(output) / "training_constraint_audit.json", {"examples": count, "conflicts": conflicts})
    if conflicts:
        raise ValueError(f"full-metadata training targets conflict with constraints: {len(conflicts)}; inspect audit")
