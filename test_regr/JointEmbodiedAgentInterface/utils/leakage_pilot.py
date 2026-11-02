"""Reproducible, inference-only shortcut audit of a historical joint checkpoint."""
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import random
import unicodedata

CONDITIONS = ("original", "anonymous-objects", "no-images", "shuffled-images",
              "shuffled-instructions", "anonymous-no-images")
VERSION = "ce-shortcut-pilot-v1"


def add_pilot_arguments(parser):
    parser.add_argument("--pilot-manifest", type=Path,
                        help="immutable task-balanced diagnostic manifest; requires --no-dfa --split test")
    parser.add_argument("--pilot-condition", choices=CONDITIONS, default="original")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_key(example):
    return digest([normalized(example.instruction), example.entities])


def images(example):
    return example.segmented_image_paths or example.image_paths


def freeze_row(example):
    """Freeze inputs and targets independently; reference corrections happen elsewhere."""
    return {"episode_id": example.episode_id, "instruction": example.instruction,
            "entities": example.entities, "reference": example.operation_sequence,
            "dependency": example.dependency, "images": [str(p) for p in images(example)],
            "task": example.task, "dimension": example.dimension}


def select_pilot(splits, size=90, seed=42):
    train_keys = {text_key(x) for x in splits["train"]}
    groups = defaultdict(list)
    for example in splits["test"]:
        groups[example.task].append(example)
    rng = random.Random(seed)
    selected = []
    for task in sorted(groups):
        rows = sorted(groups[task], key=lambda x: x.episode_id)
        rng.shuffle(rows)
        repeated = [x for x in rows if text_key(x) in train_keys]
        novel = [x for x in rows if text_key(x) not in train_keys]
        chosen = (repeated[:1] + novel[:1])
        chosen_ids = {x.episode_id for x in chosen}
        chosen += [x for x in rows if x.episode_id not in chosen_ids][:max(0, 2-len(chosen))]
        selected.extend(chosen)
    if len(selected) > size:
        raise ValueError("pilot size is smaller than two episodes per available task")
    used = {x.episode_id for x in selected}
    rest = sorted((x for x in splits["test"] if x.episode_id not in used), key=lambda x: x.episode_id)
    rng.shuffle(rest)
    selected.extend(rest[:size-len(selected)])
    if len(selected) != size:
        raise ValueError("not enough test episodes for the requested pilot")
    return selected


def donor_map(rows, *, different_instruction=False, seed=42):
    """A one-to-one derangement, including different text for instruction controls."""
    rng = random.Random(seed)
    ids = list(range(len(rows)))
    # Rejection sampling is simple and deterministic for this fixed 90-episode pilot.
    for _ in range(10000):
        rng.shuffle(ids)
        if all(i != j and (not different_instruction or
               normalized(rows[i].instruction) != normalized(rows[j].instruction))
               for i, j in enumerate(ids)):
            return {rows[i].episode_id: rows[j].episode_id for i, j in enumerate(ids)}
    raise ValueError("cannot construct distinct instruction/image donors")


def audit_overlap(splits, root):
    """CPU-only corpus audit. Image hashes distinguish bytes from decoded pixels."""
    from PIL import Image
    from test_regr.VLABenchAgentInterface.dataset_protocol import scene_fingerprint
    root = Path(root)
    records = []
    for split, examples in splits.items():
        for example in examples:
            episode = root / example.episode_id
            byte_hashes, pixel_hashes = [], []
            paths = sorted(set(example.image_paths + example.segmented_image_paths))
            for path in paths:
                byte_hashes.append(file_digest(path))
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    pixel_hashes.append(hashlib.sha256(
                        str(rgb.size).encode() + b"\0RGB\0" + rgb.tobytes()).hexdigest())
            files = {str(p.relative_to(episode)): file_digest(p) for p in sorted(episode.rglob("*"))
                     if p.is_file() and p.parts[-2] in ("input", "env_config", "output")}
            records.append({"episode_id": example.episode_id, "split": split,
                            "instruction": normalized(example.instruction),
                            "entities": digest(example.entities), "text_input": text_key(example),
                            "reference": digest(example.operation_sequence),
                            "scene": scene_fingerprint(episode), "file_hashes": files,
                            "image_bytes": byte_hashes, "image_pixels": pixel_hashes,
                            "image_byte_bundle": digest(byte_hashes),
                            "image_pixel_bundle": digest(pixel_hashes)})
        print("[pilot-audit] hashed", split, len(examples), flush=True)
    train = [r for r in records if r["split"] == "train"]
    summary = {}
    for split in ("validation", "test"):
        rows = [r for r in records if r["split"] == split]
        summary[split] = {}
        for field in ("episode_id", "instruction", "entities", "text_input", "scene",
                      "image_byte_bundle", "image_pixel_bundle"):
            known = {r[field] for r in train}
            summary[split][field] = [r["episode_id"] for r in rows if r[field] in known]
        for field in ("image_bytes", "image_pixels"):
            known = {h for r in train for h in r[field]}
            summary[split][field] = [r["episode_id"] for r in rows if known.intersection(r[field])]
        known = {(r["text_input"], r["reference"]) for r in train}
        summary[split]["text_input_and_reference"] = [r["episode_id"] for r in rows
             if (r["text_input"], r["reference"]) in known]
    return {"records": records, "cross_split_matches": summary,
            "split_ids": {k: [x.episode_id for x in v] for k, v in splits.items()}}


def prepare(args):
    from test_regr.VLABenchAgentInterface.dataset import load_planning_examples, deterministic_split
    from test_regr.VLABenchAgentInterface.repair_density_labels import updated_density_examples
    output = Path(args.output)
    if output.exists():
        raise ValueError("use a new audit directory; manifests are immutable")
    examples = load_planning_examples(args.planning_dir, observation_mode="visual")
    splits = deterministic_split(examples, seed=args.seed)
    selected = select_pilot(splits, args.size, args.seed)
    corrected = updated_density_examples(selected)
    archive = {r["episode_id"]: r for r in map(json.loads, Path(args.archive_predictions).read_text().splitlines())}
    train_keys = {text_key(x) for x in splits["train"]}
    rows = []
    for original, reference in zip(selected, corrected):
        row = freeze_row(original)
        row.update(corrected_reference=reference.operation_sequence,
                   repeated_text_input=text_key(original) in train_keys,
                   archived_prediction=archive[original.episode_id]["prediction"],
                   archived_corrected=archive[original.episode_id]["corrected"])
        rows.append(row)
    source = Path(args.source_root)
    sources = {str(p.relative_to(source)): file_digest(p)
               for folder in (source/"test_regr/JointEmbodiedAgentInterface",
                              source/"test_regr/VLABenchAgentInterface", source/"domiknows/generation")
               for p in sorted(folder.rglob("*.py"))}
    overlap = audit_overlap(splits, args.planning_dir)
    audited = {r["episode_id"]: r for r in overlap["records"]}
    for row in rows:
        row["episode_files"] = audited[row["episode_id"]]["file_hashes"]
    manifest = {"version": VERSION, "seed": args.seed, "size": args.size,
                "scope": "task-balanced diagnostic pilot, not full-test performance",
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_digest(args.checkpoint),
                "source_hashes": sources, "source_sha256": digest(sources),
                "planning_root": str(Path(args.planning_dir).resolve()),
                "split_ids": overlap["split_ids"], "rows": rows,
                "image_donors": donor_map(selected, seed=args.seed),
                "instruction_donors": donor_map(selected, different_instruction=True, seed=args.seed),
                "archive_sha256": file_digest(args.archive_predictions),
                "overlap_audit_sha256": digest(overlap)}
    manifest["manifest_sha256"] = digest(manifest)
    output.mkdir(parents=True)
    for name, value in (("manifest.json", manifest), ("overlap_audit.json", overlap)):
        (output/name).write_text(json.dumps(value, indent=2, default=str)+"\n")
    print("PILOT_PREPARED", output, Counter(x.task for x in selected), flush=True)


def load_manifest(path, checkpoint):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("manifest_sha256")
    if digest(manifest) != expected:
        raise ValueError("pilot manifest checksum mismatch")
    manifest["manifest_sha256"] = expected
    if manifest["version"] != VERSION or file_digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("pilot checkpoint/version mismatch")
    ids = [x["episode_id"] for x in manifest["rows"]]
    if len(ids) != manifest["size"] or len(set(ids)) != len(ids):
        raise ValueError("pilot requires the exact unique episode manifest")
    source = Path(__file__).resolve().parents[3]
    for relative, sha in manifest["source_hashes"].items():
        if file_digest(source/relative) != sha:
            raise ValueError("pilot source changed: "+relative)
    return manifest


def guard_settings(path, settings):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("pilot resume configuration differs; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2)+"\n")


def pilot_examples(examples, manifest, condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown pilot condition")
    lookup = {x.episode_id: x for x in examples}
    rows = []
    for row in manifest["rows"]:
        example = lookup[row["episode_id"]]
        if digest(freeze_row(example)) != digest({k: row[k] for k in freeze_row(example)}):
            raise ValueError("pilot input/reference changed since manifest creation")
        for relative, sha in row.get("episode_files", {}).items():
            if file_digest(Path(manifest["planning_root"])/example.episode_id/relative) != sha:
                raise ValueError("pilot episode file changed: "+relative)
        # Freeze scoring labels before donor instructions or images are introduced.
        example = replace(example, operation_sequence=tuple(row["corrected_reference"]))
        if condition in ("no-images", "anonymous-no-images"):
            example = replace(example, image_paths=(), segmented_image_paths=())
        elif condition == "shuffled-images":
            donor = lookup[manifest["image_donors"][example.episode_id]]
            example = replace(example, image_paths=donor.image_paths, segmented_image_paths=donor.segmented_image_paths)
        elif condition == "shuffled-instructions":
            donor = lookup[manifest["instruction_donors"][example.episode_id]]
            if normalized(donor.instruction) == normalized(example.instruction):
                raise ValueError("instruction donor has identical text")
            example = replace(example, instruction=donor.instruction)
        rows.append(example)
    return rows


class PilotObserver:
    """Record actual prompt construction and score immutable targets after generation."""
    def __init__(self, stream, examples, manifest, condition, planner):
        self.stream, self.examples, self.manifest = stream, examples, manifest
        self.condition, self.planner = condition, planner
        self.records, self.prompt, self.chat = [], None, None
        self.old_prompt, self.old_chat = planner._prompt_base, planner._chat

    def __enter__(self):
        from test_regr.VLABenchAgentInterface.models import planner_prompt
        def prompt(domain, context):
            actual = self.old_prompt(domain, context)
            if domain == "vlabench":
                plain = planner_prompt(context.get("instruction", ""), context.get("entity_table", ()),
                                       self.planner.vocabulary(domain))
                if actual != plain:
                    raise ValueError("pilot requires a plain prompt without auxiliary observation text")
                if self.condition in ("anonymous-objects", "anonymous-no-images"):
                    names = tuple(f"numbered object {i}" for i in range(len(context.get("entity_table", ()))))
                    actual = planner_prompt(context.get("instruction", ""), names, self.planner.vocabulary(domain))
                if self.prompt is not None and self.prompt != actual:
                    raise ValueError("prompt changed within an autoregressive trajectory")
                self.prompt = actual
            return actual
        def chat(prompt_text, count):
            actual = self.old_chat(prompt_text, count)
            self.chat = actual
            return actual
        self.planner._prompt_base, self.planner._chat = prompt, chat
        return self

    def __exit__(self, *args):
        self.planner._prompt_base, self.planner._chat = self.old_prompt, self.old_chat

    def observe(self, example, prediction, *, generation_error=None):
        from test_regr.VLABenchAgentInterface.planning_metrics import score_planning_metrics, strict_plan_match
        row = self.manifest["rows"][len(self.records)]
        if example.episode_id != row["episode_id"] or digest(example.operation_sequence) != digest(row["corrected_reference"]):
            raise ValueError("pilot episode order or frozen reference changed")
        if self.prompt is None or self.chat is None:
            raise ValueError("no actual prompt captured")
        record = {"episode_id": example.episode_id, "task": example.task, "dimension": example.dimension,
                  "condition": self.condition, "prompt": self.prompt, "rendered_chat": self.chat,
                  "prompt_sha256": digest(self.prompt), "instruction": example.instruction,
                  "canonical_entities": example.entities, "reference": example.operation_sequence,
                  "input_images": [{"path": str(p), "sha256": file_digest(p)} for p in images(example)],
                  "image_donor": self.manifest["image_donors"][example.episode_id] if self.condition == "shuffled-images" else None,
                  "instruction_donor": self.manifest["instruction_donors"][example.episode_id] if self.condition == "shuffled-instructions" else None,
                  "repeated_text_input": row["repeated_text_input"], "prediction": prediction,
                  "generation_error": generation_error,
                  "scores": score_planning_metrics(prediction, example.operation_sequence, example.dependency,
                                                    entity_table=example.entities),
                  "strict_plan_match": 0.0 if generation_error else strict_plan_match(
                      prediction, example.operation_sequence, entity_table=example.entities)}
        self.stream.write(json.dumps(record, default=str)+"\n"); self.stream.flush()
        self.records.append(record); self.prompt = self.chat = None

    def report(self):
        if len(self.records) != self.manifest["size"]:
            raise ValueError("incomplete pilot")
        reproduced = None
        if self.condition == "original":
            reproduced = all(digest(x["prediction"]) == digest(row["archived_prediction"])
                and all(abs(x["scores"][k]-row["archived_corrected"]["scores"][k]) < 1e-7
                        for k in ("SR", "PR", "SPR", "PM", "score"))
                and x["strict_plan_match"] == row["archived_corrected"]["strict_plan_match"]
                for x, row in zip(self.records, self.manifest["rows"]))
            if not reproduced:
                raise ValueError("original pilot does not reproduce archived predictions/scores")
        return {"manifest_sha256": self.manifest["manifest_sha256"], "condition": self.condition,
                "examples": len(self.records), "original_archive_reproduced": reproduced,
                "scope": self.manifest["scope"], "frozen_references_verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("planning-dir", "checkpoint", "source-root", "archive-predictions", "output"):
        parser.add_argument("--"+flag, required=True, type=Path)
    parser.add_argument("--size", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    prepare(parser.parse_args())
