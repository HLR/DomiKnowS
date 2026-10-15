"""Resumable, evidence-producing native and graph planner experiments.

Run with ``python -m test_regr.VLABenchAgentInterface.planning_experiment --help``.
No controller is constructed and no planner weights are updated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .dataset import load_scene_planning_examples, deterministic_split
from .native_planning import (FrozenNativePlanner, NativePlanFailure, NATIVE_LANGUAGE_VERSION,
                              grounding_for_example, validate_native_plan)
from .planning_metrics import PlanningMetrics, score_planning_metrics, TASK_DIMENSIONS
from .world_graph import canonicalize_plan, validate_plan, build_vlabench_world_graph


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scene_only_examples(root):
    """Do not expose the legacy loader's gold-argument entity-table augmentation."""
    return load_scene_planning_examples(root)


def inventory(root, output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    examples = scene_only_examples(root)
    splits = deterministic_split(examples, seed=seed)
    files = []
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and path.suffix.lower() in {".json", ".txt", ".png", ".jpg", ".jpeg", ".webp"}:
            files.append({"path": path.relative_to(root).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    for split, items in splits.items():
        write_json(output / f"{split}_manifest.json", {
            "dataset_sha256": fingerprint, "seed": seed, "split": split,
            "episode_ids": [example.episode_id for example in items],
        })
    write_json(output / "dataset_inventory.json", {
        "root": str(root), "dataset_sha256": fingerprint, "files": files,
        "examples": len(examples), "seed": seed,
        "split_counts": {split: len(items) for split, items in splits.items()},
        "folder_counts": dict(Counter(example.episode_id.split("/")[0] for example in examples)),
        "task_counts": dict(Counter(example.task for example in examples)),
        "split_dimensions": {split: dict(Counter(example.dimension or "unmapped" for example in items)) for split, items in splits.items()},
        "unknown_tasks": sorted({example.task for example in examples if example.task not in TASK_DIMENSIONS}),
        "entity_table_source": "scene configuration only; no gold argument augmentation",
        "historical_split_status": "Reconstructed from current dataset and launch seed; original training input manifest is not available",
    })
    return splits


def read_records(path):
    path = Path(path)
    records = {}
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                # A torn final append must be removed before the next append.
                path.write_text("\n".join(lines[:index]) + ("\n" if index else ""), encoding="utf-8")
                break
            if record["episode_id"] in records:
                raise ValueError("duplicate episode in persisted results")
            records[record["episode_id"]] = record
    return records


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_graph_planner(checkpoint, model_path, load_in_4bit=True):
    from .graph import PlanVocabulary
    from .models import QwenVLPlanner
    from .training import build_constraint_runtime, _load_planner_trainable_state, _planner_configuration
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("standalone_checkpoint_version") != 5:
        raise ValueError("evaluation requires a version-5 standalone checkpoint")
    world = build_vlabench_world_graph("planner_only_evaluation")
    saved = payload["vocabulary"]
    vocabulary = PlanVocabulary(**saved)
    runtime = build_constraint_runtime(world, max_entities=vocabulary.max_entities, max_operations=8, name_prefix="planner_only")
    if payload["domain_checksum"] != world.domain_checksum or payload["vocabulary_checksum"] != runtime.vocabulary.checksum:
        raise ValueError("checkpoint domain/vocabulary differs from evaluation runtime")
    planner = QwenVLPlanner.from_pretrained(vocabulary, model_path,
        use_lora=True, load_in_4bit=load_in_4bit, gradient_checkpointing=False,
        local_files_only=True, decoder_hidden_size=payload["planner_configuration"]["decoder_hidden_size"])
    if payload["planner_configuration"] != _planner_configuration(planner):
        raise ValueError("checkpoint planner configuration mismatch")
    if payload["parameter_ownership_checksum"] != planner.compute_parameter_ownership_checksum():
        raise ValueError("checkpoint parameter ownership mismatch")
    _load_planner_trainable_state(planner, payload["planner"])
    planner.eval()
    return planner, runtime


def experiment(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    examples = scene_only_examples(args.planning_dir)
    manifest = json.loads(Path(args.manifest).read_text())
    by_id = {example.episode_id: example for example in examples}
    selected = [by_id[episode_id] for episode_id in manifest["episode_ids"]]
    if args.limit:
        selected = selected[:args.limit]
    settings = {"mode": args.mode, "manifest_sha256": digest(args.manifest), "max_tokens": args.max_tokens,
        "max_operations": 8 if args.mode == "graph" else args.max_operations, "model": args.model, "load_in_4bit": args.load_in_4bit,
        "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
        "hmm_sha256": digest(args.hmm) if args.hmm else None, "weight": args.weight,
        "seed": args.seed, "limit": args.limit,
        "entity_table_source": "scene_components_only_v2",
        "native_language_version": NATIVE_LANGUAGE_VERSION,
        "grammar_profile": args.grammar_profile, "object_constraints": args.object_constraints,
        "object_constraints_applied": args.mode in {"dfa", "hmm"} and args.object_constraints == "instruction",
        "episode_ids": [e.episode_id for e in selected]}
    if args.mode == "corpus":
        settings["teacher_batch_size"] = 8
    identity_path = output / "settings.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != settings:
        raise ValueError("cannot resume with changed experiment settings")
    write_json(identity_path, settings)
    path = output / "episodes.jsonl"
    records = read_records(path)
    print(json.dumps({"stage": args.mode, "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "examples": len(selected), "completed": len(records), "output": str(output)}), flush=True)
    graph = args.mode == "graph"
    if graph:
        planner, runtime = load_graph_planner(args.checkpoint, args.model, args.load_in_4bit)
    else:
        planner = FrozenNativePlanner.load(args.model, args.load_in_4bit)
        settings["tokenizer_sha256"] = planner.checksum
        write_json(output / "model_provenance.json", {"tokenizer_sha256": planner.checksum,
            "model_files": {p.name: digest(p) for p in sorted(Path(args.model).glob("*.safetensors"))},
            "config_sha256": digest(Path(args.model) / "config.json")})
    hmm = None
    if args.hmm:
        payload = torch.load(args.hmm, map_location="cpu", weights_only=False)
        if payload["tokenizer_sha256"] != planner.checksum:
            raise ValueError("HMM tokenizer mismatch")
        hmm = tuple(payload[key].to("cuda") for key in ("initial", "transition", "emission"))
        planner.dp_cache_directory = Path(args.hmm).parent / "lookahead_cache"
        planner.dp_cache_identity = settings["hmm_sha256"]
    if args.mode == "corpus":
        for offset in range(0, len(selected), 8):
            batch = selected[offset:offset+8]
            if all(example.episode_id in records for example in batch):
                continue
            started = time.monotonic()
            # Recompute the original batch on resume to preserve sampling seeds.
            outputs = planner.generate_teacher_batch(batch, seed=args.seed+offset, max_tokens=args.max_tokens,
                                                     grammar_profile=args.grammar_profile)
            elapsed = time.monotonic()-started
            for example, (raw, tokens) in zip(batch, outputs):
                if example.episode_id in records:
                    continue
                error = None
                try:
                    canonicalize_plan(raw)
                except (ValueError, TypeError):
                    error = "teacher output is not a parseable plan"
                record = {"episode_id": example.episode_id, "raw": raw, "tokens": tokens,
                    "parse_error": error, "seconds": elapsed/len(batch)}
                append_record(path, record)
                records[example.episode_id] = record
            print(json.dumps({"stage": "corpus", "pid": os.getpid(), "completed": len(records),
                "total": len(selected), "batch_seconds": elapsed}), flush=True)
        return
    for index, example in enumerate(selected, 1):
        if example.episode_id in records:
            continue
        started = time.monotonic()
        error = None
        raw, tokens = "", []
        try:
            if graph:
                from .training import _planner_images
                images = _planner_images(example)
                try:
                    with torch.no_grad():
                        prediction = planner.generate_plan(instruction=example.instruction, images=images,
                            entity_table=example.entities, dfa=runtime.dfa, world=runtime.world_bundle, max_steps=runtime.max_tokens)
                finally:
                    for image in images:
                        image.close()
                raw, tokens = json.dumps(prediction), None
            else:
                raw, tokens = planner.generate(example, mode="raw" if args.mode == "corpus" else args.mode, max_tokens=args.max_tokens,
                    max_operations=args.max_operations, seed=args.seed + index, sample=args.mode == "corpus",
                    hmm=hmm, weight=args.weight, grammar_profile=args.grammar_profile,
                    object_constraints=args.object_constraints)
                try:
                    prediction = canonicalize_plan(raw)
                except (ValueError, TypeError) as exc:
                    prediction, error = [], str(exc)
        except NativePlanFailure as exc:
            prediction = []
            error = str(exc)
            raw, tokens = getattr(exc, "raw", raw), getattr(exc, "tokens", tokens)
        if args.mode == "corpus":
            if tokens is None or not tokens:
                raise RuntimeError("teacher generation produced no token sequence")
            record = {"episode_id": example.episode_id, "raw": raw, "tokens": tokens, "parse_error": error}
        else:
            scores = score_planning_metrics(prediction, example.operation_sequence, example.dependency)
            validation = validate_plan(prediction, entity_table=example.entities)
            record = {"episode_id": example.episode_id, "task": example.task, "dimension": example.dimension,
                "raw": raw, "tokens": tokens, "prediction": prediction, "scores": scores,
                "valid": validation.valid, "validation_errors": validation.errors, "error": error}
            if not graph:
                grounding = grounding_for_example(example, args.object_constraints)
                native_valid, native_errors = validate_native_plan(raw, example.entities,
                    grammar_profile=args.grammar_profile, max_operations=args.max_operations)
                grounded_valid, grounded_errors = validate_native_plan(raw, example.entities,
                    grammar_profile=args.grammar_profile, max_operations=args.max_operations, grounding=grounding)
                record.update(native_valid=native_valid, native_validation_errors=native_errors,
                              grounded_valid=grounded_valid, grounding_errors=grounded_errors,
                              instruction_constraints=grounding.as_dict())
        record["seconds"] = time.monotonic() - started
        append_record(path, record)
        records[example.episode_id] = record
        print(json.dumps({"stage": args.mode, "pid": os.getpid(), "index": index, "total": len(selected), "episode_id": example.episode_id, "seconds": record["seconds"], "error": error}), flush=True)
    if args.mode != "corpus":
        metrics = PlanningMetrics()
        for example in selected:
            item = records[example.episode_id]
            metrics.update(item["scores"], task=item["task"], dimension=item["dimension"])
        write_json(output / "summary.json", {"planning_metrics": metrics.result(),
            "valid": sum(records[e.episode_id]["valid"] for e in selected) / max(1, len(selected)),
            "errors": sum(bool(records[e.episode_id]["error"]) for e in selected),
            "seconds": sum(records[e.episode_id]["seconds"] for e in selected), "settings": settings})
        if not graph:
            summary = json.loads((output / "summary.json").read_text())
            summary.update(native_valid=sum(records[e.episode_id]["native_valid"] for e in selected)/max(1,len(selected)),
                           grounded_valid=sum(records[e.episode_id]["grounded_valid"] for e in selected)/max(1,len(selected)))
            write_json(output / "summary.json", summary)


def sequence_nll(head, labels, lengths):
    """Scaled forward filter: O(batch * sequence * states), no vocabulary-sized output per step."""
    emission, transition = head.emission_probs, head.transition_probs
    belief = head.initial_probs.expand(labels.shape[0], -1)
    likelihood = belief.new_zeros(labels.shape[0])
    for index in range(labels.shape[1]):
        mass = belief * emission[:, labels[:, index]].T
        scale = mass.sum(-1).clamp_min(torch.finfo(mass.dtype).tiny)
        active = index < lengths
        likelihood = likelihood + torch.where(active, scale.log(), 0)
        next_belief = (mass / scale[:, None]) @ transition
        belief = torch.where(active[:, None], next_belief, belief)
    return -likelihood.sum() / lengths.sum()


def fit_hmm(args):
    from domiknows.generation.learners.hmm.discrete.discreteHMMLearner import HMMGenerationHead
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = list(read_records(Path(args.corpus) / "episodes.jsonl").values())
    validation = list(read_records(Path(args.validation_corpus) / "episodes.jsonl").values())
    provenance = json.loads((Path(args.corpus) / "model_provenance.json").read_text())
    validation_provenance = json.loads((Path(args.validation_corpus) / "model_provenance.json").read_text())
    if provenance != validation_provenance:
        raise ValueError("training and validation teacher models/tokenizers differ")
    for directory, records in [(args.corpus, train), (args.validation_corpus, validation)]:
        settings = json.loads((Path(directory) / "settings.json").read_text())
        if settings["mode"] != "corpus" or set(settings["episode_ids"]) != {item["episode_id"] for item in records}:
            raise ValueError("distillation corpus is incomplete or has mismatched episode IDs")
    if {item["episode_id"] for item in train}.intersection(item["episode_id"] for item in validation):
        raise ValueError("training and validation teacher corpora overlap")
    head = HMMGenerationHead(label_count=args.vocab_size, state_count=128, random_seed=42).to("cuda")
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    start, best, stale, history = 0, float("inf"), 0, []
    resume = output / "progress.pt"
    identity = {"train_sha256": digest(Path(args.corpus) / "episodes.jsonl"),
        "validation_sha256": digest(Path(args.validation_corpus) / "episodes.jsonl"),
        "tokenizer_sha256": provenance["tokenizer_sha256"], "vocab_size": args.vocab_size}
    if resume.exists():
        payload = torch.load(resume, map_location="cuda", weights_only=False)
        if payload["identity"] != identity:
            raise ValueError("HMM corpus changed on resume")
        head.load_state_dict(payload["head"])
        optimizer.load_state_dict(payload["optimizer"])
        start, best, stale, history = payload["epoch"], payload["best"], payload["stale"], payload["history"]
        if stale >= 2 or start >= 10:
            return
    def batches(records, shuffle=False, epoch=0):
        indices = list(range(len(records)))
        if shuffle:
            import random
            random.Random(42 + epoch).shuffle(indices)
        for offset in range(0, len(indices), 16):
            rows = [records[i]["tokens"] for i in indices[offset:offset+16]]
            lengths = torch.tensor([len(row) for row in rows], device="cuda")
            labels = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long, device="cuda")
            for i, row in enumerate(rows):
                labels[i, :len(row)] = torch.tensor(row, device="cuda")
            yield labels, lengths
    for epoch in range(start, 10):
        head.train()
        for index, (labels, lengths) in enumerate(batches(train, True, epoch)):
            optimizer.zero_grad()
            loss = sequence_nll(head, labels, lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite HMM training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            if index % 10 == 0:
                print(json.dumps({"stage": "hmm_fit", "pid": os.getpid(), "epoch": epoch+1, "batch": index, "nll": float(loss.detach())}), flush=True)
        head.eval()
        with torch.no_grad():
            losses = [(float(sequence_nll(head, labels, lengths)), int(lengths.sum())) for labels, lengths in batches(validation)]
        val = sum(value * count for value, count in losses) / sum(count for _, count in losses)
        history.append({"epoch": epoch+1, "validation_nll": val})
        if val < best - 1e-4:
            best, stale = val, 0
            best_temporary = output / "best.tmp"
            torch.save({"initial": head.initial_probs.detach().cpu(), "transition": head.transition_probs.detach().cpu(),
                "emission": head.emission_probs.detach().cpu(), "tokenizer_sha256": identity["tokenizer_sha256"],
                "identity": identity, "epoch": epoch+1, "validation_nll": val}, best_temporary)
            os.replace(best_temporary, output / "best.pt")
        else:
            stale += 1
        temporary = output / "progress.tmp"
        torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1,
            "best": best, "stale": stale, "history": history, "identity": identity}, temporary)
        os.replace(temporary, resume)
        write_json(output / "history.json", history)
        if stale >= 2:
            break


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--planning-dir", required=True)
    inv.add_argument("--output", required=True)
    inv.add_argument("--seed", type=int, default=42)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--mode", choices=["raw", "dfa", "hmm", "graph", "corpus"], required=True)
    ev.add_argument("--planning-dir", required=True)
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--model", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--checkpoint")
    ev.add_argument("--hmm")
    ev.add_argument("--weight", type=float, default=0)
    ev.add_argument("--max-tokens", type=int, default=768)
    ev.add_argument("--max-operations", type=int, default=16)
    ev.add_argument("--grammar-profile", choices=["benchmark", "legacy"], default="benchmark")
    ev.add_argument("--object-constraints", choices=["instruction", "none"], default="instruction")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--limit", type=int)
    ev.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    fit = sub.add_parser("fit-hmm")
    fit.add_argument("--corpus", required=True)
    fit.add_argument("--validation-corpus", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--vocab-size", type=int, default=151936)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.planning_dir, args.output, args.seed)
    elif args.command == "fit-hmm":
        fit_hmm(args)
    else:
        if args.mode == "graph" and not args.checkpoint:
            parser.error("graph evaluation requires --checkpoint")
        if args.mode == "hmm" and not args.hmm:
            parser.error("HMM evaluation requires --hmm")
        experiment(args)


if __name__ == "__main__":
    main()
