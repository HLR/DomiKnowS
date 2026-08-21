"""Evaluate single-answer C2 questions with compact predicate-signature graphs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from domiknows.graph.logicalConstrain import miotaL

from .object_centered_pipeline import (
    DEFAULT_IMAGE_CACHE,
    _internvl_module,
    _qwen_module,
    active_concepts_for_instances,
    build_program,
    evaluate_executable,
    load_instances,
    required_visual_predicates,
)


def predicate_signature(instance):
    object_specs, relation_specs, relation_object_specs, knowledge_sources = required_visual_predicates([instance])
    return (
        tuple(object_specs),
        tuple(relation_specs),
        tuple(relation_object_specs),
        tuple((key, tuple(values)) for key, values in knowledge_sources.items()),
    )


def signature_id(signature):
    return hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:16]


def question_family(instance):
    predicates = {
        condition[0]
        for conditions in ([instance["query"].get("conditions", [])] + instance["query"].get("alternatives", []))
        for condition in conditions
    }
    if "KG" in predicates:
        return "knowledge"
    if "SemanticClass" in predicates:
        return "semantic"
    if predicates & {"RelationFrom", "RelationTo"}:
        return "relation"
    return "visual"


def assign_shards(groups, num_shards):
    assignments = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for signature, instances in sorted(groups.items(), key=lambda item: (-len(item[1]), repr(item[0]))):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[shard].append((signature, instances))
        loads[shard] += len(instances)
    return assignments, loads


def completed_groups(path):
    completed = set()
    if not path.is_file():
        return completed
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok":
            completed.add(record["group_id"])
    return completed


def append_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def _tensor_to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _flatten_assignment(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            if torch.is_tensor(item):
                tensors.append(item.detach().flatten())
            elif item is not None:
                tensors.append(torch.as_tensor([item], dtype=torch.float32))
        return torch.cat(tensors) if tensors else torch.empty(0)
    if torch.is_tensor(value):
        return value.detach().flatten()
    if value is None:
        return torch.empty(0)
    return torch.as_tensor([value], dtype=torch.float32)


def miota_prediction_records(group_instances, dataset, program, device, threshold=0.5, decode_policy="threshold", show_progress=False):
    records = []
    with torch.no_grad():
        rows = zip(group_instances, dataset, program.populate(dataset, device=device))
        if show_progress:
            rows = tqdm(rows, total=len(group_instances), desc="miota eval", unit="ex")
        for row_index, (instance, row, datanode) in enumerate(rows):
            datanode.inferLocal()
            active_names = datanode.getActiveExecutableConstraintNames()
            lc_name = next(iter(active_names), None)
            if lc_name is None:
                raise ValueError("No active executable constraint for prediction logging")
            context = datanode._prepareLcLossContext("P", None)
            lc = context["lc_map"].get(lc_name)
            if not isinstance(lc, miotaL):
                raise ValueError(f"Prediction logging currently expects miotaL, got {type(lc).__name__}")
            solver = context["lossCalculator"].solver
            boolean_methods = solver.myLcLossBooleanMethods
            old_count_disabled = getattr(boolean_methods.countLogger, "disabled", False)
            old_solver_disabled = getattr(solver.myLogger, "disabled", False)
            boolean_methods.countLogger.disabled = True
            solver.myLogger.disabled = True
            try:
                result = datanode.calculateSingleLcLoss(
                    lc_name,
                    tnorm="P",
                    counting_tnorm=None,
                    _context=context,
                    label=datanode.getExecutableConstraintLabel(lc_name),
                )
            finally:
                boolean_methods.countLogger.disabled = old_count_disabled
                solver.myLogger.disabled = old_solver_disabled
            scores = _flatten_assignment(result.get("selectionDistribution"))
            objects = [str(value) for value in row.get("object_ids", instance.get("objects", []))]
            gold_answers = [str(value) for value in instance.get("expected_answers", [])]
            score_values = [float(value) for value in scores.tolist()]
            if decode_policy == "top1":
                predicted_answers = [objects[max(range(len(score_values)), key=lambda idx: score_values[idx])]] if score_values else []
            elif decode_policy == "family-top1":
                # Visual attribute/name predicates benefit from multi-answer thresholding,
                # while KB/semantic/relation predicates are currently over-selected by the VLM.
                if question_family(instance) == "visual":
                    predicted_answers = [
                        object_id for object_id, score in zip(objects, score_values)
                        if float(score) >= threshold
                    ]
                else:
                    predicted_answers = [objects[max(range(len(score_values)), key=lambda idx: score_values[idx])]] if score_values else []
            else:
                predicted_answers = [
                    object_id for object_id, score in zip(objects, score_values)
                    if float(score) >= threshold
                ]
            records.append({
                "row_index": row_index,
                "source_question_id": instance.get("source_question_id"),
                "source_image_id": instance.get("source_image_id"),
                "family": question_family(instance),
                "logic_str": row.get("logic_str"),
                "objects": objects,
                "scores": [float(value) for value in scores.tolist()],
                "prediction_threshold": float(threshold),
                "predicted_answers": predicted_answers,
                "gold_answers": gold_answers,
                "correct": set(predicted_answers) == set(gold_answers),
                "query": instance.get("query"),
            })
    return records


def evaluate_dynamic_active_concepts(args, pending, failures):
    """Evaluate with one reusable union graph and per-instance active concepts."""

    flat_items = []
    for group_key, group_instances in pending:
        signature, chunk_index = group_key
        for local_index, instance in enumerate(group_instances):
            flat_items.append((group_key, signature, chunk_index, local_index, instance))
    if not flat_items:
        print(json.dumps({
            "dynamic_active_concepts": True,
            "evaluated_examples_this_run": 0,
            "translation_failures": len(failures),
            "output": str(args.output),
        }, sort_keys=True), flush=True)
        return 0

    flat_instances = [item[-1] for item in flat_items]
    qwen_options = {
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "scallop_confidence": args.scallop_confidence,
        "scallop_learnable_scale": args.scallop_learnable_scale,
        "scallop_checkpoint": args.scallop_checkpoint,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": [item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
        "max_length": args.max_length,
        "encode_batch_size": args.encode_batch_size,
        "scallop_mlp_hidden_dim": args.scallop_mlp_hidden_dim,
        "scallop_mlp_dropout": args.scallop_mlp_dropout,
        "scallop_mlp_checkpoint": args.scallop_mlp_checkpoint,
        "lora_adapter_path": args.lora_adapter_path,
    }
    context, dataset, program = build_program(
        flat_instances,
        mode=args.mode,
        model_path=args.model_path,
        image_cache=args.image_cache,
        device=args.device,
        answer_mode=args.answer_mode,
        qwen_options=qwen_options,
    )
    total_examples = 0
    total_rows = 0
    correct_rows = 0
    errors = 0
    progress = tqdm(enumerate(flat_items), total=len(flat_items), desc="dynamic active", unit="ex")
    for row_index, (group_key, signature, chunk_index, local_index, instance) in progress:
        group_id = signature_id((group_key, local_index))
        family = question_family(instance)
        try:
            context.graph.set_active_concepts(active_concepts_for_instances(context, [instance]))
            row_dataset = [dataset[row_index]]
            prediction_records = []
            if args.answer_mode == "miota":
                prediction_records = miota_prediction_records(
                    [instance],
                    row_dataset,
                    program,
                    args.device,
                    threshold=args.prediction_threshold,
                    decode_policy=args.decode_policy,
                )
                group_rows = len(prediction_records)
                group_correct = sum(1 for item in prediction_records if item["correct"])
                accuracy = 100.0 * group_correct / group_rows if group_rows else 0.0
                if args.prediction_output is not None:
                    for item in prediction_records:
                        append_record(args.prediction_output, {
                            "status": "ok",
                            "group_id": group_id,
                            "chunk_index": chunk_index,
                            "dynamic_active_concepts": True,
                            **item,
                        })
            else:
                accuracy = float(evaluate_executable(row_dataset, program, args.device))
                group_rows = len(row_dataset)
                group_correct = int(round(accuracy * group_rows / 100.0))
            record = {
                "status": "ok",
                "group_id": group_id,
                "family": family,
                "chunk_index": chunk_index,
                "local_index": local_index,
                "examples": 1,
                "executable_rows": group_rows,
                "correct": group_correct,
                "accuracy": accuracy,
                "dynamic_active_concepts": True,
                "active_concepts": len(context.graph.get_active_concepts()),
                "object_predicates": len(signature[0]),
                "relation_predicates": len(signature[1]),
                "relation_object_predicates": len(signature[2]),
                "knowledge_predicates": len(signature[3]),
            }
            total_examples += 1
            total_rows += group_rows
            correct_rows += group_correct
        except Exception as exc:
            errors += 1
            record = {
                "status": "error",
                "group_id": group_id,
                "family": family,
                "chunk_index": chunk_index,
                "local_index": local_index,
                "examples": 1,
                "dynamic_active_concepts": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        append_record(args.output, record)
        progress.set_postfix(
            examples=total_examples,
            rows=total_rows,
            accuracy=(100.0 * correct_rows / total_rows if total_rows else 0.0),
            errors=errors,
        )
    context.graph.set_active_concepts(None)
    summary = {
        "dynamic_active_concepts": True,
        "evaluated_examples_this_run": total_examples,
        "executable_rows_this_run": total_rows,
        "correct_rows_this_run": correct_rows,
        "executable_accuracy_this_run": (
            100.0 * correct_rows / total_rows if total_rows else None
        ),
        "error_groups_this_run": errors,
        "output": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if errors == 0 else 1


def evaluate(args):
    instances, failures = load_instances(
        args.task_path, args.kb_dir, args.limit,
        single_answer_only=args.answer_mode == "iota",
        image_cache=args.image_cache,
    )
    signature_groups = defaultdict(list)
    for instance in instances:
        signature_groups[predicate_signature(instance)].append(instance)
    groups = {}
    for signature, values in signature_groups.items():
        chunk_size = args.max_examples_per_graph if args.max_examples_per_graph > 0 else len(values)
        for chunk_index, start in enumerate(range(0, len(values), chunk_size)):
            groups[(signature, chunk_index)] = values[start : start + chunk_size]

    assignments, loads = assign_shards(groups, args.num_shards)
    selected = assignments[args.shard_index]
    def within_caps(group_item):
        signature, _instances = group_item
        object_specs, relation_specs, _relation_object_specs, knowledge_specs = signature[0]
        return (
            (args.min_object_predicates is None or len(object_specs) >= args.min_object_predicates)
            and (args.max_object_predicates is None or len(object_specs) <= args.max_object_predicates)
            and (args.max_relation_predicates is None or len(relation_specs) <= args.max_relation_predicates)
            and (args.max_knowledge_predicates is None or len(knowledge_specs) <= args.max_knowledge_predicates)
        )

    if args.small_groups_first:
        selected = sorted(
            selected,
            key=lambda item: (len(item[0][0][0]), len(item[0][0][1]), len(item[0][0][2]), -len(item[1])),
        )
    if args.large_groups_first:
        # Prioritize signatures with the most members so the fixed per-group
        # DomiKnowS graph-construction cost gets amortized across as many
        # examples as possible -- a random/positional N-example slice tends
        # to hit mostly singleton signatures instead.
        selected = sorted(selected, key=lambda item: -len(item[1]))
    selected = [item for item in selected if within_caps(item)]
    if args.include_family:
        allowed = set(args.include_family)
        selected = [item for item in selected if question_family(item[1][0]) in allowed]
    if args.exclude_family:
        blocked = set(args.exclude_family)
        selected = [item for item in selected if question_family(item[1][0]) not in blocked]
    if args.max_groups is not None:
        selected = selected[: args.max_groups]
    completed = completed_groups(args.output) if args.resume else set()
    pending = [item for item in selected if signature_id(item[0]) not in completed]

    print(json.dumps({
        "loaded": len(instances),
        "answer_mode": args.answer_mode,
        "dynamic_active_concepts": args.dynamic_active_concepts,
        "translation_failures": len(failures),
        "signatures": len(signature_groups),
        "graph_chunks": len(groups),
        "max_examples_per_graph": args.max_examples_per_graph,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_loads": loads,
        "min_object_predicates": args.min_object_predicates,
        "max_object_predicates": args.max_object_predicates,
        "max_relation_predicates": args.max_relation_predicates,
        "max_knowledge_predicates": args.max_knowledge_predicates,
        "small_groups_first": args.small_groups_first,
        "large_groups_first": args.large_groups_first,
        "include_family": args.include_family,
        "exclude_family": args.exclude_family,
        "selected_groups": len(selected),
        "pending_groups": len(pending),
        "selected_examples": sum(len(values) for _, values in selected),
        "pending_examples": sum(len(values) for _, values in pending),
    }, sort_keys=True), flush=True)

    if args.dynamic_active_concepts:
        return evaluate_dynamic_active_concepts(args, pending, failures)

    total_examples = 0
    total_rows = 0
    correct_rows = 0
    errors = 0
    qwen_options = {
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "scallop_confidence": args.scallop_confidence,
        "scallop_learnable_scale": args.scallop_learnable_scale,
        "scallop_checkpoint": args.scallop_checkpoint,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": [item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
        "max_length": args.max_length,
        "encode_batch_size": args.encode_batch_size,
        "scallop_mlp_hidden_dim": args.scallop_mlp_hidden_dim,
        "scallop_mlp_dropout": args.scallop_mlp_dropout,
        "scallop_mlp_checkpoint": args.scallop_mlp_checkpoint,
        "lora_adapter_path": args.lora_adapter_path,
    }
    # Loading Transformers inside an active tqdm triggers a nested progress-bar
    # bug in the CLEVER environment. Initialize the shared model first.
    qwen_warmup = None
    if args.mode == "qwen-vl" and pending:
        qwen_warmup = _qwen_module(args.model_path, args.device, relation=1, attr="object", **qwen_options)
    elif args.mode == "internvl" and pending:
        qwen_warmup = _internvl_module(args.model_path, args.device, relation=1, attr="object", **qwen_options)
    progress = tqdm(pending, desc=f"C2 shard {args.shard_index}", unit="group")
    for group_key, group_instances in progress:
        signature, chunk_index = group_key
        group_id = signature_id(group_key)
        family = question_family(group_instances[0])
        try:
            context, dataset, program = build_program(
                group_instances,
                mode=args.mode,
                model_path=args.model_path,
                image_cache=args.image_cache,
                device=args.device,
                answer_mode=args.answer_mode,
                qwen_options=qwen_options,
            )
            prediction_records = []
            if args.answer_mode == "miota":
                # Generic evaluate_condition reports a Boolean LC counter, which is
                # not a reliable exact-set metric for miotaL. Decode the selected
                # object set and compare it against the multi-hot executable label.
                prediction_records = miota_prediction_records(
                    group_instances,
                    dataset,
                    program,
                    args.device,
                    threshold=args.prediction_threshold,
                    decode_policy=args.decode_policy,
                )
                group_rows = len(prediction_records)
                group_correct = sum(1 for item in prediction_records if item["correct"])
                accuracy = 100.0 * group_correct / group_rows if group_rows else 0.0
                if args.prediction_output is not None:
                    for item in prediction_records:
                        append_record(args.prediction_output, {
                            "status": "ok",
                            "group_id": group_id,
                            "chunk_index": chunk_index,
                            **item,
                        })
            else:
                accuracy = float(evaluate_executable(dataset, program, args.device))
                group_rows = len(dataset)
                group_correct = int(round(accuracy * group_rows / 100.0))
            record = {
                "status": "ok",
                "group_id": group_id,
                "family": family,
                "chunk_index": chunk_index,
                "examples": len(group_instances),
                "executable_rows": group_rows,
                "correct": group_correct,
                "accuracy": accuracy,
                "object_predicates": len(signature[0]),
                "relation_predicates": len(signature[1]),
                "relation_object_predicates": len(signature[2]),
                "knowledge_predicates": len(signature[3]),
            }
            total_examples += len(group_instances)
            total_rows += group_rows
            correct_rows += group_correct
            del program, dataset, context
        except Exception as exc:
            errors += 1
            record = {
                "status": "error",
                "group_id": group_id,
                "family": family,
                "chunk_index": chunk_index,
                "examples": len(group_instances),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        append_record(args.output, record)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.set_postfix(
            examples=total_examples, rows=total_rows,
            accuracy=(100.0 * correct_rows / total_rows if total_rows else 0.0),
            errors=errors,
        )

    summary = {
        "evaluated_examples_this_run": total_examples,
        "executable_rows_this_run": total_rows,
        "correct_rows_this_run": correct_rows,
        "executable_accuracy_this_run": (
            100.0 * correct_rows / total_rows if total_rows else None
        ),
        "error_groups_this_run": errors,
        "output": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if errors == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--model-path")
    parser.add_argument("--mode", choices=["oracle", "scallop-local", "scallop-trained", "scallop-mlp", "qwen-vl", "internvl"], default="qwen-vl")
    parser.add_argument("--answer-mode", choices=["iota", "membership", "miota", "mixed"], default="membership")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-examples-per-graph", type=int, default=16)
    parser.add_argument("--min-object-predicates", type=int)
    parser.add_argument("--max-object-predicates", type=int)
    parser.add_argument("--max-relation-predicates", type=int)
    parser.add_argument("--max-knowledge-predicates", type=int)
    parser.add_argument("--include-family", choices=["visual", "semantic", "knowledge", "relation"], action="append")
    parser.add_argument("--exclude-family", choices=["visual", "semantic", "knowledge", "relation"], action="append")
    parser.add_argument("--small-groups-first", action="store_true")
    parser.add_argument("--large-groups-first", action="store_true")
    parser.add_argument("--dynamic-active-concepts", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--decode-policy", choices=["threshold", "top1", "family-top1"], default="threshold")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--scallop-confidence", type=float, default=8.0)
    parser.add_argument("--scallop-learnable-scale", action="store_true")
    parser.add_argument("--scallop-checkpoint", type=Path)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--scallop-mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--scallop-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--scallop-mlp-checkpoint", type=Path)
    parser.add_argument("--lora-adapter-path", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.mode not in {"oracle", "scallop-local", "scallop-mlp"} and not args.model_path:
        parser.error("--model-path is required for learned predicate modes")
    if args.mode == "scallop-trained" and args.scallop_checkpoint is None:
        parser.error("--scallop-checkpoint is required for --mode scallop-trained")
    if args.load_4bit and args.load_8bit:
        parser.error("Only one of --load-4bit/--load-8bit can be enabled")
    return args


def main():
    return evaluate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
