"""Supervised LoRA warmup for GraphQA visual atomic predicates."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .object_centered_pipeline import (
    DEFAULT_IMAGE_CACHE,
    _internvl_module,
    _oracle_object_predicate_label,
    _qwen_module,
    create_object_centered_graph,
    load_instances,
    populate_example,
    required_visual_predicates,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["qwen-vl", "internvl"], default="qwen-vl")
    parser.add_argument("--family", choices=["unary", "relation", "all"], default="unary")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-negative-concepts", type=int, default=2)
    parser.add_argument("--max-negative-objects", type=int, default=3)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache every N optimizer steps; 0 disables hot-loop cache clearing.",
    )
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.load_4bit and args.load_8bit:
        parser.error("Only one quantization mode may be enabled")
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be >= 1")
    if args.empty_cache_every < 0:
        parser.error("--empty-cache-every must be >= 0")
    return args


def append_record(path, record):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def shared_model(mode):
    if mode == "qwen-vl":
        from qwen_vl_hf import QwenVLSharedHF

        return QwenVLSharedHF.model.model
    from peftvllm import InternVLSharedHF

    return InternVLSharedHF.model.model


def module_for(args, relation, value, options):
    factory = _qwen_module if args.mode == "qwen-vl" else _internvl_module
    return factory(
        args.model_path, args.device, relation=relation, attr=value, **options
    )


def unary_steps(args, instance, row, options):
    object_specs = required_visual_predicates([instance])[0]
    facts = set(tuple(fact) for fact in instance.get("visual_facts", []))
    object_ids = [str(value) for value in instance["objects"]]
    pending = []
    for kind, value in object_specs:
        labels = torch.tensor(
            [_oracle_object_predicate_label(facts, kind, object_id, value) for object_id in object_ids],
            dtype=torch.long,
            device=args.device,
        )
        pending.append((kind, value, labels))

    positives = [item for item in pending if bool(item[2].any().item())]
    negatives = [item for item in pending if not bool(item[2].any().item())]
    if args.max_negative_concepts >= 0:
        negatives = negatives[: args.max_negative_concepts]

    for kind, value, labels in positives + negatives:
        positive_rows = torch.nonzero(labels == 1, as_tuple=False).flatten().tolist()
        negative_rows = torch.nonzero(labels == 0, as_tuple=False).flatten().tolist()
        if args.max_negative_objects >= 0:
            negative_rows = negative_rows[: args.max_negative_objects]
        selected_rows = sorted(set(positive_rows + negative_rows))
        if not selected_rows:
            continue
        boxes = row["object_boxes"]
        row_index = torch.tensor(selected_rows, dtype=torch.long, device=labels.device)
        if torch.is_tensor(boxes):
            selected_boxes = boxes.index_select(0, row_index.to(boxes.device))
        else:
            selected_boxes = [boxes[index] for index in selected_rows]
        selected_labels = labels.index_select(0, row_index)
        yield module_for(args, 1, value, options), selected_labels, f"{kind}:{value}", selected_boxes


def relation_steps(args, instance, row, options):
    relations = required_visual_predicates([instance])[1]
    facts = set(tuple(fact) for fact in instance.get("visual_facts", []))
    object_ids = [str(value) for value in instance["objects"]]
    for relation in relations:
        labels = torch.tensor(
            [int((relation, src, dst) in facts) for src in object_ids for dst in object_ids],
            dtype=torch.long,
            device=args.device,
        )
        yield module_for(args, 2, relation, options), labels, f"Relation:{relation}", row["object_boxes"]


def main():
    args = parse_args()
    instances, failures = load_instances(
        args.task_path, args.kb_dir, args.limit,
        single_answer_only=False, offset=args.offset,
    )
    print(json.dumps({"loaded": len(instances), "failures": len(failures)}), flush=True)
    options = {
        "use_llm_lora": True,
        "use_vision_lora": args.mode == "internvl",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "max_num": 1,
    }
    module_for(args, 1, "object", options)
    model = shared_model(args.mode)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("VLM exposes no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    completed_steps = 0
    pending_backward = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        for local_index, instance in enumerate(instances):
            context = create_object_centered_graph([instance])
            seed = dict(instance)
            seed["expected_answer"] = str(instance["objects"][0])
            row = populate_example(seed, context, image_cache=args.image_cache, device=args.device)
            generators = []
            if args.family in {"unary", "all"}:
                generators.append(unary_steps(args, instance, row, options))
            if args.family in {"relation", "all"}:
                generators.append(relation_steps(args, instance, row, options))
            for generator in generators:
                for module, labels, concept, boxes in generator:
                    logits = module(
                        row["pil_image"], row["image_filename"], boxes
                    )
                    loss = F.nll_loss(logits.float(), labels)
                    (loss / args.grad_accum_steps).backward()
                    pending_backward += 1
                    if pending_backward >= args.grad_accum_steps:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        pending_backward = 0
                    completed_steps += 1
                    append_record(args.log, {
                        "epoch": epoch,
                        "index": args.offset + local_index,
                        "concept": concept,
                        "loss": float(loss.detach().cpu()),
                        "labels": labels.detach().cpu().tolist(),
                        "predictions": logits.detach().argmax(dim=-1).cpu().tolist(),
                    })
                    del logits, loss
                    if (
                        args.empty_cache_every
                        and completed_steps % args.empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()
            print(
                f"epoch={epoch} instance={local_index + 1}/{len(instances)} steps={completed_steps}",
                flush=True,
            )
            del row, context
            gc.collect()
        if pending_backward:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending_backward = 0
        checkpoint = args.output / f"epoch{epoch}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        torch.save({
            "epoch": epoch,
            "steps": completed_steps,
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }, checkpoint / "training_state.pt")
        print(f"saved={checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
