"""Fine-tune a shared VLM predicate scorer through GraphQA execution loss."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from .object_centered_pipeline import (
    DEFAULT_IMAGE_CACHE,
    _internvl_module,
    _qwen_module,
    build_program,
    load_instances,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["qwen-vl", "internvl"], default="qwen-vl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--use-vision-lora",
        action="store_true",
        help="Also train InternVL vision-tower LoRA adapters. Off by default to keep 4-bit runs in memory.",
    )
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--choice-max-options", type=int, default=25)
    parser.add_argument("--answer-mode", choices=["iota", "membership", "miota", "mixed"], default="membership")
    parser.add_argument("--single-answer-only", action="store_true")
    parser.add_argument("--max-compiled-rows", type=int, default=0)
    parser.add_argument(
        "--no-global-consistency",
        action="store_true",
        help="Disable KB implication constraints while keeping derived KB concepts for execution.",
    )
    parser.add_argument("--grouped-unary", action="store_true", default=True)
    parser.add_argument("--no-grouped-unary", dest="grouped_unary", action="store_false")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.load_4bit and args.load_8bit:
        parser.error("Only one of --load-4bit and --load-8bit may be enabled")
    return args


def _append_jsonl(path, record):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _shared_vlm(mode):
    if mode == "qwen-vl":
        from qwen_vl_hf import QwenVLSharedHF

        # QwenVLHF's LoRA setup replaces the whole top-level model with the
        # PeftModel, so .model itself is already adapter-saveable.
        return QwenVLSharedHF.model.model
    from peftvllm import InternVLSharedHF

    # InternVLHF's LoRA setup only wraps the language_model submodule (see
    # peftvllm._apply_llm_lora), not the top-level model -- unlike Qwen-VL.
    # Saving InternVLSharedHF.model.model directly would dump the whole ~1.6GB
    # base model in plain HF format (no adapter_config.json), which is both
    # wasteful and not reloadable via PeftModel.from_pretrained. Target the
    # actual PEFT-wrapped submodule instead so checkpoints stay small and can
    # be chained across sharded runs via --adapter-checkpoint.
    return InternVLSharedHF.model.model.language_model


def _save_checkpoint(args, optimizer, epoch, examples):
    output = args.output / f"epoch{epoch}"
    output.mkdir(parents=True, exist_ok=True)
    model = _shared_vlm(args.mode)
    model.save_pretrained(output)
    torch.save(
        {
            "epoch": epoch,
            "examples": examples,
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        },
        output / "training_state.pt",
    )
    return output


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instances, failures = load_instances(
        args.task_path,
        args.kb_dir,
        args.limit,
        single_answer_only=args.single_answer_only or args.answer_mode == "iota",
        offset=args.offset,
        image_cache=args.image_cache,
    )
    print(json.dumps({
        "loaded": len(instances),
        "translation_failures": len(failures),
        "mode": args.mode,
        "device": args.device,
    }), flush=True)
    if not instances:
        raise ValueError("No trainable GraphQA instances were loaded")

    options = {
        "use_llm_lora": True,
        "use_vision_lora": args.mode == "internvl" and args.use_vision_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "max_length": args.max_length,
        "encode_batch_size": 1,
        "grouped_unary": args.grouped_unary,
        "choice_max_options": args.choice_max_options,
    }
    if args.adapter_checkpoint is not None:
        options["lora_adapter_path"] = str(args.adapter_checkpoint)
    # Initialize Transformers before the outer progress bar. Nested model-load
    # bars trigger a tqdm teardown bug in the CLEVER environment.
    if args.mode == "qwen-vl":
        _qwen_module(args.model_path, args.device, relation=1, attr="object", **options)
    else:
        _internvl_module(args.model_path, args.device, relation=1, attr="object", **options)
    trained = 0
    last_optimizer = None
    for epoch in range(1, args.epochs + 1):
        for local_index, instance in enumerate(instances):
            context = dataset = program = optimizer = None
            try:
                context, dataset, program = build_program(
                    [instance],
                    mode=args.mode,
                    model_path=args.model_path,
                    image_cache=args.image_cache,
                    device=args.device,
                    answer_mode=args.answer_mode,
                    qwen_options=options,
                    include_global_consistency=not args.no_global_consistency,
                )
                if args.max_compiled_rows and len(dataset) > args.max_compiled_rows:
                    record = {
                        "status": "skipped",
                        "epoch": epoch,
                        "index": args.offset + local_index,
                        "qid": instance.get("qid"),
                        "rows": len(dataset),
                        "reason": "too_many_compiled_rows",
                    }
                    _append_jsonl(args.log, record)
                    print(
                        f"epoch={epoch} instance={local_index + 1}/{len(instances)} "
                        f"trained={trained} status={record['status']} rows={len(dataset)}",
                        flush=True,
                    )
                    del program, dataset, context
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                program.to(args.device)
                trainable = [p for p in program.model.parameters() if p.requires_grad]
                if not trainable:
                    raise RuntimeError("VLM graph exposes no trainable LoRA parameters")
                # A fresh optimizer per instance, not one reused across the whole run:
                # build_program(...) rebuilds the DomiKnowS wrapper (and its parameter
                # list) from scratch every instance, so an optimizer captured once at
                # the first instance would step stale tensor references thereafter,
                # while its retained state keeps that first instance's graph pinned in
                # GPU memory for the rest of the run.
                optimizer = torch.optim.AdamW(trainable, lr=args.lr)
                program.opt = optimizer
                last_optimizer = optimizer
                program.train(
                    dataset,
                    warmup_epochs=0,
                    constraint_epochs=1,
                    device=args.device,
                    c_lr=args.lr,
                )
                trained += 1
                record = {
                    "status": "ok",
                    "epoch": epoch,
                    "index": args.offset + local_index,
                    "qid": instance.get("qid"),
                    "rows": len(dataset),
                }
            except Exception as exc:
                record = {
                    "status": "error",
                    "epoch": epoch,
                    "index": args.offset + local_index,
                    "qid": instance.get("qid"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            _append_jsonl(args.log, record)
            print(
                f"epoch={epoch} instance={local_index + 1}/{len(instances)} "
                f"trained={trained} status={record['status']}", flush=True,
            )
            del program, dataset, context, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if last_optimizer is None:
            raise RuntimeError("No VLM training instance completed; inspect --log for the first error")
        checkpoint = _save_checkpoint(args, last_optimizer, epoch, trained)
        print(f"saved={checkpoint} trained={trained}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
