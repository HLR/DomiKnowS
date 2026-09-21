"""Score one EAI or Joint checkpoint on identical, mutually held-out EAI rows.

Run this module once per checkpoint with the same data/seed/selection arguments.
The JSON outputs include decoded actions and every missing grounded goal fact.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from domiknows.generation.dfa.vocabulary import TokenVocabulary
from test_regr.EmbodiedAgentInterface.dataset import EOS_TOKEN, load_eai_dataset
from test_regr.EmbodiedAgentInterface.main import (
    build_trainable_program,
    dfa_constrained_sequence,
    labels_to_actions,
    load_eai_checkpoint,
    parse_args as eai_parse_args,
)
from test_regr.EmbodiedAgentInterface.reward import evaluate_goal_satisfaction
from test_regr.VLABenchAgentInterface.main import _controller

from .checkpoint import load_joint_checkpoint
from .main import _action_object_constraints, _ordered_union, build_parser
from .models import JointQwenVLPlanner
from .world_graph import build_joint_runtime, build_joint_world_graph


def paired_indices(count: int, *, seed: int, selection: str) -> list[int]:
    """Return reproducible row identities for paired checkpoint evaluation."""
    if selection == "canonical-validation":
        cut = min(count, max(1, int(0.8 * count)))
        return list(range(cut, count))
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    joint_cut = max(1, int(0.9 * count))
    joint_valid = indices[joint_cut:]
    if selection == "joint-validation-32":
        return joint_valid[:32]
    if selection == "mutual-holdout":
        eai_cut = min(count, max(1, int(0.8 * count)))
        return [index for index in joint_valid if index >= eai_cut]
    raise ValueError(f"unknown evaluation selection {selection!r}")


def _standalone_args(checkpoint: Path, device: str, backbone: str):
    old_argv = sys.argv
    try:
        sys.argv = [
            "eai-paired-evaluation", "--dataset", "all", "--split", "train",
            "--max-steps", "30", "--baseline-model", "causal-lm",
            "--llm-backbone-path", backbone, "--llm-device-map", "auto",
            "--use-lora", "--lora-r", "8", "--lora-alpha", "16",
            "--device", device, "--model", str(checkpoint),
        ]
        return eai_parse_args()
    finally:
        sys.argv = old_argv


def _load_standalone(checkpoint: Path, device: str, backbone: str, examples):
    args = _standalone_args(checkpoint, device, backbone)
    program, bundle = build_trainable_program(args, examples, device)
    load_eai_checkpoint(program, bundle, args, checkpoint, map_location="cpu")
    program.model.eval()

    def predict(item):
        labels = dfa_constrained_sequence(program, bundle, item, args.max_steps)
        return labels, bundle.vocabulary, bundle.world

    return predict


def _load_joint(checkpoint: Path, device: str, backbone: str, examples):
    args = build_parser().parse_args([
        "train-agent", "--two-stage", "--planner-model", backbone,
        "--device", device,
    ])
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    world = build_joint_world_graph("joint_embodied_training")
    runtime = build_joint_runtime(
        world,
        vocabulary,
        max_eai_steps=args.eai_max_steps,
        eai_object_tokens=_ordered_union(examples, "object_tokens"),
        eai_action_tokens=_ordered_union(examples, "action_tokens"),
        eai_action_sequence_tokens=_ordered_union(examples, "action_tokens"),
        eai_openable_object_tokens=_ordered_union(examples, "openable_object_tokens"),
        eai_action_object_constraint_tokens=_action_object_constraints(examples),
        max_vlabench_entities=args.max_entities,
        max_vlabench_operations=args.max_operations,
    )
    planner = JointQwenVLPlanner.from_pretrained(
        eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=backbone,
        load_in_4bit=args.load_in_4bit,
        decoder_hidden_size=args.planner_decoder_hidden_dim,
    )
    controller = _controller(args, torch.device(device))
    load_joint_checkpoint(
        checkpoint, runtime=runtime, planner=planner,
        controller=controller, map_location=device,
    )
    planner.eval()
    view = planner.for_domain("eai")

    def predict(item):
        with runtime.domain_scope("eai"):
            labels, _ = view.sample_labels(
                {
                    "instruction": item.get("causal_prompt_text", item.get("text", "")),
                    "goal": item.get("tl_goal", ""),
                },
                runtime.dfa_for("eai", item),
                max_steps=runtime.max_eai_steps,
                deterministic=True,
            )
            return labels, runtime.eai_vocabulary, runtime.world.eai

    return predict


def score_examples(examples, indices, predict):
    rows = []
    for position, index in enumerate(indices, 1):
        item = examples[index]
        labels, vocabulary, world = predict(item)
        result = evaluate_goal_satisfaction(
            labels, item, vocabulary, world_bundle=world,
        )
        missing = sorted(result["gold_state"] - result["predicted_state"])
        row = {
            "row_index": index,
            "task_id": item.get("task_id"),
            "instruction": item.get("natural_language_description") or item.get("text"),
            "tl_goal": item.get("tl_goal"),
            "gold_actions": list(item.get("target_action_tokens", ())),
            "predicted_actions": labels_to_actions(labels, vocabulary),
            "goal_success": bool(result["is_success"]),
            "goal_recall": float(result["recall"]),
            "temporal_progress": float(result["temporal_progress"]),
            "missing_goal_facts": [list(fact) for fact in missing],
            "parse_error": result["parse_error"],
        }
        rows.append(row)
        print(
            f"[paired-eai] {position}/{len(indices)} row={index} "
            f"success={row['goal_success']} recall={row['goal_recall']:.3f} "
            f"missing={len(missing)}",
            flush=True,
        )
    count = len(rows)
    return {
        "examples": count,
        "goal_success": sum(row["goal_success"] for row in rows) / count if count else 0.0,
        "goal_recall": sum(row["goal_recall"] for row in rows) / count if count else 0.0,
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("standalone", "joint"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eai-data-path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=("canonical-validation", "mutual-holdout", "joint-validation-32"),
        default="canonical-validation",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    max_steps = 30 if args.model == "standalone" else 60
    examples = load_eai_dataset(
        dataset_name="all", split="train", data_path=args.eai_data_path,
        device="cpu", max_steps=max_steps,
    )
    indices = paired_indices(len(examples), seed=args.seed, selection=args.selection)
    if not indices:
        parser.error("the two validation splits have no common examples")
    backbone = args.backbone or (
        "Qwen/Qwen3-8B" if args.model == "standalone"
        else "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    loader = _load_standalone if args.model == "standalone" else _load_joint
    predict = loader(args.checkpoint, args.device, backbone, examples)
    with torch.no_grad():
        result = score_examples(examples, indices, predict)
    result.update({
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "backbone": backbone,
        "selection": args.selection,
        "row_indices": indices,
        "seed": args.seed,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model", "examples", "goal_success", "goal_recall", "output") if key in result}, indent=2), flush=True)
    print(f"[paired-eai] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
