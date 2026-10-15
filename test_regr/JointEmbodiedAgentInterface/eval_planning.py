"""Evaluate the non-interactive VLABench planning progress score for a joint checkpoint.

This evaluates the vision-language planner on static scene inputs against ground-truth
subtask DAGs according to official Table 1 planning metrics:
  - Skill Recall (SR)
  - Parameter Recall (PR)
  - Skill-Parameter Recall (SPR)
  - Precise / Dependency Graph Match (PM)
  - Overall Non-Interactive Progress Score (0.4*SR + 0.4*PR + 0.1*SPR + 0.1*PM)
  - Six-dimension breakdown (M&T, Spatial, ComSense, Semantic, PhysLaw, Complex)
  - Macro average
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import torch

from test_regr.VLABenchAgentInterface.dataset import (
    deterministic_split,
    load_planning_examples,
)
from test_regr.VLABenchAgentInterface.main import _controller
from test_regr.VLABenchAgentInterface.training import evaluate_planner

from .checkpoint import load_joint_checkpoint
from .main import (
    DEFAULT_VLABENCH_PLANNING,
    _prepare,
    build_parser as build_main_parser,
)
from .models import DEFAULT_MODEL_ID, JointQwenVLPlanner
from .program import _runtime_adapter


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate non-interactive VLABench planning progress score for a joint checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to the joint checkpoint (e.g. joint_stage2_best.pt)",
    )
    parser.add_argument(
        "--planning-dir",
        type=Path,
        default=DEFAULT_VLABENCH_PLANNING,
        help=f"path to VLABench planning directory (default: {DEFAULT_VLABENCH_PLANNING})",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test", "train", "all"),
        default="validation",
        help="dataset split to evaluate (default: validation)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of planning examples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="path to save evaluation metrics as JSON",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="computation device (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--planner-model",
        default=DEFAULT_MODEL_ID,
        help=f"planner backbone model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--planner-decoder-hidden-dim",
        type=int,
        default=512,
        help="hidden dimension for the planner causal decoder",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load base LLM backbone in 4-bit NF4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for deterministic data splitting",
    )
    # EAI arguments required to reconstruct the identical joint runtime
    parser.add_argument("--eai-dataset", default="all")
    parser.add_argument("--eai-split", default="train")
    parser.add_argument("--eai-data-path", default=None)
    parser.add_argument("--eai-limit", type=int, default=None)
    parser.add_argument("--eai-max-steps", type=int, default=60)
    parser.add_argument("--eai-reward-mode", default="dense")
    parser.add_argument("--max-entities", type=int, default=32)
    parser.add_argument("--max-operations", type=int, default=16)
    # Controller arguments required to satisfy joint checkpoint deserialization
    parser.add_argument("--tiny-vision", action="store_true")
    parser.add_argument("--vision-dim", type=int, default=768)
    parser.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-views", type=int, default=4)
    return parser


def _prepare_eval(args, device):
    from domiknows.generation.dfa.vocabulary import TokenVocabulary
    from test_regr.EmbodiedAgentInterface.dataset import EOS_TOKEN, load_eai_dataset
    from test_regr.EmbodiedAgentInterface.reward import make_eai_reward_function
    from .main import _action_object_constraints, _ordered_union
    from .world_graph import build_joint_runtime, build_joint_world_graph

    eai_examples = load_eai_dataset(
        dataset_name=args.eai_dataset,
        split=args.eai_split,
        limit=args.eai_limit,
        data_path=args.eai_data_path,
        device="cpu",
        max_steps=args.eai_max_steps,
    )
    if not eai_examples:
        raise ValueError("EAI dataset is empty")
    vocabulary = TokenVocabulary(eai_examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    world = build_joint_world_graph("joint_embodied_training")
    for item in eai_examples:
        item["reward_function"] = make_eai_reward_function(
            item,
            vocabulary=vocabulary,
            mode=args.eai_reward_mode,
            world_bundle=world.eai,
        )

    vlabench_examples = load_planning_examples(args.vlabench_planning_dir)
    if not vlabench_examples:
        raise ValueError("VLABench planning dataset is empty")
    vlabench_splits = deterministic_split(vlabench_examples, seed=args.seed)
    runtime = build_joint_runtime(
        world,
        vocabulary,
        max_eai_steps=args.eai_max_steps,
        eai_object_tokens=_ordered_union(eai_examples, "object_tokens"),
        eai_action_tokens=_ordered_union(eai_examples, "action_tokens"),
        eai_action_sequence_tokens=_ordered_union(eai_examples, "action_tokens"),
        eai_openable_object_tokens=_ordered_union(eai_examples, "openable_object_tokens"),
        eai_action_object_constraint_tokens=_action_object_constraints(eai_examples),
        max_vlabench_entities=args.max_entities,
        max_vlabench_operations=args.max_operations,
    )
    return runtime, vlabench_splits


def run_planning_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up arguments for _prepare_eval
    args.vlabench_planning_dir = str(args.planning_dir)

    print(f"[eval-planning] Initializing joint runtime from {args.planning_dir} (device={device})...", flush=True)
    runtime, vla_splits = _prepare_eval(args, device)

    print(f"[eval-planning] Instantiating planner {args.planner_model}...", flush=True)
    planner = JointQwenVLPlanner.from_pretrained(
        eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=args.planner_model,
        load_in_4bit=args.load_in_4bit,
        decoder_hidden_size=args.planner_decoder_hidden_dim,
    )
    controller = _controller(args, device)

    print(f"[eval-planning] Loading joint checkpoint {checkpoint_path.name}...", flush=True)
    payload = load_joint_checkpoint(
        checkpoint_path,
        runtime=runtime,
        planner=planner,
        controller=controller,
        map_location=device,
    )

    planner.eval()

    # Select evaluation examples
    if args.split == "all":
        examples = load_planning_examples(args.planning_dir, limit=args.limit)
    else:
        examples = vla_splits.get(args.split, [])
        if args.limit:
            examples = examples[:args.limit]

    if not examples:
        raise ValueError(f"no planning examples found for split={args.split!r}")

    print(f"[eval-planning] Evaluating {len(examples)} examples from split '{args.split}'...", flush=True)

    with torch.no_grad():
        with runtime.domain_scope("vlabench"):
            metrics = evaluate_planner(
                planner.for_domain("vlabench"),
                examples,
                _runtime_adapter(runtime, "vlabench"),
            )

    planning_metrics = metrics.get("planning_metrics", {})
    overall = planning_metrics.get("overall", {})
    dimensions = planning_metrics.get("dimensions", {})
    macro = planning_metrics.get("Macro")
    legacy_macro = planning_metrics.get("legacy_Macro")

    report = {
        "checkpoint": str(checkpoint_path),
        "stage": payload.get("stage"),
        "epoch": payload.get("epoch"),
        "split": args.split,
        "examples_evaluated": len(examples),
        "overall": overall,
        "macro_score": macro,
        "legacy_macro_score": legacy_macro,
        "dimensions": dimensions,
        "detailed_metrics": metrics,
    }

    # Print formatted scorecard
    print("\n" + "=" * 70)
    print(f" VLABench Non-Interactive Planning Scorecard: {checkpoint_path.name}")
    print("=" * 70)
    print(f" Split: {args.split}  |  Examples: {len(examples)}  |  Stage: {payload.get('stage')} (Epoch {payload.get('epoch')})")
    print("-" * 70)
    print(f" Overall Progress Score:  {overall.get('score', 0.0):.2f} / 100.0")
    print(f" Legacy Progress Score:   {overall.get('legacy_score', 0.0):.2f} / 100.0")
    print(f" 6-Dimension Macro:       {macro if macro is not None else 'N/A'}")
    print("-" * 70)
    print(" Component Breakdown:")
    print(f"   • Skill Recall (SR):                   {overall.get('SR', 0.0):.2f}%")
    print(f"   • Parameter Recall (PR):               {overall.get('PR', 0.0):.2f}%")
    print(f"   • Skill-Parameter Recall (SPR):        {overall.get('SPR', 0.0):.2f}%")
    print(f"   • Dependency Graph Matching (PM):      {overall.get('PM', 0.0):.2f}%")
    print("-" * 70)
    print(" Dimension Breakdown:")
    for dim_name, dim_data in sorted(dimensions.items()):
        ex_count = dim_data.get("examples", 0)
        score = dim_data.get("score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        sr = dim_data.get("SR")
        pr = dim_data.get("PR")
        sr_str = f"{sr:.1f}%" if sr is not None else "N/A"
        pr_str = f"{pr:.1f}%" if pr is not None else "N/A"
        print(f"   {dim_name:<12} (N={ex_count:<3}): Score={score_str:<6} (SR={sr_str}, PR={pr_str})")
    print("=" * 70 + "\n")

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[eval-planning] Saved evaluation report to {out_path}", flush=True)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_eval_parser()
    args = parser.parse_args(argv)
    run_planning_evaluation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
