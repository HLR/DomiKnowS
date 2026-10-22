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
from contextlib import ExitStack
from typing import Any, Mapping

import torch

from test_regr.VLABenchAgentInterface.dataset import (
    deterministic_split,
    load_planning_examples,
    OBSERVATION_MODES,
)
from test_regr.VLABenchAgentInterface.knowledge import add_knowledge_argument, apply_knowledge_argument
from test_regr.VLABenchAgentInterface.knowledge.holdout import add_holdout_arguments, apply_holdout_arguments, partition
from test_regr.VLABenchAgentInterface.knowledge.task_classifier import predict_tasks
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
        choices=("validation", "test", "train", "all", "heldout"),
        default="validation",
        help="dataset split to evaluate (default: validation)",
    )
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default="visual",
                        help="QA property selection: visual VLM predictions (default) or explicit metadata-assisted constraints")
    add_knowledge_argument(parser)
    add_holdout_arguments(parser)
    parser.add_argument("--no-dfa-min-steps", type=int, default=0,
                        help="with --no-dfa: withhold the end label for this many steps (0 = fully free)")
    parser.add_argument("--no-dfa", action="store_true",
                        help="decode without the grammar DFA or any constraint (baseline below --knowledge grammar)")
    parser.add_argument("--updated-data", "--updated-labels", action="store_true",
                        help="score against corrected density labels computed in memory; default uses labels stored in --planning-dir")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of planning examples to evaluate",
    )
    parser.add_argument(
        "--dimension",
        default=None,
        help="filter evaluation to a specific dimension (e.g. PhysLaw, Complex)",
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
    parser.add_argument("--max-entities", type=int, default=64)
    parser.add_argument("--max-operations", type=int, default=12)
    parser.add_argument("--ground-complex", action=argparse.BooleanOptionalAction, default=True,
                        help="enable input-grounded Complex goals; disable for the representation-only ablation")
    parser.add_argument("--decision-trace", type=Path, help="write read-only token logits, masks, and probabilities as JSONL")
    parser.add_argument("--input-probe", choices=("original", "permuted-entities", "no-images", "shuffled-images", "shuffled-instructions"), default="original",
                        help="diagnostic ablation; non-original results are not benchmark scores")
    parser.add_argument("--allow-planning-schema-migration", action="store_true", help="explicitly warm-start the audited optional-push/longer-budget schema; exact resume remains the default")
    # Controller arguments required to satisfy joint checkpoint deserialization
    parser.add_argument("--tiny-vision", action="store_true")
    parser.add_argument("--vision-dim", type=int, default=768)
    parser.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-views", type=int, default=4)
    return parser


def _evaluation_examples(args, root):
    examples = load_planning_examples(root,
        observation_mode=getattr(args, "observation_mode", "visual"),
        **({"ground_complex": False} if not getattr(args, "ground_complex", True) else {}))
    if getattr(args, "updated_data", False):
        from test_regr.VLABenchAgentInterface.repair_density_labels import updated_density_examples
        examples = updated_density_examples(examples)
    return examples


def _prepare_eval(args, device):
    from domiknows.generation.dfa.vocabulary import TokenVocabulary
    from test_regr.EmbodiedAgentInterface.dataset import EOS_TOKEN, load_eai_dataset
    from test_regr.EmbodiedAgentInterface.reward import make_eai_reward_function
    from .main import _action_object_constraints, _ordered_union
    from .world_graph import build_joint_runtime, build_joint_world_graph

    print(f"[eval-planning] Loading EAI vocabulary (dataset={args.eai_dataset}, split={args.eai_split})...", flush=True)
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

    print(f"[eval-planning] Loading VLABench planning data from {args.vlabench_planning_dir}...", flush=True)
    vlabench_examples = _evaluation_examples(args, args.vlabench_planning_dir)
    if not vlabench_examples:
        raise ValueError("VLABench planning dataset is empty")
    seen, held_out = partition(vlabench_examples, args)
    vlabench_splits = deterministic_split(seen, seed=args.seed)
    if getattr(args, "holdout_tasks", None):
        vlabench_splits["heldout"] = held_out

    print(f"[eval-planning] Compiling joint runtime DFAs (entities={args.max_entities}, operations={args.max_operations})...", flush=True)
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
    print("[eval-planning] Joint runtime and DFAs successfully initialized.", flush=True)
    return runtime, vlabench_splits


def run_planning_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    knowledge = apply_knowledge_argument(args)
    holdout = apply_holdout_arguments(args)

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
        allow_planning_schema_migration=getattr(args, "allow_planning_schema_migration", False),
        allow_knowledge_ablation=bool(holdout) and not bool(getattr(args, "declare_heldout_knowledge", False)),
    )

    planner.eval()
    planner.free_decoding = bool(getattr(args, "no_dfa", False))
    planner.free_decoding_min_steps = int(getattr(args, "no_dfa_min_steps", 0))

    # Select evaluation examples
    if args.split == "all":
        examples = _evaluation_examples(args, args.planning_dir)
    else:
        examples = vla_splits.get(args.split, [])

    if args.dimension:
        dim_str = args.dimension.lower()
        examples = [
            e for e in examples
            if (getattr(e, "dimension", None) and dim_str in str(e.dimension).lower())
            or (getattr(e, "task", None) and dim_str in str(e.task).lower())
        ]
    if args.limit:
        examples = examples[:args.limit]

    if not examples:
        raise ValueError(f"no planning examples found for split={args.split!r}")
    if knowledge == "learned":
        examples = predict_tasks(examples, args.task_classifier, observation_mode=getattr(args, "observation_mode", "visual"),
                                 ground_complex=getattr(args, "ground_complex", True))
    from test_regr.VLABenchAgentInterface.planning_augmentation import diagnostic_examples
    examples = diagnostic_examples(examples, getattr(args, "input_probe", "original"), args.seed)

    dim_msg = f" (dimension: {args.dimension})" if args.dimension else ""
    print(f"[eval-planning] Evaluating {len(examples)} examples from split '{args.split}'{dim_msg}...", flush=True)

    with ExitStack() as stack, torch.no_grad():
        trace = None
        if getattr(args, "decision_trace", None):
            args.decision_trace.parent.mkdir(parents=True, exist_ok=True)
            trace = stack.enter_context(args.decision_trace.open("w", encoding="utf-8"))
        with runtime.domain_scope("vlabench"):
            metrics = evaluate_planner(
                planner.for_domain("vlabench"),
                examples,
                _runtime_adapter(runtime, "vlabench"),
                **({"decision_trace": trace} if trace is not None else {}),
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
        "observation_mode": getattr(args, "observation_mode", "visual"),
        "knowledge": knowledge,
        "knowledge_manifest": getattr(runtime, "knowledge_manifest", None),
        "knowledge_provenance": payload.get("knowledge_provenance", {"status": "unrecorded_legacy"}),
        "decoding": "free" if getattr(args, "no_dfa", False) else "dfa",
        "free_decoding_min_steps": int(getattr(args, "no_dfa_min_steps", 0)),
        "holdout_tasks": list(holdout),
        "declare_heldout_knowledge": bool(getattr(args, "declare_heldout_knowledge", False)),
        "task_classifier": str(args.task_classifier) if knowledge == "learned" else None,
        "reference_labels": "density-material-button-v1" if getattr(args, "updated_data", False) else "as-stored",
        "updated_data": getattr(args, "updated_data", False),
        "ground_complex": getattr(args, "ground_complex", True),
        "input_probe": getattr(args, "input_probe", "original"),
        "examples_evaluated": len(examples),
        "generation_failures": metrics.get("generation_failures", 0),
        "semantic_search_limit_failures": metrics.get("semantic_search_limit_failures", 0),
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
    print(f" Reference labels: {report['reference_labels']}  |  Observation mode: {report['observation_mode']}")
    print(f" Generation failures: {metrics.get('generation_failures', 0)}"
          f" (semantic search limit: {metrics.get('semantic_search_limit_failures', 0)})")
    unknown_tasks = planning_metrics.get("unknown_task_count", 0)
    if unknown_tasks:
        print(f" WARNING: {unknown_tasks} examples have tasks outside the dimension mapping; they are in")
        print("          the overall score but in no dimension, so Macro is not comparable to full-coverage runs.")
    print("-" * 70)
    print(f" Overall Progress Score:  {overall.get('score', 0.0):.2f} / 100.0")
    print(f" Legacy Progress Score:   {overall.get('legacy_score', 0.0):.2f} / 100.0")
    upstream_view = planning_metrics.get("official_mapping", {})
    if upstream_view:
        print(f" Upstream-mapped tasks:   {upstream_view['examples']} examples; legacy_score by dimension "
              "(upstream total_score; internal split, not the standard protocol):")
        print("   " + "  ".join(f"{name}={'N/A' if value is None else f'{value:.1f}'}"
                                for name, value in upstream_view["legacy_total_score_by_dimension"].items()))
    print(f" 6-Dimension Macro:       {macro if macro is not None else 'N/A'}")
    print("-" * 70)
    print(" Component Breakdown:")
    print(f"   Ã¢â‚¬Â¢ Skill Recall (SR):                   {overall.get('SR', 0.0):.2f}%")
    print(f"   Ã¢â‚¬Â¢ Parameter Recall (PR):               {overall.get('PR', 0.0):.2f}%")
    print(f"   Ã¢â‚¬Â¢ Skill-Parameter Recall (SPR):        {overall.get('SPR', 0.0):.2f}%")
    print(f"   Ã¢â‚¬Â¢ Dependency Graph Matching (PM):      {overall.get('PM', 0.0):.2f}%")
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
