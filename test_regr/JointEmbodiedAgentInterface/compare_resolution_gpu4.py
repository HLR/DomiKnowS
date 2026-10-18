#!/usr/bin/env python3
"""Comparative evaluation of 224x224 vs 384x384 vision resolution on VLABench simulator."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from test_regr.JointEmbodiedAgentInterface.checkpoint import load_joint_checkpoint
from test_regr.JointEmbodiedAgentInterface.main import (
    _prepare,
    build_parser,
    load_control_task_instructions,
)
from test_regr.JointEmbodiedAgentInterface.models import JointQwenVLPlanner
from test_regr.JointEmbodiedAgentInterface.program import JointReinforcementProgram
from test_regr.VLABenchAgentInterface.main import _controller, _control_loaders
from test_regr.VLABenchAgentInterface.world_graph import PRIMITIVE_TASK_PATTERNS
from test_regr.VLABenchAgentInterface.environment import create_environment


def run_eval_for_config(
    base_args,
    vision_model: str,
    image_size: int,
    device: torch.device,
    tasks: list[str],
    rollouts_per_task: int,
    seed: int,
) -> dict:
    config_args = argparse.Namespace(**vars(base_args))
    config_args.vision_model = vision_model
    config_args.image_size = image_size

    print(f"\n{'='*70}", flush=True)
    print(f"EVALUATING: {vision_model} (native image_size={image_size})", flush=True)
    print(f"{'='*70}", flush=True)

    runtime, eai_train, eai_valid, vla_splits, prepared = _prepare(config_args, device)
    planner = JointQwenVLPlanner.from_pretrained(
        eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=config_args.planner_model,
        load_in_4bit=config_args.load_in_4bit,
        decoder_hidden_size=config_args.planner_decoder_hidden_dim,
    ).to(device)

    controller = _controller(config_args, device)
    planner_opt = torch.optim.AdamW(
        (p for p in planner.parameters() if p.requires_grad),
        lr=config_args.planner_learning_rate,
    )
    controller_opt = torch.optim.AdamW(
        (p for p in controller.parameters() if p.requires_grad),
        lr=config_args.controller_learning_rate,
    )

    print(f"Loading checkpoint {config_args.resume} into {image_size}x{image_size} model...", flush=True)
    payload = load_joint_checkpoint(
        config_args.resume,
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=planner_opt,
        controller_optimizer=controller_opt,
        map_location=device,
    )
    print(f"Checkpoint loaded! Stage: {payload.get('stage')}, Epoch: {payload.get('epoch')}", flush=True)

    control_loaders = _control_loaders(config_args)
    control_task_instructions = load_control_task_instructions(config_args.control_source)

    program = JointReinforcementProgram(
        runtime,
        planner,
        controller,
        planner_optimizer=planner_opt,
        controller_optimizer=controller_opt,
        env_factory=create_environment,
        controller_task_instructions=control_task_instructions,
        controller_camera_names=getattr(config_args, "controller_camera_names", None),
        eai_supervised_examples=eai_train,
        vlabench_supervised_examples=vla_splits["train"],
        controller_anchor_loader=control_loaders["train"],
        eai_num_samples=config_args.eai_samples,
        eai_supervised_weight=0.5,
        parameter_policy="freeze_shared",
        device=device,
        num_samples=config_args.vlabench_planner_samples,
        execute_horizon=4,
        max_steps=config_args.simulator_max_steps,
        supervised_weight=config_args.stage2_planner_anchor_weight,
        controller_bc_weight=config_args.stage2_controller_bc_weight,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip=0.2,
        ppo_epochs=config_args.ppo_epochs,
        ppo_max_log_ratio=config_args.ppo_max_log_ratio,
        ppo_target_action_log_ratio=config_args.ppo_target_kl,
        ppo_backtrack_factor=config_args.ppo_backtrack_factor,
        ppo_min_learning_rate=config_args.ppo_min_learning_rate,
        value_weight=0.5,
        entropy_weight=0.01,
        max_position_step=config_args.max_position_step,
        max_rotation_step=config_args.max_rotation_step,
        ik_tolerance=config_args.ik_tolerance,
        ik_max_steps=config_args.ik_max_steps,
        max_consecutive_ik_rejections=config_args.max_consecutive_ik_rejections,
        execution_assistance=config_args.execution_assistance,
        assistance_curriculum=getattr(config_args, "assistance_curriculum", "cosine"),
        assistance_start=getattr(config_args, "assistance_start", 1.0),
        assistance_end=getattr(config_args, "assistance_end", 0.0),
        feasibility_weight=getattr(config_args, "feasibility_weight", 0.15),
        dagger_weight=getattr(config_args, "dagger_weight", 0.20),
    )

    descriptors = [{"task": task, "env_kwargs": {}} for task in tasks]
    t0 = time.time()
    results = program.evaluate_rollouts(
        descriptors,
        rollouts_per_task=rollouts_per_task,
        seed=seed,
    )
    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed

    print(f"\nCompleted {len(tasks)} tasks x {rollouts_per_task} rollouts in {elapsed:.1f}s", flush=True)
    print(f"Overall Success Rate: {results.get('success_rate', 0.0)*100:.1f}%", flush=True)
    print(f"Positive Return Rate: {results.get('positive_return_rate', 0.0)*100:.1f}%", flush=True)
    print(f"Mean Return:          {results.get('return', 0.0):.4f}", flush=True)
    print(f"IK Truncation Rate:   {results.get('ik_truncation_rate', 0.0)*100:.1f}%", flush=True)
    print(f"Unassisted Successes: {results.get('unassisted_successes', 0)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare 224 vs 384 resolution on VLABench")
    parser.add_argument("--checkpoint", default="test_regr/JointEmbodiedAgentInterface/checkpoints/full_20260927_011044/joint_stage2_epoch_004.pt")
    parser.add_argument("--planner-model", default="/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", default=["select_mahjong", "select_poker", "select_book", "select_fruit", "add_condiment"])
    parser.add_argument("--rollouts-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output", default="test_regr/JointEmbodiedAgentInterface/results/resolution_comparison_gpu4.json")
    cli_args = parser.parse_args()

    raw_args = [
        "train-agent", "--two-stage",
        "--planner-model", cli_args.planner_model,
        "--device", cli_args.device,
        "--resume", cli_args.checkpoint,
        "--load-in-4bit",
        "--vlabench-planning-dir", "test_regr/VLABenchAgentInterface/data/planning",
        "--control-source", "test_regr/VLABenchAgentInterface/data/control",
        "--eai-dataset", "all",
        "--eai-split", "train",
        "--execution-assistance", "train-only",
        "--simulator-max-steps", "400",
        "--batch-size", "1",
        "--workers", "0",
        "--output", "test_regr/JointEmbodiedAgentInterface/results",
    ]
    base_args = build_parser().parse_args(raw_args)
    base_args.resume = cli_args.checkpoint

    device = torch.device(cli_args.device)
    tasks = cli_args.tasks
    if tasks == ["all"]:
        tasks = list(PRIMITIVE_TASK_PATTERNS)

    print(f"Target Tasks ({len(tasks)}): {tasks}")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Checkpoint: {cli_args.checkpoint}")

    res_224 = run_eval_for_config(
        base_args,
        vision_model="google/siglip-base-patch16-224",
        image_size=224,
        device=device,
        tasks=tasks,
        rollouts_per_task=cli_args.rollouts_per_task,
        seed=cli_args.seed,
    )

    res_384 = run_eval_for_config(
        base_args,
        vision_model="google/siglip-base-patch16-384",
        image_size=384,
        device=device,
        tasks=tasks,
        rollouts_per_task=cli_args.rollouts_per_task,
        seed=cli_args.seed,
    )

    print("\n" + "="*85)
    print("RESOLUTION COMPARISON REPORT: 224x224 vs 384x384 (GPU 4, UNASSISTED EVALUATION)")
    print("="*85)
    print(f"{'Task':<22} | {'224 Return':<11} {'384 Return':<11} | {'224 Pos%':<9} {'384 Pos%':<9} | {'224 Succ%':<10} {'384 Succ%':<10}")
    print("-" * 85)

    per_task_224 = res_224.get("per_task", {})
    per_task_384 = res_384.get("per_task", {})

    for task in tasks:
        t224 = per_task_224.get(task, {})
        t384 = per_task_384.get(task, {})
        r224 = t224.get("return", 0.0)
        r384 = t384.get("return", 0.0)
        pos224 = t224.get("positive_return_rate", 0.0) * 100
        pos384 = t384.get("positive_return_rate", 0.0) * 100
        s224 = t224.get("success_rate", 0.0) * 100
        s384 = t384.get("success_rate", 0.0) * 100
        print(f"{task:<22} | {r224:<11.4f} {r384:<11.4f} | {pos224:<8.1f}% {pos384:<8.1f}% | {s224:<9.1f}% {s384:<9.1f}%")

    print("-" * 85)
    print(f"{'OVERALL AVERAGE':<22} | {res_224.get('return', 0.0):<11.4f} {res_384.get('return', 0.0):<11.4f} | "
          f"{res_224.get('positive_return_rate', 0.0)*100:<8.1f}% {res_384.get('positive_return_rate', 0.0)*100:<8.1f}% | "
          f"{res_224.get('success_rate', 0.0)*100:<9.1f}% {res_384.get('success_rate', 0.0)*100:<9.1f}%")
    print(f"{'IK Truncation Rate':<22} | {res_224.get('ik_truncation_rate', 0.0)*100:<11.1f}% {res_384.get('ik_truncation_rate', 0.0)*100:<11.1f}%")
    print(f"{'Elapsed Seconds':<22} | {res_224.get('elapsed_seconds', 0.0):<11.1f}s {res_384.get('elapsed_seconds', 0.0):<11.1f}s")
    print("=" * 85)

    comparison_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": cli_args.checkpoint,
        "device": str(device),
        "tasks": tasks,
        "rollouts_per_task": cli_args.rollouts_per_task,
        "baseline_224": res_224,
        "upgraded_384": res_384,
    }

    out_path = Path(cli_args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nSaved comparison JSON to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
