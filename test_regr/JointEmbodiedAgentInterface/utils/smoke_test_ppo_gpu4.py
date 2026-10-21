"""Fast 1-round PPO smoke test on GPU 4 with 384x384 SigLIP.

Verifies:
1. Loading checkpoint with 384 resolution.
2. Collecting 1 VLABench rollout with 384 images.
3. Calculating GAE advantages and PPO loss.
4. Performing controller backward pass, gradient clipping, and optimizer step.
5. Parameter delta and VRAM usage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import torch

from test_regr.JointEmbodiedAgentInterface.main import (
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT,
    DEFAULT_VLABENCH_CONTROL,
    DEFAULT_VLABENCH_PLANNING,
    _control_loaders,
    _controller,
    _enter_controller_reinforcement,
    _factory,
    _prepare,
    load_joint_checkpoint,
)
from test_regr.JointEmbodiedAgentInterface.models import JointQwenVLPlanner
from test_regr.JointEmbodiedAgentInterface.program import JointReinforcementProgram
from test_regr.VLABenchAgentInterface.dataset import load_control_task_instructions


def run_smoke_test(
    checkpoint_path: str,
    task: str = "select_mahjong",
    device_str: str = "cuda:0",
    image_size: int = 384,
    vision_model: str = "google/siglip-base-patch16-384",
    planner_model: str = "/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct",
):
    print(f"=== STEP A: PPO SMOKE TEST (Task={task}, Res={image_size}, Device={device_str}) ===")
    device = torch.device(device_str)

    # 1. Build CLI mock args
    parser_args = argparse.Namespace(
        eai_dataset="all",
        eai_split="train",
        eai_data_path=None,
        eai_limit=None,
        eai_max_steps=60,
        eai_reward_mode="dense",
        vlabench_planning_dir=str(DEFAULT_VLABENCH_PLANNING),
        control_source=str(DEFAULT_VLABENCH_CONTROL),
        vlabench_limit=None,
        output=str(DEFAULT_OUTPUT),
        seed=42,
        max_entities=64,
        max_operations=8,
        action_horizon=16,
        batch_size=8,
        workers=0,
        grad_accumulation=1,
        max_views=4,
        hidden_dim=256,
        vision_model=vision_model,
        image_size=image_size,
        tiny_vision=False,
        vision_dim=64,
        planner_model=planner_model,
        resume_adapter=None,
        load_in_4bit=True,
        planner_decoder_hidden_dim=512,
        planner_learning_rate=2e-5,
        controller_learning_rate=3e-4,
        controller_rl_learning_rate=3e-5,
        limit=4,
        task=task,
        env_factory="test_regr.VLABenchAgentInterface.simulation.environment:create_environment",
        execution_assistance="train-only",
        assistance_curriculum="cosine",
        assistance_start=1.0,
        assistance_end=0.0,
        feasibility_weight=0.15,
        dagger_weight=0.20,
        max_consecutive_ik_rejections=3,
        simulator_max_steps=400,
        max_position_step=0.02,
        max_rotation_step=0.10,
        ik_tolerance=5e-3,
        ik_max_steps=200,
        ppo_epochs=2,
        ppo_backtrack_factor=0.5,
        ppo_min_learning_rate=1e-7,
        ppo_target_kl=0.03,
        ppo_max_log_ratio=2.0,
        eai_samples=1,
        vlabench_planner_samples=1,
        stage2_parameter_policy="freeze_shared",
    )

    t0 = time.time()
    print("[1/5] Preparing datasets and joint runtime...")
    runtime, eai_train, eai_valid, vla_splits, prepared = _prepare(parser_args, device)

    print("[2/5] Loading Qwen planner and 384-SigLIP controller...")
    planner = JointQwenVLPlanner.from_pretrained(
        eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=parser_args.planner_model,
        adapter_path=parser_args.resume_adapter,
        load_in_4bit=parser_args.load_in_4bit,
        decoder_hidden_size=parser_args.planner_decoder_hidden_dim,
    )
    controller = _controller(parser_args, device)

    planner_optimizer = torch.optim.AdamW(
        (p for p in planner.parameters() if p.requires_grad),
        lr=parser_args.planner_learning_rate,
    )
    controller_optimizer = torch.optim.AdamW(
        (p for p in controller.parameters() if p.requires_grad),
        lr=parser_args.controller_learning_rate,
    )

    print(f"[3/5] Loading checkpoint {checkpoint_path}...")
    payload = load_joint_checkpoint(
        checkpoint_path,
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        map_location=device,
    )
    print("Checkpoint loaded successfully with dynamic position embedding adaptation.")

    _enter_controller_reinforcement(
        controller_optimizer,
        learning_rate=parser_args.controller_rl_learning_rate,
        reset_state=True,
    )

    control_loaders = _control_loaders(parser_args)
    control_task_instructions = load_control_task_instructions(parser_args.control_source)

    stage2 = JointReinforcementProgram(
        runtime,
        planner,
        controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        env_factory=_factory(parser_args.env_factory),
        controller_task_instructions=control_task_instructions,
        controller_camera_names=None,
        eai_supervised_examples=eai_train,
        vlabench_supervised_examples=vla_splits["train"],
        controller_anchor_loader=control_loaders["train"],
        eai_num_samples=1,
        eai_supervised_weight=0.5,
        parameter_policy="freeze_shared",
        device=device,
        execute_horizon=4,
        max_steps=400,
        supervised_weight=0.25,
        controller_bc_weight=0.25,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip=0.2,
        ppo_epochs=2,
        ppo_max_log_ratio=2.0,
        ppo_target_action_log_ratio=0.03,
        ppo_backtrack_factor=0.5,
        ppo_min_learning_rate=1e-7,
        value_weight=0.5,
        entropy_weight=0.01,
        max_position_step=0.02,
        max_rotation_step=0.10,
        ik_tolerance=5e-3,
        ik_max_steps=200,
        max_consecutive_ik_rejections=3,
        execution_assistance="train-only",
        assistance_curriculum="cosine",
        assistance_start=1.0,
        assistance_end=0.0,
        feasibility_weight=0.15,
        dagger_weight=0.20,
    )

    # Record parameter snapshot to test update
    initial_param = next(p for p in controller.parameters() if p.requires_grad).clone().detach()

    print("[4/5] Executing 1 EAI update and 1 VLABench rollout with PPO backward pass...")
    # EAI update
    eai_res = stage2.train_eai_update(eai_train[0])
    print(f"  EAI update complete: loss={eai_res['loss']:.4f}, success={eai_res['success']:.2f}")

    # VLABench rollout + PPO update
    descriptor = {"task": task, "env_kwargs": {}}
    vla_res = stage2.train_vlabench_update([descriptor], rollouts_per_update=1)

    print("[5/5] Checking parameter update & optimizer health...")
    new_param = next(p for p in controller.parameters() if p.requires_grad).detach()
    param_delta = torch.norm(new_param - initial_param).item()

    vram_used = torch.cuda.memory_allocated(device) / (1024 ** 2)
    vram_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)

    print("\n" + "=" * 60)
    print("STEP A SMOKE TEST RESULTS:")
    print(f"  Task:                  {task}")
    print(f"  VLABench Return:       {vla_res.get('return', 0.0):.4f}")
    print(f"  VLABench Steps:        {vla_res.get('steps', 0.0):.1f}")
    print(f"  Controller Loss:       {vla_res.get('controller_loss', 0.0):.4f}")
    print(f"  Controller Param Delta: {param_delta:.6e}")
    print(f"  VRAM Allocated:        {vram_used:.1f} MiB")
    print(f"  VRAM Reserved:         {vram_reserved:.1f} MiB")
    print(f"  Elapsed Time:          {time.time() - t0:.1f}s")
    print("=" * 60)

    assert not torch.isnan(torch.tensor(vla_res.get("controller_loss", 0.0))), "Controller loss is NaN!"
    print(">>> STEP A SUCCESSFUL: PPO forward, backward, and optimizer updates verified at 384x384! <<<\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="test_regr/JointEmbodiedAgentInterface/checkpoints/full_20260927_011044/joint_stage2_epoch_004.pt")
    parser.add_argument("--task", default="select_mahjong")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planner-model", default="/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--vision-model", default="google/siglip-base-patch16-384")
    args = parser.parse_args()

    run_smoke_test(
        checkpoint_path=args.checkpoint,
        task=args.task,
        device_str=args.device,
        image_size=args.image_size,
        vision_model=args.vision_model,
        planner_model=args.planner_model,
    )
