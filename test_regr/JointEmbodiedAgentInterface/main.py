"""Canonical EAI/VLABench dynamic-activation two-stage command."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import torch

from domiknows.generation.dfa.vocabulary import TokenVocabulary
from test_regr.EmbodiedAgentInterface.dataset import EOS_TOKEN, load_eai_dataset
from test_regr.EmbodiedAgentInterface.reward import (
    evaluate_goal_satisfaction,
    make_eai_reward_function,
)
from test_regr.VLABenchAgentInterface.dataset import (
    deterministic_split,
    load_control_task_instructions,
    load_planning_examples,
)
from test_regr.VLABenchAgentInterface.main import _control_loaders, _controller
from test_regr.VLABenchAgentInterface.training import (
    evaluate_controller,
    evaluate_planner,
    prepare_planner_program_examples,
)
from test_regr.VLABenchAgentInterface.world_graph import PRIMITIVE_TASK_PATTERNS

from .checkpoint import load_joint_checkpoint, save_joint_checkpoint
from .models import DEFAULT_MODEL_ID, JointQwenVLPlanner
from .program import JointReinforcementProgram, JointSolverPOIProgram, _runtime_adapter
from .world_graph import build_joint_runtime, build_joint_world_graph


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_VLABENCH_PLANNING = PACKAGE_DIR.parent / "VLABenchAgentInterface" / "data" / "planning"
DEFAULT_VLABENCH_CONTROL = PACKAGE_DIR.parent / "VLABenchAgentInterface" / "data" / "control"
DEFAULT_OUTPUT = PACKAGE_DIR / "checkpoints"


def _ordered_union(examples, key):
    vocabulary = examples[0]["generation_vocab"] if examples else ()
    return tuple(
        token for token in vocabulary
        if any(token in item.get(key, ()) for item in examples)
    )


def _action_object_constraints(examples):
    vocabulary = examples[0]["generation_vocab"] if examples else ()
    values = {}
    for item in examples:
        for action, entity in item.get("action_object_constraint_pairs", ()):
            values.setdefault(action, set()).add(entity)
    return {
        action: tuple(token for token in vocabulary if token in entities)
        for action, entities in sorted(values.items())
        if action in vocabulary
    }


def _json(value):
    print(json.dumps(value, indent=2, sort_keys=True, default=str), flush=True)


def _status(message: str) -> None:
    print(f"[joint-training] {message}", file=sys.stderr, flush=True)


def _factory(value: str):
    module_name, separator, function_name = value.partition(":")
    if not separator:
        raise ValueError("environment factory must use module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


def stage1_selection_key(metrics):
    eai, vla = metrics["eai"], metrics["vlabench"]
    minimum = min(float(eai["goal_success"]), float(vla["exact_graph_match"]))
    mean = (float(eai["goal_success"]) + float(vla["exact_graph_match"])) / 2.0
    return (
        minimum,
        mean,
        float(eai["goal_recall"]),
        float(vla["valid"]),
        -float(metrics["validation_loss"]),
    )


def stage2_selection_key(metrics):
    eai, vla = metrics["eai"], metrics["vlabench"]
    minimum = min(float(eai["success"]), float(vla["success_rate"]))
    mean = (float(eai["success"]) + float(vla["success_rate"])) / 2.0
    efficiency = 1.0 / max(1.0, float(vla.get("steps", 1.0)))
    return (minimum, mean, float(eai.get("goal_recall", eai["reward"])), float(vla["return"]), efficiency)


def stage2_checkpoint_eligible(
    metrics,
    *,
    min_vlabench_success_rate: float,
    min_successful_tasks: int = 0,
    max_ik_truncation_rate: float = 1.0,
) -> bool:
    """Require success, task coverage, and executable controller behavior."""

    vlabench = metrics["vlabench"]
    return (
        float(vlabench["success_rate"]) >= float(min_vlabench_success_rate)
        and int(vlabench.get("successful_task_count", 0)) >= int(min_successful_tasks)
        and float(vlabench.get("ik_truncation_rate", 0.0))
        <= float(max_ik_truncation_rate)
    )


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


@torch.no_grad()
def _evaluate_eai(planner, runtime, examples, *, limit=32):
    view = planner.for_domain("eai")
    exact = success = recall = positive = 0.0
    selected = list(examples)[:limit]
    with runtime.domain_scope("eai"):
        for item in selected:
            labels, _ = view.sample_labels(
                {
                    "instruction": item.get("causal_prompt_text", item.get("text", "")),
                    "goal": item.get("tl_goal", ""),
                },
                runtime.dfa_for("eai", item),
                max_steps=runtime.max_eai_steps,
                deterministic=True,
            )
            gold = [int(value) for value in torch.as_tensor(item["target_action_labels"]).tolist()]
            eos = int(runtime.eai_vocabulary.eos_label)
            if eos in gold:
                gold = gold[: gold.index(eos) + 1]
            exact += float(labels == gold)
            result = evaluate_goal_satisfaction(
                labels,
                item,
                runtime.eai_vocabulary,
                world_bundle=runtime.world.eai,
            )
            success += float(result["is_success"])
            recall += float(result["recall"])
            reward = item["reward_function"](labels, data_item=item)
            positive += float(float(torch.as_tensor(reward).mean()) > 0.0)
    count = max(1, len(selected))
    return {
        "exact_sequence": exact / count,
        "goal_success": success / count,
        "goal_recall": recall / count,
        "positive_reward_rate": positive / count,
        "examples": len(selected),
    }


def _prepare(args, device):
    _status(f"loading EAI dataset name={args.eai_dataset} split={args.eai_split}")
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
    _status(f"loaded EAI examples={len(eai_examples)}")
    vocabulary = TokenVocabulary(eai_examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    world = build_joint_world_graph("joint_embodied_training")
    for item in eai_examples:
        item["reward_function"] = make_eai_reward_function(
            item,
            vocabulary=vocabulary,
            mode=args.eai_reward_mode,
            world_bundle=world.eai,
        )

    _status(f"loading VLABench planning data from {args.vlabench_planning_dir}")
    vlabench_examples = load_planning_examples(args.vlabench_planning_dir, limit=args.vlabench_limit)
    if not vlabench_examples:
        raise ValueError("VLABench planning dataset is empty")
    _status(f"loaded VLABench planning examples={len(vlabench_examples)}")
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
    prepared = {
        name: prepare_planner_program_examples(values, _runtime_adapter(runtime, "vlabench"))
        for name, values in vlabench_splits.items()
    }
    random.Random(args.seed).shuffle(eai_examples)
    cut = max(1, int(0.9 * len(eai_examples)))
    return runtime, eai_examples[:cut], eai_examples[cut:] or eai_examples[:1], vlabench_splits, prepared


def command_train_agent(args):
    if not args.two_stage:
        raise ValueError("train-agent requires --two-stage")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _status(f"starting joint two-stage training device={device}")
    runtime, eai_train, eai_valid, vla_splits, prepared = _prepare(args, device)
    _status(
        f"prepared splits: EAI train={len(eai_train)} validation={len(eai_valid)}; "
        + "VLABench "
        + " ".join(f"{name}={len(values)}" for name, values in vla_splits.items())
    )
    _status(f"loading planner backbone {args.planner_model}")
    planner = JointQwenVLPlanner.from_pretrained(
        eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=args.planner_model,
        adapter_path=args.resume_adapter,
        load_in_4bit=args.load_in_4bit,
        decoder_hidden_size=args.planner_decoder_hidden_dim,
    )
    _status("planner backbone and graph-token decoders ready")
    _status(f"loading controller vision encoder {args.vision_model if not args.tiny_vision else 'tiny'}")
    controller = _controller(args, device)
    _status("controller ready; building control-data indices")
    planner_optimizer = torch.optim.AdamW(
        (parameter for parameter in planner.parameters() if parameter.requires_grad),
        lr=args.planner_learning_rate,
    )
    controller_optimizer = torch.optim.AdamW(
        (parameter for parameter in controller.parameters() if parameter.requires_grad),
        lr=args.controller_learning_rate,
    )
    control_loaders = _control_loaders(args)
    control_task_instructions = load_control_task_instructions(args.control_source)
    _status(
        f"loaded VLABench language controller tasks={len(control_task_instructions)}"
    )
    _status("all model and data components are ready")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    resume_stage = None
    resume_payload = None
    start_stage1 = start_stage2 = 0
    cursor = 0
    if args.resume:
        payload = load_joint_checkpoint(
            args.resume,
            runtime=runtime,
            planner=planner,
            controller=controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            map_location=device,
        )
        if payload.get("controller_migration_required"):
            _status(
                "migrated legacy absolute-pose controller; the local action head "
                "will be rebuilt by controller warm-up"
            )
        if payload.get("controller_critic_migration_required"):
            _status(
                "migrated legacy unbounded controller critic; preserved the "
                "language-conditioned action policy"
            )
        resume_stage = payload["stage"]
        resume_payload = payload
        cursor = int(payload["round_robin_cursor"])
        if resume_stage == "stage1":
            start_stage1 = int(payload["epoch"]) + 1
        elif resume_stage == "controller_warmup":
            start_stage1 = args.stage1_epochs
        elif resume_stage == "stage2":
            start_stage2 = int(payload["epoch"]) + 1
        else:
            raise ValueError(f"unknown joint checkpoint stage {resume_stage!r}")

    best_stage1 = (
        Path(args.resume).resolve()
        if resume_stage in {"stage1", "controller_warmup"}
        else None
    )
    if resume_stage != "stage2":
        stage1 = JointSolverPOIProgram(
            runtime,
            planner,
            planner_optimizer=planner_optimizer,
            controller=controller,
            controller_optimizer=controller_optimizer,
            device=device,
        )
        stage1.round_robin_cursor = cursor
        best_key = (
            stage1_selection_key(resume_payload["metrics"])
            if resume_stage == "stage1" and resume_payload is not None else None
        )
        for epoch in range(start_stage1, args.stage1_epochs):
            _status(f"Stage 1 epoch {epoch + 1}/{args.stage1_epochs} training")
            train_metrics = stage1.train_alternating_epoch(
                eai_train,
                prepared["train"],
                controller_loader=control_loaders["train"],
                rounds=args.stage1_rounds_per_epoch,
            )
            _status(f"Stage 1 epoch {epoch + 1}/{args.stage1_epochs} EAI validation")
            eai_metrics = _evaluate_eai(planner, runtime, eai_valid, limit=args.validation_limit)
            _status(f"Stage 1 epoch {epoch + 1}/{args.stage1_epochs} VLABench validation")
            with runtime.domain_scope("vlabench"):
                vla_metrics = evaluate_planner(
                    planner.for_domain("vlabench"),
                    vla_splits["validation"],
                    _runtime_adapter(runtime, "vlabench"),
                )
            metrics = {
                "train": train_metrics,
                "eai": eai_metrics,
                "vlabench": vla_metrics,
                "validation_loss": (train_metrics["eai_loss"] + train_metrics["vlabench_loss"]) / 2.0,
            }
            path = save_joint_checkpoint(
                output / f"joint_stage1_epoch_{epoch:03d}.pt",
                runtime=runtime,
                planner=planner,
                controller=controller,
                planner_optimizer=planner_optimizer,
                controller_optimizer=controller_optimizer,
                stage="stage1",
                epoch=epoch,
                round_robin_cursor=stage1.round_robin_cursor,
                metrics=metrics,
            )
            key = stage1_selection_key(metrics)
            if best_key is None or key > best_key:
                best_key, best_stage1 = key, path
            _json({"stage": "stage1", "epoch": epoch, "checkpoint": path, "metrics": metrics})
        if best_stage1 is not None:
            payload = load_joint_checkpoint(
                best_stage1,
                runtime=runtime,
                planner=planner,
                controller=controller,
                planner_optimizer=planner_optimizer,
                controller_optimizer=controller_optimizer,
                map_location=device,
            )
            cursor = int(payload["round_robin_cursor"])
            eai_metrics = payload["metrics"]["eai"]
            if (
                eai_metrics["positive_reward_rate"] < args.stage1_min_positive_reward_rate
                or eai_metrics["goal_recall"] < args.stage1_min_goal_recall
                or eai_metrics["goal_success"] < args.stage1_min_goal_success
            ):
                _json({"stage": "stage2-skipped", "reason": "EAI exploration gate", "eai": eai_metrics})
                return
            if args.controller_warmup_steps > 0 and resume_stage != "controller_warmup":
                _status(
                    f"controller behavior-cloning warm-up steps={args.controller_warmup_steps}"
                )
                warmup_metrics = stage1.train_controller_warmup(
                    control_loaders["train"],
                    steps=args.controller_warmup_steps,
                )
                warmup_metrics["validation"] = evaluate_controller(
                    controller,
                    control_loaders["validation"],
                    device=device,
                    max_batches=args.validation_limit,
                )
                metrics = dict(payload.get("metrics", {}))
                metrics["controller_warmup"] = warmup_metrics
                best_stage1 = save_joint_checkpoint(
                    output / "joint_controller_warmup.pt",
                    runtime=runtime,
                    planner=planner,
                    controller=controller,
                    planner_optimizer=planner_optimizer,
                    controller_optimizer=controller_optimizer,
                    stage="controller_warmup",
                    epoch=0,
                    round_robin_cursor=cursor,
                    metrics=metrics,
                )
                _json({
                    "stage": "controller_warmup",
                    "checkpoint": best_stage1,
                    "metrics": warmup_metrics,
                })

    tasks = list(PRIMITIVE_TASK_PATTERNS) if args.task == "all" else [args.task]
    descriptors = [{"task": task, "env_kwargs": {}} for task in tasks]
    stage2 = JointReinforcementProgram(
        runtime,
        planner,
        controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        env_factory=_factory(args.env_factory),
        controller_task_instructions=control_task_instructions,
        eai_supervised_examples=eai_train,
        vlabench_supervised_examples=vla_splits["train"],
        controller_anchor_loader=control_loaders["train"],
        eai_num_samples=args.eai_samples,
        eai_supervised_weight=0.5,
        device=device,
        num_samples=args.vlabench_planner_samples,
        execute_horizon=4,
        max_steps=args.simulator_max_steps,
        supervised_weight=0.1,
        controller_bc_weight=0.05,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip=0.2,
        ppo_epochs=4,
        value_weight=0.5,
        entropy_weight=0.01,
        max_position_step=args.max_position_step,
        max_rotation_step=args.max_rotation_step,
        ik_tolerance=args.ik_tolerance,
        ik_max_steps=args.ik_max_steps,
    )
    stage2.round_robin_cursor = cursor
    if args.stage2_eval_rollouts_per_task > 0 and start_stage2 == 0:
        _status(
            "evaluating the Stage 1/controller-warm-up baseline on fixed-seed "
            f"VLABench rollouts={args.stage2_eval_rollouts_per_task} per task"
        )
        baseline = stage2.evaluate_rollouts(
            descriptors,
            rollouts_per_task=args.stage2_eval_rollouts_per_task,
            seed=args.seed + 100000,
        )
        _json({"stage": "stage2-baseline-evaluation", "vlabench": baseline})
    best_key = None
    best_path = None
    if resume_stage == "stage2" and resume_payload is not None:
        if stage2_checkpoint_eligible(
            resume_payload["metrics"],
            min_vlabench_success_rate=args.stage2_min_vlabench_success_rate,
            min_successful_tasks=args.stage2_min_successful_tasks,
            max_ik_truncation_rate=args.stage2_max_ik_truncation_rate,
        ):
            best_key = stage2_selection_key(resume_payload["metrics"])
            best_path = Path(args.resume).resolve()
        else:
            _status(
                "resume checkpoint is not a Stage 2 best candidate: "
                f"VLABench success={float(resume_payload['metrics']['vlabench']['success_rate']):.3f} "
                f"requires {args.stage2_min_vlabench_success_rate:.3f}"
            )
    for epoch in range(start_stage2, args.stage2_epochs):
        _status(f"Stage 2 epoch {epoch + 1}/{args.stage2_epochs} training")
        metrics = stage2.train_alternating_epoch(
            eai_train,
            descriptors,
            rounds=args.stage2_rounds_per_epoch,
            vlabench_rollouts_per_update=args.vlabench_rollouts,
        )
        if args.stage2_eval_rollouts_per_task > 0:
            metrics["eai_training"] = metrics["eai"]
            eai_evaluation = _evaluate_eai(
                planner,
                runtime,
                eai_valid,
                limit=args.validation_limit,
            )
            metrics["eai"] = {
                **metrics["eai_training"],
                "success": eai_evaluation["goal_success"],
                "goal_recall": eai_evaluation["goal_recall"],
                "evaluation": eai_evaluation,
            }
            metrics["vlabench_training"] = metrics["vlabench"]
            metrics["vlabench"] = stage2.evaluate_rollouts(
                descriptors,
                rollouts_per_task=args.stage2_eval_rollouts_per_task,
                seed=args.seed + 100000,
            )
        path = save_joint_checkpoint(
            output / f"joint_stage2_epoch_{epoch:03d}.pt",
            runtime=runtime,
            planner=planner,
            controller=controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            stage="stage2",
            epoch=epoch,
            round_robin_cursor=stage2.round_robin_cursor,
            metrics=metrics,
        )
        key = stage2_selection_key(metrics)
        eligible = stage2_checkpoint_eligible(
            metrics,
            min_vlabench_success_rate=args.stage2_min_vlabench_success_rate,
            min_successful_tasks=args.stage2_min_successful_tasks,
            max_ik_truncation_rate=args.stage2_max_ik_truncation_rate,
        )
        if eligible and (best_key is None or key > best_key):
            best_key, best_path = key, path
        _json({
            "stage": "stage2",
            "epoch": epoch,
            "checkpoint": path,
            "best_candidate_eligible": eligible,
            "minimum_vlabench_success_rate": args.stage2_min_vlabench_success_rate,
            "minimum_successful_vlabench_tasks": args.stage2_min_successful_tasks,
            "maximum_ik_truncation_rate": args.stage2_max_ik_truncation_rate,
            "metrics": metrics,
        })
    if best_path is not None:
        payload = load_joint_checkpoint(
            best_path,
            runtime=runtime,
            planner=planner,
            controller=controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            map_location=device,
        )
        selected = save_joint_checkpoint(
            output / "joint_stage2_best.pt",
            runtime=runtime,
            planner=planner,
            controller=controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            stage="stage2",
            epoch=int(payload["epoch"]),
            round_robin_cursor=int(payload["round_robin_cursor"]),
            metrics=payload.get("metrics", {}),
        )
        _json({"stage": "stage2-best", "checkpoint": selected, "source": best_path})
    else:
        _json({
            "stage": "stage2-best-skipped",
            "reason": "no epoch met all VLABench success, task-coverage, and IK-feasibility gates",
            "minimum_vlabench_success_rate": args.stage2_min_vlabench_success_rate,
            "minimum_successful_vlabench_tasks": args.stage2_min_successful_tasks,
            "maximum_ik_truncation_rate": args.stage2_max_ik_truncation_rate,
        })


def build_parser():
    parser = argparse.ArgumentParser(description="Unified EAI/VLABench dynamic-activation training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    agent = subparsers.add_parser("train-agent", help="run joint SolverPOI then joint reinforcement learning")
    agent.add_argument("--two-stage", action="store_true", required=True)
    agent.add_argument("--eai-dataset", default="all")
    agent.add_argument("--eai-split", default="train")
    agent.add_argument("--eai-data-path")
    agent.add_argument("--eai-limit", type=int)
    agent.add_argument("--eai-max-steps", type=int, default=60)
    agent.add_argument("--eai-reward-mode", choices=["binary", "dense"], default="dense")
    agent.add_argument("--vlabench-planning-dir", default=str(DEFAULT_VLABENCH_PLANNING))
    agent.add_argument("--control-source", default=str(DEFAULT_VLABENCH_CONTROL))
    agent.add_argument("--vlabench-limit", type=int)
    agent.add_argument("--output", default=str(DEFAULT_OUTPUT))
    agent.add_argument("--resume")
    agent.add_argument("--resume-adapter")
    agent.add_argument("--planner-model", default=DEFAULT_MODEL_ID)
    agent.add_argument("--planner-decoder-hidden-dim", type=int, default=512)
    agent.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    agent.add_argument("--device")
    agent.add_argument("--seed", type=int, default=42)
    agent.add_argument("--stage1-epochs", type=int, default=5)
    agent.add_argument("--stage2-epochs", type=int, default=3)
    agent.add_argument("--stage1-rounds-per-epoch", type=int)
    agent.add_argument("--stage2-rounds-per-epoch", type=int, default=10)
    agent.add_argument(
        "--stage2-min-vlabench-success-rate",
        type=_unit_interval,
        default=0.10,
        help="minimum rollout success required for an epoch to become joint_stage2_best.pt",
    )
    agent.add_argument(
        "--stage2-min-successful-tasks",
        type=int,
        default=3,
        help="minimum number of VLABench task families with at least one successful rollout",
    )
    agent.add_argument(
        "--stage2-max-ik-truncation-rate",
        type=_unit_interval,
        default=0.25,
        help="maximum fraction of VLABench rollouts terminated by unrecoverable IK",
    )
    agent.add_argument("--eai-samples", type=int, default=8)
    agent.add_argument("--vlabench-planner-samples", type=int, default=4)
    agent.add_argument("--vlabench-rollouts", type=int, default=8)
    agent.add_argument(
        "--stage2-eval-rollouts-per-task",
        type=int,
        default=1,
        help="fixed-seed held-out simulator rollouts per task before RL and after each Stage 2 epoch; use 0 to disable",
    )
    agent.add_argument("--planner-learning-rate", type=float, default=2e-5)
    agent.add_argument("--controller-learning-rate", type=float, default=3e-4)
    agent.add_argument("--controller-warmup-steps", type=int, default=20000)
    agent.add_argument("--stage1-min-positive-reward-rate", type=float, default=0.05)
    agent.add_argument("--stage1-min-goal-recall", type=float, default=0.05)
    agent.add_argument("--stage1-min-goal-success", type=float, default=0.0)
    agent.add_argument("--validation-limit", type=int, default=32)
    agent.add_argument("--task", choices=["all", *PRIMITIVE_TASK_PATTERNS], default="all")
    agent.add_argument("--env-factory", default="test_regr.VLABenchAgentInterface.environment:create_environment")
    agent.add_argument("--simulator-max-steps", type=int, default=400)
    agent.add_argument("--max-position-step", type=float, default=0.02)
    agent.add_argument("--max-rotation-step", type=float, default=0.10)
    agent.add_argument("--ik-tolerance", type=float, default=1e-3)
    agent.add_argument("--ik-max-steps", type=int, default=200)
    agent.add_argument("--max-entities", type=int, default=64)
    agent.add_argument("--max-operations", type=int, default=8)
    # Existing VLABench controller loader/model settings.
    agent.add_argument("--limit", type=int)
    agent.add_argument("--batch-size", type=int, default=8)
    agent.add_argument("--workers", type=int, default=0)
    agent.add_argument("--video-decoder-cache-size", type=int, default=8)
    agent.add_argument("--action-horizon", type=int, default=16)
    agent.add_argument("--max-views", type=int, default=4)
    agent.add_argument("--hidden-dim", type=int, default=256)
    agent.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    agent.add_argument("--tiny-vision", action="store_true")
    agent.add_argument("--vision-dim", type=int, default=64)
    agent.set_defaults(handler=command_train_agent)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
