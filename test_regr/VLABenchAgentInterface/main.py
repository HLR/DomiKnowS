"""Command-line entry point for VLABench download, training, and evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

try:
    from .agent import HierarchicalVLABenchAgent
    from .dataset import (
        LeRobotWindowDataset,
        deterministic_split,
        download_processed_datasets,
        load_hf_control_records,
        load_planning_examples,
    )
    from .graph import PlanVocabulary
    from .models import FrozenSigLIPEncoder, MultiViewController, QwenVLPlanner, TinyImageEncoder
    from .training import (
        build_constraint_runtime,
        create_stage1_program,
        create_stage2_program,
        evaluate_controller,
        evaluate_planner,
        load_checkpoint,
        load_joint_checkpoint,
        prepare_planner_program_examples,
        save_checkpoint,
        save_joint_checkpoint,
        train_controller_epoch,
    )
    from .world_graph import PRIMITIVE_TASK_PATTERNS, build_vlabench_world_graph, condition_index_for_task
except ImportError:
    from agent import HierarchicalVLABenchAgent
    from dataset import LeRobotWindowDataset, deterministic_split, download_processed_datasets, load_hf_control_records, load_planning_examples
    from graph import PlanVocabulary
    from models import FrozenSigLIPEncoder, MultiViewController, QwenVLPlanner, TinyImageEncoder
    from training import build_constraint_runtime, create_stage1_program, create_stage2_program, evaluate_controller, evaluate_planner, load_checkpoint, load_joint_checkpoint, prepare_planner_program_examples, save_checkpoint, save_joint_checkpoint, train_controller_epoch
    from world_graph import PRIMITIVE_TASK_PATTERNS, build_vlabench_world_graph, condition_index_for_task


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), flush=True)


def _status(message: str) -> None:
    print(f"[vlabench-data] {message}", file=sys.stderr, flush=True)


def _device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def command_download(args) -> None:
    planning, control = download_processed_datasets(
        args.planning_dir,
        args.control_dir,
        token=args.token,
        max_workers=args.max_workers,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    _json({"planning": planning, "control": control})


def command_inspect(args) -> None:
    result = {}
    if args.planning_dir:
        examples = load_planning_examples(args.planning_dir, limit=args.limit)
        result["planning"] = {
            "episodes": len(examples),
            "skills": sorted({operation["name"] for example in examples for operation in example.operation_sequence}),
            "first": examples[0].as_reward_item(),
            "rgb_images": sum(len(example.image_paths) for example in examples),
            "segmented_images": sum(len(example.segmented_image_paths) for example in examples),
        }
    if args.control_source:
        records = load_hf_control_records(args.control_source, task=args.task, streaming=False)
        first = records[0]
        result["control"] = {
            "frames": len(records),
            "columns": sorted(first),
            "state_shape": list(torch.as_tensor(first.get("state", first.get("observation.state"))).shape),
            "action_shape": list(torch.as_tensor(first.get("actions", first.get("action"))).shape),
        }
    _json(result)


def command_build_vocab(args) -> None:
    examples = load_planning_examples(args.planning_dir, limit=args.limit)
    world = build_vlabench_world_graph("vlabench_vocab_world")
    # Validate the dataset against the graph, then derive labels only from it.
    vocabulary = PlanVocabulary.from_plans(
        (example.operation_sequence for example in examples),
        world,
        max_entities=args.max_entities,
    )
    vocabulary.save(args.output)
    _json({"output": Path(args.output).resolve(), "tokens": len(vocabulary.tokens), "checksum": vocabulary.checksum})


def _control_loaders(args):
    tasks = list(PRIMITIVE_TASK_PATTERNS) if args.task == "all" else [args.task]
    split_parts = {"train": [], "validation": [], "test": []}
    for task_index, task in enumerate(tasks, start=1):
        _status(f"loading control task {task_index}/{len(tasks)}: {task}")
        records = load_hf_control_records(args.control_source, task=task, streaming=False)
        if args.limit is not None:
            per_task_limit = max(1, args.limit // len(tasks))
            records = records.select(range(min(per_task_limit, len(records)))) if hasattr(records, "select") else list(records)[:per_task_limit]
        video_root = args.control_source if Path(args.control_source).exists() else None
        windows = LeRobotWindowDataset(
            records,
            observation_horizon=2,
            action_horizon=args.action_horizon,
            video_root=video_root,
            condition_index=condition_index_for_task(task),
            video_decoder_cache_size=getattr(args, "video_decoder_cache_size", 8),
        )
        _status(
            f"indexed control task {task_index}/{len(tasks)}: {task} "
            f"records={len(records)} windows={len(windows)} episodes={len(windows.episodes)}"
        )
        episode_split = deterministic_split(sorted(windows.episodes), seed=42)
        for name, episode_ids in episode_split.items():
            selected = set(episode_ids)
            indices = [index for index, (episode, _offset) in enumerate(windows.index) if episode in selected]
            split_parts[name].append(Subset(windows, indices))
    loaders = {
        name: DataLoader(
            ConcatDataset(parts),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
        )
        for name, parts in split_parts.items()
    }
    _status(
        "control loaders ready: "
        + " ".join(f"{name}={len(loader.dataset)}" for name, loader in loaders.items())
    )
    return loaders


def _controller(args, device):
    encoder = TinyImageEncoder(args.vision_dim) if args.tiny_vision else FrozenSigLIPEncoder(args.vision_model)
    model = MultiViewController(
        encoder,
        action_horizon=args.action_horizon,
        hidden_dim=args.hidden_dim,
        max_views=args.max_views,
    ).to(device)
    return model


def command_train_controller(args) -> None:
    device = _device(args.device)
    loaders = _control_loaders(args)
    model = _controller(args, device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    start_epoch = 0
    if args.resume:
        start_epoch = int(load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location=device)["epoch"]) + 1
    output = Path(args.output).resolve()
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_controller_epoch(
            model, loaders["train"], optimizer, device=device,
            grad_accumulation=args.grad_accumulation,
        )
        validation_metrics = evaluate_controller(model, loaders["validation"], device=device)
        metrics = {"train": train_metrics, "validation": validation_metrics}
        checkpoint = output / f"controller_epoch_{epoch:03d}.pt"
        save_checkpoint(checkpoint, model=model, optimizer=optimizer, epoch=epoch, metrics=metrics)
        _json({"epoch": epoch, "checkpoint": checkpoint, **metrics})


def _load_planner(args, vocabulary):
    return QwenVLPlanner.from_pretrained(
        vocabulary,
        args.planner_model,
        use_lora=True,
        adapter_path=args.resume_adapter,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=True,
    )


def _save_planner_epoch(path: Path, planner, optimizer, epoch: int, metrics) -> None:
    path.mkdir(parents=True, exist_ok=True)
    planner.save_pretrained(str(path))
    torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(), "metrics": metrics}, path / "trainer_state.pt")


def command_train_planner(args) -> None:
    examples = load_planning_examples(args.planning_dir, limit=args.limit)
    splits = deterministic_split(examples, seed=42)
    world = build_vlabench_world_graph("vlabench_train_world")
    runtime = build_constraint_runtime(
        world, max_entities=args.max_entities, max_operations=args.max_operations, name_prefix="vlabench_train",
    )
    vocabulary = runtime.vocabulary
    output = Path(args.output).resolve()
    vocabulary.save(output / "vocab.json")
    planner = _load_planner(args, vocabulary)
    program = create_stage1_program(runtime, planner, device=_device(args.device))
    train_items = prepare_planner_program_examples(splits["train"], runtime)
    valid_items = prepare_planner_program_examples(splits["validation"], runtime)
    program.train(
        train_items,
        valid_set=valid_items,
        test_set=None,
        train_epoch_num=args.sft_epochs,
        Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate),
        test_every_epoch=False,
    )
    validation = evaluate_planner(planner, splits["validation"], runtime)
    path = output / "planner_stage1"
    planner.save_pretrained(str(path))
    _json({"stage": "solver-poi", "checkpoint": path, "validation": validation})


def command_train_agent(args) -> None:
    """Canonical graph-first supervised -> joint reinforcement pipeline."""
    if not args.two_stage:
        raise ValueError("train-agent requires --two-stage")
    device = _device(args.device)
    examples = load_planning_examples(args.planning_dir, limit=args.limit)
    splits = deterministic_split(examples, seed=42)
    world = build_vlabench_world_graph("vlabench_agent_world")
    runtime = build_constraint_runtime(
        world,
        max_entities=args.max_entities,
        max_operations=args.max_operations,
        name_prefix="vlabench_agent",
    )
    loaders = _control_loaders(args)
    planner = _load_planner(args, runtime.vocabulary)
    controller = _controller(args, device)
    planner_optimizer = torch.optim.AdamW(
        (parameter for parameter in planner.parameters() if parameter.requires_grad),
        lr=args.rl_learning_rate,
    )
    controller_optimizer = torch.optim.AdamW(
        (parameter for parameter in controller.parameters() if parameter.requires_grad),
        lr=args.controller_learning_rate,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime.vocabulary.save(output / "vocab.json")
    resume_stage = None
    start_rl_epoch = 0
    if args.resume:
        payload = load_joint_checkpoint(
            args.resume,
            planner=planner,
            controller=controller,
            runtime=runtime,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            map_location=device,
        )
        resume_stage = payload["stage"]
        if resume_stage == "reinforcement":
            start_rl_epoch = int(payload["epoch"]) + 1

    if resume_stage is None:
        stage1 = create_stage1_program(runtime, planner, device=device)
        stage1.train(
            prepare_planner_program_examples(splits["train"], runtime),
            valid_set=prepare_planner_program_examples(splits["validation"], runtime),
            test_set=None,
            train_epoch_num=args.sft_epochs,
            Optim=lambda params: torch.optim.AdamW(params, lr=args.learning_rate),
            test_every_epoch=False,
        )
        controller_metrics = {}
        for epoch in range(args.controller_epochs):
            controller_metrics = train_controller_epoch(
                controller,
                loaders["train"],
                controller_optimizer,
                device=device,
                grad_accumulation=args.grad_accumulation,
            )
        validation = {
            "planner": evaluate_planner(planner, splits["validation"], runtime),
            "controller": evaluate_controller(controller, loaders["validation"], device=device),
        }
        checkpoint = save_joint_checkpoint(
            output / "agent_stage1.pt",
            planner=planner,
            controller=controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            runtime=runtime,
            stage="supervised",
            epoch=max(args.sft_epochs, args.controller_epochs) - 1,
            metrics={"controller_train": controller_metrics, "validation": validation},
        )
        _json({"stage": "supervised", "checkpoint": checkpoint, "validation": validation})

    tasks = list(PRIMITIVE_TASK_PATTERNS) if args.task == "all" else [args.task]
    descriptors = [{"task": task, "env_kwargs": {}} for task in tasks]
    stage2 = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        env_factory=_factory(args.env_factory),
        supervised_examples=splits["train"],
        controller_anchor_loader=loaders["train"],
        device=device,
        num_samples=args.rl_num_samples,
        execute_horizon=args.execute_horizon,
        max_steps=args.max_steps,
        supervised_weight=args.rl_supervised_weight,
        controller_bc_weight=args.controller_bc_weight,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_clip=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        value_weight=args.value_weight,
        entropy_weight=args.entropy_weight,
    )
    for epoch in range(start_rl_epoch, args.rl_epochs):
        metrics = stage2.train_joint_epoch(descriptors, rollouts_per_update=args.rollouts_per_update)
        checkpoint = save_joint_checkpoint(
            output / f"agent_rl_epoch_{epoch:03d}.pt",
            planner=planner,
            controller=controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            runtime=runtime,
            stage="reinforcement",
            epoch=epoch,
            metrics=metrics,
        )
        _json({"stage": "reinforcement", "epoch": epoch, "checkpoint": checkpoint, "metrics": metrics})


def command_evaluate_planner(args) -> None:
    examples = load_planning_examples(args.planning_dir, limit=args.limit)
    splits = deterministic_split(examples, seed=42)
    world = build_vlabench_world_graph("vlabench_eval_world")
    vocabulary = PlanVocabulary.load(args.vocab) if args.vocab else PlanVocabulary.from_world(world, args.max_entities)
    runtime = build_constraint_runtime(world, max_entities=vocabulary.max_entities, max_operations=args.max_operations, name_prefix="vlabench_eval")
    if runtime.vocabulary.checksum != vocabulary.checksum:
        raise ValueError("loaded vocabulary differs from the graph-derived runtime vocabulary")
    planner = _load_planner(args, runtime.vocabulary)
    _json(evaluate_planner(planner, splits[args.split], runtime))


def _factory(value: str):
    module_name, separator, function_name = value.partition(":")
    if not separator:
        raise ValueError("environment factory must use module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


def command_rollout(args) -> None:
    device = _device(args.device)
    vocabulary = PlanVocabulary.load(args.vocab)
    world = build_vlabench_world_graph("vlabench_rollout_world")
    runtime = build_constraint_runtime(world, max_entities=vocabulary.max_entities, max_operations=args.max_operations, name_prefix="vlabench_rollout")
    if runtime.vocabulary.checksum != vocabulary.checksum:
        raise ValueError("loaded vocabulary differs from the graph-derived runtime vocabulary")
    planner = _load_planner(args, runtime.vocabulary)
    controller = _controller(args, device)
    if args.agent_checkpoint:
        load_joint_checkpoint(
            args.agent_checkpoint,
            planner=planner,
            controller=controller,
            runtime=runtime,
            map_location=device,
        )
    elif args.controller_checkpoint:
        load_checkpoint(args.controller_checkpoint, model=controller, map_location=device)
    else:
        raise ValueError("rollout requires --agent-checkpoint or --controller-checkpoint")
    env = _factory(args.env_factory)(**json.loads(args.env_kwargs))
    agent = HierarchicalVLABenchAgent(planner, controller, runtime, device=device)
    result = agent.rollout(env, args.instruction, max_steps=args.max_steps)
    _json({
        "reward": result.reward,
        "plans": len(result.plans),
        "actions": len(result.actions),
        "plan_validity": [decision.valid for decision in result.plans],
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DomiKnowS hierarchical VLABench agent")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download processed planning and LeRobot datasets")
    download.add_argument("--planning-dir", required=True)
    download.add_argument("--control-dir", required=True)
    download.add_argument("--token", default=None)
    download.add_argument(
        "--max-workers", type=int, default=1,
        help="parallel Hugging Face file downloads (default: 1 to avoid rate limits)",
    )
    download.add_argument("--retries", type=int, default=8, help="retries for HTTP 429 and transient failures")
    download.add_argument("--retry-delay", type=float, default=5.0, help="initial exponential backoff in seconds")
    download.set_defaults(handler=command_download)

    inspect = sub.add_parser("inspect", help="validate local dataset schemas")
    inspect.add_argument("--planning-dir")
    inspect.add_argument("--control-source")
    inspect.add_argument("--task")
    inspect.add_argument("--limit", type=int, default=10)
    inspect.set_defaults(handler=command_inspect)

    vocab = sub.add_parser("build-vocab")
    vocab.add_argument("--planning-dir", required=True)
    vocab.add_argument("--output", required=True)
    vocab.add_argument("--max-entities", type=int, default=64)
    vocab.add_argument("--limit", type=int)
    vocab.set_defaults(handler=command_build_vocab)

    def control_options(command):
        command.add_argument("--control-source", required=True)
        command.add_argument("--task", choices=["all", *PRIMITIVE_TASK_PATTERNS], default="all")
        command.add_argument("--output", required=True)
        command.add_argument("--device")
        command.add_argument("--limit", type=int)
        command.add_argument("--epochs", type=int, default=20)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--video-decoder-cache-size", type=int, default=8)
        command.add_argument("--learning-rate", type=float, default=3e-4)
        command.add_argument("--grad-accumulation", type=int, default=4)
        command.add_argument("--action-horizon", type=int, default=16)
        command.add_argument("--max-views", type=int, default=4)
        command.add_argument("--hidden-dim", type=int, default=256)
        command.add_argument("--vision-model", default="google/siglip-base-patch16-224")
        command.add_argument("--tiny-vision", action="store_true")
        command.add_argument("--vision-dim", type=int, default=64)
        command.add_argument("--resume")

    controller = sub.add_parser("train-controller")
    control_options(controller)
    controller.set_defaults(handler=command_train_controller)

    def planner_options(command):
        command.add_argument("--planning-dir", required=True)
        command.add_argument("--planner-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
        command.add_argument("--resume-adapter")
        command.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--max-entities", type=int, default=64)
        command.add_argument("--max-operations", type=int, default=8)
        command.add_argument("--limit", type=int)
        command.add_argument("--device")

    planner = sub.add_parser("train-planner")
    planner_options(planner)
    planner.add_argument("--output", required=True)
    planner.add_argument("--sft-epochs", type=int, default=3)
    planner.add_argument("--learning-rate", type=float, default=2e-4)
    planner.add_argument("--grad-accumulation", type=int, default=8)
    planner.set_defaults(handler=command_train_planner)

    agent = sub.add_parser("train-agent", help="graph-first SolverPOI -> joint ReinforcementProgram/PPO training")
    agent.add_argument("--two-stage", action="store_true", required=True)
    agent.add_argument("--planning-dir", required=True)
    agent.add_argument("--control-source", required=True)
    agent.add_argument("--output", required=True)
    agent.add_argument("--task", choices=["all", *PRIMITIVE_TASK_PATTERNS], default="all")
    agent.add_argument("--env-factory", default="test_regr.VLABenchAgentInterface.environment:create_environment")
    agent.add_argument("--planner-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    agent.add_argument("--resume-adapter")
    agent.add_argument("--resume", help="joint .pt checkpoint")
    agent.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    agent.add_argument("--device")
    agent.add_argument("--limit", type=int)
    agent.add_argument("--max-entities", type=int, default=64)
    agent.add_argument("--max-operations", type=int, default=8)
    agent.add_argument("--sft-epochs", type=int, default=3)
    agent.add_argument("--controller-epochs", type=int, default=20)
    agent.add_argument("--rl-epochs", type=int, default=1)
    agent.add_argument("--learning-rate", type=float, default=2e-4)
    agent.add_argument("--rl-learning-rate", type=float, default=2e-5)
    agent.add_argument("--controller-learning-rate", type=float, default=3e-4)
    agent.add_argument("--batch-size", type=int, default=8)
    agent.add_argument("--workers", type=int, default=0)
    agent.add_argument("--video-decoder-cache-size", type=int, default=8)
    agent.add_argument("--grad-accumulation", type=int, default=4)
    agent.add_argument("--action-horizon", type=int, default=16)
    agent.add_argument("--execute-horizon", type=int, default=4)
    agent.add_argument("--max-steps", type=int, default=400)
    agent.add_argument("--max-views", type=int, default=4)
    agent.add_argument("--hidden-dim", type=int, default=256)
    agent.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    agent.add_argument("--tiny-vision", action="store_true")
    agent.add_argument("--vision-dim", type=int, default=64)
    agent.add_argument("--rl-num-samples", type=int, default=4)
    agent.add_argument("--rollouts-per-update", type=int, default=8)
    agent.add_argument("--rl-supervised-weight", type=float, default=0.1)
    agent.add_argument("--controller-bc-weight", type=float, default=0.05)
    agent.add_argument("--gamma", type=float, default=0.99)
    agent.add_argument("--gae-lambda", type=float, default=0.95)
    agent.add_argument("--ppo-clip", type=float, default=0.2)
    agent.add_argument("--ppo-epochs", type=int, default=4)
    agent.add_argument("--value-weight", type=float, default=0.5)
    agent.add_argument("--entropy-weight", type=float, default=0.01)
    agent.set_defaults(handler=command_train_agent)

    evaluate = sub.add_parser("evaluate-planner")
    planner_options(evaluate)
    evaluate.add_argument("--vocab")
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="test")
    evaluate.set_defaults(handler=command_evaluate_planner)

    rollout = sub.add_parser("rollout")
    rollout.add_argument("--env-factory", required=True)
    rollout.add_argument("--env-kwargs", default="{}")
    rollout.add_argument("--instruction", required=True)
    rollout.add_argument("--vocab", required=True)
    rollout.add_argument("--controller-checkpoint")
    rollout.add_argument("--agent-checkpoint")
    rollout.add_argument("--max-steps", type=int, default=400)
    rollout.add_argument("--max-operations", type=int, default=8)
    rollout.add_argument("--planner-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    rollout.add_argument("--resume-adapter")
    rollout.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    rollout.add_argument("--device")
    rollout.add_argument("--action-horizon", type=int, default=16)
    rollout.add_argument("--max-views", type=int, default=4)
    rollout.add_argument("--hidden-dim", type=int, default=256)
    rollout.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    rollout.add_argument("--tiny-vision", action="store_true")
    rollout.add_argument("--vision-dim", type=int, default=64)
    rollout.set_defaults(handler=command_rollout)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
