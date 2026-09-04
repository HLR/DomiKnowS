"""Resumable supervised and reward-driven training utilities."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

try:
    from .graph import PlanVocabulary, compile_planner_dfa, create_planner_generation_graph, plan_to_tokens
    from .models import controller_loss
    from .program import VLABenchHierarchicalReinforcementProgram, build_stage1_program
    from .reward import make_vlabench_reward_function, score_vlabench_plan
    from .world_graph import build_vlabench_world_graph
except ImportError:
    from graph import PlanVocabulary, compile_planner_dfa, create_planner_generation_graph, plan_to_tokens
    from models import controller_loss
    from program import VLABenchHierarchicalReinforcementProgram, build_stage1_program
    from reward import make_vlabench_reward_function, score_vlabench_plan
    from world_graph import build_vlabench_world_graph


@dataclass(frozen=True)
class PlannerConstraintRuntime:
    vocabulary: PlanVocabulary
    generation_graph: Any
    generation_bundle: Any
    dfa: Any
    world_bundle: Any
    max_operations: int
    max_tokens: int


STANDALONE_CHECKPOINT_VERSION = 2


def _planner_configuration(planner: torch.nn.Module) -> dict[str, Any]:
    return {
        "class": type(planner).__name__,
        "graph_decoder_version": getattr(planner, "graph_decoder_version", None),
        "decoder_hidden_size": getattr(planner, "decoder_hidden_size", None),
    }


def _planner_trainable_state(planner: torch.nn.Module) -> dict[str, Any]:
    names = {
        name for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    }
    state = planner.state_dict()
    return {name: state[name] for name in names}


def _load_planner_trainable_state(planner: torch.nn.Module, state: Mapping[str, Any]) -> None:
    required = {
        name for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(
            "standalone checkpoint is missing trainable planner state: "
            + ", ".join(missing[:5])
        )
    result = planner.load_state_dict(state, strict=False)
    unexpected = [
        name for name in result.unexpected_keys
        if not any(marker in name for marker in (".weight.absmax", ".weight.quant_map", ".weight.quant_state."))
    ]
    if unexpected:
        raise RuntimeError(
            "standalone checkpoint has unexpected planner state: "
            + ", ".join(unexpected[:5])
        )


def _cpu_rng_state(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().to(device="cpu", dtype=torch.uint8)


def _checkpoint_map_location(value: str | torch.device):
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError):
        return value
    return torch.device("cpu") if device.type == "cuda" else value


def build_constraint_runtime(
    world_bundle=None,
    *,
    max_entities: int = 64,
    max_operations: int = 8,
    name_prefix: str = "vlabench",
) -> PlannerConstraintRuntime:
    world_bundle = world_bundle or build_vlabench_world_graph(f"{name_prefix}_world")
    vocabulary = PlanVocabulary.from_world(world_bundle, max_entities=max_entities)
    graph, bundle = create_planner_generation_graph(
        world_bundle,
        vocabulary,
        max_operations=max_operations,
        graph_name=f"{name_prefix}_generation",
    )
    return PlannerConstraintRuntime(
        vocabulary=vocabulary,
        generation_graph=graph,
        generation_bundle=bundle,
        dfa=compile_planner_dfa(
            graph, bundle, world_bundle, vocabulary, max_operations=max_operations,
        ),
        world_bundle=world_bundle,
        max_operations=max_operations,
        max_tokens=max_operations * 5 + 1,
    )


def prepare_planner_program_examples(examples: Iterable[Any], runtime: PlannerConstraintRuntime) -> list[dict[str, Any]]:
    """Encode graph-owned labels and multimodal contexts for DomiKnowS programs."""
    prepared = []
    for example in examples:
        entity_table = tuple(_example_value(example, "entities", ()))
        tokens = plan_to_tokens(
            _example_value(example, "operation_sequence"),
            entity_table,
            world=runtime.world_bundle,
        )
        labels = [runtime.vocabulary.label_for_token(token) for token in tokens]
        if len(labels) > runtime.max_tokens:
            raise ValueError(f"plan requires {len(labels)} labels but max_tokens={runtime.max_tokens}")
        labels.extend([runtime.vocabulary.eos_label] * (runtime.max_tokens - len(labels)))
        image_paths = _example_value(example, "segmented_image_paths", ()) or _example_value(example, "image_paths", ())
        item = {
            "planner_context": {
                "instruction": _example_value(example, "instruction", ""),
                "images": tuple(image_paths),
                "entity_table": entity_table,
            },
            "token_positions": torch.arange(runtime.max_tokens, dtype=torch.long),
            "target_plan_labels": torch.tensor(labels, dtype=torch.long),
            "operation_sequence": _example_value(example, "operation_sequence"),
            "instruction": _example_value(example, "instruction", ""),
            "entities": entity_table,
        }
        item["reward_function"] = make_vlabench_reward_function(
            item,
            mode="dense",
            world_bundle=runtime.world_bundle,
        )
        prepared.append(item)
    return prepared


def create_stage1_program(runtime: PlannerConstraintRuntime, planner, *, device="cpu"):
    return build_stage1_program(runtime, planner, device=device)


def create_stage2_program(
    runtime: PlannerConstraintRuntime,
    planner,
    controller,
    *,
    planner_optimizer,
    controller_optimizer,
    env_factory,
    supervised_examples=(),
    controller_anchor_loader=None,
    device="cpu",
    **kwargs,
):
    return VLABenchHierarchicalReinforcementProgram(
        runtime,
        planner,
        controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        env_factory=env_factory,
        supervised_examples=supervised_examples,
        controller_anchor_loader=controller_anchor_loader,
        device=device,
        **kwargs,
    )
def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def train_controller_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device,
    grad_accumulation: int = 1,
    mixed_precision: bool = True,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    device = torch.device(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "pose_loss": 0.0, "gripper_loss": 0.0}
    steps = 0
    for steps, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        task_index = batch["task_index"].to(device)
        target = batch["actions"].to(device)
        with _autocast(device, mixed_precision):
            plan_context = batch.get("plan_context")
            inputs = (images, state, task_index)
            if plan_context is not None:
                inputs += (plan_context.to(device),)
            prediction = model(*inputs)
            loss, metrics = controller_loss(
                prediction,
                target,
                state=state,
                pose_step_scale=getattr(model, "pose_step_scale", None),
            )
            scaled_loss = loss / max(1, grad_accumulation)
        scaled_loss.backward()
        if steps % max(1, grad_accumulation) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += metrics[key]
    if steps and steps % max(1, grad_accumulation):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {key: value / max(1, steps) for key, value in totals.items()}


def train_controller_steps(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    device: str | torch.device,
    grad_accumulation: int = 1,
    mixed_precision: bool = True,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run bounded controller BC without traversing millions of windows."""

    requested = int(steps)
    if requested <= 0:
        return {"loss": 0.0, "pose_loss": 0.0, "gripper_loss": 0.0, "steps": 0}
    device = torch.device(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "pose_loss": 0.0, "gripper_loss": 0.0}
    iterator = iter(loader)
    try:
        from tqdm.auto import tqdm
        progress = tqdm(range(1, requested + 1), desc="Controller BC warm-up")
    except ImportError:
        progress = range(1, requested + 1)
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("controller warm-up loader is empty") from exc
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        task_index = batch["task_index"].to(device)
        target = batch["actions"].to(device)
        plan_context = batch.get("plan_context")
        inputs = (images, state, task_index)
        if plan_context is not None:
            inputs += (plan_context.to(device),)
        with _autocast(device, mixed_precision):
            loss, metrics = controller_loss(
                model(*inputs),
                target,
                state=state,
                pose_step_scale=getattr(model, "pose_step_scale", None),
            )
            scaled_loss = loss / max(1, grad_accumulation)
        scaled_loss.backward()
        if step % max(1, grad_accumulation) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += metrics[key]
        if hasattr(progress, "set_postfix") and (step == 1 or step % 100 == 0):
            progress.set_postfix(loss=f"{totals['loss'] / step:.4f}")
    if requested % max(1, grad_accumulation):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        **{key: value / requested for key, value in totals.items()},
        "steps": requested,
    }


@torch.no_grad()
def evaluate_controller(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: str | torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    device = torch.device(device)
    was_training = model.training
    model.eval()
    pose_error = gripper_correct = samples = 0.0
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            plan_context = batch.get("plan_context")
            inputs = (
                batch["images"].to(device),
                batch["state"].to(device),
                batch["task_index"].to(device),
            )
            if plan_context is not None:
                inputs += (plan_context.to(device),)
            prediction = model(*inputs)
            target = batch["actions"].to(device)
            pose_error += float(torch.abs(prediction[..., :-1] - target[..., :-1]).sum())
            gripper_correct += float(((torch.sigmoid(prediction[..., -1]) >= 0.5) == (target[..., -1] >= 0.5)).sum())
            samples += float(target[..., :-1].numel())
    finally:
        model.train(was_training)
    gripper_total = samples / 6.0 if samples else 0.0
    return {
        "pose_mae": pose_error / max(1.0, samples),
        "gripper_accuracy": gripper_correct / max(1.0, gripper_total),
    }


def _example_value(example: Any, name: str, default: Any = None) -> Any:
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def _planner_images(example: Any) -> list[Image.Image]:
    paths = _example_value(example, "segmented_image_paths", ()) or _example_value(example, "image_paths", ())
    return [Image.open(path).convert("RGB") for path in paths]


def train_planner_sft_epoch(
    planner: torch.nn.Module,
    examples: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    grad_accumulation: int = 4,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    planner.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    count = 0
    for count, example in enumerate(examples, start=1):
        images = _planner_images(example)
        try:
            loss = planner.supervised_loss(
                instruction=_example_value(example, "instruction", ""),
                images=images,
                entity_table=_example_value(example, "entities", ()),
                target_plan=_example_value(example, "operation_sequence"),
            )
            (loss / max(1, grad_accumulation)).backward()
            if count % max(1, grad_accumulation) == 0:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
        finally:
            for image in images:
                image.close()
    if count and count % max(1, grad_accumulation):
        torch.nn.utils.clip_grad_norm_(planner.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total / max(1, count), "examples": count}


def train_planner_reinforcement_epoch(
    planner: torch.nn.Module,
    examples: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    runtime: PlannerConstraintRuntime,
    *,
    num_samples: int = 4,
    estimator: str = "importance_weighted",
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Fine-tune discrete plans with DomiKnowS reward/loss utilities."""
    from domiknows.reinforcement.sampling import importance_weighted_loss, reinforce_loss

    if num_samples < 2 and estimator == "reinforce":
        raise ValueError("mean-baseline REINFORCE requires at least two samples")
    if estimator not in {"importance_weighted", "reinforce"}:
        raise ValueError("unsupported reinforcement estimator")
    planner.train()
    total_loss = total_reward = 0.0
    count = 0
    for count, example in enumerate(examples, start=1):
        item = example.as_reward_item() if hasattr(example, "as_reward_item") else dict(example)
        reward_fn = make_vlabench_reward_function(
            item,
            mode="dense",
            world_bundle=runtime.world_bundle,
        )
        images = _planner_images(example)
        try:
            logprobs = []
            rewards = []
            for _ in range(num_samples):
                output, logprob = planner.sample_with_logprob(
                    instruction=_example_value(example, "instruction", ""),
                    images=images,
                    entity_table=_example_value(example, "entities", ()),
                    dfa=runtime.dfa,
                    world=runtime.world_bundle,
                    max_steps=runtime.max_tokens,
                )
                logprobs.append(logprob.reshape(()))
                rewards.append(reward_fn(output, data_item=item).to(logprob.device).reshape(()))
            logprob_tensor = torch.stack(logprobs)
            reward_tensor = torch.stack(rewards)
            loss = (
                importance_weighted_loss(logprob_tensor, reward_tensor)
                if estimator == "importance_weighted"
                else reinforce_loss(logprob_tensor, reward_tensor, baseline="mean")
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach())
            total_reward += float(reward_tensor.mean().detach())
        finally:
            for image in images:
                image.close()
    return {
        "loss": total_loss / max(1, count),
        "reward": total_reward / max(1, count),
        "examples": count,
    }


@torch.no_grad()
def evaluate_planner(planner: torch.nn.Module, examples: Iterable[Any], runtime: PlannerConstraintRuntime) -> dict[str, float]:
    was_training = planner.training
    planner.eval()
    totals = {
        "reward": 0.0,
        "skill_match": 0.0,
        "entity_match": 0.0,
        "skill_with_entity_match": 0.0,
        "exact_graph_match": 0.0,
        "valid": 0.0,
    }
    count = 0
    try:
        for count, example in enumerate(examples, start=1):
            images = _planner_images(example)
            try:
                output = planner.generate_plan(
                    instruction=_example_value(example, "instruction", ""),
                    images=images,
                    entity_table=_example_value(example, "entities", ()),
                    dfa=runtime.dfa,
                    world=runtime.world_bundle,
                    max_steps=runtime.max_tokens,
                )
            finally:
                for image in images:
                    image.close()
            result = score_vlabench_plan(
                output,
                _example_value(example, "operation_sequence"),
                _example_value(example, "dependency", "Sequential"),
                entity_table=_example_value(example, "entities", ()),
                world_bundle=runtime.world_bundle,
            )
            totals["reward"] += result.total
            totals["valid"] += float(result.valid)
            for key in ("skill_match", "entity_match", "skill_with_entity_match", "exact_graph_match"):
                totals[key] += getattr(result, key)
    finally:
        planner.train(was_training)
    return {key: value / max(1, count) for key, value in totals.items()} | {"examples": count}


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metrics": dict(metrics or {}),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload


def save_joint_checkpoint(
    path: str | Path,
    *,
    planner: torch.nn.Module,
    controller: torch.nn.Module,
    planner_optimizer: torch.optim.Optimizer | None,
    controller_optimizer: torch.optim.Optimizer | None,
    runtime: PlannerConstraintRuntime,
    stage: str,
    epoch: int,
    metrics: Mapping[str, Any] | None = None,
    next_round: int | None = None,
) -> Path:
    """Atomically save the complete hierarchical training state."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "standalone_checkpoint_version": STANDALONE_CHECKPOINT_VERSION,
        "stage": str(stage),
        "epoch": int(epoch),
        "domain_checksum": runtime.world_bundle.domain_checksum,
        "vocabulary": asdict(runtime.vocabulary),
        "vocabulary_checksum": runtime.vocabulary.checksum,
        "planner_configuration": _planner_configuration(planner),
        "planner": _planner_trainable_state(planner),
        "controller": controller.state_dict(),
        "planner_optimizer": planner_optimizer.state_dict() if planner_optimizer is not None else None,
        "controller_optimizer": controller_optimizer.state_dict() if controller_optimizer is not None else None,
        "metrics": dict(metrics or {}),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if next_round is not None:
        if int(next_round) < 0:
            raise ValueError("next reinforcement round cannot be negative")
        payload["next_round"] = int(next_round)
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_joint_checkpoint(
    path: str | Path,
    *,
    planner: torch.nn.Module,
    controller: torch.nn.Module,
    runtime: PlannerConstraintRuntime,
    planner_optimizer: torch.optim.Optimizer | None = None,
    controller_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(
        Path(path),
        map_location=_checkpoint_map_location(map_location),
        weights_only=False,
    )
    if payload.get("standalone_checkpoint_version") != STANDALONE_CHECKPOINT_VERSION:
        raise ValueError(
            "checkpoint predates the efficient standalone graph decoder; restart Stage 1"
        )
    if payload.get("domain_checksum") != runtime.world_bundle.domain_checksum:
        raise ValueError("checkpoint domain checksum differs from the current world graph")
    if payload.get("vocabulary_checksum") != runtime.vocabulary.checksum:
        raise ValueError("checkpoint vocabulary checksum differs from the current graph vocabulary")
    if payload.get("planner_configuration") != _planner_configuration(planner):
        raise ValueError("checkpoint planner configuration differs from the current graph decoder")
    _load_planner_trainable_state(planner, payload["planner"])
    controller.load_state_dict(payload["controller"])
    if planner_optimizer is not None and payload.get("planner_optimizer") is not None:
        planner_optimizer.load_state_dict(payload["planner_optimizer"])
    if controller_optimizer is not None and payload.get("controller_optimizer") is not None:
        controller_optimizer.load_state_dict(payload["controller_optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(_cpu_rng_state(payload["torch_rng"]))
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        for device, state in enumerate(
            list(payload["cuda_rng"])[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(_cpu_rng_state(state), device=device)
    return payload
