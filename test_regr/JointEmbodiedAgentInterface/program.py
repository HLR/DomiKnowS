"""Alternating supervised and reinforcement programs over the joint graph."""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch

from domiknows.program import SolverPOIProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.reinforcement.reinforcement_program import ReinforcementProgram

from test_regr.VLABenchAgentInterface.models import controller_loss
from test_regr.VLABenchAgentInterface.program import (
    EOSMaskedCrossEntropyLoss,
    VLABenchHierarchicalReinforcementProgram,
    attach_planner_sensors,
)

from .models import JointQwenVLPlanner
from .world_graph import JointDomainRuntime


@dataclass(frozen=True)
class DomainUpdate:
    domain: str
    planner_loss: float
    controller_loss: float = 0.0
    reward: float | None = None


def _emit_progress(message: str) -> None:
    """Write progress that remains visible when stdout/stderr are redirected."""

    print(f"[joint-training] {message}", file=sys.stderr, flush=True)


class _TrainingProgress:
    def __init__(self, label: str, total: int, *, interval_seconds: float = 30.0):
        self.label = label
        self.total = max(0, int(total))
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.started = self.last_report = time.monotonic()
        _emit_progress(f"{self.label}: 0/{self.total} started")

    def update(self, completed: int, **metrics: float) -> None:
        now = time.monotonic()
        completed = int(completed)
        if completed < self.total and completed != 1 and now - self.last_report < self.interval_seconds:
            return
        elapsed = max(now - self.started, 1e-9)
        rate = completed / elapsed
        remaining = max(0, self.total - completed)
        eta = remaining / rate if rate > 0 else float("inf")
        values = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        suffix = f" {values}" if values else ""
        _emit_progress(
            f"{self.label}: {completed}/{self.total} "
            f"elapsed={elapsed / 60.0:.1f}m eta={eta / 60.0:.1f}m{suffix}"
        )
        self.last_report = now


def _context(item: Mapping[str, Any], domain: str) -> Mapping[str, Any]:
    value = item.get("planner_context")
    if isinstance(value, Mapping):
        return value
    if domain == "eai":
        return {
            "instruction": item.get("causal_prompt_text") or item.get("instruction_text") or item.get("instruction", ""),
            "goal": item.get("goal") or item.get("tl_goal", ""),
        }
    return {
        "instruction": item.get("instruction", ""),
        "images": item.get("images", item.get("segmented_image_paths", ())),
        "entity_table": item.get("entities", ()),
    }


def _labels(item: Mapping[str, Any], domain: str) -> torch.Tensor:
    key = "target_action_labels" if domain == "eai" else "target_plan_labels"
    if key not in item:
        raise KeyError(f"{domain} supervised item is missing {key!r}")
    return torch.as_tensor(item[key], dtype=torch.long)


def _cycle(values: Sequence[Any]):
    while True:
        order = list(values)
        random.shuffle(order)
        yield from order


def _runtime_adapter(runtime: JointDomainRuntime, domain: str):
    if domain == "eai":
        return SimpleNamespace(
            vocabulary=runtime.eai_vocabulary,
            generation_graph=runtime.root,
            generation_bundle=runtime.eai_generation,
            dfa=runtime.eai_dfa,
            world_bundle=runtime.world.eai,
            max_operations=runtime.max_eai_steps,
            max_tokens=runtime.max_eai_steps,
        )
    return SimpleNamespace(
        vocabulary=runtime.vlabench_vocabulary,
        generation_graph=runtime.root,
        generation_bundle=runtime.vlabench_generation,
        dfa=runtime.vlabench_dfa,
        world_bundle=runtime.world.vlabench,
        max_operations=runtime.max_vlabench_operations,
        max_tokens=runtime.max_vlabench_operations * 5 + 1,
    )


class _ActiveDomainEOSLoss(torch.nn.Module):
    def __init__(self, runtime: JointDomainRuntime):
        super().__init__()
        self.runtime = runtime

    def forward(self, input, target, *args, **kwargs):
        domain = self.runtime.active_domain or "eai"
        vocabulary = (
            self.runtime.eai_vocabulary if domain == "eai"
            else self.runtime.vlabench_vocabulary
        )
        return EOSMaskedCrossEntropyLoss(vocabulary.eos_label)(input, target, *args, **kwargs)


class JointSolverPOIProgram(SolverPOIProgram):
    """One SolverPOI lifecycle with both sensor branches attached once."""

    def __init__(
        self,
        runtime: JointDomainRuntime,
        planner: JointQwenVLPlanner,
        *,
        planner_optimizer: torch.optim.Optimizer,
        controller: torch.nn.Module | None = None,
        controller_optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ):
        eai_view = planner.for_domain("eai")
        vlabench_view = planner.for_domain("vlabench")
        poi = []
        for domain, view in (("eai", eai_view), ("vlabench", vlabench_view)):
            poi.extend(attach_planner_sensors(_runtime_adapter(runtime, domain), view, device=device))
        # Preserve order while removing repeated relation/property objects.
        poi = list(dict.fromkeys(poi))
        super().__init__(
            runtime.root,
            poi=poi,
            inferTypes=["local/argmax"],
            loss=MacroAverageTracker(_ActiveDomainEOSLoss(runtime)),
            device=device,
            metric={},
        )
        self.runtime = runtime
        self.planner_head = planner
        self.eai_planner = eai_view
        self.vlabench_planner = vlabench_view
        self.planner_optimizer = planner_optimizer
        self.controller = controller
        self.controller_optimizer = controller_optimizer
        self.round_robin_cursor = 0

    def _planner_step(self, domain: str, item: Mapping[str, Any]) -> float:
        # Validation routes through the same shared object and may leave it in
        # eval mode.  Prefix-level activation checkpointing is enabled only
        # during training, so restore the lifecycle explicitly before every
        # optimizer-bearing planner turn.
        self.planner_head.train()
        with self.runtime.domain_scope(domain):
            loss = self.planner_head.supervised_loss(
                domain,
                context=_context(item, domain),
                target_labels=_labels(item, domain).to(self.planner_head.device),
            )
            self.planner_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.planner_head.parameters(), 1.0)
            self.planner_optimizer.step()
            return float(loss.detach())

    def _controller_step(self, batch: Mapping[str, torch.Tensor] | None) -> float:
        if batch is None or self.controller is None or self.controller_optimizer is None:
            return 0.0
        with self.runtime.domain_scope("vlabench"):
            self.controller.train()
            device = next(self.controller.parameters()).device
            prediction = self.controller(
                batch["images"].to(device),
                batch["state"].to(device),
                batch["task_index"].to(device),
            )
            loss = controller_loss(prediction, batch["actions"].to(device))[0]
            self.controller_optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                raise ValueError("controller behavior-cloning loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.controller.parameters(),
                1.0,
                error_if_nonfinite=True,
            )
            self.controller_optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in self.controller.parameters()
                if parameter.requires_grad
            ):
                raise RuntimeError("controller behavior-cloning update produced non-finite parameters")
            return float(loss.detach())

    def train_controller_warmup(
        self,
        controller_loader: Iterable[Mapping[str, torch.Tensor]],
        *,
        steps: int,
    ) -> dict[str, float]:
        """Run controller-only BC before online reinforcement learning."""

        steps = int(steps)
        if steps < 0:
            raise ValueError("controller warm-up steps cannot be negative")
        if steps == 0:
            return {"loss": 0.0, "steps": 0}
        stream = iter(controller_loader)
        total = 0.0
        progress = _TrainingProgress("Controller BC warm-up", steps)
        for index in range(steps):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(controller_loader)
                batch = next(stream)
            total += self._controller_step(batch)
            progress.update(index + 1, controller_bc_loss=total / (index + 1))
        return {"loss": total / steps, "steps": steps}

    def train_round(
        self,
        eai_item: Mapping[str, Any],
        vlabench_item: Mapping[str, Any],
        *,
        controller_batch: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[DomainUpdate, DomainUpdate]:
        """Run exactly one EAI turn followed by one equally weighted VLA turn."""
        eai_loss = self._planner_step("eai", eai_item)
        vlabench_loss = self._planner_step("vlabench", vlabench_item)
        bc_loss = self._controller_step(controller_batch)
        self.round_robin_cursor += 1
        return (
            DomainUpdate("eai", eai_loss),
            DomainUpdate("vlabench", vlabench_loss, controller_loss=bc_loss),
        )

    def train_alternating_epoch(
        self,
        eai_examples: Sequence[Mapping[str, Any]],
        vlabench_examples: Sequence[Mapping[str, Any]],
        *,
        controller_loader: Iterable[Mapping[str, torch.Tensor]] | None = None,
        rounds: int | None = None,
    ) -> dict[str, float]:
        if not eai_examples or not vlabench_examples:
            raise ValueError("joint Stage 1 requires non-empty EAI and VLABench loaders")
        rounds = int(rounds) if rounds is not None else min(len(eai_examples), len(vlabench_examples))
        eai_stream, vlabench_stream = _cycle(eai_examples), _cycle(vlabench_examples)
        controller_stream = iter(controller_loader) if controller_loader is not None else None
        totals = {"eai_loss": 0.0, "vlabench_loss": 0.0, "controller_bc_loss": 0.0}
        progress = _TrainingProgress("Stage 1 rounds", rounds)
        for round_index in range(rounds):
            batch = None
            if controller_stream is not None:
                try:
                    batch = next(controller_stream)
                except StopIteration:
                    controller_stream = iter(controller_loader)
                    batch = next(controller_stream)
            eai, vlabench = self.train_round(next(eai_stream), next(vlabench_stream), controller_batch=batch)
            totals["eai_loss"] += eai.planner_loss
            totals["vlabench_loss"] += vlabench.planner_loss
            totals["controller_bc_loss"] += vlabench.controller_loss
            completed = round_index + 1
            progress.update(
                completed,
                eai_loss=totals["eai_loss"] / completed,
                vlabench_loss=totals["vlabench_loss"] / completed,
                controller_bc_loss=totals["controller_bc_loss"] / completed,
            )
        return {key: value / max(1, rounds) for key, value in totals.items()} | {"rounds": rounds}


class JointReinforcementProgram(VLABenchHierarchicalReinforcementProgram):
    """Alternating EAI REINFORCE and VLABench planner/PPO optimization.

    EAI and VLABench rewards are intentionally stored and reported separately.
    The only coupling is that their alternating gradients update the same LoRA
    backbone; no reward scalar crosses the domain boundary.
    """

    def __init__(
        self,
        runtime: JointDomainRuntime,
        planner: JointQwenVLPlanner,
        controller: torch.nn.Module,
        *,
        planner_optimizer: torch.optim.Optimizer,
        controller_optimizer: torch.optim.Optimizer,
        env_factory,
        eai_supervised_examples: Sequence[Mapping[str, Any]] = (),
        vlabench_supervised_examples: Sequence[Any] = (),
        controller_anchor_loader=None,
        eai_num_samples: int = 8,
        eai_supervised_weight: float = 0.5,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        self.joint_runtime = runtime
        self.joint_planner = planner
        self.eai_planner = planner.for_domain("eai")
        self.vlabench_planner = planner.for_domain("vlabench")
        self.eai_supervised_examples = tuple(eai_supervised_examples)
        self.eai_num_samples = int(eai_num_samples)
        self.eai_supervised_weight = float(eai_supervised_weight)
        self.round_robin_cursor = 0
        kwargs.setdefault("progress_callback", _emit_progress)
        super().__init__(
            _runtime_adapter(runtime, "vlabench"),
            self.vlabench_planner,
            controller,
            planner_optimizer=planner_optimizer,
            controller_optimizer=controller_optimizer,
            env_factory=env_factory,
            supervised_examples=vlabench_supervised_examples,
            controller_anchor_loader=controller_anchor_loader,
            device=device,
            **kwargs,
        )
        # Public identity used by checkpointing and acceptance tests.
        self.planner_head = planner

    def collect_episode(self, descriptor):
        with self.joint_runtime.domain_scope("vlabench"):
            # The parent implementation invokes methods through planner_head;
            # temporarily expose its non-owning VLABench view.
            planner = self.planner_head
            self.planner_head = self.vlabench_planner
            try:
                return super().collect_episode(descriptor)
            finally:
                self.planner_head = planner

    def _planner_anchor(self):
        with self.joint_runtime.domain_scope("vlabench"):
            planner = self.planner_head
            self.planner_head = self.vlabench_planner
            try:
                return super()._planner_anchor()
            finally:
                self.planner_head = planner

    def _eai_anchor(self) -> torch.Tensor:
        if not self.eai_supervised_examples or not self.eai_supervised_weight:
            return torch.zeros((), device=self.joint_planner.device)
        item = random.choice(self.eai_supervised_examples)
        return self.joint_planner.supervised_loss(
            "eai",
            context=_context(item, "eai"),
            target_labels=_labels(item, "eai").to(self.joint_planner.device),
        )

    def train_eai_update(self, item: Mapping[str, Any]) -> dict[str, float]:
        self.joint_planner.train()
        reward_fn = item.get("reward_function")
        if not callable(reward_fn):
            raise ValueError("EAI reinforcement items require reward_function")
        context = _context(item, "eai")
        pairs = []
        with self.joint_runtime.domain_scope("eai"):
            encoded_context = self.eai_planner.encode_context(context)
            for _ in range(self.eai_num_samples):
                policy_dfa = self.joint_runtime.dfa_for("eai", item)
                labels, logprob = self.eai_planner.sample_labels_from_context(
                    encoded_context,
                    policy_dfa,
                    max_steps=self.joint_runtime.max_eai_steps,
                )
                reward = reward_fn(labels, data_item=item)
                from test_regr.EmbodiedAgentInterface.reward import evaluate_goal_satisfaction
                goal = evaluate_goal_satisfaction(
                    labels, item, self.joint_runtime.eai_vocabulary,
                    world_bundle=self.joint_runtime.world.eai,
                )
                pairs.append((
                    logprob,
                    float(torch.as_tensor(reward).float().mean()),
                    float(goal["recall"]),
                    float(goal["is_success"]),
                ))
            logprobs = torch.stack([pair[0] for pair in pairs])
            rewards = torch.tensor([pair[1] for pair in pairs], device=logprobs.device, dtype=logprobs.dtype)
            advantages = rewards - rewards.mean()
            loss = -(logprobs * advantages.detach()).mean()
            loss = loss + self.eai_supervised_weight * self._eai_anchor()
            self.planner_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.joint_planner.parameters(), 1.0)
            self.planner_optimizer.step()
        return {
            "loss": float(loss.detach()),
            "reward": float(rewards.mean()),
            "goal_recall": sum(pair[2] for pair in pairs) / len(pairs),
            "success": sum(pair[3] for pair in pairs) / len(pairs),
            "samples": len(pairs),
        }

    def train_vlabench_update(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        *,
        rollouts_per_update: int = 8,
    ) -> dict[str, float]:
        self.joint_planner.train()
        self.controller.train()
        with self.joint_runtime.domain_scope("vlabench"):
            planner = self.planner_head
            self.planner_head = self.vlabench_planner
            try:
                return super().train_joint_epoch(
                    descriptors,
                    rollouts_per_update=rollouts_per_update,
                )
            finally:
                self.planner_head = planner

    def train_alternating_epoch(
        self,
        eai_examples: Sequence[Mapping[str, Any]],
        vlabench_descriptors: Sequence[Mapping[str, Any]],
        *,
        rounds: int = 10,
        vlabench_rollouts_per_update: int = 8,
    ) -> dict[str, Any]:
        if not eai_examples or not vlabench_descriptors:
            raise ValueError("joint Stage 2 requires both domain sources")
        self.joint_planner.train()
        self.controller.train()
        eai_totals = {"loss": 0.0, "reward": 0.0, "goal_recall": 0.0, "success": 0.0}
        vla_totals = {
            "planner_loss": 0.0,
            "controller_loss": 0.0,
            "return": 0.0,
            "success_rate": 0.0,
            "valid_rate": 0.0,
            "steps": 0.0,
        }
        vla_task_totals: dict[str, dict[str, float]] = {}
        vla_episode_count = 0
        start_cursor = self.round_robin_cursor
        progress = _TrainingProgress("Stage 2 rounds", int(rounds))
        for offset in range(int(rounds)):
            _emit_progress(f"Stage 2 round {offset + 1}/{rounds}: EAI policy update")
            eai = self.train_eai_update(random.choice(eai_examples))
            descriptor = vlabench_descriptors[
                (start_cursor + offset) % len(vlabench_descriptors)
            ]
            _emit_progress(
                f"Stage 2 round {offset + 1}/{rounds}: "
                f"VLABench task={descriptor.get('task', 'unknown')} "
                f"rollouts={vlabench_rollouts_per_update}"
            )
            vla = self.train_vlabench_update(
                [descriptor],
                rollouts_per_update=vlabench_rollouts_per_update,
            )
            for key in eai_totals:
                eai_totals[key] += float(eai[key])
            for key in vla_totals:
                vla_totals[key] += float(vla[key])
            vla_episode_count += int(vla["episodes"])
            for task_name, task_metrics in vla.get("per_task", {}).items():
                episodes = int(task_metrics["episodes"])
                totals = vla_task_totals.setdefault(
                    task_name,
                    {
                        "episodes": 0.0,
                        "successes": 0.0,
                        "valid": 0.0,
                        "return": 0.0,
                        "steps": 0.0,
                    },
                )
                totals["episodes"] += episodes
                totals["successes"] += int(task_metrics["successes"])
                totals["valid"] += float(task_metrics["valid_rate"]) * episodes
                totals["return"] += float(task_metrics["return"]) * episodes
                totals["steps"] += float(task_metrics["steps"]) * episodes
            self.round_robin_cursor += 1
            progress.update(
                offset + 1,
                eai_reward=eai_totals["reward"] / (offset + 1),
                vlabench_return=vla_totals["return"] / (offset + 1),
            )
        count = max(1, int(rounds))
        per_task = {
            task_name: {
                "episodes": int(totals["episodes"]),
                "successes": int(totals["successes"]),
                "success_rate": totals["successes"] / totals["episodes"],
                "valid_rate": totals["valid"] / totals["episodes"],
                "return": totals["return"] / totals["episodes"],
                "steps": totals["steps"] / totals["episodes"],
            }
            for task_name, totals in sorted(vla_task_totals.items())
        }
        return {
            "eai": {key: value / count for key, value in eai_totals.items()},
            "vlabench": {
                **{key: value / count for key, value in vla_totals.items()},
                "episodes": vla_episode_count,
                "per_task": per_task,
            },
            "rounds": int(rounds),
            "round_robin_cursor": self.round_robin_cursor,
        }


assert issubclass(JointReinforcementProgram, ReinforcementProgram)
