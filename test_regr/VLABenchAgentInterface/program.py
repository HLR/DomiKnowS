"""DomiKnowS two-stage programs and joint planner/controller reinforcement."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from domiknows.reinforcement.reinforcement_program import ReinforcementProgram

try:
    from .environment import ee_action_to_env_action, numbered_views_from_observation, reset_reward_tracking
    from .graph import dfa_accepts_plan
    from .models import controller_loss
    from .world_graph import (
        condition_index_for_pattern,
        materialize_plan,
        split_subtasks,
        validate_plan,
        verify_plan_constraints,
    )
except ImportError:
    from environment import ee_action_to_env_action, numbered_views_from_observation, reset_reward_tracking
    from graph import dfa_accepts_plan
    from models import controller_loss
    from world_graph import condition_index_for_pattern, materialize_plan, split_subtasks, validate_plan, verify_plan_constraints


class EOSMaskedCrossEntropyLoss(torch.nn.Module):
    """Cross entropy through the first EOS, excluding EOS padding afterward."""

    def __init__(self, eos_label: int):
        super().__init__()
        self.eos_label = int(eos_label)

    def forward(self, input, target, *_args, **_kwargs):
        logits = input.reshape(-1, input.shape[-1])
        labels = torch.as_tensor(target, dtype=torch.long, device=input.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        keep = ((labels == self.eos_label).cumsum(dim=-1) <= 1).reshape(-1)
        labels = labels.reshape(-1)
        if keep.numel() != logits.shape[0]:
            raise ValueError(f"logit/label shape mismatch: {tuple(input.shape)} vs {tuple(target.shape)}")
        return F.cross_entropy(logits[keep], labels[keep])


def attach_planner_sensors(runtime, planner, *, device="cpu"):
    """Attach graph inputs and the compact planner learner once."""
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
    from domiknows.sensor.pytorch.sensors import ReaderSensor

    bundle = runtime.generation_bundle
    text, token, generated = bundle.text, bundle.token, bundle.generated_token
    if "planner_context" not in text:
        text["planner_context"] = ReaderSensor(keyword="planner_context")
        token["position"] = ReaderSensor(keyword="token_positions")
        token[bundle.contains] = EdgeSensor(
            text["planner_context"],
            token["position"],
            relation=bundle.contains,
            forward=lambda _context, positions: torch.ones_like(positions).unsqueeze(-1).float(),
        )
        token["target_plan_label"] = ReaderSensor(keyword="target_plan_labels")
        token[generated] = ModuleLearner(
            bundle.contains,
            text["planner_context"],
            "target_plan_label",
            module=planner,
            device=device,
        )
        token[generated] = ReaderSensor(keyword="target_plan_labels", label=True)
    return [text, token, generated, token[bundle.contains], token[generated]]


def build_stage1_program(runtime, planner, *, device="cpu"):
    """Build the EAI-equivalent SolverPOI exact-match program."""
    from domiknows.program import SolverPOIProgram
    from domiknows.program.metric import MacroAverageTracker

    poi = attach_planner_sensors(runtime, planner, device=device)
    program = SolverPOIProgram(
        runtime.generation_graph,
        poi=poi,
        inferTypes=["local/argmax"],
        loss=MacroAverageTracker(EOSMaskedCrossEntropyLoss(runtime.vocabulary.eos_label)),
        device=device,
        metric={},
    )
    program.planner_head = planner
    return program


@dataclass
class ControllerTransition:
    images: torch.Tensor
    state: torch.Tensor
    task_index: torch.Tensor
    actions: torch.Tensor
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    reward: float
    done: bool
    executed: int
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class JointEpisode:
    planner_logprobs: list[torch.Tensor]
    controller: list[ControllerTransition]
    total_return: float
    success: bool
    valid: bool
    steps: int
    planner_returns: list[float] | None = None


def generalized_advantage_estimate(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have equal length")
    advantages = [0.0] * len(rewards)
    running = 0.0
    for index in reversed(range(len(rewards))):
        next_value = 0.0 if index + 1 == len(values) else float(values[index + 1])
        continuation = 0.0 if dones[index] else 1.0
        delta = float(rewards[index]) + gamma * next_value * continuation - float(values[index])
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index] = running
    returns = [advantage + float(value) for advantage, value in zip(advantages, values)]
    return advantages, returns


def ppo_clipped_loss(new_logprob, old_logprob, advantage, *, clip: float = 0.2):
    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def _last(timestep: Any) -> bool:
    value = getattr(timestep, "last", None)
    return bool(value()) if callable(value) else bool(value)


def _signal(env, name: str) -> float:
    function = getattr(env, name, None)
    if function is None:
        return 0.0
    try:
        value = function(threshold=0.1, discrete=False) if name == "get_intention_score" else function()
    except TypeError:
        value = function(threshold=0.1) if name == "get_intention_score" else function()
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _controller_inputs(observations, task_index: int, device, camera_views: int = 3):
    history = list(observations)[-2:]
    if len(history) == 1:
        history.insert(0, history[0])
    image_history, state_history = [], []
    for observation in history:
        rgb = np.asarray(observation["rgb"])[:camera_views]
        image_history.append(torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0)
        state = np.asarray(observation.get("state", observation.get("ee_state", observation.get("q_state")))).reshape(-1)[:7]
        state_history.append(torch.as_tensor(state, dtype=torch.float32))
    return (
        torch.stack(image_history).unsqueeze(0).to(device),
        torch.stack(state_history).unsqueeze(0).to(device),
        torch.tensor([task_index], dtype=torch.long, device=device),
    )


class VLABenchHierarchicalReinforcementProgram(ReinforcementProgram):
    """DomiKnowS planner REINFORCE plus continuous controller PPO."""

    def __init__(
        self,
        runtime,
        planner,
        controller,
        *,
        planner_optimizer,
        controller_optimizer,
        env_factory: Callable[..., Any],
        supervised_examples: Sequence[Any] = (),
        controller_anchor_loader: Iterable[Mapping[str, torch.Tensor]] | None = None,
        device="cpu",
        num_samples: int = 4,
        execute_horizon: int = 4,
        max_steps: int = 400,
        supervised_weight: float = 0.1,
        controller_bc_weight: float = 0.05,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_clip: float = 0.2,
        ppo_epochs: int = 4,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        progress_callback: Callable[[str], None] | None = None,
    ):
        poi = attach_planner_sensors(runtime, planner, device=device)
        super().__init__(
            runtime.generation_graph,
            targets=[runtime.generation_bundle.generated_token],
            num_samples=num_samples,
            estimator="reinforce",
            poi=poi,
            device=device,
        )
        self.runtime = runtime
        self.planner_head = planner
        self.controller = controller
        self.planner_optimizer = planner_optimizer
        self.controller_optimizer = controller_optimizer
        self.env_factory = env_factory
        self.supervised_examples = tuple(supervised_examples)
        self.controller_anchor_loader = controller_anchor_loader
        self.device_name = torch.device(device)
        self.execute_horizon = int(execute_horizon)
        self.max_steps = int(max_steps)
        self.supervised_weight = float(supervised_weight)
        self.controller_bc_weight = float(controller_bc_weight)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.ppo_clip = float(ppo_clip)
        self.ppo_epochs = int(ppo_epochs)
        self.value_weight = float(value_weight)
        self.entropy_weight = float(entropy_weight)
        self.progress_callback = progress_callback

    def _report_progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _valid_plan(self, plan, entities) -> bool:
        validation = validate_plan(plan, entity_table=entities, skill_arguments=self.runtime.world_bundle.skill_arguments)
        if not validation.valid:
            return False
        if not dfa_accepts_plan(
            self.runtime.dfa,
            self.runtime.generation_bundle,
            plan,
            entities,
            world=self.runtime.world_bundle,
        ):
            return False
        try:
            root = materialize_plan(plan, entities, self.runtime.world_bundle)
            result = verify_plan_constraints(root, self.runtime.world_bundle)
            return result is None or result.score >= 1.0
        except Exception:
            return False

    def collect_episode(self, descriptor: Mapping[str, Any]) -> JointEpisode:
        kwargs = dict(descriptor.get("env_kwargs", {}))
        if descriptor.get("task") is not None:
            kwargs.setdefault("task", descriptor["task"])
        env = self.env_factory(**kwargs)
        planner_logprobs: list[torch.Tensor] = []
        planner_transition_indices: list[int | None] = []
        transitions: list[ControllerTransition] = []
        valid, success, steps = True, False, 0
        previous_progress = previous_intention = 0.0
        last_progress_report = time.monotonic()
        try:
            timestep = env.reset()
            reset_reward_tracking(env)
            observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
            observations = [observation]
            previous_progress = initial_progress = _signal(env, "get_task_progress")
            previous_intention = initial_intention = _signal(env, "get_intention_score")
            instruction = descriptor.get("instruction")
            if not instruction:
                instruction = env.task.get_instruction() if hasattr(getattr(env, "task", None), "get_instruction") else ""

            while steps < self.max_steps:
                now = time.monotonic()
                if now - last_progress_report >= 30.0:
                    self._report_progress(
                        f"VLABench episode task={descriptor.get('task', 'unknown')} "
                        f"steps={steps}/{self.max_steps}"
                    )
                    last_progress_report = now
                views, entities = numbered_views_from_observation(env, observation)
                selected_plan = None
                selected_logprob = None
                try:
                    for _ in range(self.num_samples):
                        try:
                            candidate, candidate_logprob = self.planner_head.sample_with_logprob(
                                instruction=instruction,
                                images=views,
                                entity_table=entities,
                                dfa=self.runtime.dfa,
                                world=self.runtime.world_bundle,
                                max_steps=self.runtime.max_tokens,
                            )
                        except (RuntimeError, TypeError, ValueError):
                            continue
                        if self._valid_plan(candidate, entities):
                            if selected_plan is None:
                                selected_plan = candidate
                                selected_logprob = candidate_logprob
                        else:
                            # Constraint-invalid samples are useful negative
                            # planner evidence but can never reach the controller.
                            planner_logprobs.append(candidate_logprob)
                            planner_transition_indices.append(None)
                finally:
                    for image in views:
                        image.close()
                if selected_plan is None:
                    valid = False
                    break
                plan = selected_plan
                planner_logprobs.append(selected_logprob)
                planner_transition_indices.append(len(transitions))
                subtasks = split_subtasks([operation["name"] for operation in plan])
                task_index = condition_index_for_pattern(subtasks[0]) if subtasks else 0
                inputs = _controller_inputs(observations, task_index, self.device_name)
                actions, logprobs, _entropy, values = self.controller.sample_action_chunk(*inputs)
                executed = 0
                chunk_reward = 0.0
                for candidate in actions[0, : self.execute_horizon]:
                    try:
                        command = ee_action_to_env_action(env, candidate.detach().cpu().numpy())
                    except (ValueError, TypeError, AttributeError):
                        valid = False
                        break
                    timestep = env.step(command)
                    steps += 1
                    executed += 1
                    observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
                    observations.append(observation)
                    progress = _signal(env, "get_task_progress")
                    intention = _signal(env, "get_intention_score")
                    chunk_reward += 0.25 * (progress - previous_progress) + 0.10 * (intention - previous_intention)
                    previous_progress, previous_intention = progress, intention
                    if _last(timestep):
                        success = True
                        break
                    if steps >= self.max_steps:
                        break
                if executed:
                    transitions.append(ControllerTransition(
                        images=inputs[0].detach().cpu(),
                        state=inputs[1].detach().cpu(),
                        task_index=inputs[2].detach().cpu(),
                        actions=actions.detach().cpu(),
                        old_logprob=logprobs[0, :executed].sum().detach().cpu(),
                        old_value=values[0].detach().cpu(),
                        reward=chunk_reward,
                        done=success or not valid or steps >= self.max_steps,
                        executed=executed,
                    ))
                if success or not valid or steps >= self.max_steps:
                    break

            final_progress = previous_progress
            final_intention = previous_intention
            efficiency = max(0.0, 1.0 - steps / max(1, self.max_steps)) if success else 0.0
            target_total = float(np.clip(
                0.60 * float(success)
                + 0.25 * final_progress
                + 0.10 * final_intention
                + 0.05 * efficiency,
                0.0,
                1.0,
            ))
            # Delta shaping omits the initial scores.  The terminal correction
            # adds those initial terms plus success/efficiency (and clipping),
            # making the stored rewards telescope exactly to target_total.
            terminal = target_total - sum(item.reward for item in transitions)
            if transitions:
                transitions[-1].reward += terminal
                transitions[-1].done = True
            total = sum(item.reward for item in transitions)
            if not valid:
                total = 0.0
                for item in transitions:
                    item.reward = 0.0
            values = [float(item.old_value) for item in transitions]
            advantages, returns = generalized_advantage_estimate(
                [item.reward for item in transitions], values, [item.done for item in transitions],
                gamma=self.gamma, gae_lambda=self.gae_lambda,
            )
            for item, advantage, return_value in zip(transitions, advantages, returns):
                item.advantage = advantage
                item.return_value = return_value
            planner_returns = [
                0.0 if start is None else sum(item.reward for item in transitions[start:])
                for start in planner_transition_indices
            ]
            return JointEpisode(planner_logprobs, transitions, total, success, valid, steps, planner_returns)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _planner_anchor(self):
        if not self.supervised_examples or not self.supervised_weight:
            return torch.zeros((), device=self.device_name)
        example = random.choice(self.supervised_examples)
        value = example.as_reward_item() if hasattr(example, "as_reward_item") else example
        images = []
        from PIL import Image
        for path in getattr(example, "segmented_image_paths", ()) or getattr(example, "image_paths", ()):
            images.append(Image.open(path).convert("RGB"))
        try:
            return self.planner_head.supervised_loss(
                instruction=value.get("instruction", ""),
                images=images,
                entity_table=value.get("entities", ()),
                target_plan=value.get("operation_sequence"),
                world=self.runtime.world_bundle,
            )
        finally:
            for image in images:
                image.close()

    def _update_planner(self, episodes: Sequence[JointEpisode]) -> float:
        pairs = []
        for episode in episodes:
            returns = episode.planner_returns
            if returns is None:
                returns = [episode.total_return] * len(episode.planner_logprobs)
            pairs.extend(zip(episode.planner_logprobs, returns))
        if not pairs:
            return 0.0
        logprobs = torch.stack([item[0] for item in pairs])
        rewards = torch.tensor([item[1] for item in pairs], dtype=logprobs.dtype, device=logprobs.device)
        advantage = rewards - rewards.mean()
        loss = -(logprobs * advantage.detach()).mean() + self.supervised_weight * self._planner_anchor()
        self.planner_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner_head.parameters(), 1.0)
        self.planner_optimizer.step()
        return float(loss.detach())

    def _controller_anchor(self):
        if self.controller_anchor_loader is None or not self.controller_bc_weight:
            return torch.zeros((), device=self.device_name)
        try:
            batch = next(iter(self.controller_anchor_loader))
        except StopIteration:
            return torch.zeros((), device=self.device_name)
        prediction = self.controller(
            batch["images"].to(self.device_name),
            batch["state"].to(self.device_name),
            batch["task_index"].to(self.device_name),
        )
        return controller_loss(prediction, batch["actions"].to(self.device_name))[0]

    def _update_controller(self, episodes: Sequence[JointEpisode]) -> float:
        transitions = [item for episode in episodes for item in episode.controller]
        if not transitions:
            return 0.0
        advantages = torch.tensor([item.advantage for item in transitions], device=self.device_name)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        total = 0.0
        for _ in range(self.ppo_epochs):
            losses = []
            for index, item in enumerate(transitions):
                logprob, entropy, value = self.controller.evaluate_action_chunk(
                    item.images.to(self.device_name),
                    item.state.to(self.device_name),
                    item.task_index.to(self.device_name),
                    item.actions.to(self.device_name),
                )
                new_logprob = logprob[0, : item.executed].sum()
                policy_loss = ppo_clipped_loss(
                    new_logprob.reshape(1), item.old_logprob.to(self.device_name).reshape(1), advantages[index].reshape(1),
                    clip=self.ppo_clip,
                )
                value_loss = F.mse_loss(value.reshape(()), torch.tensor(item.return_value, device=self.device_name))
                losses.append(policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy[0, : item.executed].mean())
            loss = torch.stack(losses).mean() + self.controller_bc_weight * self._controller_anchor()
            self.controller_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
            self.controller_optimizer.step()
            total += float(loss.detach())
        return total / self.ppo_epochs

    def train_joint_epoch(self, descriptors: Sequence[Mapping[str, Any]], *, rollouts_per_update: int = 8):
        if not descriptors:
            raise ValueError("joint training requires at least one simulator descriptor")
        episodes = []
        for rollout_index in range(rollouts_per_update):
            descriptor = random.choice(descriptors)
            self._report_progress(
                f"VLABench rollout {rollout_index + 1}/{rollouts_per_update} "
                f"task={descriptor.get('task', 'unknown')} started"
            )
            episode = self.collect_episode(descriptor)
            episodes.append(episode)
            self._report_progress(
                f"VLABench rollout {rollout_index + 1}/{rollouts_per_update} "
                f"finished valid={episode.valid} success={episode.success} "
                f"steps={episode.steps} return={episode.total_return:.4f}"
            )
        planner_loss = self._update_planner(episodes)
        controller_loss_value = self._update_controller(episodes)
        return {
            "planner_loss": planner_loss,
            "controller_loss": controller_loss_value,
            "return": sum(item.total_return for item in episodes) / len(episodes),
            "success_rate": sum(item.success for item in episodes) / len(episodes),
            "valid_rate": sum(item.valid for item in episodes) / len(episodes),
            "steps": sum(item.steps for item in episodes) / len(episodes),
            "episodes": len(episodes),
        }
