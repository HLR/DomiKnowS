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
    from .dataset import control_task_index_for_instruction
    from .environment import InverseKinematicsError, bound_ee_action, ee_action_to_env_action, numbered_views_from_observation, quaternion_to_euler, reset_reward_tracking
    from .graph import dfa_accepts_plan
    from .models import controller_loss
    from .world_graph import (
        condition_index_for_pattern,
        controller_plan_context,
        materialize_plan,
        split_subtasks,
        validate_plan,
        verify_plan_constraints,
    )
except ImportError:
    from dataset import control_task_index_for_instruction
    from environment import InverseKinematicsError, bound_ee_action, ee_action_to_env_action, numbered_views_from_observation, quaternion_to_euler, reset_reward_tracking
    from graph import dfa_accepts_plan
    from models import controller_loss
    from world_graph import condition_index_for_pattern, controller_plan_context, materialize_plan, split_subtasks, validate_plan, verify_plan_constraints


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
    plan_context: torch.Tensor | None = None
    feasibility_cost: float = 0.0
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class JointEpisode:
    planner_logprobs: list[torch.Tensor | "PlannerReplayDecision"]
    controller: list[ControllerTransition]
    total_return: float
    success: bool
    valid: bool
    steps: int
    planner_returns: list[float] | None = None
    ik_failures: int = 0
    ik_recoveries: int = 0
    termination_reason: str = "unknown"


@dataclass
class PlannerReplayDecision:
    """CPU replay record for a planner decision sampled during simulation."""

    prepared_context: Mapping[str, Any]
    labels: tuple[int, ...]
    dfa: Any
    max_steps: int

    def logprob(self, planner) -> torch.Tensor:
        replay = getattr(planner, "replay_labels_logprob", None)
        if not callable(replay):
            raise TypeError("planner does not support bounded-memory trajectory replay")
        return replay(
            self.prepared_context,
            self.labels,
            self.dfa,
            max_steps=self.max_steps,
        )


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


def ppo_clipped_loss(
    new_logprob,
    old_logprob,
    advantage,
    *,
    clip: float = 0.2,
    max_log_ratio: float = 2.0,
):
    if not (
        bool(torch.isfinite(new_logprob).all())
        and bool(torch.isfinite(old_logprob).all())
        and bool(torch.isfinite(advantage).all())
    ):
        raise ValueError("PPO inputs must be finite")
    if not np.isfinite(max_log_ratio) or max_log_ratio <= 0:
        raise ValueError("PPO max log ratio must be finite and positive")
    # The ordinary PPO objective remains unbounded for negative advantages when
    # the new policy assigns far more probability than the behavior policy.
    # Cap that trust-region excursion before exp so one stale trajectory cannot
    # dominate the controller update numerically.
    ratio = torch.exp(
        (new_logprob - old_logprob).clamp(-max_log_ratio, max_log_ratio)
    )
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def _last(timestep: Any) -> bool:
    value = getattr(timestep, "last", None)
    return bool(value()) if callable(value) else bool(value)


def _recoverable_simulator_error(exc: BaseException) -> bool:
    """Identify randomized MuJoCo state failures that affect one rollout."""
    name = type(exc).__name__
    module = type(exc).__module__
    message = str(exc)
    return (
        name == "PhysicsError"
        and (module.startswith("dm_control") or "Physics state is invalid" in message)
    ) or "mjWARN_BADQACC" in message


def _observation_state(observation: Mapping[str, Any]) -> np.ndarray:
    # Official VLABench publishes ee_state=[xyz, wxyz, gripper].  q_state is
    # joint space and must never be used as a Cartesian pose.  Keep support for
    # seven-component synthetic/legacy xyz-Euler observations used by tests.
    value = observation.get("ee_state", observation.get("state"))
    if value is None:
        raise KeyError("simulator observation contains no EE state")
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size == 8:
        return np.concatenate((value[:3], quaternion_to_euler(value[3:7]), value[7:8]))
    if value.size >= 7:
        return value[:7]
    raise ValueError("simulator EE state must contain xyz plus orientation and gripper")


def _signal(env, name: str) -> float:
    function = getattr(env, name, None)
    if function is None:
        return 0.0
    try:
        try:
            value = function(threshold=0.1, discrete=False) if name == "get_intention_score" else function()
        except TypeError:
            value = function(threshold=0.1) if name == "get_intention_score" else function()
    except (AttributeError, KeyError, LookupError, ZeroDivisionError):
        # Progress and intention are optional shaping signals.  Known upstream
        # primitive tasks expose the methods while leaving their backing state
        # incomplete.  A missing signal contributes zero; simulator execution
        # and the authoritative success reward remain active.
        return 0.0
    value = float(value)
    return value if np.isfinite(value) else 0.0


class _EntityPointerDFA:
    """Lazy DFA view that removes unknown observation-local pointers."""

    def __init__(self, base_dfa, *, valid_labels, all_labels):
        self.base_dfa = base_dfa
        self.valid_labels = frozenset(int(value) for value in valid_labels)
        self.invalid_labels = frozenset(int(value) for value in all_labels) - self.valid_labels
        self.start_state = base_dfa.start_state
        self.alphabet = base_dfa.alphabet

    def step(self, state, symbol):
        if int(symbol) in self.invalid_labels:
            return None
        return self.base_dfa.step(state, symbol)

    def is_accepting(self, state):
        return self.base_dfa.is_accepting(state)

    def accepts(self, sequence):
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def allowed_tokens(self, state, remaining_steps=None):
        try:
            allowed = self.base_dfa.allowed_tokens(state, remaining_steps=remaining_steps)
        except TypeError:
            allowed = self.base_dfa.allowed_tokens(state)
        return set(allowed).difference(self.invalid_labels)

    def __getattr__(self, name):
        return getattr(self.base_dfa, name)


def _entity_pointer_dfa(base_dfa, vocabulary, entity_count: int):
    """Mask object-pointer labels that are absent from this observation."""
    count = min(max(0, int(entity_count)), int(vocabulary.max_entities))
    if count >= int(vocabulary.max_entities):
        return base_dfa
    all_labels = [
        vocabulary.label_for_token(f"obj:{index}")
        for index in range(vocabulary.max_entities)
    ]
    valid_labels = [
        vocabulary.label_for_token(f"obj:{index}")
        for index in range(count)
    ]
    return _EntityPointerDFA(
        base_dfa,
        valid_labels=valid_labels,
        all_labels=all_labels,
    )


def _controller_inputs(observations, task_index: int, device, camera_views: int = 3, plan_context=None):
    history = list(observations)[-2:]
    if len(history) == 1:
        history.insert(0, history[0])
    image_history, state_history = [], []
    for observation in history:
        rgb = np.asarray(observation["rgb"])[:camera_views]
        image_history.append(torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0)
        state = _observation_state(observation)
        state_history.append(torch.as_tensor(state, dtype=torch.float32))
    return (
        torch.stack(image_history).unsqueeze(0).to(device),
        torch.stack(state_history).unsqueeze(0).to(device),
        torch.tensor([task_index], dtype=torch.long, device=device),
        torch.tensor([plan_context or (0, 0, 0)], dtype=torch.long, device=device),
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
        controller_task_instructions: Mapping[int, str] | None = None,
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
        ppo_max_log_ratio: float = 2.0,
        ppo_target_action_log_ratio: float = 0.2,
        max_controller_loss: float = 50.0,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        feasibility_weight: float = 0.05,
        max_position_step: float = 0.02,
        max_rotation_step: float = 0.10,
        ik_tolerance: float = 1e-3,
        ik_max_steps: int = 200,
        max_consecutive_ik_rejections: int = 3,
        simulator_init_retries: int = 3,
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
        self.controller_task_instructions = dict(controller_task_instructions or {})
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
        self.ppo_max_log_ratio = float(ppo_max_log_ratio)
        self.ppo_target_action_log_ratio = float(ppo_target_action_log_ratio)
        self.max_controller_loss = float(max_controller_loss)
        self.value_weight = float(value_weight)
        self.entropy_weight = float(entropy_weight)
        self.feasibility_weight = float(feasibility_weight)
        self.max_position_step = float(max_position_step)
        self.max_rotation_step = float(max_rotation_step)
        self.ik_tolerance = float(ik_tolerance)
        self.ik_max_steps = int(ik_max_steps)
        if not np.isfinite(self.ppo_max_log_ratio) or self.ppo_max_log_ratio <= 0:
            raise ValueError("PPO max log ratio must be finite and positive")
        if (
            not np.isfinite(self.ppo_target_action_log_ratio)
            or self.ppo_target_action_log_ratio <= 0
        ):
            raise ValueError("PPO target action log-ratio must be finite and positive")
        if not np.isfinite(self.max_controller_loss) or self.max_controller_loss <= 0:
            raise ValueError("maximum controller loss must be finite and positive")
        if self.max_position_step <= 0 or self.max_rotation_step <= 0:
            raise ValueError("controller execution step limits must be positive")
        if not np.isfinite(self.ik_tolerance) or self.ik_tolerance <= 0:
            raise ValueError("IK tolerance must be finite and positive")
        if self.ik_max_steps <= 0:
            raise ValueError("IK max steps must be positive")
        self.max_consecutive_ik_rejections = int(max_consecutive_ik_rejections)
        if self.max_consecutive_ik_rejections <= 0:
            raise ValueError("maximum consecutive IK rejections must be positive")
        self.simulator_init_retries = max(1, int(simulator_init_retries))
        self.progress_callback = progress_callback
        self._entity_dfa_cache: dict[int, Any] = {}

    def _report_progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _dfa_for_entities(self, entities) -> Any:
        count = len(entities)
        cached = self._entity_dfa_cache.get(count)
        if cached is None:
            cached = _entity_pointer_dfa(self.runtime.dfa, self.runtime.vocabulary, count)
            self._entity_dfa_cache[count] = cached
        return cached

    def _plan_rejection_reason(self, plan, entities, *, dfa=None) -> str | None:
        validation = validate_plan(plan, entity_table=entities, skill_arguments=self.runtime.world_bundle.skill_arguments)
        if not validation.valid:
            return "schema:" + (validation.errors[0] if validation.errors else "invalid")
        if not dfa_accepts_plan(
            dfa or self.runtime.dfa,
            self.runtime.generation_bundle,
            plan,
            entities,
            world=self.runtime.world_bundle,
        ):
            return "dfa"
        try:
            root = materialize_plan(plan, entities, self.runtime.world_bundle)
            result = verify_plan_constraints(root, self.runtime.world_bundle)
            return None if result is None or result.score >= 1.0 else "semantic_constraint"
        except Exception as exc:
            return f"constraint_error:{type(exc).__name__}"

    def _valid_plan(self, plan, entities, *, dfa=None) -> bool:
        return self._plan_rejection_reason(plan, entities, dfa=dfa) is None

    def collect_episode(self, descriptor: Mapping[str, Any]) -> JointEpisode:
        kwargs = dict(descriptor.get("env_kwargs", {}))
        if descriptor.get("task") is not None:
            kwargs.setdefault("task", descriptor["task"])
        env = None
        for attempt in range(self.simulator_init_retries):
            try:
                env = self.env_factory(**kwargs)
                break
            except Exception as exc:
                if not _recoverable_simulator_error(exc):
                    raise
                self._report_progress(
                    f"VLABench simulator initialization failed for task={descriptor.get('task', 'unknown')} "
                    f"attempt={attempt + 1}/{self.simulator_init_retries}: {type(exc).__name__}"
                )
        if env is None:
            return JointEpisode([], [], 0.0, False, False, 0, [])
        planner_logprobs: list[torch.Tensor | PlannerReplayDecision] = []
        planner_transition_indices: list[int | None] = []
        transitions: list[ControllerTransition] = []
        valid, success, steps = True, False, 0
        operation_cursor = 0
        consecutive_ik_rejections = 0
        ik_failures = ik_recoveries = 0
        termination_reason = "max_steps"
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
            controller_task_index = None
            if self.controller_task_instructions:
                try:
                    controller_task_index = control_task_index_for_instruction(
                        instruction,
                        self.controller_task_instructions,
                    )
                except KeyError as exc:
                    self._report_progress(
                        f"VLABench controller rejected unknown instruction for "
                        f"task={descriptor.get('task', 'unknown')}: {exc}"
                    )
                    valid = False
                    termination_reason = "unknown_instruction"

            while steps < self.max_steps:
                if not valid:
                    break
                now = time.monotonic()
                if now - last_progress_report >= 30.0:
                    self._report_progress(
                        f"VLABench episode task={descriptor.get('task', 'unknown')} "
                        f"steps={steps}/{self.max_steps}"
                    )
                    last_progress_report = now
                views, entities = numbered_views_from_observation(env, observation)
                sampling_dfa = self._dfa_for_entities(entities)
                selected_plan = None
                selected_logprob = None
                rejection_counts: dict[str, int] = {}
                try:
                    encoded_context = None
                    prepared_context = None
                    encode_context = getattr(self.planner_head, "encode_context", None)
                    prepare_replay = getattr(self.planner_head, "prepare_replay_context", None)
                    encode_replay = getattr(self.planner_head, "encode_replay_context", None)
                    planner_context = {
                        "instruction": instruction,
                        "images": views,
                        "entity_table": entities,
                    }
                    if callable(prepare_replay) and callable(encode_replay):
                        prepared_context = prepare_replay(planner_context)
                        # Collection needs sampled labels, not a retained Qwen
                        # graph. The trajectories are replayed one at a time
                        # during the policy update.
                        with torch.no_grad():
                            encoded_context = encode_replay(prepared_context)
                    elif callable(encode_context):
                        encoded_context = encode_context(planner_context)
                    for _ in range(self.num_samples):
                        try:
                            sampled = self.planner_head.sample_with_logprob(
                                instruction=instruction,
                                images=views,
                                entity_table=entities,
                                dfa=sampling_dfa,
                                world=self.runtime.world_bundle,
                                max_steps=self.runtime.max_tokens,
                                encoded_context=encoded_context,
                                return_labels=prepared_context is not None,
                            )
                            if len(sampled) == 3:
                                candidate, candidate_logprob, candidate_labels = sampled
                                candidate_evidence = PlannerReplayDecision(
                                    prepared_context=prepared_context,
                                    labels=tuple(int(label) for label in candidate_labels),
                                    dfa=sampling_dfa,
                                    max_steps=self.runtime.max_tokens,
                                )
                            else:
                                candidate, candidate_logprob = sampled
                                candidate_evidence = candidate_logprob
                        except (RuntimeError, TypeError, ValueError) as exc:
                            reason = f"sample_error:{type(exc).__name__}"
                            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                            continue
                        reason = self._plan_rejection_reason(candidate, entities, dfa=sampling_dfa)
                        if reason is None:
                            if selected_plan is None:
                                selected_plan = candidate
                                selected_logprob = candidate_evidence
                        else:
                            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                            # Constraint-invalid samples are useful negative
                            # planner evidence but can never reach the controller.
                            planner_logprobs.append(candidate_evidence)
                            planner_transition_indices.append(None)
                finally:
                    for image in views:
                        image.close()
                if selected_plan is None:
                    details = ", ".join(
                        f"{name}={count}" for name, count in sorted(rejection_counts.items())
                    ) or "no candidates"
                    self._report_progress(
                        f"VLABench planner rejected all {self.num_samples} candidates: {details}"
                    )
                    valid = False
                    termination_reason = "invalid_plan"
                    break
                plan = selected_plan
                planner_logprobs.append(selected_logprob)
                planner_transition_indices.append(len(transitions))
                subtasks = split_subtasks([operation["name"] for operation in plan])
                task_index = (
                    controller_task_index
                    if controller_task_index is not None
                    else condition_index_for_pattern(subtasks[0]) if subtasks else 0
                )
                # VLABench's progress/intention signals are often flat until a
                # primitive completes.  Without a fallback the controller is
                # conditioned on `pick` for the entire rollout, although the
                # demonstration windows switch operation context by episode
                # phase.  Keep semantic advancement when available and use the
                # same normalized phase convention as the offline dataset.
                phase_cursor = min(
                    max(0, len(plan) - 1),
                    int(steps * max(1, len(plan)) / max(1, self.max_steps)),
                )
                operation_cursor = max(operation_cursor, phase_cursor)
                inputs = _controller_inputs(
                    observations,
                    task_index,
                    self.device_name,
                    # Offline control demonstrations have no segmentation-to-
                    # graph pointer correspondence and therefore train the
                    # entity padding row.  Feeding numbered online pointers
                    # here selects otherwise untrained embedding rows.  The
                    # language task id and images retain object identity while
                    # skill and operation position remain graph-conditioned.
                    plan_context=controller_plan_context(plan, operation_cursor),
                )
                try:
                    try:
                        actions, logprobs, _entropy, values = self.controller.sample_action_chunk(*inputs)
                    except TypeError as exc:
                        # Preserve small legacy/fake controllers while the
                        # production controller consumes graph plan context.
                        if "positional" not in str(exc) and "argument" not in str(exc):
                            raise
                        actions, logprobs, _entropy, values = self.controller.sample_action_chunk(*inputs[:3])
                except ValueError as exc:
                    self._report_progress(
                        f"VLABench controller policy rejected task={descriptor.get('task', 'unknown')} "
                        f"step={steps}: {exc}"
                    )
                    valid = False
                    termination_reason = "invalid_policy"
                    break
                # PPO must retain the latent actions sampled by the behavior
                # policy. The safety envelope below is part of the environment
                # transition, not a second policy sample; evaluating its clipped
                # outputs under the Normal/Bernoulli policy produces an invalid
                # importance ratio.
                policy_actions = actions.detach().clone()
                executed = 0
                chunk_reward = 0.0
                chunk_ik_failures = 0
                chunk_advanced = False
                ik_truncated = False
                for action_index, candidate in enumerate(actions[0, : self.execute_horizon]):
                    try:
                        bounded = bound_ee_action(
                            candidate.detach().cpu().numpy(),
                            _observation_state(observation),
                            max_position_step=self.max_position_step,
                            max_rotation_step=self.max_rotation_step,
                        )
                        current = np.asarray(_observation_state(observation), dtype=np.float64)
                        command = None
                        last_ik_error = None
                        # A zero-scale target is merely a hold command. Treating
                        # it as recovery creates long no-op loops that look
                        # executable while providing no controller progress.
                        for recovery_scale in (1.0, 0.5, 0.25, 0.125):
                            recovered = np.asarray(bounded, dtype=np.float64).copy()
                            recovered[:3] = current[:3] + recovery_scale * (recovered[:3] - current[:3])
                            angle_delta = np.arctan2(
                                np.sin(recovered[3:6] - current[3:6]),
                                np.cos(recovered[3:6] - current[3:6]),
                            )
                            recovered[3:6] = current[3:6] + recovery_scale * angle_delta
                            try:
                                command = ee_action_to_env_action(
                                    env,
                                    recovered,
                                    ik_tolerance=self.ik_tolerance,
                                    ik_max_steps=self.ik_max_steps,
                                )
                                if recovery_scale < 1.0:
                                    ik_recoveries += 1
                                    self._report_progress(
                                        f"VLABench IK recovered task={descriptor.get('task', 'unknown')} "
                                        f"step={steps} scale={recovery_scale:g}"
                                    )
                                break
                            except InverseKinematicsError as exc:
                                last_ik_error = exc
                                ik_failures += 1
                                chunk_ik_failures += 1
                        if command is None:
                            consecutive_ik_rejections += 1
                            ik_truncated = (
                                consecutive_ik_rejections
                                >= self.max_consecutive_ik_rejections
                            )
                            if ik_truncated:
                                self._report_progress(
                                    f"VLABench rollout IK-truncated task={descriptor.get('task', 'unknown')} "
                                    f"step={steps} after {consecutive_ik_rejections} rejected chunks: "
                                    f"{last_ik_error}"
                                )
                                termination_reason = "ik_failure"
                            else:
                                self._report_progress(
                                    f"VLABench IK rejected action chunk task={descriptor.get('task', 'unknown')} "
                                    f"step={steps} attempt={consecutive_ik_rejections}/"
                                    f"{self.max_consecutive_ik_rejections}; resampling"
                                )
                            break
                    except (ValueError, TypeError, AttributeError, KeyError) as exc:
                        self._report_progress(
                            f"VLABench controller action rejected task={descriptor.get('task', 'unknown')} "
                            f"step={steps}: {type(exc).__name__}: {exc}"
                        )
                        valid = False
                        termination_reason = "invalid_action"
                        break
                    timestep = env.step(command)
                    consecutive_ik_rejections = 0
                    steps += 1
                    executed += 1
                    observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
                    observations.append(observation)
                    progress = _signal(env, "get_task_progress")
                    intention = _signal(env, "get_intention_score")
                    delta_progress = progress - previous_progress
                    delta_intention = intention - previous_intention
                    chunk_reward += 0.25 * delta_progress + 0.10 * delta_intention
                    if (
                        not chunk_advanced
                        and (delta_progress > 1e-6 or delta_intention > 1e-6)
                        and operation_cursor + 1 < len(plan)
                    ):
                        operation_cursor += 1
                        chunk_advanced = True
                        # Potential-based graph-subgoal shaping moves credit to
                        # the operation boundary. The terminal correction below
                        # preserves the authoritative final rollout formula.
                        chunk_reward += 0.05 / max(1, len(plan))
                    previous_progress, previous_intention = progress, intention
                    if _last(timestep):
                        success = True
                        termination_reason = "success"
                        break
                    if steps >= self.max_steps:
                        break
                # Even a first-action IK rejection is retained for the
                # feasibility auxiliary objective. It has no simulator reward
                # or PPO advantage, but the actor can reduce that action's
                # likelihood instead of repeatedly rediscovering it.
                likelihood_steps = executed or int(chunk_ik_failures > 0)
                if likelihood_steps:
                    transitions.append(ControllerTransition(
                        images=inputs[0].detach().cpu(),
                        state=inputs[1].detach().cpu(),
                        task_index=inputs[2].detach().cpu(),
                        plan_context=inputs[3].detach().cpu(),
                        actions=policy_actions.cpu(),
                        old_logprob=logprobs[0, :likelihood_steps].sum().detach().cpu(),
                        old_value=values[0].detach().cpu(),
                        reward=chunk_reward,
                        done=success or not valid or ik_truncated or steps >= self.max_steps,
                        executed=likelihood_steps,
                        feasibility_cost=float(chunk_ik_failures),
                    ))
                if success or not valid or ik_truncated or steps >= self.max_steps:
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
            return JointEpisode(
                planner_logprobs,
                transitions,
                total,
                success,
                valid,
                steps,
                planner_returns,
                ik_failures,
                ik_recoveries,
                termination_reason,
            )
        except Exception as exc:
            if _recoverable_simulator_error(exc):
                self._report_progress(
                    f"VLABench simulator physics failure task={descriptor.get('task', 'unknown')} "
                    f"steps={steps}: {type(exc).__name__}"
                )
                return JointEpisode(planner_logprobs, [], 0.0, False, False, steps, [
                    0.0 for _ in planner_logprobs
                ])
            raise
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
        # Evaluation and simulator collection may temporarily put the policy in
        # inference mode.  cuDNN RNNs only retain the reserve-space needed for
        # backward when their forward pass is executed in training mode, so
        # restore the planner before replaying graph-token trajectories.
        self.planner_head.train()
        self.planner_optimizer.zero_grad(set_to_none=True)
        reward_values = torch.tensor([float(item[1]) for item in pairs], dtype=torch.float32)
        advantages = reward_values - reward_values.mean()
        count = len(pairs)
        policy_value = 0.0

        # Legacy and small test planners can still return live log-probability
        # tensors. Backward them together because candidates may share one
        # encoded-context graph.
        live_terms = []
        for (evidence, _reward), advantage in zip(pairs, advantages):
            if isinstance(evidence, PlannerReplayDecision) or float(advantage) == 0.0:
                continue
            live_terms.append(
                -(evidence * advantage.to(device=evidence.device, dtype=evidence.dtype)) / count
            )
        if live_terms:
            live_loss = torch.stack(live_terms).sum()
            live_loss.backward()
            policy_value += float(live_loss.detach())

        # Replay Qwen decisions sequentially. At most one vision-language
        # autograd graph is resident, so memory does not grow with rollout
        # count or episode length while return-to-go still updates LoRA.
        for (evidence, _reward), advantage in zip(pairs, advantages):
            if not isinstance(evidence, PlannerReplayDecision) or float(advantage) == 0.0:
                continue
            logprob = evidence.logprob(self.planner_head)
            term = -(logprob * advantage.to(device=logprob.device, dtype=logprob.dtype)) / count
            term.backward()
            policy_value += float(term.detach())

        anchor = self.supervised_weight * self._planner_anchor()
        if anchor.requires_grad:
            anchor.backward()
        torch.nn.utils.clip_grad_norm_(self.planner_head.parameters(), 1.0)
        self.planner_optimizer.step()
        return policy_value + float(anchor.detach())

    def _controller_anchor(self):
        if self.controller_anchor_loader is None or not self.controller_bc_weight:
            return torch.zeros((), device=self.device_name)
        try:
            batch = next(iter(self.controller_anchor_loader))
        except StopIteration:
            return torch.zeros((), device=self.device_name)
        inputs = (
            batch["images"].to(self.device_name),
            batch["state"].to(self.device_name),
            batch["task_index"].to(self.device_name),
        )
        plan_context = batch.get("plan_context")
        if plan_context is not None:
            inputs += (plan_context.to(self.device_name),)
        prediction = self.controller(*inputs)
        return controller_loss(
            prediction,
            batch["actions"].to(self.device_name),
            state=inputs[1],
            pose_step_scale=getattr(self.controller, "pose_step_scale", None),
        )[0]

    def _update_controller(self, episodes: Sequence[JointEpisode]) -> float:
        # PPO and the behavior-cloning anchor construct fresh autograd graphs.
        # Do not inherit evaluation mode from an earlier rollout evaluation.
        self.controller.train()
        entries = [
            (item, episode.total_return > 0.0)
            for episode in episodes
            for item in episode.controller
        ]
        transitions = [item for item, _informative in entries]
        if not transitions:
            return 0.0
        advantages = torch.zeros(len(entries), device=self.device_name)
        informative_indices = [index for index, (_item, informative) in enumerate(entries) if informative]
        if informative_indices:
            informative_advantages = torch.tensor(
                [entries[index][0].advantage for index in informative_indices],
                device=self.device_name,
            )
            informative_advantages = (
                informative_advantages - informative_advantages.mean()
            ) / informative_advantages.std(unbiased=False).clamp_min(1e-6)
            advantages[informative_indices] = informative_advantages
        total = 0.0
        completed_epochs = 0
        for ppo_epoch in range(self.ppo_epochs):
            losses = []
            action_log_ratios = []
            for index, item in enumerate(transitions):
                logprob, entropy, value = self.controller.evaluate_action_chunk(
                    item.images.to(self.device_name),
                    item.state.to(self.device_name),
                    item.task_index.to(self.device_name),
                    item.actions.to(self.device_name),
                    item.plan_context.to(self.device_name) if item.plan_context is not None else None,
                )
                new_logprob = logprob[0, : item.executed].sum()
                old_logprob = item.old_logprob.to(self.device_name).reshape(())
                if entries[index][1]:
                    action_log_ratios.append(
                        (new_logprob.detach() - old_logprob).abs()
                        / max(1, item.executed)
                    )
                policy_loss = (
                    ppo_clipped_loss(
                        new_logprob.reshape(1),
                        old_logprob.reshape(1),
                        advantages[index].reshape(1),
                        clip=self.ppo_clip,
                        max_log_ratio=self.ppo_max_log_ratio,
                    )
                    if entries[index][1]
                    else torch.zeros((), device=self.device_name)
                )
                target_value = torch.tensor(
                    item.return_value,
                    device=self.device_name,
                ).clamp(-1.0, 1.0)
                value_loss = F.smooth_l1_loss(value.reshape(()), target_value)
                entropy_bonus = (
                    self.entropy_weight * entropy[0, : item.executed].mean()
                    if entries[index][1] else torch.zeros((), device=self.device_name)
                )
                losses.append(
                    policy_loss
                    + self.value_weight * value_loss
                    - entropy_bonus
                    + self.feasibility_weight
                    * float(item.feasibility_cost)
                    * torch.exp(
                        ((new_logprob - old_logprob) / max(1, item.executed)).clamp(-2.0, 2.0)
                    )
                )
            mean_action_log_ratio = (
                float(torch.stack(action_log_ratios).mean())
                if action_log_ratios else 0.0
            )
            if (
                ppo_epoch
                and mean_action_log_ratio > self.ppo_target_action_log_ratio
            ):
                self._report_progress(
                    "VLABench controller PPO early stop: "
                    f"mean action log-ratio={mean_action_log_ratio:.4f} "
                    f"exceeds {self.ppo_target_action_log_ratio:.4f}"
                )
                break
            loss = torch.stack(losses).mean() + self.controller_bc_weight * self._controller_anchor()
            self.controller_optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                self._report_progress("VLABench controller PPO update skipped: non-finite loss")
                break
            if abs(float(loss.detach())) > self.max_controller_loss:
                self._report_progress(
                    "VLABench controller PPO update skipped: "
                    f"loss={float(loss.detach()):.4f} exceeds {self.max_controller_loss:.4f}"
                )
                break
            loss.backward()
            trainable = [
                parameter for parameter in self.controller.parameters()
                if parameter.requires_grad
            ]
            try:
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    1.0,
                    error_if_nonfinite=True,
                )
            except RuntimeError:
                self.controller_optimizer.zero_grad(set_to_none=True)
                self._report_progress("VLABench controller PPO update skipped: non-finite gradient")
                break
            backup = [parameter.detach().clone() for parameter in trainable]
            self.controller_optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
                with torch.no_grad():
                    for parameter, previous in zip(trainable, backup):
                        parameter.copy_(previous)
                # A non-finite Adam moment would immediately poison the next
                # update even after restoring weights. Reset only controller
                # optimizer state; model parameters remain at the last finite
                # point and supervised/PPO training can continue safely.
                self.controller_optimizer.state.clear()
                self.controller_optimizer.zero_grad(set_to_none=True)
                self._report_progress(
                    "VLABench controller PPO update rolled back: optimizer produced non-finite parameters"
                )
                break
            total += float(loss.detach())
            completed_epochs += 1
        return total / max(1, completed_epochs)

    def train_joint_epoch(self, descriptors: Sequence[Mapping[str, Any]], *, rollouts_per_update: int = 8):
        if not descriptors:
            raise ValueError("joint training requires at least one simulator descriptor")
        episodes = []
        episode_tasks = []
        for rollout_index in range(rollouts_per_update):
            descriptor = random.choice(descriptors)
            task_name = str(descriptor.get("task", "unknown"))
            self._report_progress(
                f"VLABench rollout {rollout_index + 1}/{rollouts_per_update} "
                f"task={task_name} started"
            )
            episode = self.collect_episode(descriptor)
            episodes.append(episode)
            episode_tasks.append(task_name)
            self._report_progress(
                f"VLABench rollout {rollout_index + 1}/{rollouts_per_update} "
                f"finished valid={episode.valid} success={episode.success} "
                f"steps={episode.steps} return={episode.total_return:.4f}"
            )
        planner_loss = self._update_planner(episodes)
        controller_loss_value = self._update_controller(episodes)
        task_totals: dict[str, dict[str, float]] = {}
        for task_name, episode in zip(episode_tasks, episodes):
            totals = task_totals.setdefault(
                task_name,
                {
                    "episodes": 0.0, "successes": 0.0, "valid": 0.0,
                    "return": 0.0, "steps": 0.0, "ik_failures": 0.0,
                    "ik_recoveries": 0.0, "ik_truncations": 0.0,
                    "execution_complete": 0.0,
                },
            )
            totals["episodes"] += 1.0
            totals["successes"] += float(episode.success)
            totals["valid"] += float(episode.valid)
            totals["return"] += float(episode.total_return)
            totals["steps"] += float(episode.steps)
            totals["ik_failures"] += float(episode.ik_failures)
            totals["ik_recoveries"] += float(episode.ik_recoveries)
            totals["ik_truncations"] += float(episode.termination_reason == "ik_failure")
            totals["execution_complete"] += float(
                episode.valid and episode.termination_reason != "ik_failure"
            )
        per_task = {
            task_name: {
                "episodes": int(totals["episodes"]),
                "successes": int(totals["successes"]),
                "success_rate": totals["successes"] / totals["episodes"],
                "valid_rate": totals["valid"] / totals["episodes"],
                "return": totals["return"] / totals["episodes"],
                "steps": totals["steps"] / totals["episodes"],
                "ik_failures": int(totals["ik_failures"]),
                "ik_recoveries": int(totals["ik_recoveries"]),
                "ik_recovery_rate": totals["ik_recoveries"] / max(1.0, totals["ik_failures"]),
                "ik_truncation_rate": totals["ik_truncations"] / totals["episodes"],
                "execution_complete_rate": totals["execution_complete"] / totals["episodes"],
            }
            for task_name, totals in sorted(task_totals.items())
        }
        return {
            "planner_loss": planner_loss,
            "controller_loss": controller_loss_value,
            "return": sum(item.total_return for item in episodes) / len(episodes),
            "success_rate": sum(item.success for item in episodes) / len(episodes),
            "valid_rate": sum(item.valid for item in episodes) / len(episodes),
            "steps": sum(item.steps for item in episodes) / len(episodes),
            "episodes": len(episodes),
            "successful_task_count": sum(any(item.success for name, item in zip(episode_tasks, episodes) if name == task) for task in set(episode_tasks)),
            "ik_failures": sum(item.ik_failures for item in episodes),
            "ik_recoveries": sum(item.ik_recoveries for item in episodes),
            "ik_truncation_rate": sum(item.termination_reason == "ik_failure" for item in episodes) / len(episodes),
            "execution_complete_rate": sum(
                item.valid and item.termination_reason != "ik_failure" for item in episodes
            ) / len(episodes),
            "per_task": per_task,
        }

    @torch.no_grad()
    def evaluate_rollouts(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        *,
        rollouts_per_task: int = 1,
        seed: int = 1729,
    ) -> dict[str, Any]:
        """Run fixed-seed simulator evaluation without optimizer updates."""

        if not descriptors or rollouts_per_task <= 0:
            raise ValueError("evaluation requires descriptors and positive rollouts_per_task")
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        controller_training = self.controller.training
        planner_training = self.planner_head.training
        self.controller.eval()
        self.planner_head.eval()
        episodes: list[JointEpisode] = []
        names: list[str] = []
        try:
            for task_offset, descriptor in enumerate(descriptors):
                task_name = str(descriptor.get("task", "unknown"))
                for rollout_offset in range(int(rollouts_per_task)):
                    rollout_seed = int(seed) + 1009 * task_offset + rollout_offset
                    random.seed(rollout_seed)
                    np.random.seed(rollout_seed % (2**32 - 1))
                    torch.manual_seed(rollout_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(rollout_seed)
                    episodes.append(self.collect_episode(descriptor))
                    names.append(task_name)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                for device, state in enumerate(cuda_state[: torch.cuda.device_count()]):
                    torch.cuda.set_rng_state(state, device=device)
            self.controller.train(controller_training)
            self.planner_head.train(planner_training)

        task_totals: dict[str, dict[str, float]] = {}
        for task_name, episode in zip(names, episodes):
            totals = task_totals.setdefault(task_name, {
                "episodes": 0.0, "successes": 0.0, "valid": 0.0,
                "return": 0.0, "steps": 0.0, "ik_failures": 0.0,
                "ik_recoveries": 0.0, "ik_truncations": 0.0,
                "execution_complete": 0.0,
            })
            totals["episodes"] += 1.0
            totals["successes"] += float(episode.success)
            totals["valid"] += float(episode.valid)
            totals["return"] += float(episode.total_return)
            totals["steps"] += float(episode.steps)
            totals["ik_failures"] += float(episode.ik_failures)
            totals["ik_recoveries"] += float(episode.ik_recoveries)
            totals["ik_truncations"] += float(episode.termination_reason == "ik_failure")
            totals["execution_complete"] += float(
                episode.valid and episode.termination_reason != "ik_failure"
            )
        per_task = {
            name: {
                "episodes": int(values["episodes"]),
                "successes": int(values["successes"]),
                "success_rate": values["successes"] / values["episodes"],
                "valid_rate": values["valid"] / values["episodes"],
                "return": values["return"] / values["episodes"],
                "steps": values["steps"] / values["episodes"],
                "ik_failures": int(values["ik_failures"]),
                "ik_recoveries": int(values["ik_recoveries"]),
                "ik_recovery_rate": values["ik_recoveries"] / max(1.0, values["ik_failures"]),
                "ik_truncation_rate": values["ik_truncations"] / values["episodes"],
                "execution_complete_rate": values["execution_complete"] / values["episodes"],
            }
            for name, values in sorted(task_totals.items())
        }
        count = len(episodes)
        return {
            "return": sum(item.total_return for item in episodes) / count,
            "success_rate": sum(item.success for item in episodes) / count,
            "valid_rate": sum(item.valid for item in episodes) / count,
            "steps": sum(item.steps for item in episodes) / count,
            "episodes": count,
            "successful_task_count": sum(int(value["successes"] > 0) for value in per_task.values()),
            "ik_failures": sum(item.ik_failures for item in episodes),
            "ik_recoveries": sum(item.ik_recoveries for item in episodes),
            "ik_truncation_rate": sum(item.termination_reason == "ik_failure" for item in episodes) / count,
            "execution_complete_rate": sum(
                item.valid and item.termination_reason != "ik_failure" for item in episodes
            ) / count,
            "per_task": per_task,
            "evaluation_seed": int(seed),
        }
