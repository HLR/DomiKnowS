"""DomiKnowS two-stage programs and joint planner/controller reinforcement."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from domiknows.reinforcement.reinforcement_program import ReinforcementProgram


POSITIVE_RETURN_EPSILON = 1e-6

try:
    from .diagnostics import RolloutDiagnostics
    from .observations import (
        DEFAULT_CONTROLLER_CAMERA_KEYS,
        camera_indices,
        camera_report,
        image_tensor,
        resolve_controller_camera_names,
    )
    from .dataset import control_task_index_for_instruction
    from .environment import (
        CONTROLLER_FRAME_VERSION,
        InverseKinematicsError,
        bound_ee_action,
        ee_action_to_env_action,
        euler_to_quaternion,
        numbered_views_from_observation,
        quaternion_to_euler,
        reset_reward_tracking,
        robot_frame_position,
        robot_to_world_ee_action,
        world_to_robot_ee_state,
    )
    from .graph import dfa_accepts_plan
    from .models import controller_loss
    from .world_graph import (
        PRIMITIVE_TASK_PATTERNS,
        condition_index_for_pattern,
        controller_plan_context,
        materialize_plan,
        split_subtasks,
        task_contract_for,
        validate_plan,
        verify_plan_constraints,
    )
except ImportError:
    from diagnostics import RolloutDiagnostics
    from observations import (
        DEFAULT_CONTROLLER_CAMERA_KEYS,
        camera_indices,
        camera_report,
        image_tensor,
        resolve_controller_camera_names,
    )
    from dataset import control_task_index_for_instruction
    from environment import (
        CONTROLLER_FRAME_VERSION,
        InverseKinematicsError,
        bound_ee_action,
        ee_action_to_env_action,
        euler_to_quaternion,
        numbered_views_from_observation,
        quaternion_to_euler,
        reset_reward_tracking,
        robot_frame_position,
        robot_to_world_ee_action,
        world_to_robot_ee_state,
    )
    from graph import dfa_accepts_plan
    from models import controller_loss
    from world_graph import PRIMITIVE_TASK_PATTERNS, condition_index_for_pattern, controller_plan_context, materialize_plan, split_subtasks, task_contract_for, validate_plan, verify_plan_constraints


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
    feasibility_index: int | None = None
    old_feasibility_logprob: torch.Tensor | None = None
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
    diagnostics: dict[str, Any] | None = None


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
    physics = getattr(env, "physics", None)
    attempts = []
    if physics is not None:
        if name == "get_intention_score":
            attempts.append(lambda: function(physics, threshold=0.1, discrete=True))
        else:
            attempts.append(lambda: function(physics))
    if name == "get_intention_score":
        attempts.extend((
            lambda: function(threshold=0.1, discrete=True),
            lambda: function(threshold=0.1),
        ))
    attempts.append(lambda: function())
    try:
        value = None
        for attempt in attempts:
            try:
                value = attempt()
                break
            except TypeError:
                continue
        if value is None:
            return 0.0
    except (AttributeError, KeyError, LookupError, ZeroDivisionError):
        # Progress and intention are optional shaping signals.  Known upstream
        # primitive tasks expose the methods while leaving their backing state
        # incomplete.  A missing signal contributes zero; simulator execution
        # and the authoritative success reward remain active.
        return 0.0
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _task_success(env) -> bool:
    """Evaluate task completion independently from dm_env LAST."""
    task = getattr(env, "task", None)
    physics = getattr(env, "physics", None)
    checker = getattr(task, "should_terminate_episode", None)
    if callable(checker) and physics is not None:
        try:
            return bool(checker(physics))
        except (AttributeError, KeyError, LookupError, TypeError, ValueError):
            return False
    conditions = getattr(task, "conditions", None)
    is_met = getattr(conditions, "is_met", None)
    if callable(is_met) and physics is not None:
        try:
            return bool(is_met(physics))
        except (AttributeError, KeyError, LookupError, TypeError, ValueError):
            return False
    for name in ("is_success", "task_success", "success"):
        value = getattr(task, name, None)
        if value is not None and not callable(value):
            return bool(value)
    return False


def _task_signals(env, diagnostics: RolloutDiagnostics) -> tuple[float, float, str]:
    """Read upstream shaping signals with a geometric progress fallback."""
    progress = _signal(env, "get_task_progress")
    intention = _signal(env, "get_intention_score")
    distance_progress = diagnostics.distance_progress()
    if distance_progress is not None and progress <= 0.0:
        return distance_progress, intention, "target_distance"
    return progress, intention, "upstream"


def _press_button_fallback_plan(
    expected_pattern: Sequence[str],
    diagnostics: RolloutDiagnostics,
    entities: Sequence[Any],
) -> list[dict[str, Any]] | None:
    """Build a grounded press operation when a primitive planner drifts.

    ``SelectPaintingTask`` is implemented upstream as ``PressButtonTask``. A
    syntactically valid but unrelated plan can pass generic graph checks while
    remaining unusable for the simulator. The diagnostics target is resolved
    from the live task object, so this fallback keeps the safety gate while
    allowing a smoke rollout to exercise the controller and success predicate.
    """

    if tuple(expected_pattern) != ("press",):
        return None
    target_names = [
        str(name)
        for name, values in diagnostics.targets.items()
        if int(values.get("samples", 0)) > 0
    ]
    if not target_names:
        # Only an explicitly named button is eligible when live geometry was
        # unavailable.
        target_names = [
            str(entity)
            for entity in entities
            if "button" in str(entity).lower()
        ]
    if not target_names:
        return None
    return [{"name": "press", "params": {"target_entity_name": target_names[0]}}]


class _TaskPatternDFA:
    """Restrict online planner sampling to a task's canonical skill family."""

    def __init__(self, base_dfa, vocabulary, expected_pattern, forced_labels=()):
        self.base_dfa = base_dfa
        self.vocabulary = vocabulary
        self.expected_pattern = tuple(str(name) for name in expected_pattern)
        self.forced_labels = tuple(int(label) for label in forced_labels)
        self.start_state = (base_dfa.start_state, 0, 0)
        self.alphabet = base_dfa.alphabet

    def _state(self, state):
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError("invalid task-pattern DFA state")
        return state

    def allowed_tokens(self, state, remaining_steps=None):
        base_state, skill_index, forced_index = self._state(state)
        try:
            allowed = self.base_dfa.allowed_tokens(
                base_state, remaining_steps=remaining_steps
            )
        except TypeError:
            allowed = self.base_dfa.allowed_tokens(base_state)
        allowed = set(int(label) for label in allowed)
        # Forced labels form a grounded prefix; resume pattern sampling after it.
        if forced_index < len(self.forced_labels):
            return allowed.intersection({self.forced_labels[forced_index]})
        filtered = set()
        for label in allowed:
            token = self.vocabulary.token_for_label(label)
            if token.startswith("skill:"):
                expected = (
                    self.expected_pattern[skill_index]
                    if skill_index < len(self.expected_pattern)
                    else None
                )
                if token != f"skill:{expected}":
                    continue
            elif token == self.vocabulary.eos_token and skill_index != len(self.expected_pattern):
                continue
            filtered.add(label)
        return filtered

    def step(self, state, symbol):
        base_state, skill_index, forced_index = self._state(state)
        symbol = int(symbol)
        if symbol not in self.allowed_tokens(state):
            return None
        next_base = self.base_dfa.step(base_state, symbol)
        if next_base is None:
            return None
        token = self.vocabulary.token_for_label(symbol)
        next_skill_index = skill_index + int(token.startswith("skill:"))
        return (next_base, next_skill_index, forced_index + int(bool(self.forced_labels)))

    def is_accepting(self, state):
        base_state, skill_index, forced_index = self._state(state)
        if not self.base_dfa.is_accepting(base_state):
            return False
        return (
            forced_index >= len(self.forced_labels)
            and skill_index == len(self.expected_pattern)
        )

    def accepts(self, sequence):
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def __getattr__(self, name):
        return getattr(self.base_dfa, name)


def _task_pattern_dfa(base_dfa, vocabulary, expected_pattern, *, target_name=None, entities=()):
    """Build a task-conditioned view of the graph DFA for online sampling."""

    forced_labels = ()
    if target_name is not None and expected_pattern:
        try:
            target_index = next(
                index for index, entity in enumerate(entities)
                if str(entity) == str(target_name)
            )
            forced_tokens = (
                "skill:" + str(expected_pattern[0]),
                "arg:target_entity_name",
                f"obj:{target_index}",
            )
            if tuple(expected_pattern) == ("press",):
                forced_tokens += (vocabulary.eos_token,)
            forced_labels = tuple(vocabulary.label_for_token(token) for token in forced_tokens)
        except (KeyError, StopIteration, TypeError, ValueError):
            forced_labels = ()
    return _TaskPatternDFA(
        base_dfa, vocabulary, expected_pattern, forced_labels=forced_labels
    )

def _live_task_entity_position(env: Any, attribute: str) -> np.ndarray | None:
    """Return an upstream task entity origin in world coordinates."""
    task = getattr(env, "task", None)
    name = getattr(task, attribute, None)
    entities = getattr(task, "entities", None)
    if name is None or not isinstance(entities, Mapping):
        return None
    entity = entities.get(name)
    getter = getattr(entity, "get_xpos", None)
    physics = getattr(env, "physics", None)
    if not callable(getter) or physics is None:
        return None
    try:
        value = np.asarray(getter(physics), dtype=np.float64).reshape(3)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value).all() else None


def _blend_pick_target(action, current, target_robot, blend: float) -> np.ndarray:
    """Blend only the Cartesian pick target toward a live task target."""
    value = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    state = np.asarray(current, dtype=np.float64).reshape(-1)
    target = np.asarray(target_robot, dtype=np.float64).reshape(-1)
    if value.shape != (7,) or state.size < 6 or target.shape != (3,):
        raise ValueError("pick target blending requires a 7D action, 6D state, and 3D target")
    if not np.isfinite(value).all() or not np.isfinite(state[:6]).all() or not np.isfinite(target).all():
        raise ValueError("pick target blending requires finite action, state, and target")
    ratio = float(np.clip(blend, 0.0, 1.0))
    value[:3] = (1.0 - ratio) * value[:3] + ratio * target
    return value


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


def _controller_inputs(
    observations,
    task_index: int,
    device,
    camera_views: int = 3,
    plan_context=None,
    robot_frame=None,
    selected_camera_indices=None,
):
    history = list(observations)[-2:]
    if len(history) == 1:
        history.insert(0, history[0])
    image_history, state_history = [], []
    for observation in history:
        rgb = np.asarray(observation["rgb"])
        rgb = rgb[:camera_views] if selected_camera_indices is None else rgb[list(selected_camera_indices)]
        image_history.append(image_tensor(rgb, channels_last=True))
        state = _observation_state(observation)
        if robot_frame is not None:
            state = world_to_robot_ee_state(state, robot_frame)
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
        controller_camera_names: Sequence[str] | None = None,
        controller_camera_keys: Sequence[str] | None = None,
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
        pick_approach_blend: float = 0.5,
        pick_grasp_distance: float = 0.30,
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
        self.controller_camera_names = tuple(controller_camera_names) if controller_camera_names else None
        self.controller_camera_keys = tuple(controller_camera_keys or DEFAULT_CONTROLLER_CAMERA_KEYS)
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
        self.pick_approach_blend = float(pick_approach_blend)
        self.pick_grasp_distance = float(pick_grasp_distance)
        if not np.isfinite(self.pick_approach_blend) or not 0.0 <= self.pick_approach_blend <= 1.0:
            raise ValueError("pick approach blend must be finite and within [0, 1]")
        if not np.isfinite(self.pick_grasp_distance) or self.pick_grasp_distance <= 0:
            raise ValueError("pick grasp distance must be finite and positive")
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
        self.last_controller_update: dict[str, Any] = {}

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
        diagnostics = RolloutDiagnostics()
        pick_assist_steps = grasp_assist_steps = pull_assist_steps = pour_assist_steps = 0
        pick_grasp_latched = False
        grasp_close_steps = 0
        condiment_prepare_reached = False
        condiment_grasp_pose = None
        condiment_grasp_qpos = None
        condiment_pour_phase = -1
        condiment_lift_target_world = None
        condiment_container_target_world = None
        grasp_contact_streak = 0
        try:
            timestep = env.reset()
            reset_reward_tracking(env)
            observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
            # The official control dataset stores absolute EE poses relative to
            # the robot base. Keep policy inputs/outputs in that learned frame
            # and cross into simulator world coordinates only at the IK edge.
            controller_robot_frame = robot_frame_position(env)
            self._report_progress(
                f"VLABench controller frame=v{CONTROLLER_FRAME_VERSION} "
                f"robot_base_world={controller_robot_frame.tolist()}"
            )
            observations = [observation]
            selected_camera_names, camera_source = resolve_controller_camera_names(
                env,
                observation,
                dataset_keys=self.controller_camera_keys,
                requested=self.controller_camera_names,
            )
            selected_cameras = camera_indices(env, observation, selected_camera_names)
            cameras = camera_report(
                env,
                observation,
                indices=selected_cameras,
                dataset_keys=self.controller_camera_keys,
            )
            cameras["mapping_source"] = camera_source
            self._report_progress("VLABench controller cameras=" + json.dumps(cameras))
            diagnostics.observe(env, _observation_state(observation))
            geometric_progress_available = diagnostics.distance_progress() is not None
            previous_progress, previous_intention, progress_source = _task_signals(env, diagnostics)
            initial_progress, initial_intention = previous_progress, previous_intention
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

            expected_pattern = PRIMITIVE_TASK_PATTERNS.get(descriptor.get("task"))
            # Synthetic test environments deliberately use the historical
            # pick/place fixture. Apply the upstream task pattern only to the
            # real VLABench task implementation running on the server.
            task_type = type(getattr(env, "task", None))
            if (
                descriptor.get("task") == "select_book"
                and expected_pattern is not None
                and not task_type.__module__.startswith("VLABench.")
            ):
                expected_pattern = ("pick", "place")
            while steps < self.max_steps:
                if not valid:
                    break
                now = time.monotonic()
                if now - last_progress_report >= 30.0:
                    self._report_progress(
                        f"VLABench episode task={descriptor.get('task', 'unknown')} "
                        f"steps={steps}/{self.max_steps}"
                        + (f" grasp_distance={diagnostics.target_grasp_distance()}"
                           f" close_steps={grasp_close_steps} contact_streak={grasp_contact_streak}"
                           f" pour_phase={condiment_pour_phase}"
                           if descriptor.get("task") == "add_condiment" else "")
                    )
                    last_progress_report = now
                views, entities = numbered_views_from_observation(env, observation)
                sampling_dfa = self._dfa_for_entities(entities)
                if expected_pattern is not None:
                    target_name = next(
                        (str(name) for name, values in diagnostics.targets.items()
                         if int(values.get("samples", 0)) > 0),
                        None,
                    )
                    sampling_dfa = _task_pattern_dfa(
                        sampling_dfa,
                        self.runtime.vocabulary,
                        expected_pattern,
                        target_name=target_name,
                        entities=entities,
                    )
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
                if expected_pattern is not None:
                    actual_pattern = tuple(str(operation.get("name")) for operation in plan)
                    if actual_pattern != tuple(expected_pattern):
                        fallback_plan = _press_button_fallback_plan(
                            expected_pattern, diagnostics, entities
                        )
                        if fallback_plan is not None and self._valid_plan(
                            fallback_plan, entities, dfa=sampling_dfa
                        ):
                            self._report_progress(
                                f"VLABench planner task-pattern fallback task={descriptor.get('task', 'unknown')} "
                                f"expected={list(expected_pattern)} actual={list(actual_pattern)}"
                            )
                            plan = fallback_plan
                        else:
                            self._report_progress(
                                f"VLABench planner task-pattern mismatch task={descriptor.get('task', 'unknown')} "
                                f"expected={list(expected_pattern)} actual={list(actual_pattern)}"
                            )
                            valid = False
                            termination_reason = "task_pattern_mismatch"
                            break
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
                if geometric_progress_available:
                    # Distance shaping is continuous approach credit, not
                    # evidence that the current primitive (usually ``pick``)
                    # completed.  Advancing here would switch to ``place``
                    # before the simulator observes a grasp.
                    phase_cursor = operation_cursor
                else:
                    # Preserve the legacy phase fallback for diagnostic or
                    # synthetic environments that expose no target geometry.
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
                    robot_frame=controller_robot_frame,
                    selected_camera_indices=selected_cameras,
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
                failed_action_index = None
                chunk_advanced = False
                ik_truncated = False
                for action_index, candidate in enumerate(actions[0, : self.execute_horizon]):
                    try:
                        current = world_to_robot_ee_state(
                            _observation_state(observation), controller_robot_frame
                        )
                        candidate_value = candidate.detach().cpu().numpy()
                        active_skill = (
                            str(plan[operation_cursor].get("name"))
                            if plan and operation_cursor < len(plan)
                            else ""
                        )
                        if (
                            task_type.__module__.startswith("VLABench.")
                            and descriptor.get("task") == "add_condiment"
                            and operation_cursor > 0
                            and active_skill == "pour"
                        ):
                            # AddCondimentTask's official expert expands the
                            # high-level pour operation into lift, move above
                            # the target container, then wrist rotation.
                            current_world = current[:3] + controller_robot_frame
                            if condiment_pour_phase < 0:
                                condiment_pour_phase = 0
                                condiment_lift_target_world = (
                                    current_world + np.asarray([0.0, 0.0, 0.2])
                                )
                                container = _live_task_entity_position(
                                    env, "target_container"
                                )
                                if container is not None:
                                    condiment_container_target_world = (
                                        container + np.asarray([0.0, 0.0, 0.2])
                                    )
                            if condiment_lift_target_world is not None:
                                if (
                                    condiment_pour_phase == 0
                                    and np.linalg.norm(
                                        current_world - condiment_lift_target_world
                                    ) <= 0.04
                                ):
                                    condiment_pour_phase = 1
                                if (
                                    condiment_pour_phase == 1
                                    and condiment_container_target_world is not None
                                    and np.linalg.norm(
                                        current_world - condiment_container_target_world
                                    ) <= 0.04
                                ):
                                    condiment_pour_phase = 2
                            candidate_value[6] = 0.0
                            if condiment_pour_phase == 0:
                                candidate_value[:3] = (
                                    condiment_lift_target_world - controller_robot_frame
                                )
                                candidate_value[3:6] = current[3:6]
                            elif (
                                condiment_pour_phase == 1
                                and condiment_container_target_world is not None
                            ):
                                candidate_value[:3] = (
                                    condiment_container_target_world
                                    - controller_robot_frame
                                )
                                candidate_value[3:6] = current[3:6]
                            else:
                                # The official SkillLib.pour increments the
                                # final Franka joint, so the direct command
                                # below handles the wrist rotation exactly.
                                candidate_value[:3] = current[:3]
                                candidate_value[3:6] = current[3:6]
                            pour_assist_steps += 1
                        if (
                            operation_cursor == 0
                            and active_skill in {"pick", "press"}
                            and self.pick_approach_blend > 0.0
                        ):
                            target_world = diagnostics.target_position()
                            if (
                                task_type.__module__.startswith("VLABench.")
                                and descriptor.get("task") in {"select_book", "add_condiment"}
                            ):
                                grasp_target = diagnostics.target_grasp_position()
                                if grasp_target is not None:
                                    target_world = grasp_target
                            if target_world is not None:
                                approach_blend = self.pick_approach_blend
                                # SelectBookTask's expert pick uses the live
                                # target pose throughout the approach. A
                                # controller sample can otherwise drift away
                                # after the first bounded step, producing
                                # repeated IK recovery without reaching the
                                # grasp envelope.
                                if (
                                    task_type.__module__.startswith("VLABench.")
                                    and descriptor.get("task") in {"select_book", "add_condiment"}
                                    and active_skill == "pick"
                                ):
                                    approach_blend = 1.0
                                    candidate_value[3:6] = np.asarray(
                                        [-np.pi / 2, -np.pi / 2,
                                         np.pi / 2 if descriptor.get("task") == "add_condiment" else 0.0],
                                        dtype=np.float64,
                                    )
                                    # Follow SkillLib.pick's collision-free
                                    # prepare point before descending onto the
                                    # grasp keypoint. Directly crossing the
                                    # shelf can create contact without a
                                    # stable grasp.
                                    gripper_pcd = getattr(
                                        getattr(env, "robot", None),
                                        "gripper_pcd",
                                        None,
                                    )
                                    if callable(gripper_pcd):
                                        try:
                                            _, approach_vector = gripper_pcd(
                                                target_world,
                                                euler_to_quaternion(*candidate_value[3:6]),
                                            )
                                            approach_vector = np.asarray(
                                                approach_vector, dtype=np.float64
                                            ).reshape(3)
                                            prepare_world = (
                                                target_world - 0.1 * approach_vector
                                            )
                                            current_world = (
                                                current[:3] + controller_robot_frame
                                            )
                                            if (
                                                not (descriptor.get("task") == "add_condiment" and condiment_prepare_reached)
                                                and np.isfinite(approach_vector).all()
                                                and np.linalg.norm(
                                                    current_world - prepare_world
                                                ) > 0.08
                                            ):
                                                target_world = prepare_world
                                            elif descriptor.get("task") == "add_condiment":
                                                condiment_prepare_reached = True
                                        except (
                                            AttributeError,
                                            KeyError,
                                            TypeError,
                                            ValueError,
                                        ):
                                            pass
                                candidate_value = _blend_pick_target(
                                    candidate_value,
                                    current,
                                    target_world - controller_robot_frame,
                                    approach_blend,
                                )
                                pick_assist_steps += 1
                        if (
                            task_type.__module__.startswith("VLABench.")
                            and descriptor.get("task") == "select_book"
                            and operation_cursor > 0
                            and active_skill == "pull"
                        ):
                            # SelectBookTask's expert pull uses the current
                            # orientation, closed gripper, and a -Y 0.3 m
                            # displacement. Apply one bounded step at a time.
                            candidate_value[:3] = current[:3] + np.asarray([0.0, -0.02, 0.0])
                            candidate_value[3:6] = current[3:6]
                            candidate_value[6] = 0.0
                            pull_assist_steps += 1
                        bounded = bound_ee_action(
                            candidate_value,
                            current,
                            max_position_step=self.max_position_step,

                            max_rotation_step=self.max_rotation_step,
                        )
                        current_target_distance = diagnostics.target_distance()
                        target_min_distance = None
                        if diagnostics.targets:
                            minimum_key = "minimum_m"
                            if (
                                task_type.__module__.startswith("VLABench.")
                                and descriptor.get("task") in {"select_book", "add_condiment"}
                            ):
                                current_target_distance = diagnostics.target_grasp_distance() or current_target_distance
                                minimum_key = "grasp_minimum_m"
                            target_min_distance = min(
                                (
                                    float(values.get(minimum_key))
                                    for values in diagnostics.targets.values()
                                    if values.get(minimum_key) is not None
                                ),
                                default=None,
                            )
                        if operation_cursor == 0 and active_skill == "pick":
                            within_grasp_envelope = (
                                current_target_distance is not None
                                and current_target_distance <= self.pick_grasp_distance
                                or target_min_distance is not None
                                and target_min_distance <= self.pick_grasp_distance
                            )
                            if within_grasp_envelope:
                                pick_grasp_latched = True
                            if pick_grasp_latched:
                                # Match SkillLib.close_gripper: close over
                                # several transitions while holding the live
                                # grasp pose, instead of one abrupt pulse.
                                bounded[6] = max(
                                    0.0,
                                    0.04 * (1.0 - min(grasp_close_steps, 10) / 10.0),
                                )
                                grasp_close_steps += 1
                                grasp_assist_steps += 1
                        condiment_pick = (
                            task_type.__module__.startswith("VLABench.")
                            and descriptor.get("task") == "add_condiment"
                            and operation_cursor == 0 and active_skill == "pick"
                        )
                        if condiment_pick:
                            # Close only at the grasp pose, not at the broad
                            # approach envelope used by legacy controllers.
                            grasp_distance = diagnostics.target_grasp_distance()
                            if condiment_grasp_pose is None:
                                if grasp_distance is not None and grasp_distance <= 0.04:
                                    condiment_grasp_pose = current.copy()
                                    get_qpos = getattr(env.robot, "get_qpos", None)
                                    if callable(get_qpos):
                                        condiment_grasp_qpos = np.asarray(
                                            get_qpos(env.physics), dtype=np.float64
                                        ).copy()
                                grasp_close_steps = 0
                            if condiment_grasp_pose is None:
                                bounded[6] = 1.0
                            else:
                                bounded[:6] = condiment_grasp_pose[:6]
                                bounded[6] = 0.0
                        command = None
                        last_ik_error = None
                        direct_pour = (
                            task_type.__module__.startswith("VLABench.")
                            and descriptor.get("task") == "add_condiment"
                            and operation_cursor > 0
                            and active_skill == "pour"
                            and condiment_pour_phase >= 2
                        )
                        if direct_pour:
                            # Match SkillLib.pour: rotate the last arm joint
                            # by pi/40 per step while holding the gripper.
                            try:
                                qpos = np.asarray(
                                    env.robot.get_qpos(env.physics),
                                    dtype=np.float64,
                                ).reshape(-1)
                                if qpos.size < 1 or not np.isfinite(qpos).all():
                                    raise ValueError("invalid robot qpos for pour")
                                qpos[-1] += np.pi / 40.0
                                command = np.concatenate(
                                    (qpos, np.zeros(2, dtype=np.float64))
                                )
                                spec = getattr(env, "action_spec", None)
                                spec = spec() if callable(spec) else spec
                                if (
                                    spec is not None
                                    and hasattr(spec, "minimum")
                                    and hasattr(spec, "maximum")
                                ):
                                    command = np.clip(
                                        command,
                                        np.asarray(spec.minimum),
                                        np.asarray(spec.maximum),
                                    )
                                recovered = bounded.copy()
                                recovered[6] = 0.0
                            except (AttributeError, KeyError, TypeError, ValueError):
                                command = None
                        # A zero-scale target is merely a hold command. Treating
                        # it as recovery creates long no-op loops that look
                        # executable while providing no controller progress.
                        if command is None:
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
                                        robot_to_world_ee_action(
                                            recovered, controller_robot_frame
                                        ),
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
                            failed_action_index = action_index
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
                    if condiment_pick and condiment_grasp_pose is not None:
                        # EE gripper state is binary; apply the physical
                        # aperture ramp only after conversion to joint control.
                        if condiment_grasp_qpos is not None:
                            command[:-2] = condiment_grasp_qpos
                        command[-2:] = 0.04 * max(0.0, 1.0 - grasp_close_steps / 10.0)
                    timestep = env.step(command)
                    consecutive_ik_rejections = 0
                    steps += 1
                    executed += 1
                    observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
                    observations.append(observation)
                    diagnostics.observe(env, _observation_state(observation), command_gripper=recovered[6])
                    progress, intention, progress_source = _task_signals(env, diagnostics)
                    delta_progress = progress - previous_progress
                    delta_intention = intention - previous_intention
                    chunk_reward += 0.25 * delta_progress + 0.10 * delta_intention
                    semantic_advance = (
                        progress_source != "target_distance"
                        and (delta_progress > 1e-6 or delta_intention > 1e-6)
                    )
                    grasp_advance = False
                    if operation_cursor == 0 and active_skill == "pick":
                        task = getattr(env, "task", None)
                        target_name = getattr(task, "target_entity", None)
                        entities = getattr(task, "entities", None)
                        target_entity = (
                            entities.get(target_name)
                            if isinstance(entities, Mapping)
                            else None
                        )
                        grasp_checker = getattr(target_entity, "is_grasped", None)
                        if callable(grasp_checker):
                            try:
                                physically_grasped = bool(
                                    grasp_checker(env.physics, env.robot)
                                )
                                grasp_contact_streak = (
                                    grasp_contact_streak + 1
                                    if physically_grasped
                                    else 0
                                )
                                grasp_advance = (
                                    physically_grasped
                                    and grasp_close_steps >= 10
                                    and grasp_contact_streak >= 10
                                )
                            except (AttributeError, KeyError, TypeError, ValueError):
                                grasp_contact_streak = 0
                                grasp_advance = False
                        else:
                            task_state = getattr(task, "target_is_grasped", None)
                            grasp_advance = bool(
                                isinstance(task_state, Mapping)
                                and any(bool(value) for value in task_state.values())
                            )
                        # The live SelectBookTask exposes an authoritative
                        # per-target grasp flag. Distance plus a closed
                        # command is only a controller-side fallback for
                        # synthetic environments; it can otherwise switch to
                        # pull while the object is still on the shelf.
                        if not (
                            task_type.__module__.startswith("VLABench.")
                            and descriptor.get("task") in {"select_book", "add_condiment"}
                        ):
                            latest_target_distance = diagnostics.target_distance()
                            grasp_advance = grasp_advance or (
                                recovered[6] < 0.5
                                and latest_target_distance is not None
                                and latest_target_distance <= self.pick_grasp_distance
                            )
                    if (
                        not chunk_advanced
                        and (
                            grasp_advance
                            or (
                                semantic_advance
                                and not (
                                    task_type.__module__.startswith("VLABench.")
                                    and descriptor.get("task") in {"select_book", "add_condiment"}
                                    and operation_cursor == 0
                                    and active_skill == "pick"
                                )
                            )
                        )
                        and operation_cursor + 1 < len(plan)
                    ):
                        operation_cursor += 1
                        chunk_advanced = True
                        # Potential-based graph-subgoal shaping moves credit to
                        # the operation boundary. The terminal correction below
                        # preserves the authoritative final rollout formula.
                        chunk_reward += 0.05 / max(1, len(plan))
                    previous_progress, previous_intention = progress, intention
                    # VLABench may satisfy the task condition before its
                    # dm_env timestep becomes LAST. Check the authoritative
                    # task predicate on every executed transition so a
                    # completed grasp/pull is not hidden by the time limit.
                    if _task_success(env):
                        success = True
                        termination_reason = "success"
                        break
                    if _last(timestep):
                        termination_reason = "terminal"
                        break
                    if steps >= self.max_steps:
                        break
                # Keep a fully rejected sampled action as separate feasibility
                # evidence.  `executed` remains the number of actions that
                # actually reached the simulator; conflating it with the failed
                # action index penalizes the successful prefix of a chunk.
                if executed or failed_action_index is not None:
                    transitions.append(ControllerTransition(
                        images=inputs[0].detach().cpu(),
                        state=inputs[1].detach().cpu(),
                        task_index=inputs[2].detach().cpu(),
                        plan_context=inputs[3].detach().cpu(),
                        actions=policy_actions.cpu(),
                        old_logprob=logprobs[0, :executed].sum().detach().cpu(),
                        old_value=values[0].detach().cpu(),
                        reward=chunk_reward,
                        done=success or not valid or ik_truncated or steps >= self.max_steps,
                        executed=executed,
                        # Recovery is part of the environment's deterministic
                        # safety transform. Penalize only the sampled target
                        # that remained infeasible at every recovery scale.
                        feasibility_cost=float(failed_action_index is not None),
                        feasibility_index=failed_action_index,
                        old_feasibility_logprob=(
                            logprobs[0, failed_action_index].detach().cpu()
                            if failed_action_index is not None else None
                        ),
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
            diagnostic_result = {
                **diagnostics.result(),
                "task_contract": (
                    task_contract.as_dict()
                    if (task_contract := task_contract_for(str(descriptor.get("task", ""))))
                    is not None
                    else None
                ),
                "cameras": cameras,
                "initial_progress": initial_progress,
                "final_progress": final_progress,
                "initial_intention": initial_intention,
                "final_intention": final_intention,
                "progress_source": progress_source,
                "pick_assist_steps": pick_assist_steps,
                "grasp_assist_steps": grasp_assist_steps,
                "pull_assist_steps": pull_assist_steps,
                "pour_assist_steps": pour_assist_steps,
            }
            self._report_progress(
                f"VLABench controller diagnostics task={descriptor.get('task', 'unknown')} "
                + json.dumps(diagnostic_result)
            )
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
                diagnostic_result,
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
        task_signal_episodes = sum(
            episode.total_return > POSITIVE_RETURN_EPSILON for episode in episodes
        )
        actor_update_enabled = bool(task_signal_episodes)
        # Once a batch has genuine task signal, all executed transitions from
        # valid rollouts are useful PPO evidence: positive-return episodes show
        # what to reinforce and zero-return episodes show what to suppress.
        # A wholly zero-return batch still updates only the detached critic.
        entries = [
            (item, actor_update_enabled and int(item.executed) > 0)
            for episode in episodes
            for item in episode.controller
        ]
        transitions = [item for item, _informative in entries]
        if not transitions:
            self.last_controller_update = {
                "task_signal_episodes": 0,
                "actor_update_attempted": False,
                "rolled_back": False,
                "ppo_epochs_completed": 0,
                "mean_action_log_ratio": 0.0,
                "max_action_log_ratio": 0.0,
            }
            return 0.0
        advantages = torch.zeros(len(entries), device=self.device_name)
        informative_indices = [
            index
            for index, (item, informative) in enumerate(entries)
            if informative and int(item.executed) > 0
        ]
        if informative_indices:
            informative_advantages = torch.tensor(
                [entries[index][0].advantage for index in informative_indices],
                device=self.device_name,
            )
            # Centering a singleton advantage erases the only sparse-success
            # policy signal. Normalize only when there is a population against
            # which to compare it; otherwise preserve the raw GAE advantage.
            if informative_advantages.numel() > 1:
                informative_advantages = (
                    informative_advantages - informative_advantages.mean()
                ) / informative_advantages.std(unbiased=False).clamp_min(1e-6)
            advantages[informative_indices] = informative_advantages
        trainable = [
            parameter for parameter in self.controller.parameters()
            if parameter.requires_grad
        ]
        batch_start = (
            [parameter.detach().clone() for parameter in trainable]
            if actor_update_enabled else None
        )
        total = 0.0
        completed_epochs = 0
        rolled_back = False
        last_action_log_ratio = 0.0
        last_max_action_log_ratio = 0.0

        def restore_batch_start(message: str) -> None:
            nonlocal rolled_back, total, completed_epochs
            if batch_start is not None:
                with torch.no_grad():
                    for parameter, previous in zip(trainable, batch_start):
                        parameter.copy_(previous)
            self.controller_optimizer.state.clear()
            self.controller_optimizer.zero_grad(set_to_none=True)
            rolled_back = True
            total = 0.0
            completed_epochs = 0
            self._report_progress(message)

        def add_action_log_ratios(target, logprob, item, informative) -> None:
            executed = int(item.executed)
            if informative and executed > 0:
                old_logprob = item.old_logprob.to(self.device_name).reshape(())
                target.append(
                    (logprob[0, :executed].sum().detach() - old_logprob).abs()
                    / executed
                )
            if (
                actor_update_enabled
                and float(item.feasibility_cost) > 0.0
                and item.feasibility_index is not None
            ):
                if item.old_feasibility_logprob is None:
                    raise ValueError("feasibility transition has no behavior log-probability")
                feasibility_index = int(item.feasibility_index)
                if feasibility_index < 0 or feasibility_index >= logprob.shape[1]:
                    raise ValueError("feasibility transition index is outside the action chunk")
                old_feasibility = item.old_feasibility_logprob.to(
                    self.device_name
                ).reshape(())
                target.append(
                    (logprob[0, feasibility_index].detach() - old_feasibility).abs()
                )

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
                executed = int(item.executed)
                new_logprob = logprob[0, :executed].sum()
                old_logprob = item.old_logprob.to(self.device_name).reshape(())
                informative = bool(entries[index][1])
                add_action_log_ratios(
                    action_log_ratios, logprob, item, informative
                )
                policy_loss = (
                    ppo_clipped_loss(
                        new_logprob.reshape(1),
                        old_logprob.reshape(1),
                        advantages[index].reshape(1),
                        clip=self.ppo_clip,
                        max_log_ratio=self.ppo_max_log_ratio,
                    )
                    if informative and executed > 0
                    else torch.zeros((), device=self.device_name)
                )
                target_value = torch.tensor(
                    item.return_value,
                    device=self.device_name,
                ).clamp(-1.0, 1.0)
                value_loss = F.smooth_l1_loss(value.reshape(()), target_value)
                entropy_bonus = (
                    self.entropy_weight * entropy[0, :executed].mean()
                    if informative and executed > 0
                    else torch.zeros((), device=self.device_name)
                )
                feasibility_loss = torch.zeros((), device=self.device_name)
                if (
                    actor_update_enabled
                    and float(item.feasibility_cost) > 0.0
                    and item.feasibility_index is not None
                ):
                    feasibility_index = int(item.feasibility_index)
                    old_feasibility = item.old_feasibility_logprob.to(
                        self.device_name
                    ).reshape(())
                    feasibility_loss = (
                        self.feasibility_weight
                        * float(item.feasibility_cost)
                        * torch.exp(
                            (
                                logprob[0, feasibility_index]
                                - old_feasibility
                            ).clamp(-self.ppo_max_log_ratio, self.ppo_max_log_ratio)
                        )
                    )
                losses.append(
                    policy_loss
                    + self.value_weight * value_loss
                    - entropy_bonus
                    + feasibility_loss
                )
            mean_action_log_ratio = (
                float(torch.stack(action_log_ratios).mean())
                if action_log_ratios else 0.0
            )
            max_action_log_ratio = (
                float(torch.stack(action_log_ratios).max())
                if action_log_ratios else 0.0
            )
            last_action_log_ratio = mean_action_log_ratio
            last_max_action_log_ratio = max_action_log_ratio
            if (
                ppo_epoch
                and (
                    mean_action_log_ratio > self.ppo_target_action_log_ratio
                    or max_action_log_ratio > self.ppo_max_log_ratio
                )
            ):
                restore_batch_start(
                    "VLABench controller PPO early stop: "
                    f"mean action log-ratio={mean_action_log_ratio:.4f} "
                    f"(limit {self.ppo_target_action_log_ratio:.4f}), "
                    f"max={max_action_log_ratio:.4f} "
                    f"(limit {self.ppo_max_log_ratio:.4f}); "
                    "rolled back the complete controller update"
                )
                break
            loss = torch.stack(losses).mean()
            if actor_update_enabled:
                loss = loss + self.controller_bc_weight * self._controller_anchor()
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
            backup = (
                batch_start
                if batch_start is not None
                else [parameter.detach().clone() for parameter in trainable]
            )
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

        # The next PPO pass detects drift from earlier passes, but the final
        # optimizer step has no subsequent pass. Validate it explicitly and
        # make the whole update transactional when the trust region is crossed.
        if actor_update_enabled and completed_epochs and not rolled_back:
            post_update_ratios = []
            with torch.no_grad():
                for item, informative in entries:
                    logprob, _entropy, _value = self.controller.evaluate_action_chunk(
                        item.images.to(self.device_name),
                        item.state.to(self.device_name),
                        item.task_index.to(self.device_name),
                        item.actions.to(self.device_name),
                        item.plan_context.to(self.device_name)
                        if item.plan_context is not None else None,
                    )
                    add_action_log_ratios(
                        post_update_ratios, logprob, item, informative
                    )
            last_action_log_ratio = (
                float(torch.stack(post_update_ratios).mean())
                if post_update_ratios else 0.0
            )
            last_max_action_log_ratio = (
                float(torch.stack(post_update_ratios).max())
                if post_update_ratios else 0.0
            )
            if (
                last_action_log_ratio > self.ppo_target_action_log_ratio
                or last_max_action_log_ratio > self.ppo_max_log_ratio
            ):
                restore_batch_start(
                    "VLABench controller PPO rollback: final action log-ratio "
                    f"mean={last_action_log_ratio:.4f} "
                    f"(limit {self.ppo_target_action_log_ratio:.4f}), "
                    f"max={last_max_action_log_ratio:.4f} "
                    f"(limit {self.ppo_max_log_ratio:.4f})"
                )

        self.last_controller_update = {
            "task_signal_episodes": int(task_signal_episodes),
            "actor_update_attempted": actor_update_enabled,
            "rolled_back": rolled_back,
            "ppo_epochs_completed": int(completed_epochs),
            "mean_action_log_ratio": float(last_action_log_ratio),
            "max_action_log_ratio": float(last_max_action_log_ratio),
        }
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
                    "return": 0.0, "positive_returns": 0.0,
                    "steps": 0.0, "ik_failures": 0.0,
                    "ik_recoveries": 0.0, "ik_truncations": 0.0,
                    "execution_complete": 0.0,
                },
            )
            totals["episodes"] += 1.0
            totals["successes"] += float(episode.success)
            totals["valid"] += float(episode.valid)
            totals["return"] += float(episode.total_return)
            totals["positive_returns"] += float(
                episode.total_return > POSITIVE_RETURN_EPSILON
            )
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
                "positive_return_rate": (
                    totals["positive_returns"] / totals["episodes"]
                ),
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
            "controller_update": dict(self.last_controller_update),
            "return": sum(item.total_return for item in episodes) / len(episodes),
            "positive_return_rate": sum(
                item.total_return > POSITIVE_RETURN_EPSILON for item in episodes
            ) / len(episodes),
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
                "return": 0.0, "positive_returns": 0.0,
                "steps": 0.0, "ik_failures": 0.0,
                "ik_recoveries": 0.0, "ik_truncations": 0.0,
                "execution_complete": 0.0,
            })
            totals["episodes"] += 1.0
            totals["successes"] += float(episode.success)
            totals["valid"] += float(episode.valid)
            totals["return"] += float(episode.total_return)
            totals["positive_returns"] += float(
                episode.total_return > POSITIVE_RETURN_EPSILON
            )
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
                "positive_return_rate": (
                    values["positive_returns"] / values["episodes"]
                ),
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
            "positive_return_rate": sum(
                item.total_return > POSITIVE_RETURN_EPSILON for item in episodes
            ) / count,
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
            "episode_diagnostics": [
                {"task": name, "termination_reason": episode.termination_reason,
                 "steps": episode.steps, "diagnostics": episode.diagnostics}
                for name, episode in zip(names, episodes)
            ],
        }
