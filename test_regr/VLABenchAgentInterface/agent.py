"""Runtime composition of the constrained planner and continuous controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

try:
    from .environment import ee_action_to_env_action, numbered_views_from_observation
    from .graph import dfa_accepts_plan
    from .reward import RewardBreakdown, RolloutRewardAccumulator
    from .training import PlannerConstraintRuntime
    from .world_graph import condition_index_for_pattern, materialize_plan, split_subtasks, validate_plan, verify_plan_constraints
except ImportError:
    from environment import ee_action_to_env_action, numbered_views_from_observation
    from graph import dfa_accepts_plan
    from reward import RewardBreakdown, RolloutRewardAccumulator
    from training import PlannerConstraintRuntime
    from world_graph import condition_index_for_pattern, materialize_plan, split_subtasks, validate_plan, verify_plan_constraints


@dataclass(frozen=True)
class PlanDecision:
    plan: tuple[dict[str, Any], ...]
    raw_output: str
    valid: bool
    errors: tuple[str, ...]
    dfa_valid: bool
    constraint_score: float | None


@dataclass(frozen=True)
class RolloutResult:
    reward: RewardBreakdown
    plans: tuple[PlanDecision, ...]
    actions: tuple[np.ndarray, ...]


def _truthy_last(timestep: Any) -> bool:
    last = getattr(timestep, "last", None)
    return bool(last()) if callable(last) else bool(last)


class HierarchicalVLABenchAgent:
    """Constrained skill replanning around a receding-horizon controller."""

    def __init__(
        self,
        planner: Any,
        controller: torch.nn.Module,
        runtime: PlannerConstraintRuntime,
        *,
        device: str | torch.device = "cpu",
        execute_horizon: int = 4,
        camera_views: int = 3,
    ):
        self.planner = planner
        self.controller = controller
        self.runtime = runtime
        self.device = torch.device(device)
        self.execute_horizon = int(execute_horizon)
        self.camera_views = int(camera_views)

    def plan(self, instruction: str, images: Sequence[Image.Image], entity_table: Sequence[str]) -> PlanDecision:
        raw = self.planner.generate_plan(
            instruction=instruction,
            images=images,
            entity_table=entity_table,
            dfa=self.runtime.dfa,
            world=self.runtime.world_bundle,
            max_steps=self.runtime.max_tokens,
        )
        validation = validate_plan(
            raw,
            entity_table=entity_table,
            skill_arguments=self.runtime.vocabulary.skill_argument_map,
        )
        plan = list(validation.canonical_plan)
        dfa_valid = bool(plan) and dfa_accepts_plan(
            self.runtime.dfa,
            self.runtime.generation_bundle,
            plan,
            entity_table,
            world=self.runtime.world_bundle,
        )
        constraint_score = None
        errors = list(validation.errors)
        if plan:
            try:
                root = materialize_plan(plan, entity_table, self.runtime.world_bundle)
                evaluation = verify_plan_constraints(root, self.runtime.world_bundle)
                constraint_score = None if evaluation is None else evaluation.score
                if constraint_score is not None and constraint_score < 1.0:
                    errors.append(f"DomiKnowS plan constraint score={constraint_score:.6f}")
            except Exception as exc:
                constraint_score = 0.0
                errors.append(str(exc))
        if not dfa_valid:
            errors.append("symbolic plan is rejected by the DomiKnowS generation DFA")
        valid = validation.valid and dfa_valid and (constraint_score is None or constraint_score >= 1.0)
        return PlanDecision(tuple(plan), str(raw), valid, tuple(errors), dfa_valid, constraint_score)

    def _controller_input(self, observation_history: Sequence[Mapping[str, Any]], task_index: int):
        histories = list(observation_history)[-2:]
        if len(histories) == 1:
            histories.insert(0, histories[0])
        image_history = []
        state_history = []
        for observation in histories:
            rgb = np.asarray(observation["rgb"])[: self.camera_views]
            images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
            image_history.append(images)
            state = np.asarray(observation.get("q_state", observation.get("ee_state"))).reshape(-1)[:7]
            state_history.append(torch.as_tensor(state, dtype=torch.float32))
        return (
            torch.stack(image_history).unsqueeze(0).to(self.device),
            torch.stack(state_history).unsqueeze(0).to(self.device),
            torch.tensor([task_index], dtype=torch.long, device=self.device),
        )

    @torch.no_grad()
    def action_chunk(self, observation_history: Sequence[Mapping[str, Any]], plan: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        subtasks = split_subtasks([str(operation["name"]) for operation in plan])
        condition_index = condition_index_for_pattern(subtasks[0]) if subtasks else 0
        inputs = self._controller_input(observation_history, condition_index)
        return self.controller.predict_action_chunk(*inputs)[0]

    @staticmethod
    def _safe_action(env: Any, action: torch.Tensor) -> tuple[np.ndarray, bool]:
        value = action.detach().float().cpu().numpy()
        try:
            return ee_action_to_env_action(env, value), True
        except (TypeError, ValueError, AttributeError):
            return np.zeros_like(value), False

    def rollout(self, env: Any, instruction: str, *, max_steps: int = 400) -> RolloutResult:
        timestep = env.reset()
        observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
        history = [observation]
        reward = RolloutRewardAccumulator(max_steps)
        plans: list[PlanDecision] = []
        actions: list[np.ndarray] = []
        valid = True
        success = False

        while reward.steps < max_steps:
            numbered, entities = numbered_views_from_observation(env, observation, max_views=self.camera_views)
            decision = self.plan(instruction, numbered, entities)
            plans.append(decision)
            for image in numbered:
                image.close()
            if not decision.valid or not decision.plan:
                valid = False
                break
            # Receding-horizon execution: use the next high-level skill and
            # obtain a fresh plan after four low-level controls.
            chunk = self.action_chunk(history, decision.plan)
            for candidate in chunk[: self.execute_horizon]:
                action, action_valid = self._safe_action(env, candidate)
                valid = valid and action_valid
                if not action_valid:
                    break
                timestep = env.step(action)
                actions.append(action)
                observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
                history.append(observation)
                progress = env.get_task_progress() if hasattr(env, "get_task_progress") else 0.0
                intention = env.get_intention_score(threshold=0.1) if hasattr(env, "get_intention_score") else 0.0
                reward.update(progress=progress, intention=intention, valid=valid)
                if _truthy_last(timestep):
                    success = True
                    break
            if success or not valid:
                break

        final_progress = env.get_task_progress() if hasattr(env, "get_task_progress") else reward.progress
        final_intention = env.get_intention_score(threshold=0.1) if hasattr(env, "get_intention_score") else reward.intention
        return RolloutResult(
            reward.finalize(success, progress=final_progress, intention=final_intention, valid=valid),
            tuple(plans),
            tuple(actions),
        )
