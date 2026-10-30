"""Runtime composition of the constrained planner and continuous controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

try:
    from .environment import ee_action_to_env_action, numbered_views_from_observation
    from ..generation_runtime import dfa_accepts_plan
    from ..reward import RewardBreakdown
    from .rollout_reward import RolloutRewardAccumulator
    from ..training import PlannerConstraintRuntime
    from ..world_graph import condition_index_for_pattern, controller_plan_context, materialize_plan, split_subtasks, validate_plan, verify_plan_constraints
except ImportError:
    from environment import ee_action_to_env_action, numbered_views_from_observation
    from generation_runtime import dfa_accepts_plan
    from reward import RewardBreakdown, RolloutRewardAccumulator
    from training import PlannerConstraintRuntime
    from world_graph import condition_index_for_pattern, controller_plan_context, materialize_plan, split_subtasks, validate_plan, verify_plan_constraints


@dataclass(frozen=True)
class PlanDecision:
    plan: tuple[dict[str, Any], ...]
    raw_output: str
    valid: bool
    errors: tuple[str, ...]
    dfa_valid: bool
    constraint_score: float | None
    grounding_report: Mapping[str, Any] | None = None
    status: str = "accepted"
    research_report: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RolloutResult:
    reward: RewardBreakdown
    plans: tuple[PlanDecision, ...]
    actions: tuple[np.ndarray, ...]
    measured_task_success: bool | None = None


def _truthy_last(timestep: Any) -> bool:
    last = getattr(timestep, "last", None)
    return bool(last()) if callable(last) else bool(last)


def _signal(env: Any, name: str) -> float:
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
    for attempt in attempts:
        try:
            value = float(attempt())
            return value if np.isfinite(value) else 0.0
        except (AttributeError, KeyError, LookupError, TypeError, ValueError, ZeroDivisionError):
            continue
    return 0.0


def _task_success(env: Any) -> bool:
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
        controller_task_instructions=None,
    ):
        self.planner = planner
        self.controller = controller
        self.runtime = runtime
        self.device = torch.device(device)
        self.execute_horizon = int(execute_horizon)
        self.camera_views = int(camera_views)
        self.controller_task_instructions=controller_task_instructions

    def plan(self, instruction: str, images: Sequence[Image.Image], entity_table: Sequence[str],
             *, task_context: Mapping[str, Any] | None = None) -> PlanDecision:
        context = {**(task_context or {}), "instruction": instruction, "entity_table": entity_table, "entities": entity_table}
        from ..research.runtime import admission
        from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
        preflight=admission(context,self.runtime.world_bundle)
        if not preflight.accepted:
            return PlanDecision((),"",False,(preflight.status,),False,None,status=preflight.status,research_report=preflight.as_dict())
        try:
            policy = self.runtime.dfa_for("vlabench", context)
        except SemanticSearchLimitExceeded:
            return PlanDecision((),"",False,("budget_exhausted",),False,None,status="budget_exhausted")
        try:
            raw = self.planner.generate_plan(instruction=instruction,images=images,entity_table=entity_table,
                dfa=policy,world=self.runtime.world_bundle,max_steps=self.runtime.max_tokens)
        except SemanticSearchLimitExceeded:
            return PlanDecision((),"",False,("budget_exhausted",),False,None,status="budget_exhausted")
        except ValueError as error:
            from ..research.runtime import blocked_admission,config_for
            if not config_for(context,self.runtime.world_bundle).enabled:raise
            blocked=blocked_admission(context,self.runtime.world_bundle)
            return PlanDecision((),"",False,(str(error),),False,None,status=blocked.status,research_report=blocked.as_dict())
        validation = validate_plan(
            raw,
            entity_table=entity_table,
            skill_arguments=self.runtime.vocabulary.skill_argument_map,
            policy=policy,
        )
        plan = list(validation.canonical_plan)
        dfa_valid = validation.valid and dfa_accepts_plan(
            policy,
            self.runtime.generation_bundle,
            plan,
            entity_table,
            world=self.runtime.world_bundle,
        )
        constraint_score = None
        grounding = None
        errors = list(validation.errors)
        if validation.valid:
            try:
                root = materialize_plan(plan, entity_table, self.runtime.world_bundle, policy=policy)
                evaluation = verify_plan_constraints(root, self.runtime.world_bundle)
                constraint_score = None if evaluation is None else evaluation.score
                grounding = root.attributes.get("grounding_report")
                if constraint_score is not None and constraint_score < 1.0:
                    errors.append(f"DomiKnowS plan constraint score={constraint_score:.6f}")
            except Exception as exc:
                constraint_score = 0.0
                errors.append(str(exc))
        if not dfa_valid:
            errors.append("symbolic plan is rejected by the DomiKnowS generation DFA")
        valid = validation.valid and dfa_valid and (constraint_score is None or constraint_score >= 1.0)
        research=admission(context,self.runtime.world_bundle,operation=plan[0]) if plan else preflight
        if not research.accepted:valid=False
        return PlanDecision(tuple(plan), str(raw), valid, tuple(errors), dfa_valid, constraint_score, grounding,
                            research.status if not research.accepted else "accepted" if valid else "infeasible",research.as_dict())

    def _controller_input(self, observation_history: Sequence[Mapping[str, Any]], task_index: int, plan_context=(0, 0, 0)):
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
            torch.tensor([plan_context], dtype=torch.long, device=self.device),
        )

    @torch.no_grad()
    def action_chunk(self, observation_history: Sequence[Mapping[str, Any]], plan: Sequence[Mapping[str, Any]],*,entities=None,instruction=None,env=None) -> torch.Tensor:
        subtasks = split_subtasks([str(operation["name"]) for operation in plan])
        condition_index = condition_index_for_pattern(subtasks[0]) if subtasks else 0
        plan_context=controller_plan_context(plan,0)
        if entities is not None:
            from ..research.runtime import controller_context
            plan_context=controller_context(plan,0,entities)
            if self.controller_task_instructions is None:raise ValueError("Research controller needs its trained instruction vocabulary")
            from ..dataset import control_task_index_for_instruction
            condition_index=control_task_index_for_instruction(instruction,self.controller_task_instructions)
        if entities is not None:
            if env is None:raise ValueError("Research controller requires its observed robot frame and camera mapping")
            from .program import _controller_inputs
            from .environment import robot_frame_position
            from ..observations import resolve_controller_camera_names,camera_indices
            names,_=resolve_controller_camera_names(env,observation_history[-1])
            inputs=_controller_inputs(observation_history,condition_index,self.device,plan_context=plan_context,
                robot_frame=robot_frame_position(env),selected_camera_indices=camera_indices(env,observation_history[-1],names))
        else:inputs = self._controller_input(observation_history,condition_index,plan_context)
        return self.controller.predict_action_chunk(*inputs)[0]

    @staticmethod
    def _safe_action(env: Any, action: torch.Tensor) -> tuple[np.ndarray, bool]:
        value = action.detach().float().cpu().numpy()
        try:
            return ee_action_to_env_action(env, value), True
        except (TypeError, ValueError, AttributeError):
            return np.zeros_like(value), False

    def rollout(self, env: Any, instruction: str, *, max_steps: int = 400,
                task_context: Mapping[str, Any] | None = None) -> RolloutResult:
        timestep = env.reset()
        observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
        history = [observation]
        reward = RolloutRewardAccumulator(max_steps)
        plans: list[PlanDecision] = []
        actions: list[np.ndarray] = []
        valid = True
        success = False
        from ..research.runtime import reconcile,admit_with_refresh,geometry_admission,config_for,record_measured_completion
        from ..research.sensors import attach_sensor_adapter
        research_context={**(task_context or {}),"research_config":config_for(task_context,self.runtime.world_bundle).as_dict()}
        attach_sensor_adapter(env,research_context)
        research_context=reconcile(env,research_context,step=0,changed=("reset",),observation=observation)
        planning_refresh_steps=set()

        while reward.steps < max_steps:
            research_context=reconcile(env,research_context,step=reward.steps,observation=observation)
            numbered, entities = numbered_views_from_observation(env, observation, max_views=self.camera_views)
            from ..articulation import refreshed_articulation_context
            fresh_context = refreshed_articulation_context(env, research_context, entities)
            fresh_context={**fresh_context,"entities":entities,"entity_table":entities}
            fresh_context,_=admit_with_refresh(env,fresh_context,self.runtime.world_bundle,step=reward.steps)
            research_context=fresh_context
            decision = self.plan(instruction, numbered, entities, task_context=fresh_context)
            plans.append(decision)
            for image in numbered:
                image.close()
            if not decision.valid:
                if decision.status in ("needs_observation","needs_clarification") and reward.steps not in planning_refresh_steps:
                    planning_refresh_steps.add(reward.steps)
                    if hasattr(env,"get_observation"):observation=env.get_observation(require_pcd=False)
                    research_context=reconcile(env,research_context,step=reward.steps,observation=observation)
                    history.append(observation)
                    continue
                valid = False
                break
            if not decision.plan:
                success = _task_success(env)
                break
            # Receding-horizon execution: use the next high-level skill and
            # obtain a fresh plan after four low-level controls.
            if config_for(research_context,self.runtime.world_bundle).enabled or research_context.get("research_matched"):
                chunk=self.action_chunk(history,decision.plan,entities=entities,instruction=instruction,env=env)
            else:chunk = self.action_chunk(history, decision.plan)
            fresh_context,admitted=admit_with_refresh(env,fresh_context,self.runtime.world_bundle,
                operation=decision.plan[0],step=reward.steps,execution=True)
            if not admitted.accepted:
                valid=False;break
            research_context=fresh_context
            for candidate in chunk[: self.execute_horizon]:
                fresh_context=research_context
                fresh_context,admitted=admit_with_refresh(env,fresh_context,self.runtime.world_bundle,
                    operation=decision.plan[0],step=reward.steps,execution=True)
                if not admitted.accepted:
                    valid=False;break
                if config_for(research_context,self.runtime.world_bundle).enabled or research_context.get("research_matched"):
                    from .environment import robot_to_world_ee_action,robot_frame_position
                    candidate=torch.as_tensor(robot_to_world_ee_action(candidate.detach().cpu().numpy(),robot_frame_position(env)),device=candidate.device)
                action, action_valid = self._safe_action(env, candidate)
                valid = valid and action_valid
                if not action_valid:
                    break
                fresh_context,geometry=geometry_admission(env,fresh_context,self.runtime.world_bundle,action)
                if not geometry.accepted:
                    valid=False;break
                timestep = env.step(action)
                actions.append(action)
                observation = env.get_observation(require_pcd=False) if hasattr(env, "get_observation") else timestep.observation
                history.append(observation)
                research_context=reconcile(env,fresh_context,step=reward.steps+1,changed=("robot","camera"),observation=observation)
                completed=False
                if config_for(research_context,self.runtime.world_bundle).enabled:
                    research_context,completed=record_measured_completion(research_context,decision.plan[0])
                progress = _signal(env, "get_task_progress")
                intention = _signal(env, "get_intention_score")
                reward.update(progress=progress, intention=intention, valid=valid)
                if (_truthy_last(timestep) or config_for(research_context,self.runtime.world_bundle).enabled or research_context.get("research_matched")) and _task_success(env):
                    success = True
                    break
                if completed:break
            if success or not valid:
                break

        final_progress = _signal(env, "get_task_progress")
        from ..research.runtime import admission
        terminal=admission(research_context,self.runtime.world_bundle,completion=True,execution=True)
        success=success and terminal.accepted
        valid=valid and terminal.accepted
        final_intention = _signal(env, "get_intention_score")
        return RolloutResult(
            reward.finalize(success, progress=final_progress, intention=final_intention, valid=valid),
            tuple(plans),
            tuple(actions),
            _task_success(env),
        )
