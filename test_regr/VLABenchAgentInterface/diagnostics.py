"""Controller diagnostics; task success remains the simulator's responsibility."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch


AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def wrapped_pose_delta(pose, previous):
    delta = pose - previous
    return torch.cat((delta[..., :3], torch.atan2(
        torch.sin(delta[..., 3:6]), torch.cos(delta[..., 3:6])
    )), dim=-1)


class ControllerMetrics:
    """Sample-weighted errors, including raw (unclipped) demonstration deltas."""

    def __init__(self, scale=None):
        self.scale = torch.as_tensor(
            (0.02,) * 3 + (0.1,) * 3 if scale is None else scale, dtype=torch.float64,
        ).cpu()
        if self.scale.shape != (6,) or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("pose step scale must contain six finite positive values")
        self.count = self.first_count = 0
        self.sums = {name: torch.zeros(6, dtype=torch.float64) for name in (
            "pose_mae", "delta_mae", "normalized_delta_mae", "first_normalized_delta_mae",
            "hold_normalized_delta_mae", "target_step_exceedance_rate",
        )}
        self.confusion = torch.zeros(2, 2, dtype=torch.int64)
        self.transitions = Counter()
        self.raw_pose_error = 0.0
        self.image_min, self.image_max = float("inf"), float("-inf")

    def update(self, prediction, target, state, images):
        prediction, target, state = [value.detach().double().cpu() for value in (prediction, target, state)]
        if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 7:
            raise ValueError("controller metrics require matching [batch, horizon, 7] actions")
        if not all(torch.isfinite(value).all() for value in (prediction, target, state)):
            raise ValueError("controller metrics received non-finite poses")
        base = state[:, -1:, :6]
        pd = wrapped_pose_delta(prediction[..., :6], torch.cat((base, prediction[:, :-1, :6]), 1))
        td = wrapped_pose_delta(target[..., :6], torch.cat((base, target[:, :-1, :6]), 1))
        error = torch.abs(pd - td)
        self.count += target.shape[0] * target.shape[1]
        self.first_count += target.shape[0]
        self.raw_pose_error += float((prediction[..., :6] - target[..., :6]).abs().sum())
        self.sums["pose_mae"] += wrapped_pose_delta(prediction[..., :6], target[..., :6]).abs().sum((0, 1))
        self.sums["delta_mae"] += error.sum((0, 1))
        self.sums["normalized_delta_mae"] += (error / self.scale).sum((0, 1))
        self.sums["first_normalized_delta_mae"] += (error[:, 0] / self.scale).sum(0)
        self.sums["hold_normalized_delta_mae"] += (td.abs() / self.scale).sum((0, 1))
        self.sums["target_step_exceedance_rate"] += (td.abs() > self.scale).sum((0, 1))
        predicted_grip, target_grip = prediction[..., 6] >= 0, target[..., 6] >= 0.5
        for truth in (0, 1):
            for predicted in (0, 1):
                self.confusion[truth, predicted] += ((target_grip == truth) & (predicted_grip == predicted)).sum()
        initial_grip = state[:, -1:, 6] >= 0.5
        true_previous = torch.cat((initial_grip, target_grip[:, :-1]), 1)
        predicted_previous = torch.cat((initial_grip, predicted_grip[:, :-1]), 1)
        actual_transition = target_grip != true_previous
        predicted_transition = predicted_grip != predicted_previous
        for name, mask in (("target", actual_transition), ("predicted", predicted_transition),
                           ("matched", actual_transition & predicted_transition & (predicted_grip == target_grip))):
            self.transitions[name] += int(mask.sum())
        self.image_min = min(self.image_min, float(images.min()))
        self.image_max = max(self.image_max, float(images.max()))

    def result(self) -> dict[str, Any]:
        def ratio(numerator, denominator):
            return float(numerator / denominator) if denominator else None

        classes = {}
        for index, name in enumerate(("closed", "open")):
            classes[name] = {
                "support": int(self.confusion[index].sum()),
                "recall": ratio(self.confusion[index, index], self.confusion[index].sum()),
                "precision": ratio(self.confusion[index, index], self.confusion[:, index].sum()),
            }
        recalls = [item["recall"] for item in classes.values() if item["recall"] is not None]
        return {
            # Preserve historical fields, documenting the old mixed-unit metric.
            "pose_mae": self.raw_pose_error / max(1, 6 * self.count),
            "gripper_accuracy": float(self.confusion.trace()) / max(1, self.count),
            "action_samples": self.count,
            "axis_metrics": {
                name: dict(zip(AXES, (values / max(1, self.first_count if name.startswith("first_") else self.count)).tolist()))
                for name, values in self.sums.items()
            },
            "pose_step_scale": dict(zip(AXES, self.scale.tolist())),
            "gripper": {
                "confusion_true_predicted_closed_open": self.confusion.tolist(),
                "classes": classes,
                "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
                "majority_baseline_accuracy": ratio(int(self.confusion.sum(1).max()), self.count),
                "transitions": {
                    **dict(self.transitions),
                    "precision": ratio(self.transitions["matched"], self.transitions["predicted"]),
                    "recall": ratio(self.transitions["matched"], self.transitions["target"]),
                },
            },
            "image_range": [self.image_min, self.image_max] if self.count else None,
        }


class RolloutDiagnostics:
    """Measure motion toward task targets independently of the planner's choice."""

    def __init__(self):
        self.targets = {}
        self.counts = Counter()
        self.previous_grip = self.previous_command = self.previous_xyz = None
        self.path_length = 0.0

    def observe(self, env, world_state, *, command_gripper=None):
        state = np.asarray(world_state)
        grip = bool(state[6] >= 0.5)
        self.counts["observations"] += 1
        self.counts["observed_open"] += int(grip)
        if self.previous_grip is not None:
            self.counts["observed_gripper_transitions"] += int(grip != self.previous_grip)
        if self.previous_xyz is not None:
            self.path_length += float(np.linalg.norm(state[:3] - self.previous_xyz))
        self.previous_grip, self.previous_xyz = grip, state[:3].copy()
        if command_gripper is not None:
            command = bool(command_gripper >= 0.5)
            self.counts["commands"] += 1
            self.counts["commanded_open"] += int(command)
            # Include the first command's transition from the initial observed state.
            if self.previous_command is not None:
                self.counts["commanded_gripper_transitions"] += int(command != self.previous_command)
            self.previous_command = command
        elif self.previous_command is None:
            self.previous_command = grip
        task = getattr(env, "task", None)
        # PressButtonTask stores the semantic target style in target_entity
        # but evaluates success against target_button. Prefer the actionable
        # button whenever the task exposes it; otherwise use the normal object
        # target for manipulation tasks.
        targets = getattr(task, "target_button", None)
        if targets is None:
            targets = getattr(task, "target_entity", ())
        if isinstance(targets, str):
            targets = (targets,)
        elif targets is None:
            targets = ()
        elif not isinstance(targets, (list, tuple, set, frozenset)):
            targets = (targets,)
        entities = getattr(task, "entities", {})

        def normalize(name):
            return "".join(character for character in str(name).lower() if character.isalnum())

        def resolve_entity(target):
            if not isinstance(target, str):
                return getattr(target, "name", str(target)), target
            exact = entities.get(target)
            if exact is not None:
                return target, exact
            target_key = normalize(target)
            candidates = []
            for key, value in entities.items():
                key_text = str(key)
                base = key_text[:-8] if key_text.lower().endswith("_painting") else key_text
                if normalize(key_text) == target_key or normalize(base) == target_key:
                    candidates.append((key_text, value))
            if len(candidates) == 1:
                return candidates[0]
            return target, None

        for target in targets:
            name, entity = resolve_entity(target)
            getter = getattr(entity, "get_xpos", None)
            entry = self.targets.setdefault(name, {"samples": 0, "unavailable": 0})
            if not callable(getter):
                entry["unavailable"] += 1
                continue
            try:
                xyz = np.asarray(getter(env.physics), dtype=float).reshape(3)
                distance = float(np.linalg.norm(xyz - state[:3]))
                if not np.isfinite(distance):
                    raise ValueError("non-finite target position")
            except (AttributeError, KeyError, TypeError, ValueError):
                entry["unavailable"] += 1
                continue
            entry.setdefault("initial_m", distance)
            entry["final_m"] = distance
            entry["final_position"] = xyz.tolist()
            entry["minimum_m"] = min(distance, entry.get("minimum_m", distance))
            keypoint_getter = getattr(entity, "get_grasped_keypoints", None)
            if callable(keypoint_getter):
                try:
                    keypoints = np.asarray(keypoint_getter(env.physics), dtype=float).reshape(-1, 3)
                    keypoints = keypoints[np.isfinite(keypoints).all(axis=1)]
                    if len(keypoints):
                        # SkillLib.pick samples a valid keypoint. Use the
                        # nearest finite keypoint for a short, reachable live
                        # approach while preserving the expert grasp geometry.
                        grasp_position = keypoints[np.argmin(
                            np.linalg.norm(keypoints - state[:3], axis=1)
                        )]
                        grasp_distance = float(np.linalg.norm(grasp_position - state[:3]))
                        if np.isfinite(grasp_distance):
                            entry["grasp_position"] = grasp_position.tolist()
                            entry["grasp_final_m"] = grasp_distance
                            entry["grasp_minimum_m"] = min(
                                grasp_distance, entry.get("grasp_minimum_m", grasp_distance)
                            )
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
            entry["samples"] += 1

    def result(self):
        return {
            "target_distance_reference": "world EE to task target entity origin; not grasp distance",
            "distance_progress": self.distance_progress(),
            "targets": {name: {**values, "improvement_m": values["initial_m"] - values["final_m"]}
                        if values["samples"] else dict(values) for name, values in self.targets.items()},
            "target_status": "available" if any(v["samples"] for v in self.targets.values()) else "unavailable",
            "ee_path_length_m": self.path_length,
            **dict(self.counts),
        }

    def distance_progress(self):
        """Return normalized progress toward the nearest observed task target.

        VLABench primitive tasks often expose ``get_task_progress`` but leave
        it at zero until a discrete primitive completes.  This value is a
        shaping fallback only; task success is still determined by the
        simulator's success predicate.  Returning ``None`` preserves the
        distinction between no geometric evidence and zero progress.
        """
        ratios = []
        for values in self.targets.values():
            if not values.get("samples"):
                continue
            initial = float(values.get("initial_m", np.nan))
            final = float(values.get("final_m", np.nan))
            if not np.isfinite(initial) or not np.isfinite(final):
                continue
            if initial <= np.finfo(float).eps:
                ratios.append(1.0 if final <= np.finfo(float).eps else 0.0)
            else:
                ratios.append(float(np.clip((initial - final) / initial, 0.0, 1.0)))
        return max(ratios) if ratios else None

    def target_position(self):
        """Return the latest observed origin of the first available target."""
        for values in self.targets.values():
            position = values.get("final_position")
            if position is not None:
                result = np.asarray(position, dtype=np.float64).reshape(-1)
                if result.shape == (3,) and np.isfinite(result).all():
                    return result
        return None

    def target_grasp_position(self):
        """Return the live grasp keypoint when the task entity exposes one."""
        for values in self.targets.values():
            position = values.get("grasp_position")
            if position is not None:
                result = np.asarray(position, dtype=np.float64).reshape(-1)
                if result.shape == (3,) and np.isfinite(result).all():
                    return result
        return None

    def target_grasp_distance(self):
        """Return the EE distance to the latest live grasp keypoint."""
        distances = [
            float(values["grasp_final_m"])
            for values in self.targets.values()
            if values.get("grasp_final_m") is not None
            and np.isfinite(values.get("grasp_final_m", np.nan))
        ]
        return min(distances) if distances else None

    def target_distance(self):
        """Return the latest EE-to-target distance, when geometry is available."""
        distances = [
            float(values["final_m"])
            for values in self.targets.values()
            if values.get("samples") and np.isfinite(values.get("final_m", np.nan))
        ]
        return min(distances) if distances else None
