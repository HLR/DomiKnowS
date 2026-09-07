"""Short paired demonstration/controller rollouts from a restored held-out scene.

Restoration is explicit: LeRobot EE states alone cannot restore object poses,
robot joints, velocities, contacts, or task bookkeeping. A caller must provide
the original episode configuration/state through restore(env, descriptor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .diagnostics import RolloutDiagnostics
from .environment import (InverseKinematicsError, bound_ee_action, ee_action_to_env_action,
                          robot_frame_position, robot_to_world_ee_action, world_to_robot_ee_state)
from .observations import camera_indices, camera_report, image_tensor
from .program import _controller_inputs, _observation_state


def paired_camera_mapping_verified(report, *, image_tolerance: float) -> bool:
    """Require every dataset slot to match its paired live frame."""
    views = report.get("views", ())
    return bool(views) and all(
        bool(view.get("same_shape"))
        and view.get("pixel_mae") is not None
        and float(view["pixel_mae"]) <= float(image_tolerance)
        for view in views
    )


def compare_cameras(dataset_images, dataset_keys, env, observation, camera_mapping, output_dir):
    """Export paired images and compare the restored same-frame pixels per camera."""
    if len(dataset_keys) != len(dataset_images) or not dataset_keys:
        raise ValueError("demonstration camera keys are required; anonymous image slots cannot be verified")
    if set(camera_mapping) != set(dataset_keys):
        raise ValueError("camera mapping must name every dataset camera exactly once")
    indices = camera_indices(env, observation, [camera_mapping[key] for key in dataset_keys])
    report = camera_report(env, observation, indices=indices, dataset_keys=dataset_keys)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    live_images = image_tensor(np.asarray(observation["rgb"]), channels_last=True)
    for slot, (reference, index) in enumerate(zip(dataset_images, indices)):
        live = live_images[index]
        same_shape = reference.shape == live.shape
        report["views"][slot]["same_shape"] = same_shape
        report["views"][slot]["pixel_mae"] = float((reference - live).abs().mean()) if same_shape else None
        for name, value in (("dataset", reference), ("live", live)):
            path = output_dir / f"camera-{slot}-{name}.png"
            pixels = (value.detach().cpu().movedim(0, -1).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(path)
            report["views"][slot][f"{name}_image"] = str(path.resolve())
    return indices, report


@torch.no_grad()
def replay_heldout_demo(controller, dataset, descriptor, *, env_factory, restore,
                        camera_mapping, output_dir, device="cpu", steps=32,
                        execute_horizon=4, position_tolerance=0.005,
                        rotation_tolerance=0.05, image_tolerance=0.05,
                        ik_tolerance=5e-3, ik_max_steps=200):
    """Compare recorded-action execution with a policy fed fresh live observations.

    The caller selects an episode from its existing validation/test split. Both
    arms must restore the exact same scene; fail closed on pose/camera mismatch.
    Skill context comes from held-out demonstration phase, isolating the policy
    from planner errors. No training or optimizer calls occur here.
    """
    if steps <= 0 or execute_horizon <= 0 or min(position_tolerance, rotation_tolerance, image_tolerance) < 0:
        raise ValueError("replay requires positive horizons and nonnegative tolerances")
    if execute_horizon > dataset.action_horizon:
        raise ValueError("replay execute horizon cannot exceed the demonstration action horizon")
    episode, offset = int(descriptor["episode_index"]), int(descriptor.get("offset", 0))
    if offset < 0 or offset >= len(dataset.episodes[episode]) - 1:
        raise ValueError("replay offset needs a recorded successor state")
    indices_by_offset = {frame: index for index, (ep, frame) in enumerate(dataset.index) if ep == episode}
    reference = dataset[indices_by_offset[offset]]
    row = dataset.records[dataset.episodes[episode][offset]]
    keys = dataset.camera_keys(row)
    result = {"episode_index": episode, "offset": offset, "status": "completed",
              "context": "held-out demonstration phase; planner bypassed", "arms": {}}
    was_training = controller.training
    controller.eval()
    try:
        for arm in ("demonstration", "controller"):
            env = env_factory(**dict(descriptor.get("env_kwargs", {})))
            trace, diagnostics = [], RolloutDiagnostics()
            try:
                env.reset()
                # This hook restores task state as well as the physical scene.
                # An ordinary reset or setting only the EE pose is insufficient.
                restore(env, descriptor)
                observation = env.get_observation(require_pcd=False)
                frame = robot_frame_position(env)
                state = world_to_robot_ee_state(_observation_state(observation), frame)
                error = state[:6] - reference["state"][-1, :6].numpy()
                error[3:] = np.arctan2(np.sin(error[3:]), np.cos(error[3:]))
                camera_ids, cameras = compare_cameras(
                    reference["images"][-1], keys, env, observation, camera_mapping,
                    Path(output_dir) / arm,
                )
                camera_matches = paired_camera_mapping_verified(
                    cameras, image_tolerance=image_tolerance
                )
                matches = (np.max(np.abs(error[:3])) <= position_tolerance
                           and np.max(np.abs(error[3:])) <= rotation_tolerance
                           and bool(state[6] >= 0.5) == bool(reference["state"][-1, 6] >= 0.5)
                           and camera_matches)
                cameras["paired_frame_verified"] = camera_matches
                cameras["status"] = "verified" if matches else "restore_or_camera_mismatch"
                arm_result = {"initial_pose_error": error.tolist(), "cameras": cameras,
                              "trace": trace, "status": "short_horizon_complete"}
                result["arms"][arm] = arm_result
                if not matches:
                    arm_result["status"] = result["status"] = "restore_or_camera_mismatch"
                    continue
                history = [observation]
                diagnostics.observe(env, _observation_state(observation))
                chunk = None
                budget = min(int(steps), len(dataset.episodes[episode]) - offset - 1)
                for step in range(budget):
                    window = dataset[indices_by_offset[offset + step]]
                    if arm == "demonstration":
                        candidate = window["actions"][0].numpy()
                    else:
                        if chunk is None or step % execute_horizon == 0:
                            inputs = _controller_inputs(history, int(window["task_index"]), device,
                                robot_frame=frame, selected_camera_indices=camera_ids,
                                plan_context=window["plan_context"].tolist())
                            chunk = controller.predict_action_chunk(*inputs)[0].detach().cpu().numpy()
                        candidate = chunk[step % execute_horizon]
                    current = world_to_robot_ee_state(_observation_state(observation), frame)
                    bounded = bound_ee_action(candidate, current)
                    try:
                        command = ee_action_to_env_action(env, robot_to_world_ee_action(bounded, frame),
                                                         ik_tolerance=ik_tolerance, ik_max_steps=ik_max_steps)
                    except InverseKinematicsError:
                        arm_result["status"] = "ik_failure"
                        break
                    timestep = env.step(command)
                    observation = env.get_observation(require_pcd=False)
                    history = (history + [observation])[-2:]
                    observed = world_to_robot_ee_state(_observation_state(observation), frame)
                    next_state = dataset[indices_by_offset[offset + step + 1]]["state"][-1].numpy()
                    tracking = observed[:6] - next_state[:6]
                    tracking[3:] = np.arctan2(np.sin(tracking[3:]), np.cos(tracking[3:]))
                    diagnostics.observe(env, _observation_state(observation), command_gripper=bounded[6])
                    trace.append({"step": step, "candidate": candidate.tolist(), "bounded": bounded.tolist(),
                                  "observed": observed.tolist(), "reference_next_state": next_state.tolist(),
                                  "tracking_error": tracking.tolist()})
                    last = getattr(timestep, "last", None)
                    if bool(last() if callable(last) else last):
                        # Terminal is not labeled success here without task evidence.
                        arm_result["status"] = "environment_terminal"
                        break
                arm_result["steps"] = len(trace)
                arm_result["diagnostics"] = diagnostics.result()
            finally:
                env.close()
    finally:
        controller.train(was_training)
    return result
