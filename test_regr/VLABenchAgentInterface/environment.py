"""Thin lazy adapter around the official VLABench environment registry."""

from __future__ import annotations

import importlib
import math
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


def numbered_views_from_observation(
    env: Any,
    observation: Mapping[str, Any],
    *,
    max_views: int = 3,
) -> tuple[list[Image.Image], tuple[str, ...]]:
    """Render stable entity-pointer labels over simulator segmentation views.

    Fake/diagnostic environments may omit segmentation.  In that case the RGB
    views are still returned, but the entity table remains authoritative and
    no pointer is drawn without geometric evidence.
    """
    rgb = np.asarray(observation["rgb"])
    entities = getattr(getattr(env, "task", None), "entities", {}) or {}
    entity_names = tuple(sorted(str(name) for name in entities))
    segmentation_value = observation.get("segmentation")
    segmentation = None if segmentation_value is None else np.asarray(segmentation_value)
    geom_ids: dict[str, set[int]] = {}
    physics = getattr(env, "physics", None)
    for name in entity_names:
        ids: set[int] = set()
        for geom in getattr(entities[name], "geoms", ()):
            try:
                ids.add(int(physics.bind(geom).element_id))
            except Exception:
                continue
        geom_ids[name] = ids

    views: list[Image.Image] = []
    for view_index in range(min(max_views, len(rgb))):
        image = Image.fromarray(rgb[view_index].astype(np.uint8)).convert("RGB")
        if segmentation is not None:
            draw = ImageDraw.Draw(image)
            mask = segmentation[view_index, ..., 0] if segmentation.ndim == 4 else segmentation[view_index]
            for pointer, name in enumerate(entity_names):
                ids = geom_ids[name]
                if not ids:
                    continue
                ys, xs = np.where(np.isin(mask, list(ids)))
                if not len(xs):
                    continue
                x, y = int(np.median(xs)), int(np.median(ys))
                draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=(255, 230, 0), outline=(0, 0, 0), width=2)
                draw.text((x - 4 * len(str(pointer)), y - 7), str(pointer), fill=(0, 0, 0))
        views.append(image)
    return views, entity_names


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the wxyz quaternion expected by VLABench/MuJoCo IK."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def quaternion_to_euler(quaternion) -> np.ndarray:
    """Convert a VLABench wxyz quaternion to xyz Euler radians."""

    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must be a finite wxyz vector")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion must have nonzero norm")
    w, x, y, z = value / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(pitch_term, -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def bound_ee_action(
    action,
    current_state,
    *,
    max_position_step: float = 0.05,
    max_rotation_step: float = 0.25,
) -> np.ndarray:
    """Rate-limit an absolute EE target around the current simulator pose.

    The controller remains a Normal actor over absolute xyz/Euler coordinates,
    but an unbounded exploratory draw can be far outside the local IK basin.
    Bound the command actually sent to the simulator, using wrapped Euler
    deltas so crossing ``-pi/pi`` does not become a full rotation.
    """
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    current = np.asarray(current_state, dtype=np.float64).reshape(-1)
    if value.shape != (7,) or current.size < 6:
        raise ValueError("EE action/state must contain 7 and at least 6 values")
    if not np.isfinite(value).all() or not np.isfinite(current[:6]).all():
        raise ValueError("EE action/state must be finite")
    if max_position_step <= 0 or max_rotation_step <= 0:
        raise ValueError("EE step limits must be positive")
    bounded = value.copy()
    bounded[:3] = current[:3] + np.clip(
        value[:3] - current[:3], -float(max_position_step), float(max_position_step)
    )
    angular_delta = (value[3:6] - current[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    bounded[3:6] = current[3:6] + np.clip(
        angular_delta, -float(max_rotation_step), float(max_rotation_step)
    )
    bounded[3:6] = (bounded[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    bounded[6] = float(value[6] >= 0.5)
    return bounded


def ee_action_to_env_action(env, action) -> np.ndarray:
    """Convert dataset EE action [xyz, rpy, grip] to VLABench joint control."""
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("controller action must be a finite 7D EE action")
    quaternion = euler_to_quaternion(*value[3:6])
    status, joints = env.robot.get_qpos_from_ee_pos(
        physics=env.physics,
        pos=value[:3],
        quat=quaternion,
    )
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    if not bool(status) or not np.isfinite(joints).all():
        raise ValueError("VLABench inverse-kinematics conversion failed")
    gripper = np.full(2, 0.04 if value[6] >= 0.5 else 0.0, dtype=np.float64)
    command = np.concatenate((joints, gripper))
    spec = getattr(env, "action_spec", None)
    spec = spec() if callable(spec) else spec
    if spec is not None and hasattr(spec, "minimum") and hasattr(spec, "maximum"):
        command = np.clip(command, np.asarray(spec.minimum), np.asarray(spec.maximum))
    return command


def reset_reward_tracking(env: Any) -> None:
    """Initialize optional VLABench shaping state omitted by some tasks.

    Several upstream primitive tasks inherit progress/intention accessors but
    do not call the corresponding reset hooks during ``env.reset()``.  Invoke
    a hook only when its state attribute is absent, preserving state managed
    correctly by other tasks.
    """
    task = getattr(env, "task", None)
    if task is None:
        return
    for attribute, method_name in (
        ("target_is_grasped", "reset_task_progress"),
        ("intention_distance", "reset_intention_distance"),
    ):
        if hasattr(task, attribute):
            continue
        reset = getattr(task, method_name, None)
        if callable(reset):
            reset()
    # Some upstream tasks create these dictionaries but omit the selected
    # target (for example when a target book is also listed as randomly
    # ignored).  The inherited reward accessors index the target directly and
    # otherwise raise KeyError before the rollout starts.  Missing entries mean
    # "no observed progress/intention yet", so initialize only those entries.
    target = getattr(task, "target_entity", ())
    if isinstance(target, str):
        targets = (target,)
    elif isinstance(target, (list, tuple, set, frozenset)):
        targets = tuple(target)
    elif target is None:
        targets = ()
    else:
        targets = (target,)
    progress = getattr(task, "target_is_grasped", None)
    intention = getattr(task, "intention_distance", None)
    for name in targets:
        if isinstance(progress, dict):
            progress.setdefault(name, False)
        if isinstance(intention, dict):
            intention.setdefault(name, np.inf)


def create_environment(
    task: str,
    *,
    robot: str = "franka",
    time_limit: int = 400,
    **kwargs,
):
    """Create an official VLABench environment without importing it at train time."""
    try:
        # VLABench uses decorator side effects to populate its process-global
        # registry.  Importing envs alone leaves valid identifiers such as
        # ``franka`` and the task names absent from that registry.
        importlib.import_module("VLABench.robots")
        importlib.import_module("VLABench.tasks")
        load_env = importlib.import_module("VLABench.envs").load_env
    except ImportError as exc:
        raise RuntimeError(
            "VLABench is required only for online rollout; install the OpenMOSS/VLABench clone editable"
        ) from exc
    return load_env(task, robot=robot, time_limit=time_limit, **kwargs)
