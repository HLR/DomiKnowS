"""Thin lazy adapter around the official VLABench environment registry."""

from __future__ import annotations

import importlib
from functools import partial
from copy import deepcopy
from threading import RLock
import math
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


class InverseKinematicsError(ValueError):
    """A finite Cartesian target lies outside the current IK basin."""


CONTROLLER_FRAME_VERSION = 2
_ENV_CONFIG_LOCK = RLock()

# Official Franka Emika Panda joint ranges (radians). VLABench's MJCF comments
# them out, so the incremental IK is the only place they are known.
FRANKA_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
FRANKA_JOINT_RANGES = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
# Null-space posture motion allowed per IK call (radians, joint-space norm).
POSTURE_STEP_LIMIT = 0.05


def robot_frame_position(env: Any) -> np.ndarray:
    """Return the robot-base origin used by the official LeRobot dataset.

    VLABench observations and IK targets are world-frame, while its LeRobot
    ``state`` and ``actions`` columns subtract this origin.  Synthetic test
    environments have no robot geometry and intentionally fall back to the
    world origin.
    """
    getter = getattr(env, "get_robot_frame_position", None)
    if callable(getter):
        value = getter()
    else:
        robot = getattr(env, "robot", None)
        get_base = getattr(robot, "get_base_position", None)
        physics = getattr(env, "physics", None)
        if callable(get_base) and physics is not None:
            value = get_base(physics)
        else:
            config = getattr(robot, "robot_config", {}) or {}
            value = config.get("position", np.zeros(3, dtype=np.float64))
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("VLABench robot frame position must be a finite xyz vector")
    return result.copy()


def world_to_robot_ee_state(state, robot_frame) -> np.ndarray:
    """Translate a world-frame xyz-Euler EE state into dataset coordinates."""
    value = np.asarray(state, dtype=np.float64).reshape(-1)
    frame = np.asarray(robot_frame, dtype=np.float64).reshape(-1)
    if value.size < 7 or frame.shape != (3,):
        raise ValueError("EE state and robot frame must contain 7 and 3 values")
    if not np.isfinite(value[:7]).all() or not np.isfinite(frame).all():
        raise ValueError("EE state and robot frame must be finite")
    result = value[:7].copy()
    result[:3] -= frame
    return result


def robot_to_world_ee_action(action, robot_frame) -> np.ndarray:
    """Translate a dataset-frame absolute EE action into an IK world target."""
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    frame = np.asarray(robot_frame, dtype=np.float64).reshape(-1)
    if value.shape != (7,) or frame.shape != (3,):
        raise ValueError("EE action and robot frame must contain 7 and 3 values")
    if not np.isfinite(value).all() or not np.isfinite(frame).all():
        raise ValueError("EE action and robot frame must be finite")
    result = value.copy()
    result[:3] += frame
    return result


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
    max_position_step: float = 0.02,
    max_rotation_step: float = 0.10,
) -> np.ndarray:
    """Rate-limit an absolute EE target around the current simulator pose.

    The current controller already samples tanh-bounded local increments. This
    independent execution envelope remains a defense for legacy checkpoints,
    custom controllers, and floating-point drift, using wrapped Euler deltas
    so crossing ``-pi/pi`` does not become a full rotation.
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


def posture_regularized_qpos(
    physics,
    site,
    joints,
    target_pos,
    target_quat,
    *,
    nominal_qpos,
    joint_ranges=None,
    tol: float = 1e-3,
    max_steps: int = 200,
    rot_weight: float = 1.0,
    regularization_threshold: float = 0.1,
    regularization_strength: float = 3e-2,
    max_update_norm: float = 2.0,
    progress_thresh: float = 20.0,
    posture_gain: float = 0.5,
    posture_step_limit: float = POSTURE_STEP_LIMIT,
    periodic_joints=(),
) -> tuple[bool, np.ndarray]:
    """Damped least-squares site IK with a bounded null-space posture bias.

    Mirrors ``dm_control.utils.inverse_kinematics`` (same error metric,
    regularisation, and progress test) but adds a one-shot null-space step
    toward ``nominal_qpos`` -- stronger for joints outside ``joint_ranges`` --
    so that hundreds of incremental solves do not wind the redundant arm into
    contorted, self-colliding postures. The step is projected into the task
    null space and capped at ``posture_step_limit`` per call, so the returned
    site pose still meets ``tol`` and the physical arm never has to jump.
    """
    import mujoco

    # Resolve indices on the live physics (mjcf bindings belong to it), then
    # iterate on a scratch copy so the simulator state is untouched.
    site_id = int(physics.bind(site).element_id)
    dof_ids = [int(np.asarray(physics.bind(joint).dofadr).reshape(-1)[0]) for joint in joints]
    qpos_ids = [int(np.asarray(physics.bind(joint).qposadr).reshape(-1)[0]) for joint in joints]
    scratch = physics.copy(share_model=True)
    model, data = scratch.model.ptr, scratch.data.ptr
    count = len(joints)
    nominal = np.asarray(nominal_qpos, dtype=np.float64).reshape(-1)[:count]
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
    target_quat = target_quat / np.linalg.norm(target_quat)
    jac_pos = np.empty((3, model.nv))
    jac_rot = np.empty((3, model.nv))
    site_quat = np.empty(4)
    neg_site_quat = np.empty(4)
    err_quat = np.empty(4)
    err_rot = np.empty(3)
    def pose_error():
        mujoco.mj_fwdPosition(model, data)
        err_pos = target_pos - data.site_xpos[site_id]
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
        mujoco.mju_negQuat(neg_site_quat, site_quat)
        mujoco.mju_mulQuat(err_quat, target_quat, neg_site_quat)
        mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)
        return err_pos, err_rot.copy(), float(np.linalg.norm(err_pos) + rot_weight * np.linalg.norm(err_rot))

    posture_applied = False
    success = False
    for iteration in range(int(max_steps)):
        err_pos, err_rot, err_norm = pose_error()
        # The posture step is taken even when the pose already matches, so a
        # held target still lets the arm relax toward the nominal posture.
        if err_norm < tol and posture_applied:
            success = True
            break
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        jac = np.vstack((jac_pos[:, dof_ids], rot_weight * jac_rot[:, dof_ids]))
        err = np.concatenate((err_pos, rot_weight * err_rot))
        damping = regularization_strength if err_norm > regularization_threshold else 1e-6
        gram = jac @ jac.T + damping * np.eye(jac.shape[0])
        pseudo_inverse = jac.T @ np.linalg.solve(gram, np.eye(jac.shape[0]))
        update = pseudo_inverse @ err
        update_norm = float(np.linalg.norm(update))
        if posture_applied and update_norm > 0 and err_norm / update_norm > progress_thresh:
            break
        if update_norm > max_update_norm:
            update *= max_update_norm / update_norm
        if not posture_applied:
            posture_applied = True
            current = data.qpos[qpos_ids]
            difference = nominal - current
            # A full turn of the wrist roll reproduces the same site pose and
            # cannot be undone inside the task null space, so compare it
            # modulo 2*pi instead of dragging the whole arm after it.
            for index in periodic_joints:
                difference[index] = (difference[index] + np.pi) % (2.0 * np.pi) - np.pi
            gradient = posture_gain * difference
            if joint_ranges is not None:
                for index, (low, high) in enumerate(joint_ranges[:count]):
                    if index in periodic_joints:
                        continue
                    if current[index] < low:
                        gradient[index] += 2.0 * (low - current[index])
                    elif current[index] > high:
                        gradient[index] += 2.0 * (high - current[index])
            null_space = np.eye(count) - pseudo_inverse @ jac
            posture_step = null_space @ gradient
            posture_norm = float(np.linalg.norm(posture_step))
            if posture_norm > posture_step_limit:
                posture_step *= posture_step_limit / posture_norm
            update = update + posture_step
        data.qpos[qpos_ids] += update
    if not success:
        success = pose_error()[2] < tol
    return success, np.asarray(data.qpos[qpos_ids], dtype=np.float64).copy()


def _franka_ik_arguments(env):
    """Return (site, joints, nominal) for a Franka robot, else ``None``."""
    robot = getattr(env, "robot", None)
    model = getattr(robot, "mjcf_model", None)
    physics = getattr(env, "physics", None)
    if model is None or physics is None or getattr(physics, "copy", None) is None:
        return None
    try:
        site = model.find("site", "end_effector")
        joints = [model.find("joint", name) for name in FRANKA_JOINT_NAMES]
        nominal = np.asarray(getattr(robot, "default_qpos", ()), dtype=np.float64).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return None
    if site is None or any(joint is None for joint in joints) or nominal.size < 7:
        return None
    return site, joints, nominal[:7]


def ee_action_to_env_action(
    env,
    action,
    *,
    ik_tolerance: float = 1e-3,
    ik_max_steps: int = 200,
) -> np.ndarray:
    """Convert dataset EE action [xyz, rpy, grip] to VLABench joint control."""
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("controller action must be a finite 7D EE action")
    if not np.isfinite(ik_tolerance) or ik_tolerance <= 0:
        raise ValueError("IK tolerance must be finite and positive")
    if int(ik_max_steps) <= 0:
        raise ValueError("IK max steps must be positive")
    quaternion = euler_to_quaternion(*value[3:6])
    franka = _franka_ik_arguments(env)
    if franka is not None:
        site, arm_joints, nominal = franka
        try:
            status, solution = posture_regularized_qpos(
                env.physics,
                site,
                arm_joints,
                value[:3],
                quaternion,
                nominal_qpos=nominal,
                joint_ranges=FRANKA_JOINT_RANGES,
                tol=float(ik_tolerance),
                max_steps=int(ik_max_steps),
                periodic_joints=(6,),
            )
        except ImportError:
            franka = None
    if franka is None:
        status, solution = env.robot.get_qpos_from_ee_pos(
            physics=env.physics,
            pos=value[:3],
            quat=quaternion,
            tol=float(ik_tolerance),
            max_steps=int(ik_max_steps),
        )
    joints = np.asarray(solution, dtype=np.float64).reshape(-1)
    if not bool(status) or not np.isfinite(joints).all():
        raise InverseKinematicsError("VLABench inverse-kinematics conversion failed")
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
        env_module = importlib.import_module("VLABench.envs")
        load_env = env_module.load_env
    except ImportError as exc:
        raise RuntimeError(
            "VLABench is required only for online rollout; install the OpenMOSS/VLABench clone editable"
        ) from exc
    # The standalone controller consumes VLABench's progress and intention
    # signals as shaping rewards.  The upstream default (``efficient``) is
    # intended for data collection and leaves those signals uninitialised and
    # unupdated.  Online rollouts must use the evaluation contract unless a
    # caller explicitly supplies another mode for a specialised diagnostic.
    kwargs.setdefault("run_mode", "eval")
    # Upstream mutates TASK_CONFIG['default'] and ROBOT_CONFIG in place.
    # Private copies prevent task order and constructor failures from leaking
    # configuration. Existing environments retain their own referenced objects.
    with _ENV_CONFIG_LOCK:
        originals = {
            name: getattr(env_module, name)
            for name in ("TASK_CONFIG", "ROBOT_CONFIG")
            if hasattr(env_module, name)
        }
        # Composer otherwise creates an entropy-seeded RandomState of its own.
        # Drawing from the evaluator-seeded NumPy stream keeps task layout and
        # physics randomization reproducible across baseline/candidate checks.
        constructor = getattr(env_module, "LM4ManipDMEnv", None)
        simulator_seed = kwargs.pop("simulator_seed", None)
        if simulator_seed is None:
            simulator_seed = int(np.random.randint(0, 2**31 - 1))
        try:
            for name, value in originals.items():
                setattr(env_module, name, deepcopy(value))
            if constructor is not None:
                env_module.LM4ManipDMEnv = partial(constructor, random_state=int(simulator_seed))
            return load_env(task, robot=robot, time_limit=time_limit, **deepcopy(kwargs))
        finally:
            if constructor is not None:
                env_module.LM4ManipDMEnv = constructor
            for name, value in originals.items():
                setattr(env_module, name, value)
