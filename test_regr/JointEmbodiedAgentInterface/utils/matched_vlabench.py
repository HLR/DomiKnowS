"""Source-scene diagnostic for Joint. Expert information stays inside this module."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from test_regr.VLABenchAgentInterface.knowledge.tables import PRIMITIVE_TASK_PATTERNS
from test_regr.VLABenchAgentInterface.simulation.environment import create_environment, reset_reward_tracking
from test_regr.VLABenchAgentInterface.world_domain import SKILL_ARGUMENTS


TASKS = ("select_chemistry_tube", "select_fruit", "insert_flower", "add_condiment")


def source_cases():
    for index, task in enumerate(TASKS):
        for offset in range(3):
            yield task, 724000 + 100 * index + offset


def scene_id(env):
    # Include the task semantics as well as settled physics and object ordering.
    labels = [env.task.get_instruction(), env.task.target_entity,
              getattr(env.task, "target_container", None), sorted(env.task.entities)]
    digest = hashlib.sha256(json.dumps(labels).encode())
    digest.update(np.round(env.physics.get_state(), 8).tobytes())
    return digest.hexdigest()


def expert_plan(env, task, entities):
    plan = []
    for skill in PRIMITIVE_TASK_PATTERNS[task]:
        params = {}
        for role in SKILL_ARGUMENTS[skill]:
            target = getattr(env.task, "target_entity" if role == "target_entity_name" else "target_container")
            params[role] = entities.index(target)
        plan.append({"name": skill, "params": params})
    return plan


def resolve(operation, entities):
    result = {}
    for role, pointer in operation["params"].items():
        if type(pointer) is int and 0 <= pointer < len(entities):
            result[role] = entities[pointer]
        elif isinstance(pointer, str) and pointer in entities:
            result[role] = pointer
        else:
            raise ValueError(f"Unresolved {role} pointer {pointer!r}")
    return result


def skill_calls(env, task, skill, args):
    from VLABench.utils.skill_lib import SkillLib
    from VLABench.utils.utils import euler_to_quaternion

    if skill == "pick":
        options = {}
        if task == "insert_flower":
            from test_regr.VLABenchAgentInterface.simulation.program import _flower_grasp_euler, _live_task_flower_stem_axis
            subject = env.task.entities[args["target_entity_name"]]
            axis = _live_task_flower_stem_axis(env, "target_entity", entity=subject)
            options = {"prior_eulers": [_flower_grasp_euler(axis).tolist()]}
        if task == "add_condiment":
            from test_regr.VLABenchAgentInterface.simulation.program import (
                CONDIMENT_GRASP_EULER, _live_task_condiment_grasp_point)
            subject = env.task.entities[args["target_entity_name"]]
            position = _live_task_condiment_grasp_point(env, "target_entity", entity=subject)
            if position is not None:
                options = dict(target_pos=position, target_quat=euler_to_quaternion(*CONDIMENT_GRASP_EULER),
                               max_n_substep=4)
        return [(SkillLib.pick, {**args, **options})]
    if skill == "insert":
        container = env.task.entities[args["target_container_name"]]
        point = np.array(container.get_place_point(env.physics)[-1]) + [0, 0, .05]
        quat = euler_to_quaternion(-np.pi/2, np.pi/2, 0)
        return [(SkillLib.lift, {"target_quat": quat}), (SkillLib.moveto, {"target_pos": point, "target_quat": quat,
                                                                         "gripper_state": np.zeros(2), "max_n_substep": 3}),
                (SkillLib.lift, {"lift_height": -.2})]
    if skill == "pour":
        container = env.task.entities[args["target_container_name"]]
        point = np.array(container.get_xpos(env.physics)) + [0, 0, .2]
        return [(SkillLib.lift, {"lift_height": .2, "gripper_state": np.zeros(2)}),
                (SkillLib.moveto, {"target_pos": point, "gripper_state": np.zeros(2), "max_n_substep": 3}),
                (SkillLib.pour, {})]
    if skill in ("open_door", "close_door"):
        return [(getattr(SkillLib, skill), {"target_container_name": args["target_entity_name"]})]
    if skill == "open_laptop":
        return [(SkillLib.open_laptop, args)]
    if skill == "press":
        return [(SkillLib.press, {"target_pos": env.task.entities[args["target_entity_name"]].get_xpos(env.physics).copy()})]
    if skill == "push" and "target_container_name" in args:
        return [(SkillLib.push, {"target_pos": env.task.entities[args["target_container_name"]].get_xpos(env.physics).copy()})]
    if skill in ("lift", "place", "pull", "push"):
        return [(getattr(SkillLib, skill), args)]
    raise ValueError(f"No diagnostic execution adapter for skill {skill}")


class StepLimit(Exception):
    pass


class Trace:
    def __init__(self, env, task, seed):
        self.env = env
        self.row = dict(task=task, seed=seed, scene=scene_id(env), requested=env.task.target_entity,
                        requested_container=getattr(env.task, "target_container", None),
                        instruction=env.task.get_instruction(), operations=[], steps=0, native_success=False)
        self.subject = None
        self.contact = False
        self.start_z = {k: float(v.get_xpos(env.physics)[2]) for k,v in env.task.entities.items()}
        self.commands = []
        self.step = env.step
        env.step = self.observe_step

    def select(self, skill, args):
        if "target_entity_name" in args:
            name = args["target_entity_name"]
            if name != self.subject:
                self.contact = False
            self.subject = name
        operation = {"skill": skill, "arguments": args, "subject": self.subject}
        if self.row["operations"] and all(self.row["operations"][-1][k] == v for k,v in operation.items()):
            return
        operation.update(start_step=self.row["steps"], contact=False, closed=False, release=False,
                         closed_contact=False, held_lift=False, max_height_gain_m=None, min_grasp_point_m=None)
        self.row["operations"].append(operation)

    def observe_step(self, action):
        if self.row["steps"] >= 400:
            raise StepLimit()
        timestep = self.step(action)
        self.commands.append(np.asarray(action).copy())
        self.row["steps"] += 1
        # AboveCondition writes through a live xpos view while testing success.
        self.env.physics.forward()
        if self.subject is None:
            return timestep
        env = self.env
        obj = env.task.entities[self.subject]
        op = self.row["operations"][-1]
        # Franka's upstream open-state flag is inverted. Read both finger joints directly.
        widths = [float(env.physics.bind(env.robot.mjcf_model.find("joint", f"finger_joint{i}")).qpos[0])
                  for i in (1, 2)]
        op["closed"] |= max(widths) < .035
        robot_ids = {int(env.physics.bind(g).element_id) for g in env.robot.gripper_geoms}
        object_ids = {int(env.physics.bind(g).element_id) for g in obj.geoms}
        contact = any((c.geom1 in robot_ids and c.geom2 in object_ids) or
                      (c.geom2 in robot_ids and c.geom1 in object_ids) for c in env.physics.data.contact)
        op["contact"] |= contact
        op["closed_contact"] |= contact and max(widths) < .035
        op["release"] |= self.contact and not contact
        self.contact = contact
        delta = float(obj.get_xpos(env.physics)[2]) - self.start_z[self.subject]
        op["held_lift"] |= contact and delta > .01
        op["final_subject_world"] = np.asarray(obj.get_xpos(env.physics)).tolist()
        op["max_height_gain_m"] = max(delta, op["max_height_gain_m"] or 0.)
        getter = getattr(obj, "get_grasped_keypoints", None)
        if callable(getter):
            points = np.asarray(getter(env.physics), dtype=float).reshape(-1, 3)
            if len(points):
                distance = float(np.linalg.norm(points - env.robot.get_end_effector_pos(env.physics), axis=1).min())
                old = op["min_grasp_point_m"]
                if np.isfinite(distance) and (old is None or distance < old):
                    op["min_grasp_point_m"] = distance
        return timestep


class ExpertEnvironment:
    """Skip unused demonstration images while preserving the expert's actual observation reads."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observation(self, **kwargs):
        env = self.env

        class Observation(dict):
            def __missing__(self, key):
                self.update(env.get_observation(**kwargs))
                return dict.__getitem__(self, key)

        return Observation()


def execute_plan(env, task, plan, trace):
    from test_regr.VLABenchAgentInterface.simulation.program import _task_success
    entities = sorted(env.task.entities)
    proxy = ExpertEnvironment(env)
    row = trace.row
    for operation in plan:
        skill = operation["name"]
        row["first_failed_skill"] = skill
        args = resolve(operation, entities)
        trace.select(skill, args)
        for call, kwargs in skill_calls(env, task, skill, args):
            result = call(proxy, **kwargs)
            row["native_success"] = _task_success(env)
            if row["native_success"]:
                row["first_failed_skill"] = None
                return
            # SkillLib's intermediate stage flag checks a 1 cm wrist tolerance.
            # Its native flower/condiment scripts continue through transit even
            # when the wrist misses that threshold with the object still held.
            if result is None or result[3] or (skill == "pick" and not result[2]):
                row["failed_macro"] = call.__name__
                return
        if skill == "insert" and trace.subject is not None:
            # Close the remaining error in object coordinates. A stem can be
            # grasped off-centre even when the wrist reaches the place waypoint.
            obj = env.task.entities[trace.subject]
            vase = env.task.entities[args["target_container_name"]]
            sites = np.array([env.physics.bind(s).xpos.copy() for s in vase.key_sites(env.physics)])
            if len(sites):
                goal = (sites.max(axis=0) + sites.min(axis=0)) / 2
                quat = env.robot.get_end_effector_quat(env.physics).copy()
                for _ in range(100):
                    offset = np.asarray(goal) - np.asarray(obj.get_xpos(env.physics))
                    if np.linalg.norm(offset[:2]) > .025:
                        offset[2] = max(0., offset[2])
                    wrist = np.asarray(env.robot.get_end_effector_pos(env.physics))
                    target = wrist + np.clip(offset, -.02, .02)
                    ok, joint = env.robot.get_qpos_from_ee_pos(env.physics, target, quat)
                    if not ok:
                        break
                    ts = env.step(np.r_[joint, 0., 0.])
                    row["native_success"] = _task_success(env)
                    if row["native_success"] or ts.last():
                        if row["native_success"]:
                            row["first_failed_skill"] = None
                        return


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import open3d
    open3d.utility.random.seed(seed)


class MatchedArm:
    def __init__(self, control, planner, controller):
        self.control = control
        self.planner = planner
        self.controller = controller
        self.trace = None

    def begin(self, env):
        self.trace = Trace(env, self.control["task"], self.control["seed"])
        if self.trace.row["scene"] != self.control["scene"]:
            raise ValueError(f"Unmatched simulator reset for {self.control['task']}:{self.control['seed']}")
        self.trace.row.update(planner=self.planner, controller=self.controller, plans=[])

    def plan(self, env, entities):
        return expert_plan(env, self.control["task"], entities)

    def record_plan(self, plan):
        self.trace.row["plans"].append({"step": self.trace.row["steps"], "plan": plan})

    def select(self, operation, entities):
        self.trace.select(operation["name"], resolve(operation, entities))

    def execute(self, env, plan):
        from test_regr.VLABenchAgentInterface.simulation.program import JointEpisode
        try:
            execute_plan(env, self.control["task"], plan, self.trace)
        except (ValueError, KeyError, AttributeError, StepLimit) as exc:
            self.trace.row["execution_error"] = f"{type(exc).__name__}: {exc}"
        row = self.trace.row
        return JointEpisode([], [], 0., row["native_success"], True, row["steps"],
                            termination_reason="success" if row["native_success"] else "expert_execution_failed")


def read_controls(path, tasks=TASKS):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    rows = [row for row in rows if row["task"] in tasks]
    if [(r["task"], r.get("original_seed", r["seed"])) for r in rows] != [(t, s) for t, s in source_cases() if t in tasks]:
        raise ValueError("Positive controls must cover every selected source reset in order")
    for row in rows:
        delta = row["seed"] - row.get("original_seed", row["seed"])
        if not 0 <= delta <= 18 or delta % 3:
            raise ValueError("Replacement reset is outside the source diagnostic schedule")
        if (row["planner"] != "expert" or row["controller"] != "expert" or not row["native_success"]
                or row.get("replay_success") is False):
            raise ValueError(f"Unresolved positive control on {row['task']}:{row['seed']}")
    return rows


def run_comparisons(stage2, controls, output, tasks=TASKS, controller_check=False):
    import torch
    from .constrained_stage2 import _pilot_policy
    controls = read_controls(controls, tasks)
    with output.open("x", encoding="utf8") as stream, _pilot_policy(stage2), torch.no_grad():
        for control in controls:
            arms = (("expert", "learned"),) if controller_check else (
                ("learned", "expert"), ("expert", "learned"), ("learned", "learned"))
            for planner, controller in arms:
                seed = control["seed"]
                reset_seed(seed)
                torch.manual_seed(seed)
                arm = MatchedArm(control, planner, controller)
                episode = stage2.collect_episode({"task": control["task"], "env_kwargs": {"simulator_seed": seed}},
                                                 diagnostic=arm)
                if arm.trace is None:
                    raise RuntimeError(f"Matched scene unavailable: {episode.termination_reason}")
                row = arm.trace.row
                row.update(native_success=bool(episode.success), termination=episode.termination_reason,
                           valid=bool(episode.valid))
                if controller == "learned" or "first_failed_skill" not in row:
                    row["first_failed_skill"] = None
                    if not episode.success:
                        operations = row["operations"]
                        picks = [op for op in row["operations"] if op["skill"] == "pick"]
                        if operations and operations[0]["skill"] != "pick":
                            row["first_failed_skill"] = operations[0]["skill"]
                        elif operations and operations[0]["subject"] != row["requested"]:
                            row["first_failed_skill"] = "pick"
                        elif picks and all(not op["closed_contact"] for op in picks):
                            row["first_failed_skill"] = "pick"
                        elif row["operations"]:
                            # Contact alone does not establish that a pick
                            # completed. Preserve ambiguity in that case.
                            row["first_failed_skill"] = "undetermined"
                        else:
                            row["first_failed_skill"] = "planning"
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                del episode


def record_control(output, task, seed):
    reset_seed(seed)
    env = create_environment(task, simulator_seed=seed)
    try:
        env.reset()
        reset_reward_tracking(env)
        trace = Trace(env, task, seed)
        plan = expert_plan(env, task, sorted(env.task.entities))
        try:
            execute_plan(env, task, plan, trace)
        except StepLimit:
            trace.row["termination"] = "max_steps"
        trace.row.update(planner="expert", controller="expert", plan=plan)
        if trace.row["native_success"]:
            recording = output.with_name(f"{output.stem}-{task}-{seed}.npz")
            np.savez_compressed(recording, commands=trace.commands)
            trace.row["demonstration"] = str(recording)
        return trace.row
    finally:
        env.close()


def positive_controls(output, repair_resets=False):
    attempts = output.with_suffix(".attempts.jsonl")
    with output.open("x", encoding="utf8") as accepted, attempts.open("x", encoding="utf8") as audit:
        for task, original in source_cases():
            # A disjoint deterministic sequence per slot. Decisions depend only
            # on the expert control and same-scene demonstration replay.
            for seed in range(original, original + (19 if repair_resets else 1), 3):
                row = record_control(output, task, seed)
                row["original_seed"] = original
                row["replay_success"] = replay_control(row) if row["native_success"] else None
                audit.write(json.dumps(row) + "\n")
                audit.flush()
                print(task, seed, row["native_success"], row["replay_success"], flush=True)
                if row["replay_success"]:
                    break
            accepted.write(json.dumps(row) + "\n")
            accepted.flush()


def replay_control(row, *, audit_controller=False):
    """Replay measured joint controls on the exact source reset that produced them."""
    from test_regr.VLABenchAgentInterface.simulation.program import _task_success
    reset_seed(row["seed"])
    env = create_environment(row["task"], simulator_seed=row["seed"])
    try:
        env.reset()
        reset_reward_tracking(env)
        if scene_id(env) != row["scene"]:
            raise ValueError("Recorded demonstration reset does not match")
        if audit_controller:
            from test_regr.VLABenchAgentInterface.simulation.environment import (
                InverseKinematicsError, bound_ee_action, ee_action_to_env_action, quaternion_to_euler,
                robot_frame_position, robot_to_world_ee_action, world_to_robot_ee_state)
            clone = env.physics.copy(share_model=True)
            joints = [env.robot.mjcf_model.find("joint", f"joint{i}") for i in range(1, 8)]
            frame = robot_frame_position(env)
            audit = dict(scene=row["scene"], steps=0, native_success=False, clipped_position_steps=0,
                         clipped_rotation_steps=0, frame_roundtrip_error=0., ik_failures=0)
        with np.load(row["demonstration"]) as recording:
            for command in recording["commands"]:
                if audit_controller:
                    clone.data.qpos[:] = env.physics.data.qpos
                    for joint, angle in zip(joints, command[:7]):
                        clone.bind(joint).qpos = angle
                    clone.forward()
                    target = np.r_[env.robot.get_end_effector_pos(clone),
                                   quaternion_to_euler(env.robot.get_end_effector_quat(clone)),
                                   float(command[-1] > .03)]
                    current = np.r_[env.robot.get_end_effector_pos(env.physics),
                                    quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                    float(env.robot.get_ee_open_state(env.physics))]
                    local_target = world_to_robot_ee_state(target, frame)
                    local_state = world_to_robot_ee_state(current, frame)
                    bounded = bound_ee_action(local_target, local_state, max_position_step=.02, max_rotation_step=.10)
                    actual_target = robot_to_world_ee_action(bounded, frame)
                    roundtrip = robot_to_world_ee_action(local_target, frame)
                    audit["frame_roundtrip_error"] = max(audit["frame_roundtrip_error"], float(abs(roundtrip-target).max()))
                    audit["clipped_position_steps"] += int(np.linalg.norm(actual_target[:3]-target[:3]) > 1e-6)
                    angles = actual_target[3:6] - target[3:6]
                    audit["clipped_rotation_steps"] += int(np.linalg.norm(np.arctan2(np.sin(angles),np.cos(angles))) > 1e-6)
                    try:
                        command = ee_action_to_env_action(env, actual_target, ik_tolerance=.005)
                    except InverseKinematicsError:
                        audit["ik_failures"] += 1
                        return audit
                    audit["steps"] += 1
                ts = env.step(command)
                env.physics.forward()
                if _task_success(env):
                    if audit_controller:
                        audit["native_success"] = True
                        return audit
                    return True
                if ts.last():
                    break
        return audit if audit_controller else False
    finally:
        env.close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("output", type=Path)
    cli.add_argument("--repair-resets", action="store_true",
                     help="Try the next source reset in the slot after a failed expert control or replay")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Use a fresh output path for the matched controls")
    positive_controls(args.output, args.repair_resets)
