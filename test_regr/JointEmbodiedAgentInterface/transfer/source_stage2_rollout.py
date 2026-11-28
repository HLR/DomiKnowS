"""Freeze and execute source VLABench scenes with a fixed native skill bridge."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from .protocol import digest, freeze, public_input, read, sha


PROTOCOL = "source-native-skills-v1"
HORIZON = 400
TASKS = ("select_chemistry_tube", "select_fruit", "insert_flower", "add_condiment")


def executor_fingerprint():
    from inspect import getfile
    from VLABench.utils.skill_lib import SkillLib
    from ..utils import matched_vlabench
    from test_regr.VLABenchAgentInterface.simulation import environment, program

    return digest([PROTOCOL, HORIZON, *map(sha, (
        __file__, matched_vlabench.__file__, environment.__file__, program.__file__, getfile(SkillLib),
    ))])


@contextmanager
def scene(task, seed):
    from ..utils.matched_vlabench import reset_seed
    from test_regr.VLABenchAgentInterface.simulation.environment import create_environment, reset_reward_tracking

    reset_seed(seed)
    env = create_environment(task, simulator_seed=seed)
    try:
        env.reset()
        reset_reward_tracking(env)
        yield env
    finally:
        env.close()


def observation(env):
    from test_regr.VLABenchAgentInterface.simulation.environment import numbered_views_from_observation

    obs = env.get_observation(require_pcd=False)
    if obs.get("segmentation") is None:
        raise ValueError("native source scene has no segmentation for image tags")
    views, entities = numbered_views_from_observation(env, obs, max_views=2)
    if len(views) != 2:
        raise ValueError("native source scene requires two camera views")
    images = []
    for view in views:
        with view, BytesIO() as stream:
            view.save(stream, format="PNG")
            images.append(stream.getvalue())
    public = {"instruction": env.task.get_instruction(), "entities": list(entities)}
    if not isinstance(public["instruction"], str) or not public["instruction"]:
        raise ValueError("source observation has no public task instruction")
    return public, images


def fingerprint(env):
    return sha256(np.round(env.physics.get_state(), 8).tobytes()).hexdigest()


def execute(env, row, plan):
    from ..utils.matched_vlabench import StepLimit, Trace, execute_plan
    from test_regr.VLABenchAgentInterface.simulation.program import _task_success

    trace = Trace(env, row["task"], row["seed"])
    reason = "completed"
    try:
        execute_plan(env, row["task"], plan, trace)
        if "failed_macro" in trace.row:
            reason = "macro_failed"
    except StepLimit:
        reason = "horizon"
    except (LookupError, ValueError, TypeError, AttributeError) as exc:
        reason = "manipulation_failed"
        trace.row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        env.step = trace.step
    # Native success is read only from the simulator. The trace is score-side.
    success = bool(_task_success(env))
    return {"success": success, "steps": trace.row["steps"], "termination_reason": reason,
            "trace": trace.row}


def verify_reset(env, row):
    from test_regr.VLABenchAgentInterface.simulation.program import _task_success

    public, images = observation(env)
    frozen = public_input(row)
    if (fingerprint(env) != row["scene_sha256"] or
            any(public[key] != frozen[key] for key in public) or
            [sha256(image).hexdigest() for image in images] != [row["files"][path] for path in frozen["images"]] or
            _task_success(env)):
        raise ValueError(f"source reset drift or initial native success: {row['episode_id']}")


def control(row, plan):
    with scene(row["task"], row["seed"]) as env:
        verify_reset(env, row)
        return execute(env, row, plan)


def controls(row, env, entities):
    from ..utils.matched_vlabench import expert_plan

    correct = expert_plan(env, row["task"], entities)
    source = correct[0]["params"]["target_entity_name"]
    obj = env.task.entities[entities[source]]
    comparable = [i for i, name in enumerate(entities)
                  if i != source and type(env.task.entities[name]) is type(obj)]
    if not comparable:
        raise ValueError(f"no same-kind wrong-source control: {row['episode_id']}")
    wrong = deepcopy(correct)
    for op in wrong:
        if op["params"].get("target_entity_name") == source:
            op["params"]["target_entity_name"] = comparable[0]
    return correct, wrong


def calibration_records(path, per_task):
    if path is None:
        return {}
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    grouped = {task: [row for row in rows if row["task"] == task] for task in TASKS}
    if (any(len(group) != per_task for group in grouped.values()) or len(rows) != len(TASKS) * per_task
            or any(row["native_success"] is not True or row["replay_success"] is not True for row in rows)
            or len({row["seed"] for row in rows}) != len(rows)):
        raise ValueError("archived controls must cover every calibration task and replay successfully")
    return grouped


def prepare(output, seed, scenes_per_task, archived_controls=None):
    if output.exists():
        raise ValueError("source preparation requires a fresh output directory")
    archive = calibration_records(archived_controls, scenes_per_task)
    rows, qualifications = [], []
    executor = executor_fingerprint()
    for split in ("calibration", "training", "validation"):
        for task in TASKS:
            for index in range(scenes_per_task):
                identity = f"vlabench/{task}/{split}-{index:03d}"
                archived = archive[task][index] if archive and split == "calibration" else None
                reset = archived["seed"] if archived else int(digest([PROTOCOL, seed, identity])[:8], 16)
                folder = output / identity
                folder.mkdir(parents=True)
                with scene(task, reset) as env:
                    if archived:
                        from ..utils.matched_vlabench import scene_id

                        if scene_id(env) != archived["scene"]:
                            raise ValueError(f"archived calibration scene changed: {identity}")
                    public, images = observation(env)
                    paths = [folder / f"view{i}.png" for i in range(len(images))]
                    for path, image in zip(paths, images, strict=True):
                        path.write_bytes(image)
                    generation = folder / "generation_input.json"
                    freeze(generation, {**public, "images": list(map(str, paths))})
                    record = {"episode_id": identity, "task": task, "split": split, "seed": reset,
                              "scene_sha256": fingerprint(env), "generation_input": str(generation),
                              "files": {str(path): sha(path) for path in (*paths, generation)}}
                    if split == "calibration":
                        plans = controls(record, env, public["entities"])
                freeze(folder / "record.json", record)
                rows.append(record)
                if split == "calibration":
                    positive, negative = (control(record, plan) for plan in plans)
                    qualifications.append({"episode_id": identity, "correct": positive, "wrong_source": negative})
                print(f"[source-stage2-prepare] {identity}", flush=True)
    report = {"passed": all(row["correct"]["success"] and not row["wrong_source"]["success"]
                            for row in qualifications),
              "executor_sha256": executor, "controls": qualifications}
    if archived_controls:
        report["positive_control_archive_sha256"] = sha(archived_controls)
    qualification = output / "qualification.json"
    freeze(qualification, report)
    manifest = {"protocol": PROTOCOL, "executor_sha256": executor, "horizon": HORIZON, "episodes": rows,
                "qualification": str(qualification), "qualification_sha256": sha(qualification)}
    verify_manifest(manifest)
    freeze(output / "manifest.json", manifest)


def verify_manifest(manifest):
    if manifest["protocol"] != PROTOCOL or manifest["horizon"] != HORIZON:
        raise ValueError("source rollout protocol differs")
    rows = manifest["episodes"]
    splits = ("calibration", "training", "validation")
    counts = [sum(row["split"] == split and row["task"] == task for row in rows)
              for split in splits for task in TASKS]
    if not counts[0] or len(set(counts)) != 1 or sum(counts) != len(rows):
        raise ValueError("source manifests require balanced and complete task coverage")
    for field in ("episode_id", "seed", "scene_sha256"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"source split overlap: {field}")
    image_splits = {}
    for row in rows:
        for image in public_input(row)["images"]:
            previous = image_splits.setdefault(row["files"][image], row["split"])
            if previous != row["split"]:
                raise ValueError("source splits share an input image")
    if sha(manifest["qualification"]) != manifest["qualification_sha256"]:
        raise ValueError("source qualification changed")
    proof = read(manifest["qualification"])
    controls = proof["controls"]
    if (proof["executor_sha256"] != manifest["executor_sha256"] or not proof["passed"] or
            len(controls) != counts[0] * len(TASKS) or
            {row["episode_id"] for row in controls} !=
            {row["episode_id"] for row in rows if row["split"] == "calibration"} or
            any(row["correct"]["success"] is not True or row["wrong_source"]["success"] is not False
                for row in controls)):
        raise ValueError("source executor failed correct-plan / wrong-source qualification")
    return rows


def rollout(request, result):
    data = read(request)
    if sha(data["manifest"]) != data["manifest_sha256"]:
        raise ValueError("source rollout manifest changed")
    manifest = read(data["manifest"])
    if manifest["executor_sha256"] != executor_fingerprint():
        raise ValueError("source executor differs from calibration")
    rows = {row["episode_id"]: row for row in verify_manifest(manifest) if row["split"] == data["split"]}
    if data["split"] not in {"training", "validation"}:
        raise ValueError("calibration cases cannot supply RL rewards")
    outcomes = [{"rollout_id": job["rollout_id"], **control(rows[job["episode_id"]], job["prediction"])}
                for job in data["jobs"]]
    freeze(result, {"request_sha256": sha(request), "rollouts": outcomes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes-per-task", type=int, default=3)
    parser.add_argument("--calibration-controls", type=Path, help="Previously replayed calibration JSONL")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.output and args.scenes_per_task >= 2 and not args.request and not args.result:
        prepare(args.output.resolve(), args.seed, args.scenes_per_task, args.calibration_controls)
    elif args.request and args.result and not args.output:
        rollout(args.request, args.result)
    else:
        parser.error("supply --output and at least two scenes per task, or --request and --result")


if __name__ == "__main__":
    main()
