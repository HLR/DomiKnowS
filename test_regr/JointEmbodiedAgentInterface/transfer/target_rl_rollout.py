"""Execute target planner rollouts in an isolated, qualified simulator runtime."""
import argparse
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "transfer"

from .protocol import freeze, read, sha


def executor_proof(dataset, source, calibration, evaluation_manifest):
    gate = read(calibration)
    if (gate["manifest_sha256"] != sha(evaluation_manifest) or gate["episodes"] != 100
            or not .9 <= gate["success_rate"] <= 1):
        raise ValueError("target executor calibration does not qualify")
    if dataset == "libero":
        fields = {"executor_source_sha256": "fixed_executor.py",
                  "bridge_source_sha256": "libero_campaign.py"}
    elif dataset == "rlbench":
        fields = {"executor_sha256": "rlbench_executor.py", "bridge_sha256": "rlbench_bridge.py",
                  "runner_sha256": "rlbench_campaign.py"}
    else:
        raise ValueError("unsupported target simulator")
    expected = {name: gate[key] for key, name in fields.items()}
    for name, fingerprint in gate.get("source_sha256", {}).items():
        if Path(name).name != name or expected.get(name, fingerprint) != fingerprint:
            raise ValueError("inconsistent qualified executor source manifest")
        expected[name] = fingerprint
    if dataset == "libero":
        # This module supplies the reset seed and canonical physical bindings.
        expected["libero_protocol.py"] = sha(Path(source) / "libero_protocol.py")
    if dataset == "rlbench":
        name = "rlbench_reset_validation.py"
        if "reset_validation" in gate:
            fingerprint = gate["reset_validation"]["validator_sha256"]
            if expected.get(name, fingerprint) != fingerprint:
                raise ValueError("inconsistent qualified reset verifier")
            expected[name] = fingerprint
        if name not in expected:
            raise ValueError("qualified RLBench reset verifier is required")
    if any(sha(Path(source) / name) != fingerprint for name, fingerprint in expected.items()):
        raise ValueError("qualified target executor source changed")
    return expected


def run_libero(bridge, jobs, records, stack):
    env = None
    current = None
    for job in sorted(jobs, key=lambda job: records[job["episode_id"]]["bddl"]):
        record = records[job["episode_id"]]
        if current != record["bddl"]:
            stack.close()
            env = bridge.environment(record["bddl"], render=False)
            stack.callback(env.close)
            current = record["bddl"]
        obs = bridge.reset_episode(env, record)
        private = read(record["scoring_private"])
        plan = job["prediction"]
        result = bridge.FixedExecutor(env, private["entities"], horizon=1000).execute(plan, obs)
        success = bool(env.check_success())
        from libero_protocol import abstract_score
        yield {**result, **abstract_score(plan, private), "rollout_id": job["rollout_id"], "success": success}


def run_rlbench(module, jobs, records, stack):
    from pyrep.const import ObjectType
    from pyrep.objects.object import Object
    from rlbench_reset_validation import validate_reset

    bridge = module.RLBenchBridge(render=False)
    stack.callback(bridge.close)
    for job in jobs:
        record = records[job["episode_id"]]
        state = read(record["state"])
        entities = read(record["scoring_private"])["entities"]
        bridge.reset(record["task"], record["variation"], record["seed"], state=state, entities=entities)
        restored = {row["name"]: row for row in bridge.snapshot_state()}
        robot_names = {obj.get_name() for obj in bridge.env._robot.arm.get_objects_in_tree()}
        dummy_names = {row["name"] for row in state if Object.get_object(row["name"]).get_type() == ObjectType.DUMMY}
        reset = validate_reset(state, restored, robot_names=robot_names,
                               robot_base=bridge.env._robot.arm.get_name(),
                               task_base=bridge.task._task.get_base().get_name(),
                               dummy_names=dummy_names, pose_error=module.pose_error)
        result = module.RLBenchExecutor(bridge, horizon=1000).execute(job["prediction"])
        yield {**result, **reset, "rollout_id": job["rollout_id"], "success": bool(bridge.native_success())}


def execute(request):
    dataset = request["dataset"]
    source = Path(request["executor_source"]).resolve()
    proof = executor_proof(dataset, source, request["calibration"], request["evaluation_manifest"])
    if (proof != request["executor_sha256"] or sha(request["calibration"]) != request["calibration_sha256"]
            or sha(request["manifest"]) != request["manifest_sha256"]
            or sha(request["evaluation_manifest"]) != request["evaluation_manifest_sha256"]):
        raise ValueError("native rollout contract changed")
    manifest = read(request["manifest"])
    records = {row["episode_id"]: row for row in manifest["episodes"] if row["split"] == request["split"]}
    jobs = request["jobs"]
    if len({job["rollout_id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate target rollout IDs")
    for episode_id in {job["episode_id"] for job in jobs}:
        row = records[episode_id]
        for field in ("state", "scoring_private"):
            path = row[field]
            if sha(path) != row["files"][path]:
                raise ValueError("native reset or physical binding changed")
        if dataset == "libero" and sha(row["bddl"]) != request["bddl_sha256"][row["bddl"]]:
            raise ValueError("frozen LIBERO task definition changed")
    if not jobs:
        return []
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module(dataset + "_campaign")
        if Path(module.__file__).resolve().parent != source:
            raise ValueError("native runner loaded from the wrong source directory")
        with ExitStack() as stack:
            if dataset == "libero":
                return list(run_libero(module, jobs, records, stack))
            if module.runtime_provenance() != read(request["evaluation_manifest"])["runtime"]:
                raise ValueError("RLBench runtime differs from qualified evaluation")
            return list(run_rlbench(module, jobs, records, stack))
    finally:
        sys.path.remove(str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.result, {"request_sha256": sha(args.request), "rollouts": execute(read(args.request))})


if __name__ == "__main__":
    main()
