"""Describe split provenance without treating requested holdouts as proof of exclusion."""

from test_regr.VLABenchAgentInterface.knowledge.holdout import holdout_tasks


def training_data_protocol(args, resume_payload=None):
    tasks = sorted(set(holdout_tasks(args)))
    previous = (resume_payload or {}).get("metrics", {}).get("training_data_protocol", {})
    verified = not getattr(args, "resume_adapter", None) and (not getattr(args, "resume", None) or (
        bool(previous.get("holdout_history_verified"))
        and previous.get("vlabench_holdout_tasks") == tasks
    ))
    return {"vlabench_split": "deterministic_episode_split", "split_seed": args.seed,
            "vlabench_holdout_tasks": tasks, "holdout_history_verified": verified}


def evaluation_protocol(args, payload):
    requested = sorted(set(holdout_tasks(args)))
    previous = payload.get("metrics", {}).get("training_data_protocol", {})
    heldout = args.split == "heldout"
    verified = bool(heldout and requested and previous.get("holdout_history_verified")
                    and set(requested).issubset(previous.get("vlabench_holdout_tasks", ())))
    scope = ("requested_task_holdout" if heldout else "training_episodes" if args.split == "train"
             else "all_episodes_including_training" if args.split == "all"
             else "episode_split_within_seen_tasks")
    return {"scope": scope, "requested_holdout_tasks": requested,
            "checkpoint_training_data_protocol": previous or None,
            "excluded_from_joint_training_verified": verified,
            "input_probe": getattr(args, "input_probe", "original")}
