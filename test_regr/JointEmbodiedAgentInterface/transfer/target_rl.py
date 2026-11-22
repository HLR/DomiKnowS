"""Warm-start the Joint graph planner with native target-environment rewards."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "transfer"

from .data import audit_disjoint
from .matched_baseline.protocol import common_prompt, raw_format_valid
from .prepare_data import fingerprints
from .protocol import digest, episodes, freeze, public_input, read, sha
from .repair.input_protocol import IMAGE_INSTRUCTION, constraint_context, contract_record, generation_context
from .repair.protocol import reject_reserved
from .target_rl_rollout import executor_proof


MAX_LABELS = 61


def checked_fingerprints(manifest, dataset, split=None):
    result = []
    for row in manifest["episodes"]:
        for name in ("state", "generation_input", "scoring_private"):
            if sha(row[name]) != row["files"][row[name]]:
                raise ValueError("frozen target input changed: " + row[name])
        public = public_input(row)
        result.append({**fingerprints(row, dataset), "split": split or row["split"],
                       "input_fingerprint": digest({"instruction": public["instruction"],
                           "images": [row["files"][path] for path in public["images"]]})})
    return result


def target_splits(dataset, target_path, eval_path, reserved_paths, reservations):
    from .repair.reservations import guard

    target = read(target_path)
    evaluation = read(eval_path)
    if target["dataset"] != dataset or not target["integrity"]["passed"]:
        raise ValueError("target preparation has not passed its integrity gate")
    train = [row for row in target["episodes"] if row["split"] == "training"]
    valid = [row for row in target["episodes"] if row["split"] == "validation"]
    if not train or not valid or len(train) + len(valid) != len(target["episodes"]):
        raise ValueError("target manifest requires disjoint training and validation episodes only")
    reserved_paths = list(dict.fromkeys([eval_path, *reserved_paths]))
    if target["reserved_manifest_sha256"] not in {sha(p) for p in reserved_paths}:
        raise ValueError("supply the original holdout manifest used to prepare the target data")
    guard(target, reservations)
    if dataset == "libero":
        if any(35 <= row["state_index"] <= 39 for row in target["episodes"]):
            raise ValueError("LIBERO confirmation states 35–39 are reserved")
        for path in reserved_paths:
            reject_reserved(target, {"episodes": read(path)["episodes"]})
    reserved = [row for path in reserved_paths
                for row in checked_fingerprints(read(path), dataset, split="reserved")]
    proof = audit_disjoint(checked_fingerprints(target, dataset), reserved)
    return train, valid, episodes(evaluation), proof


def predict(planner, runtime, rows, policy_name, samples, greedy):
    import torch
    from ..generalization.shared_constraints import bind_shared
    from ..models import FreeDecoding
    from ..utils.policy_reference import deterministic_policy
    from .anonymous_grounding.planner_binding.verification_protocol import exact_policy_search
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan, plan_to_tokens

    vocab = runtime.vlabench_vocabulary
    planner.free_decoding = policy_name == "none"
    traces = []
    with torch.no_grad(), deterministic_policy(planner, inference=True), runtime.domain_scope("vlabench"):
        for row in rows:
            public = public_input(row)
            bound = generation_context(public, vocab)
            bindings = None if planner.free_decoding else constraint_context(public)
            policy = (FreeDecoding(vocab.label_count, vocab.eos_label) if planner.free_decoding
                      else bind_shared(vocab, 12, bindings))
            input_hash = digest({"prompt": bound["_matched_core"],
                                 "images": [row["files"][path] for path in bound["images"]],
                                 "policy": policy_name, "constraint_bindings": bindings})
            for sample in range(samples):
                retries = []
                with exact_policy_search(policy, retries):
                    labels, _ = planner.sample_labels("vlabench", bound, policy, max_steps=MAX_LABELS,
                                                      deterministic=greedy)
                prediction = []
                error = None
                try:
                    if not labels or labels[-1] != vocab.eos_label:
                        raise ValueError("plan reached its token limit without EOS")
                    decoded = labels_to_plan(labels, vocab, world=runtime.world.vlabench)
                    canonical = plan_to_tokens(decoded, bound["entity_table"], world=runtime.world.vlabench)
                    if ([vocab.token_for_label(label) for label in labels] != canonical
                            or not raw_format_valid(decoded, vocab, len(bound["entity_table"]))):
                        raise ValueError("noncanonical or invalid graph-label plan")
                    prediction = decoded
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                traces.append(({"rollout_id": f'{row["episode_id"]}#{sample}',
                                "episode_id": row["episode_id"], "labels": labels, "prediction": prediction,
                                "input_hash": input_hash, "generation_error": error,
                                "search_retries": retries}, bound, policy))
    return traces


def score(traces, folder, contract, command):
    rows = [trace[0] for trace in traces]
    # Invalid emissions remain in all denominators and receive zero reward.
    jobs = [{key: row[key] for key in ("rollout_id", "episode_id", "prediction")}
            for row in rows if row["generation_error"] is None]
    folder.mkdir(parents=True, exist_ok=False)
    request = folder / "request.json"
    result = folder / "native.json"
    freeze(request, {**contract, "jobs": jobs})
    freeze(folder / "predictions.json", rows)
    native = {}
    if jobs:
        argv = [*command, "--request", str(request.resolve()), "--result", str(result.resolve())]
        freeze(folder / "command.json", argv)
        with (folder / "native.log").open("x", encoding="utf-8") as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        response = read(result)
        native = {row["rollout_id"]: row for row in response["rollouts"]}
        if (response["request_sha256"] != sha(request) or len(native) != len(response["rollouts"])
                or set(native) != {job["rollout_id"] for job in jobs}
                or any(type(row["success"]) is not bool for row in native.values())):
            raise ValueError("native worker result does not match the requested rollout batch")
    records = [{**row, "native": native.get(row["rollout_id"]),
                "success": native.get(row["rollout_id"], {}).get("success", False)} for row in rows]
    freeze(folder / "scores.json", records)
    return records


def advantages(records):
    """Center within a reset, including failed decodes with reward zero."""
    rewards = {}
    for row in records:
        rewards.setdefault(row["episode_id"], []).append(float(row["success"]))
    means = {episode: sum(values) / len(values) for episode, values in rewards.items()}
    return [float(row["success"]) - means[row["episode_id"]] for row in records]


def metrics(records):
    successes = sum(row["success"] for row in records)
    native = [row["native"] for row in records if row["native"] is not None]
    result = {"episodes": len(records), "successes": successes, "native_success_rate": successes / len(records),
              "valid_plans": sum(row["generation_error"] is None for row in records),
              "executor_failures": sum(row["termination_reason"] != "completed" for row in native)}
    if any("goal_recall" in row for row in native):
        result["abstract_goal_recall"] = sum(row["goal_recall"] for row in native) / len(records)
    return result


def paired_report(before, after):
    baseline = {row["episode_id"]: row for row in before}
    adapted = {row["episode_id"]: row for row in after}
    if (len(baseline) != len(before) or len(adapted) != len(after) or baseline.keys() != adapted.keys()
            or any(baseline[k]["input_hash"] != adapted[k]["input_hash"] for k in baseline)):
        raise ValueError("paired evaluation input or reset coverage differs")
    gains = sum(adapted[k]["success"] and not baseline[k]["success"] for k in baseline)
    losses = sum(baseline[k]["success"] and not adapted[k]["success"] for k in baseline)
    return {"baseline": metrics(before), "adapted": metrics(after), "gains": gains, "losses": losses,
            "success_rate_delta": (gains - losses) / len(before)}


def update(planner, runtime, optimizer, reference, traces, records, source_row, eai_item, anchor_weights):
    import torch
    from ..utils.planner_context import planner_context
    from ..utils.policy_reference import deterministic_policy, regularized_policy_replay
    from .anonymous_grounding.planner_binding.verification_protocol import exact_policy_search
    from .repair.common import source_context
    from .repair.objective import eai_loss, planning_loss
    from .repair.protocol import normalize_anonymous_prediction
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan

    weights = advantages(records)
    if not any(weight and record["labels"] for weight, record in zip(weights, records, strict=True)):
        return False
    planner.train()
    optimizer.zero_grad(set_to_none=True)
    with runtime.domain_scope("vlabench"):
        for (trace, context, policy), weight in zip(traces, weights, strict=True):
            if not trace["labels"]:
                continue
            with exact_policy_search(policy, []), deterministic_policy(planner):
                pg, kl = regularized_policy_replay(planner, reference, "vlabench", context, trace["labels"],
                                                   policy, torch.tensor(weight), max_steps=MAX_LABELS)
                ((pg + .05 * kl) / len(weights)).backward()
        row, visible = source_row
        labels = row["item"]["target_plan_labels"]
        bound = source_context(planner, row)
        vocab = runtime.vlabench_vocabulary
        plan = labels_to_plan(labels.tolist(), vocab, world=runtime.world.vlabench)
        plan, _ = normalize_anonymous_prediction(plan, len(bound["entity_table"]), vocab.argument_keys)
        vla, _ = planning_loss(planner, bound, plan, labels, visible, "joint")
        (anchor_weights[0] * vla).backward()
    with runtime.domain_scope("eai"):
        eai, _ = eai_loss(planner, planner_context("eai", eai_item), eai_item["target_action_labels"])
        (anchor_weights[1] * eai).backward()
    torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]],
                                  1., error_if_nonfinite=True)
    optimizer.step()
    return True


def source_anchors(config, runtime, splits):
    from .anonymous_grounding.train import records_for

    observations = read(config["source_observations"])
    bindings = {row["episode_id"]: row for row in observations["episodes"]}
    if not observations["complete"] or observations["test_used"] or observations["references_read"]:
        raise ValueError("source replay observations are not qualified")
    for row in bindings.values():
        if any(sha(path) != expected for path, expected in row["files"].items()):
            raise ValueError("source replay observation changed")
    features = {row["episode_id"]: row for row in read(config["features"])["episodes"]}
    training = records_for(runtime, splits["train"], bindings, features, 0, augmented=False)[2]
    validation = records_for(runtime, splits["validation"], bindings, features, 0, augmented=False)[2]
    return training, validation


def visible_entities(row):
    import torch

    feature = row["features"]
    if sha(feature["path"]) != feature["sha256"]:
        raise ValueError("source pointer visibility feature changed")
    return torch.load(feature["path"], map_location="cpu", weights_only=True)["visible"][row["order"]]


def train(args):
    from .repair.protocol import verify

    c = read(args.source_config)
    verify(c)
    source_root = Path(c["output"])
    selection = read(source_root / "frozen_selection.json")["systems"]["joint-trained"]
    if not selection["qualified"] or sha(selection["path"]) != selection["sha256"]:
        raise ValueError("qualified source-selected Joint checkpoint required")
    reserved = list(args.holdout_manifest)
    if args.dataset == "libero":
        reserved += [Path(c["historical_manifest"]), source_root / "confirmation/manifest.json"]
    training, valid, test, integrity = target_splits(
        args.dataset, args.target_manifest, args.evaluation_manifest, reserved, args.reservations,
    )
    proof = executor_proof(args.dataset, args.executor_source, args.calibration, args.evaluation_manifest)
    worker = args.worker or [sys.executable, str(Path(__file__).with_name("target_rl_rollout.py").resolve())]
    common = {"dataset": args.dataset, "executor_source": str(args.executor_source.resolve()),
              "executor_sha256": proof, "calibration": str(args.calibration.resolve()),
              "calibration_sha256": sha(args.calibration),
              "evaluation_manifest": str(args.evaluation_manifest.resolve()),
              "evaluation_manifest_sha256": sha(args.evaluation_manifest)}
    if args.dataset == "libero":
        common["bddl_sha256"] = {path: sha(path) for path in sorted({
            row["bddl"] for row in [*training, *valid, *test]
        })}
    target_contract = {**common, "manifest": str(args.target_manifest.resolve()),
                       "manifest_sha256": sha(args.target_manifest)}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    settings = output / "settings.json"
    freeze(settings, {**target_contract, **contract_record(IMAGE_INSTRUCTION), "policy": args.policy,
                      "source_config_sha256": sha(args.source_config), "source_checkpoint": selection,
                      "split_integrity": integrity, "holdouts": {str(path): sha(path) for path in reserved},
                      "worker": worker, "worker_sha256": sha(Path(__file__).with_name("target_rl_rollout.py")),
                      "trainer_sha256": sha(__file__),
                      "epochs": args.epochs, "scenes_per_batch": args.scenes_per_batch,
                      "samples": args.samples, "seed": 42, "lora_lr": args.lora_lr, "head_lr": args.head_lr,
                      "kl_weight": .05, "anchor_weights": [.25, .5],
                      "max_eai_loss": args.max_eai_loss, "max_progress_loss": args.max_progress_loss})
    if args.validate_only:
        return
    import torch
    from ..utils.policy_reference import FrozenPolicyReference
    from .repair.common import build_runtime, fresh_planner, load_checkpoint, save_checkpoint
    from .repair.train import validation

    _, (runtime, eai_train, eai_valid, splits, _) = build_runtime(c, output / "runtime", args.device)
    planner = fresh_planner(c, runtime, "joint")
    load_checkpoint(selection["path"], planner, "joint")
    for name, parameter in planner.named_parameters():
        parameter.requires_grad_(parameter.requires_grad and
                                 ("lora_" in name or name.startswith("label_heads.vlabench.")))
    planner._base_parameter_requires_grad = {p: p.requires_grad for p in planner.parameters()}
    lora = [p for name, p in planner.named_parameters() if p.requires_grad and "lora_" in name]
    head = [p for p in planner.label_heads["vlabench"].parameters() if p.requires_grad]
    if not lora or not head:
        raise ValueError("requires trainable LoRA and Joint graph head")
    optimizer = torch.optim.AdamW([{"params": lora, "lr": args.lora_lr},
                                   {"params": head, "lr": args.head_lr}], weight_decay=.01)
    reference = FrozenPolicyReference(planner, optimizer)
    source_rows, source_valid = source_anchors(c, runtime, splits)
    baseline_source = validation(planner, runtime, source_valid, splits["validation"], eai_valid,
                                 "joint", output / "source_baseline")
    validation_contract = {**target_contract, "split": "validation"}
    with common_prompt(planner):
        initial = score(predict(planner, runtime, valid, args.policy, 1, True),
                        output / "target_baseline", validation_contract, worker)
    history = []
    rng = random.Random(42)
    for epoch in range(1, args.epochs + 1):
        folder = output / f"epoch_{epoch:03d}"
        order = rng.sample(training, len(training))
        updates = 0
        with common_prompt(planner):
            for offset in range(0, len(order), args.scenes_per_batch):
                batch = order[offset:offset + args.scenes_per_batch]
                traces = predict(planner, runtime, batch, args.policy, args.samples, False)
                results = score(traces, folder / f"batch_{offset:04d}",
                                {**target_contract, "split": "training"}, worker)
                if any(advantages(results)):
                    row = rng.choice(source_rows)
                    updates += update(planner, runtime, optimizer, reference, traces, results,
                                      (row, visible_entities(row)), rng.choice(eai_train), (.25, .5))
                print(f"[target-rl] epoch={epoch} scenes={offset + len(batch)}/{len(order)} "
                      f"updates={updates} successes={sum(r['success'] for r in results)}", flush=True)
            adapted = score(predict(planner, runtime, valid, args.policy, 1, True),
                            folder / "target_validation", validation_contract, worker)
        source = validation(planner, runtime, source_valid, splits["validation"], eai_valid,
                            "joint", folder / "source_validation")
        checkpoint = folder / "planner.pt"
        save_checkpoint(checkpoint, planner, runtime, "joint", epoch, sha(settings),
                        provenance={"method": "target-rl", "initial_checkpoint_sha256": selection["sha256"],
                                    "reference_sha256": reference.fingerprint})
        result = {"epoch": epoch, "updates": updates, "source": source, "target": paired_report(initial, adapted),
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                  "eligible": bool(updates) and
                  source["eai_successes"] >= baseline_source["eai_successes"] - args.max_eai_loss and
                  source["progress_score"] >= baseline_source["progress_score"] - args.max_progress_loss}
        freeze(folder / "result.json", result)
        history.append(result)
    best = max((row for row in history if row["eligible"]),
               key=lambda row: (row["target"]["adapted"]["successes"], -row["epoch"]), default=None)
    freeze(output / "selection.json", {"selected": best, "history": history})
    if best is None:
        freeze(output / "report.json", {"qualified": False, "reason": "no updated epoch passed source retention"})
        return

    test_contract = {**common, "split": "evaluation", "manifest": common["evaluation_manifest"],
                     "manifest_sha256": common["evaluation_manifest_sha256"]}
    if sha(best["checkpoint"]) != best["checkpoint_sha256"] or sha(selection["path"]) != selection["sha256"]:
        raise ValueError("selected model changed before final evaluation")
    with common_prompt(planner):
        load_checkpoint(selection["path"], planner, "joint")
        before = score(predict(planner, runtime, test, args.policy, 1, True), output / "test_source",
                       test_contract, worker)
        load_checkpoint(best["checkpoint"], planner, "joint")
        after = score(predict(planner, runtime, test, args.policy, 1, True), output / "test_adapted",
                      test_contract, worker)
    freeze(output / "report.json", {"qualified": True, "selected_epoch": best["epoch"],
                                    "test": paired_report(before, after)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-config", "target-manifest", "evaluation-manifest", "executor-source", "calibration", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--holdout-manifest", action="append", default=[], type=Path)
    parser.add_argument("--reservations", type=Path)
    parser.add_argument("--dataset", choices=("libero", "rlbench"), required=True)
    parser.add_argument("--policy", choices=("none", "shared-verified"), default="none")
    parser.add_argument("--worker", type=json.loads, help="JSON argv prefix for the simulator worker")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--scenes-per-batch", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-eai-loss", type=int, default=0)
    parser.add_argument("--max-progress-loss", type=float, default=1.)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.samples < 2 or args.scenes_per_batch < 1 or
            not (0 < args.lora_lr < float("inf") and 0 < args.head_lr < float("inf")) or
            args.max_eai_loss < 0 or not 0 <= args.max_progress_loss < float("inf")):
        parser.error("invalid training budget, learning rate, or retention threshold")
    if args.worker is not None and (not isinstance(args.worker, list) or not args.worker
                                    or not all(isinstance(item, str) and item for item in args.worker)):
        parser.error("--worker must be a nonempty JSON array of argument strings")
    train(args)


if __name__ == "__main__":
    main()
