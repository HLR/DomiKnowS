"""Run a source-only Joint planner pilot and a separate matched LIBERO check."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys

import torch

from ..generalization.executor_policy import Action, DESTINATION, EXECUTOR, ExecutorProfile, SUBJECT, pick_place_profile
from ..utils import observed_eai
from ..utils.policy_reference import FrozenPolicyReference, deterministic_policy, regularized_policy_replay
from .anonymous_grounding.planner_binding.verification_protocol import exact_policy_search
from .batch_gradients import BatchGradientManager
from .data import audit_disjoint
from .fresh_metadata.checkpoint import checkpoint_record, load_planner_snapshot
from .matched_baseline.protocol import common_prompt
from .protocol import digest, episodes, freeze, read, sha
from .repair.common import build_runtime, fresh_planner, load_checkpoint, save_checkpoint
from .repair.input_protocol import contract_record, generation_context, IMAGE_INSTRUCTION
from .repair.protocol import verify
from .source_stage2_es import Population
from .source_stage2_rollout import verify_manifest
from .target_rl import MAX_LABELS, advantages, checked_fingerprints, paired_report, predict, score, source_anchors, visible_entities
from .target_rl_rollout import executor_proof


RECIPE = {
    "version": "joint-source-planner-stage2-v1", "seed": 42, "generations": 2, "draws": 4,
    "lora_lr": 5e-6, "head_lr": 1e-4, "kl_weight": .05, "eai_replay_weight": .5,
    "vlabench_replay_weight": .25, "target_rewards": False, "controllers_trainable": False,
}
ES_SIGMA = .001
SOURCE_PROFILE = ExecutorProfile((
    Action("pick", (SUBJECT,), grasp=True),
    Action("insert", (DESTINATION,), requires_held=True),
    Action("pour", (DESTINATION,), requires_held=True),
), capacities=(("right", 1),), initial_holdings=(("right", None),))


def source_provenance():
    files = (Path(__file__), Path(__file__).with_name("source_stage2_rollout.py"),
             Path(__file__).with_name("source_stage2_es.py"),
             Path(__file__).with_name("target_rl.py"), Path(observed_eai.__file__),
             Path(observed_eai.__file__).with_name("policy_reference.py"))
    return {path.name: sha(path) for path in files}


def worker(argv, module):
    if argv is None:
        return [sys.executable, "-m", f"{__package__}.{module}"]
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
        raise ValueError("worker must be a nonempty JSON argument array")
    return argv


def eai_rollouts(planner, runtime, items, draws, *, greedy):
    records, replay = [], []
    with torch.no_grad(), deterministic_policy(planner, inference=True), runtime.domain_scope("eai"):
        planner.free_decoding = False
        view = planner.for_domain("eai")
        for position, (identity, item) in enumerate(items, 1):
            context, goal = observed_eai.input_and_goal(item, runtime.world.eai)
            prompt = planner._chat(planner._prompt_base("eai", context), 0)
            policy = runtime.dfa_for("eai", context)
            for index in range(draws):
                retries = []
                with exact_policy_search(policy, retries):
                    labels, result = observed_eai.sample(view, context, policy, goal, runtime, deterministic=greedy)
                records.append({"episode_id": identity, "draw": index, "labels": list(labels),
                    "input_hash": digest([prompt, context]), "rendered_prompt": prompt,
                    "success": bool(result["is_success"]), "recall": result["recall"],
                    "unresolved": list(result["unresolved"]), "generation_error": result.get("generation_error"),
                    "execution_error": result.get("execution_error"),
                    "search_retries": retries})
                replay.append((records[-1], context, policy))
            if greedy and position % 8 == 0:
                print(f"[source-stage2] EAI validation {position}/{len(items)}", flush=True)
    return records, replay


def eai_scores(rows):
    return {"episodes": len(rows), "successes": sum(row["success"] for row in rows),
            "mean_goal_recall": sum(row["recall"] for row in rows) / len(rows),
            "unresolved": sum(bool(row["unresolved"]) for row in rows),
            "execution_errors": sum(bool(row.get("execution_error")) for row in rows),
            "generation_errors": sum(bool(row["generation_error"]) for row in rows)}


def capture_policy(manager, planner, runtime, reference, domain, traces, outcomes, *, es=False):
    maximum = runtime.max_eai_steps if domain == "eai" else MAX_LABELS
    weights = [0.] * len(outcomes) if es else advantages(outcomes)
    replays = signal = 0
    kl_sum = 0.0
    for (trace, context, policy), weight in zip(traces, weights, strict=True):
        if trace["generation_error"] or not trace["labels"]:
            continue
        with exact_policy_search(policy, []):
            pg, kl = regularized_policy_replay(
                planner, reference, domain, context, trace["labels"], policy, torch.tensor(weight),
                max_steps=maximum,
            )
            loss = (pg + RECIPE["kl_weight"] * kl) / len(weights)
            if not bool(loss.isfinite()):
                raise RuntimeError("nonfinite planner replay loss")
            manager.capture(domain, loss)
            replays += 1
            signal += int(weight != 0)
            kl_sum += float(kl.detach())
    return {"trajectories": replays, "nonzero_advantages": signal, "mean_kl": kl_sum / max(1, replays)}


def anchor_vla(planner, runtime, row):
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
    from .repair.objective import planning_loss
    from .repair.protocol import normalize_anonymous_prediction
    from ..utils.planner_context import planner_context

    vocab = runtime.vlabench_vocabulary
    source = planner_context("vlabench", row["item"])
    context = generation_context({"instruction": source["instruction"], "images": source["images"],
                                  "entities": source["entity_table"]}, vocab)
    labels = row["item"]["target_plan_labels"]
    plan = labels_to_plan(labels.tolist(), vocab, world=runtime.world.vlabench)
    plan, _ = normalize_anonymous_prediction(plan, len(context["entity_table"]), vocab.argument_keys)
    loss, _ = planning_loss(planner, context, plan, labels, visible_entities(row), "joint")
    return loss


def vla_rollouts(planner, runtime, rows, draws, *, greedy, output, contract, command, profile=SOURCE_PROFILE):
    with common_prompt(planner):
        traces = predict(planner, runtime, rows, EXECUTOR, draws, greedy, profile=profile)
    return score(traces, output, contract, command), traces


def collect_es_traces(parameters, population, collect):
    def draw(candidate):
        traces = collect()
        for record, _, _ in traces:
            record["es_candidate"] = candidate
            if "rollout_id" in record:
                record["rollout_id"] = f'{record["episode_id"]}#{candidate}'
            else:
                record["draw"] = candidate
        return traces

    return [trace for candidate in population.collect(parameters, ES_SIGMA, draw) for trace in candidate]


def evolution_rollouts(planner, runtime, manager, eai_items, vla_batch, step, contract, command, epoch, index):
    with runtime.domain_scope("eai", parameter_policy="freeze_inactive"):
        eai_parameters = [parameter for parameter in manager.parameters if parameter.requires_grad]
        eai_population = Population.from_budget(RECIPE["draws"], RECIPE["seed"], epoch, index, "eai")
        eai_traces = collect_es_traces(eai_parameters, eai_population,
            lambda: eai_rollouts(planner, runtime, eai_items, 1, greedy=False)[1])
    with common_prompt(planner), runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
        vla_parameters = [parameter for parameter in manager.parameters if parameter.requires_grad]
        vla_population = Population.from_budget(RECIPE["draws"], RECIPE["seed"], epoch, index, "vlabench")
        vla_traces = collect_es_traces(vla_parameters, vla_population,
            lambda: predict(planner, runtime, vla_batch, EXECUTOR, 1, False, profile=SOURCE_PROFILE))
    eai_records = [row for row, _, _ in eai_traces]
    vla_records = score(vla_traces, step / "native", contract, command)
    domains = {
        "eai": (eai_population, eai_parameters, eai_population.reward_differences(eai_records)),
        "vlabench": (vla_population, vla_parameters, vla_population.reward_differences(vla_records)),
    }
    return eai_records, eai_traces, vla_records, vla_traces, domains


def training_schedule(rows, anchor_rows, rng):
    groups, anchors = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    for row in anchor_rows:
        item = row["item"]
        anchors[item.get("task") or item.get("task_name")].append(row)
    keys = sorted(anchors, key=str)
    actions = ("left of", "right of", "next to", "between", "closest", "farthest",
               "leftmost", "rightmost", "in front of", "behind")
    from ..utils.planner_context import planner_context

    spatial = [row for row in anchor_rows if any(term in
               planner_context("vlabench", row["item"])["instruction"].lower() for term in actions)]
    if not groups or not keys or not spatial:
        raise ValueError("pilot requires source task cohorts and observed spatial replay examples")
    rounds = []
    for round_index, batch in enumerate(zip(*(rng.sample(group, len(group)) for group in groups.values()), strict=True)):
        generation_replay = [rng.choice(spatial), rng.choice(anchors[keys[round_index % len(keys)]])]
        rounds.append((batch, generation_replay))
    return rounds


def validate(planner, runtime, eai, vla, output, contract, command):
    eai_rows, _ = eai_rollouts(planner, runtime, eai, 1, greedy=True)
    freeze(output / "eai.json", eai_rows)
    vla_rows, _ = vla_rollouts(planner, runtime, vla, 1, greedy=True, output=output / "native",
                              contract=contract, command=command)
    return eai_rows, vla_rows


def qualify(baseline, after, updates, *, optimizer_mode="reinforce"):
    before_eai, after_eai = eai_scores(baseline[0]), eai_scores(after[0])
    if [(row["episode_id"], row["input_hash"]) for row in baseline[0]] != [
            (row["episode_id"], row["input_hash"]) for row in after[0]]:
        raise ValueError("matched EAI inputs changed")
    vla = paired_report(baseline[1], after[1], clustered=True)
    reasons = []
    if not all(updates.values()):
        reasons.append("missing_es_reward_signal" if optimizer_mode == "es" else "missing_policy_gradient_signal")
    if after_eai["successes"] < before_eai["successes"]:
        reasons.append("eai_success_regression")
    if after_eai["mean_goal_recall"] + .02 < before_eai["mean_goal_recall"]:
        reasons.append("eai_recall_regression")
    if vla["success_rate_delta"] < 0:
        reasons.append("vlabench_native_regression")
    return {"optimizer_mode": optimizer_mode, "eai": {"baseline": before_eai, "after": after_eai}, "vlabench": vla,
            "updated_batches": updates, "qualified": not reasons, "rejections": reasons}


def transfer_partition(holdout, archive, config, exclusions):
    test = episodes(read(holdout))
    audit_path = holdout.with_name("split_audit.json")
    audit = read(audit_path)
    if (audit["passed"] is not True or audit.get("seed_namespace") != "source-stage2-holdout-v1"
            or audit["official_states_per_task"] != 50 or
            {row["episode_id"] for row in audit["scenes"]} != {row["episode_id"] for row in test}):
        raise ValueError("fresh LIBERO holdout audit did not certify all evaluation resets")
    audited = {Path(path).resolve(): checksum for path, checksum in audit["additional_exclusions"].items()}
    if any(audited.get(path.resolve()) != sha(path) for path in exclusions):
        raise ValueError("a previous manifest was absent from the holdout preparation audit")
    prior = read(archive / "protocol.json")["external_manifest"]
    excluded = {Path(prior).resolve(), Path(config["historical_manifest"]).resolve(),
                *(path.resolve() for path in exclusions)}
    if (not exclusions or any(row["state_index"] < 50 or row.get("sampling_seed") is None for row in test)
            or holdout.resolve() in excluded):
        raise ValueError("supply fresh native LIBERO scenes and all prior scored manifests")
    prepared = checked_fingerprints({"episodes": test}, "libero")
    reserved = [row for path in excluded for row in checked_fingerprints(read(path), "libero", "reserved")]
    evidence = {"manifest_sha256": sha(holdout), "sampling_audit_sha256": sha(audit_path),
                "exclusions": {str(path): sha(path) for path in sorted(excluded)},
                "partition": audit_disjoint(prepared, reserved)}
    return test, evidence


def train(args):
    from .repair.objective import eai_loss
    from ..program import JointSolverPOIProgram
    from ..utils.stage2_training import SharedUpdateEvidence

    config, initialization = checkpoint_record(args.archive, 6)
    verify(config)
    source = read(args.source_manifest)
    rows = verify_manifest(source)
    _, holdout = transfer_partition(args.holdout, args.archive, config, args.excluded_manifest)
    command = worker(args.worker, "source_stage2_rollout")
    contract = {"manifest": str(args.source_manifest.resolve()), "manifest_sha256": sha(args.source_manifest)}
    settings = {"recipe": RECIPE, "initialization": initialization, "source": contract, "holdout": holdout,
                "executor_sha256": source["executor_sha256"], "executor_profile": asdict(SOURCE_PROFILE),
                "inputs": contract_record(IMAGE_INSTRUCTION), "eai_reward": observed_eai.OBSERVED,
                "auxiliary_heads": "disabled", "optimizer_mode": args.optimizer,
                "code_sha256": source_provenance(), "worker": command}
    es = args.optimizer == "es"
    if es:
        settings["evolution_strategies"] = {"sigma": ES_SIGMA, "pairs": RECIPE["draws"] // 2,
                                           "reward": "mean source success", "pair_sampling_seeds": "shared"}
    if args.validate_only:
        print(json.dumps(settings))
        return
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    freeze(root / "settings.json", settings)
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    freeze(root / "launch.json", {"pid": os.getpid(), "argv": sys.argv,
                                 "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                                 "gpu_uuid": str(gpu.uuid) if getattr(gpu, "uuid", None) else None})
    _, (runtime, train_eai, valid_eai, splits, _) = build_runtime(config, root / "runtime", args.device)
    if len(valid_eai) != 88:
        raise ValueError("full 88-example EAI validation is required")
    planner = load_planner_snapshot(fresh_planner(config, runtime, "joint"), initialization)
    lora = [p for name, p in planner.named_parameters() if "lora_" in name and p.requires_grad]
    heads = [p for p in planner.label_heads.parameters() if p.requires_grad]
    if not lora or not heads:
        raise ValueError("source pilot requires trainable shared LoRA and graph heads")
    optimizer = torch.optim.AdamW([{"params": lora, "lr": RECIPE["lora_lr"]},
                                   {"params": heads, "lr": RECIPE["head_lr"]}], weight_decay=.01)
    program = JointSolverPOIProgram(runtime, planner, planner_optimizer=optimizer,
                                    device=args.device, controller=None)
    planner = program.planner_head
    reference = FrozenPolicyReference(planner, optimizer)
    manager = BatchGradientManager(planner, optimizer)
    valid = list(zip((f"eai/validation/{i:03d}" for i in range(88)), valid_eai, strict=True))
    training = [row for row in rows if row["split"] == "training"]
    validation = [row for row in rows if row["split"] == "validation"]
    validation_contract = {**contract, "split": "validation"}
    print("[source-stage2] measuring the frozen Epoch 6 baseline", flush=True)
    baseline = validate(planner, runtime, valid, validation, root / "baseline", validation_contract, command)
    print("[source-stage2] repeating baseline before any optimizer update", flush=True)
    repeated = validate(planner, runtime, valid, validation, root / "baseline_repeat", validation_contract, command)
    identity = lambda records: [[row["input_hash"], row["labels"], row["success"]] for row in records]
    if any(identity(a) != identity(b) for a, b in zip(baseline, repeated, strict=True)):
        raise ValueError("frozen E6 baseline did not reproduce before Stage 2 updates")
    freeze(root / "baseline_verified.json", {"checkpoint_sha256": initialization["sha256"],
                                           "reference_sha256": reference.fingerprint})
    print("[source-stage2] verifying source replay inputs", flush=True)
    vla_anchors, _ = source_anchors(config, runtime, splits)
    rng = random.Random(RECIPE["seed"])
    updates = {"eai": 0, "vlabench": 0}
    shared_changed = False
    for epoch in range(1, RECIPE["generations"] + 1):
        folder = root / f"generation_{epoch}"
        for index, (batch, anchors) in enumerate(training_schedule(training, vla_anchors, rng)):
            step = folder / f"step_{index}"
            eai_items = [(f"eai/train/{i:03d}", train_eai[i]) for i in rng.sample(range(len(train_eai)), len(batch))]
            training_contract = {**contract, "split": "training"}
            es_results = {}
            if es:
                eai_records, eai_traces, vla_records, vla_traces, es_results = evolution_rollouts(
                    planner, runtime, manager, eai_items, batch, step, training_contract, command, epoch, index,
                )
                reward_evidence = {domain: result[2] for domain, result in es_results.items()}
                freeze(step / "es_population.json", reward_evidence)
                has_signal = any(metrics["informative_pairs"] for metrics in reward_evidence.values())
            else:
                eai_records, eai_traces = eai_rollouts(planner, runtime, eai_items, RECIPE["draws"], greedy=False)
                vla_records, vla_traces = vla_rollouts(planner, runtime, batch, RECIPE["draws"], greedy=False,
                    output=step / "native", contract=training_contract, command=command)
                has_signal = any(advantages(eai_records)) or any(advantages(vla_records))
            freeze(step / "eai_rollouts.json", eai_records)
            if not has_signal:
                reason = "tied_reward_in_every_mirrored_pair" if es else "tied_rewards_in_every_reset"
                freeze(step / "update.json", {"skipped": reason})
                continue
            evidence = SharedUpdateEvidence(planner, optimizer)
            # Qwen enables activation checkpointing in training mode. Keep
            # dropout disabled during both the forward and recomputation.
            with deterministic_policy(planner, training=True), manager.step_context():
                with runtime.domain_scope("eai", parameter_policy="freeze_inactive"):
                    if any(p.requires_grad for p in planner.label_heads["vlabench"].parameters()):
                        raise RuntimeError("inactive VLABench label head is trainable in EAI scope")
                    eai_policy = capture_policy(manager, planner, runtime, reference, "eai", eai_traces, eai_records, es=es)
                    if es:
                        population, parameters, metrics = es_results["eai"]
                        population.capture_reward(manager, "eai", parameters, ES_SIGMA, metrics)
                    for _, item in eai_items:
                        context, _ = observed_eai.input_and_goal(item, runtime.world.eai)
                        loss, _ = eai_loss(planner, context, item["target_action_labels"])
                        manager.capture("eai", RECIPE["eai_replay_weight"] * loss / len(eai_items))
                with common_prompt(planner), runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
                    if any(p.requires_grad for p in planner.label_heads["eai"].parameters()):
                        raise RuntimeError("inactive EAI label head is trainable in VLABench scope")
                    vla_policy = capture_policy(manager, planner, runtime, reference, "vlabench", vla_traces, vla_records, es=es)
                    if es:
                        population, parameters, metrics = es_results["vlabench"]
                        population.capture_reward(manager, "vlabench", parameters, ES_SIGMA, metrics)
                    for row in anchors:
                        manager.capture("vlabench", RECIPE["vlabench_replay_weight"] *
                                        anchor_vla(planner, runtime, row) / len(anchors))
                gradients = manager.resolve_gradients(method="pcgrad")
                if any(gradient is not None and not bool(gradient.isfinite().all()) for gradient in gradients):
                    raise RuntimeError("nonfinite PCGrad update")
                manager.apply_resolved_gradients(gradients)
                torch.nn.utils.clip_grad_norm_(manager.parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                changes = evidence.report()
            shared_changed |= changes["shared_parameters_changed"]
            for domain, evidence_row in (("eai", eai_policy), ("vlabench", vla_policy)):
                signal = es_results[domain][2]["informative_pairs"] if es else evidence_row["nonzero_advantages"]
                updates[domain] += int(bool(signal))
            freeze(step / "update.json", {"eai_policy": eai_policy, "vlabench_policy": vla_policy,
                                         "shared_update": changes})
            print(f"[source-stage2] generation={epoch} batch={index + 1} signal={updates}", flush=True)
        save_checkpoint(folder / "planner.pt", planner, runtime, "joint", epoch, sha(root / "settings.json"),
                        provenance={"method": RECIPE["version"], "optimizer_mode": args.optimizer,
                                    "initial_checkpoint_sha256": initialization["sha256"],
                                    "reference_sha256": reference.fingerprint})
    final = validate(planner, runtime, valid, validation, root / "final", validation_contract, command)
    result = qualify(baseline, final, updates, optimizer_mode=args.optimizer)
    if not shared_changed:
        result["rejections"].append("no_shared_parameter_change")
        result["qualified"] = False
    checkpoint = root / f"generation_{RECIPE['generations']}/planner.pt"
    freeze(root / "report.json", {**result, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint)})


def evaluate(args):
    config, e6 = checkpoint_record(args.archive, 6)
    config5, e5 = checkpoint_record(args.epoch5_archive, 5, verify_base=False)
    verify(config)
    if (config5["model"] != config["model"] or e5["pretrained_base_files"] != e6["pretrained_base_files"]):
        raise ValueError("E5 and E6 pretrained bases differ")
    pilots = {"stage2": args.pilot}
    if args.compare_pilot:
        pilots["parallel_stage2"] = args.compare_pilot
    reports = {name: read(folder / "report.json") for name, folder in pilots.items()}
    settings = {name: read(folder / "settings.json") for name, folder in pilots.items()}
    modes = {name: value.get("optimizer_mode", "reinforce") for name, value in settings.items()}
    for name, report in reports.items():
        if (not report["qualified"] or settings[name]["initialization"] != e6 or
                modes[name] != report.get("optimizer_mode", "reinforce") or
                sha(report["checkpoint"]) != report["checkpoint_sha256"]):
            raise ValueError("matched evaluation requires qualified source-only Stage 2 pilots")
    if args.compare_pilot:
        matched = ("recipe", "source", "inputs", "eai_reward", "auxiliary_heads", "executor_sha256", "executor_profile")
        if set(modes.values()) != {"es", "reinforce"} or any(
                settings["stage2"][key] != settings["parallel_stage2"][key] for key in matched):
            raise ValueError("parallel comparison requires matched ES and REINFORCE settings")
    test, holdout = transfer_partition(args.holdout, args.archive, config, args.excluded_manifest)
    if any(holdout != value["holdout"] for value in settings.values()):
        raise ValueError("LIBERO holdout differs from the one frozen before updates")
    proof = executor_proof("libero", args.executor_source, args.calibration, args.holdout)
    contract = {"dataset": "libero", "split": "evaluation", "manifest": str(args.holdout.resolve()),
                "manifest_sha256": sha(args.holdout), "evaluation_manifest": str(args.holdout.resolve()),
                "evaluation_manifest_sha256": sha(args.holdout), "executor_source": str(args.executor_source.resolve()),
                "executor_sha256": proof, "calibration": str(args.calibration.resolve()),
                "calibration_sha256": sha(args.calibration),
                "bddl_sha256": {row["bddl"]: row["source_bddl_sha256"] for row in test}}
    command = worker(args.worker, "target_rl_rollout")
    protocol = {"contract": contract, "policy": EXECUTOR, "input_protocol": IMAGE_INSTRUCTION,
                "holdout": holdout,
                "checkpoints": {"e5": e5, "e6": e6, **{name: report["checkpoint_sha256"] for name, report in reports.items()}},
                "optimizer_modes": modes, "worker": command}
    if args.validate_only:
        print(json.dumps(protocol))
        return
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    freeze(root / "protocol.json", protocol)
    _, (runtime, *_) = build_runtime(config, root / "runtime", args.device)
    planner = fresh_planner(config, runtime, "joint")
    outcomes = {}
    for name, record in (("e5", e5), ("e6", e6), *((name, None) for name in pilots)):
        if record is not None:
            load_planner_snapshot(planner, record)
        else:
            load_checkpoint(reports[name]["checkpoint"], planner, "joint", sha(pilots[name] / "settings.json"))
        outcomes[name], _ = vla_rollouts(planner, runtime, test, 1, greedy=True, profile=pick_place_profile(),
                                       output=root / name, contract=contract, command=command)
    private = {row["episode_id"]: read(row["scoring_private"]) for row in test}
    grounding = {}
    for name, records in outcomes.items():
        successes = cases = 0
        for row in records:
            spec = private[row["episode_id"]]
            targets = {str(goal[1]).lower() for goal in spec["goals"] if len(goal) == 3}
            if len(targets) != 1:
                continue
            cases += 1
            selected = next((op["params"][SUBJECT] for op in row["prediction"] if op["name"] == "pick"), None)
            successes += int(selected is not None and str(spec["entities"][selected]).lower() in targets)
        grounding[name] = {"correct_first_source": successes, "unambiguous_source_cases": cases}
    contrasts = {"e6_minus_e5": paired_report(outcomes["e5"], outcomes["e6"], clustered=True)}
    for name in pilots:
        for baseline in ("e5", "e6"):
            contrasts[f"{name}_minus_{baseline}"] = paired_report(outcomes[baseline], outcomes[name], clustered=True)
    if args.compare_pilot:
        by_mode = {mode: outcomes[name] for name, mode in modes.items()}
        contrasts["es_minus_reinforce"] = paired_report(by_mode["reinforce"], by_mode["es"], clustered=True)
    freeze(root / "paired_report.json", {**contrasts, "optimizer_modes": modes, "grounding": grounding})


def evaluate_eai_decoders(args):
    from .fresh_metadata.checkpoint import load_frozen_planner
    from .fresh_metadata.eai_latent_audit import audit_case, summarize
    from .shared_reranker.head import SharedPlanReranker, fit_shared_reranker

    config, record = checkpoint_record(args.archive, 6)
    verify(config)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    _, (runtime, eai_train, eai_valid, _, _) = build_runtime(config, root / "runtime", args.device)
    if len(eai_valid) != 88 or not 1 <= args.train_cases <= len(eai_train):
        raise ValueError("invalid decoder evaluation split or training budget")
    indices = sorted(random.Random(RECIPE["seed"]).sample(range(len(eai_train)), args.train_cases))
    freeze(root / "protocol.json", {"checkpoint": record, "train_indices": indices,
                                    "identities_sha256": sha(root / "runtime/eai_identities.json"),
                                    "replay_seed": RECIPE["seed"], "validation_episodes": 88})
    planner = load_frozen_planner(config, record, runtime)
    planner.free_decoding = False
    pools = []
    training = []
    for position, index in enumerate(indices):
        training.append(audit_case(planner, runtime, eai_train[index], index,
                                   check_batch=position == 0, replay_pools=pools, archived_prompt=False))
        print(f"[source-stage2-eai] training candidates {position + 1}/{len(indices)}", flush=True)
    freeze(root / "train_cases.json", training)
    freeze(root / "replay_pools.json", [asdict(pool) for pool in pools])
    torch.manual_seed(RECIPE["seed"])
    head = SharedPlanReranker()
    try:
        learned = fit_shared_reranker(head, pools, source_train_ids={f"eai/train/{i:03d}" for i in indices})
        path = root / "shared_reranker.pt"
        head.save(path)
        trained_sha = sha(path)
    except ValueError as exc:
        if str(exc) != "no replay-labeled success/failure pairs with distinguishable observations":
            raise
        learned, trained_sha, head = {"reason": str(exc)}, None, None
    freeze(root / "fitting.json", {**learned, "head_sha256": trained_sha})
    validation = []
    for index, item in enumerate(eai_valid):
        validation.append(audit_case(planner, runtime, item, index,
                                     check_batch=index == 0, shared_reranker=head, archived_prompt=False))
        print(f"[source-stage2-eai] validation {index + 1}/88", flush=True)
    freeze(root / "validation_cases.json", validation)
    freeze(root / "result.json", {**summarize(validation), "head_sha256": trained_sha,
                                    "beam_oracle_fields": "coverage diagnostic only"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("train")
    measure = commands.add_parser("evaluate")
    for command in (run, measure):
        command.add_argument("--archive", required=True, type=Path, help="Frozen Epoch 6 comparison archive")
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--worker", type=json.loads)
        command.add_argument("--validate-only", action="store_true")
        command.add_argument("--holdout", type=Path, required=True)
        command.add_argument("--excluded-manifest", required=True, action="append", type=Path)
    run.add_argument("--source-manifest", required=True, type=Path)
    run.add_argument("--optimizer", choices=("reinforce", "es"), default="reinforce")
    for key in ("epoch5-archive", "pilot", "executor-source", "calibration"):
        measure.add_argument("--" + key, required=True, type=Path)
    measure.add_argument("--compare-pilot", type=Path, help="Qualified pilot from the other optimizer arm")
    decoders = commands.add_parser("eai-decoders")
    decoders.add_argument("--archive", required=True, type=Path)
    decoders.add_argument("--output", required=True, type=Path)
    decoders.add_argument("--device", default="cuda:0")
    decoders.add_argument("--train-cases", type=int, default=32)
    args = parser.parse_args(argv)
    {"train": train, "evaluate": evaluate, "eai-decoders": evaluate_eai_decoders}[args.action](args)


if __name__ == "__main__":
    main()
