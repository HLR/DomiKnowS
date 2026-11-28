"""Two-head Stage 2 optimizer comparison on fixed source cohorts."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch

from .generalization_protocol import file_hash
from . import observed_eai
from .stage2_training import eai_retention_rejections, selection_rejections


MODES = ("reinforce", "constrained-es", "constrained-reinforce")
DOMAINS = ("eai", "vlabench")
GENERATIONS = 2
PAIRS = 4
SIGMA = 0.001
STEP_SIZE = 0.0005
EVALUATION_SEED = 100042
SETTINGS = {
    "version": 3, "generations": GENERATIONS, "pairs": PAIRS, "sigma": SIGMA,
    "step_size": STEP_SIZE, "eai_per_domain": 8, "vlabench_resets": 4,
}


def _fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _seed(*keys):
    return int(_fingerprint(keys)[:15], 16)


@contextmanager
def _rollout_rng(scene_seed, draw_seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        random.seed(scene_seed)
        np.random.seed(scene_seed % (2**32 - 1))
        torch.manual_seed(draw_seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


@dataclass
class Rollout:
    success: bool
    replay: tuple = ()


def _noise(parameters, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for param in parameters:
        yield torch.randn(param.shape, generator=generator, dtype=torch.float32).to(param.device)


def _weights(results):
    rewards = torch.tensor(results, dtype=torch.float64).mean(dim=1)
    spread = float(rewards.std(unbiased=False))
    return rewards, spread, (rewards - rewards.mean()) / spread if spread else rewards * 0


def _generation_metrics(results, rewards, spread, changed, *, tied=None, **evidence):
    return {"candidate_success_counts": [sum(row) for row in results],
            "candidate_fitness": rewards.tolist(), "fitness_std": spread,
            "aggregate_fitness_tied": spread == 0, "updated": changed,
            "tied": spread == 0 if tied is None else tied, **evidence}


def es_generation(head, cohort, rollout, seed, *, progress):
    """Evaluate mirrored seeded variants and replay noise into the center."""
    params = tuple(head.parameters())
    centers = tuple(param.detach().clone() for param in params)
    noise_seeds = [_seed(seed, "noise", pair) for pair in range(PAIRS)]
    candidates = [(direction, sign) for direction in noise_seeds for sign in (1, -1)]
    outcomes = []
    with torch.no_grad():
        for candidate, (direction, sign) in enumerate(candidates):
            try:
                for param, center, noise in zip(params, centers, _noise(params, direction)):
                    param.copy_(center + sign * SIGMA * noise)
                successes = [
                    int(rollout(case, _seed(seed, "draw", candidate // 2, index), False).success)
                    for index, case in enumerate(cohort)
                ]
                outcomes.append(successes)
                progress(f"ES candidate {candidate + 1}/{len(candidates)} success={sum(successes)}/{len(cohort)}")
            finally:
                for param, center in zip(params, centers):
                    param.copy_(center)
        rewards, spread, weights = _weights(outcomes)
        changed = False
        informative_pairs = sum(bool(positive != negative)
                                for positive, negative in zip(rewards[::2], rewards[1::2], strict=True))
        if informative_pairs:
            deltas = [torch.zeros_like(param) for param in params]
            for index, direction in enumerate(noise_seeds):
                coefficient = float(weights[2 * index] - weights[2 * index + 1]) * STEP_SIZE / len(candidates)
                for delta, noise in zip(deltas, _noise(params, direction)):
                    delta.add_(noise, alpha=coefficient)
            if not all(torch.isfinite(delta).all() for delta in deltas):
                raise RuntimeError("nonfinite constrained ES update")
            for param, center, delta in zip(params, centers, deltas, strict=True):
                param.add_(delta)
                changed |= not torch.equal(param, center)
    return _generation_metrics(outcomes, rewards, spread, changed, tied=informative_pairs == 0,
                               informative_pairs=informative_pairs, noise_seeds=noise_seeds)


def reinforce_generation(head, cohort, rollout, seed, *, progress):
    """Apply terminal-reward REINFORCE with a leave-in baseline per reset."""
    params = tuple(head.parameters())
    outcomes = [[] for _ in range(2 * PAIRS)]
    scale = 1 / (len(outcomes) * len(cohort))
    generator_loss = 0.0
    mixed_resets = 0
    head.zero_grad(set_to_none=True)
    try:
        for index, case in enumerate(cohort):
            batch = [rollout(case, _seed(seed, "reinforce_draw", candidate, index), True)
                     for candidate in range(len(outcomes))]
            for row, result in zip(outcomes, batch):
                row.append(int(result.success))
            mean = sum(result.success for result in batch) / len(batch)
            mixed_resets += int(0 < mean < 1)
            for result in batch:
                advantage = (float(result.success) - mean) * scale
                if advantage:
                    for replay in result.replay:
                        loss = -advantage * replay()
                        if loss.requires_grad:
                            loss.backward()
                            generator_loss += float(loss.detach())
            progress(f"REINFORCE reset {index + 1}/{len(cohort)} success={sum(r.success for r in batch)}/{len(batch)}")
        rewards, spread, _ = _weights(outcomes)
        gradients = [param.grad for param in params if param.grad is not None]
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("nonfinite constrained REINFORCE update")
        changed = False
        if any(bool(torch.count_nonzero(gradient)) for gradient in gradients):
            with torch.no_grad():
                for param in params:
                    if param.grad is not None:
                        before = param.detach().clone()
                        param.add_(param.grad, alpha=-STEP_SIZE)
                        changed |= not torch.equal(before, param)
    finally:
        head.zero_grad(set_to_none=True)
    return _generation_metrics(outcomes, rewards, spread, changed, policy_loss=generator_loss,
                               tied=mixed_resets == 0, mixed_resets=mixed_resets)


def _eai_cases(items, world):
    cases = []
    for item in items:
        try:
            cases.append((item, observed_eai.input_and_goal(item, world)[1]))
        except ValueError as exc:
            raise ValueError(f"invalid EAI Stage 2 input for task {item.get('task_id')!r}: {exc}") from exc
    return cases


def _eai_identity(case):
    item, _ = case
    return _fingerprint([str(item.get(key, "")) for key in
                         ("task_id", "tl_goal", "transition_model", "causal_prompt_text")])


def _score_eai(stage2, case, *, deterministic, replay):
    item, goal = case
    runtime, view = stage2.joint_runtime, stage2.eai_planner
    context, _ = observed_eai.input_and_goal(item, runtime.world.eai)
    max_steps = runtime.max_eai_steps
    with torch.no_grad():
        dfa = runtime.dfa_for("eai", context)
        labels, result = observed_eai.sample(view, context, dfa, goal, runtime, deterministic=deterministic)
        if not replay or result.get("generation_error"):
            return result, ()
        prepared = view.prepare_replay_context(context)

    def logprob():
        return view.replay_labels_logprob(prepared, labels, dfa, max_steps=max_steps)

    return result, (logprob,)


def _eai_rollout(stage2, case, seed, replay):
    with _rollout_rng(seed, seed):
        result, logprobs = _score_eai(stage2, case, deterministic=False, replay=replay)
        return Rollout(result["is_success"], logprobs)


def _vla_rollout(stage2, case, draw_seed, replay):
    descriptor, reset = case
    with _rollout_rng(reset, draw_seed), torch.no_grad():
        episode = stage2.collect_episode(descriptor)
        records = tuple(record for record in episode.planner_logprobs
                        if callable(getattr(record, "logprob", None))) if replay else ()
        outcome = bool(episode.valid and episode.success)
        del episode  # drop controller images before collecting another rollout
    return Rollout(outcome, tuple(
        lambda record=record: record.logprob(stage2.vlabench_planner) for record in records))


@contextmanager
def _pilot_policy(stage2):
    planner, controller = stage2.joint_planner, stage2.controller
    planner_mode, controller_mode = planner.training, controller.training
    assistance = stage2._execution_assistance_override
    planner.eval()
    controller.eval()
    stage2._execution_assistance_override = False
    try:
        yield
    finally:
        stage2._execution_assistance_override = assistance
        planner.train(planner_mode)
        controller.train(controller_mode)


@contextmanager
def _active_head(stage2, domain):
    planner = stage2.joint_planner
    head = planner.label_heads[domain]
    with stage2.joint_runtime.domain_scope(domain, parameter_policy="freeze_shared"):
        trainable = {id(param) for param in head.parameters()}
        saved = [(param, param.requires_grad) for param in planner.parameters()]
        for param, was_trainable in saved:
            if was_trainable != (id(param) in trainable):
                param.requires_grad_(id(param) in trainable)
        try:
            yield head
        finally:
            for param, was_trainable in saved:
                if param.requires_grad != was_trainable:
                    param.requires_grad_(was_trainable)


def _evaluation(stage2, validation, descriptors, *, progress):
    progress("constrained Stage 2 fixed evaluation: EAI 88 examples, VLABench 100 resets")
    with _active_head(stage2, "eai"):
        eai = [_score_eai(stage2, case, deterministic=True, replay=False)[0] for case in validation]
    vla = stage2.evaluate_rollouts(descriptors, rollouts_per_task=10, seed=EVALUATION_SEED,
                                  assistance=False, valid_success_only=True)
    episodes = vla.pop("episode_diagnostics")
    if len(episodes) != 100 or vla["assist_steps"]:
        raise RuntimeError("fixed VLABench evaluation requires 100 unassisted episodes")
    vla["successes"] = sum(episode["success"] for episode in episodes)
    return {"eai": {"success": sum(result["is_success"] for result in eai) / len(eai),
                    "goal_recall": sum(result["recall"] for result in eai) / len(eai),
                    "successes": sum(result["is_success"] for result in eai), "examples": len(eai),
                    "unresolved_examples": sum(bool(result["unresolved"]) for result in eai),
                    "no_accepting_plan": sum(result.get("generation_error") == "no_accepting_plan" for result in eai),
                    "semantic_search_limits": sum(result.get("generation_error") == "semantic_search_limit" for result in eai)},
            "vlabench": vla, "stage2_evaluation": {"source": "fixed_seed_rollouts", "seed": EVALUATION_SEED}}


def _training_cohorts(eai, descriptors, seed):
    sources = {domain: [case for case in eai if case[1].domain == domain and not case[1].unresolved]
               for domain in ("igibson", "virtualhome")}
    rng = random.Random(_seed(seed, "eai_cohort"))
    count = SETTINGS["eai_per_domain"]
    if any(len(cases) < count * GENERATIONS for cases in sources.values()):
        raise ValueError("constrained Stage 2 needs 16 independently groundable training examples from each EAI source")
    for cases in sources.values():
        rng.shuffle(cases)
    names = {descriptor["task"] for descriptor in descriptors}
    if len(names) != len(descriptors) or len(names) < SETTINGS["vlabench_resets"]:
        raise ValueError("constrained Stage 2 requires distinct VLABench source tasks")
    order = sorted(descriptors, key=lambda case: case["task"])
    random.Random(_seed(seed, "vla_cohort")).shuffle(order)
    resets = SETTINGS["vlabench_resets"]
    return [
        {"eai": [case for cases in sources.values() for case in cases[generation * count:(generation + 1) * count]],
         "vlabench": [(order[(generation * resets + index) % len(order)],
                       1_000_000 + _seed(seed, "vla_reset", generation, index) % 2_000_000_000)
                      for index in range(resets)]}
        for generation in range(GENERATIONS)
    ]


def _atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_pilot(stage2, *, mode, seed, eai_train, eai_valid, descriptors, evaluation_descriptors,
              output, initialization, resume, settings, save_checkpoint, progress):
    """Run both active heads, committing each domain generation before proceeding."""
    if len(eai_valid) != 88 or len(evaluation_descriptors) != 10:
        raise ValueError("constrained Stage 2 needs the full 88 EAI examples and 10 VLABench evaluation tasks")
    world = stage2.joint_runtime.world.eai
    train, valid = _eai_cases(eai_train, world), _eai_cases(eai_valid, world)
    coverage = {domain: {"training_examples": sum(goal.domain == domain for _, goal in train),
                         "groundable_training_examples": sum(goal.domain == domain and not goal.unresolved for _, goal in train),
                         "validation_examples": sum(goal.domain == domain for _, goal in valid)}
                for domain in ("igibson", "virtualhome")}
    progress(f"constrained Stage 2 EAI grounding coverage={coverage}")
    cohorts = _training_cohorts(train, descriptors, seed)
    protocol = _fingerprint({"version": 1, "evaluation_seed": EVALUATION_SEED,
                             "validation": [_eai_identity(case) for case in valid],
                             "evaluation_descriptors": evaluation_descriptors,
                             "settings": SETTINGS, "rollout_settings": settings})
    cohort_id = _fingerprint([{"eai": [_eai_identity(case) for case in generation["eai"]],
                               "vlabench": generation["vlabench"]} for generation in cohorts])
    state = {"mode": mode, "training_seed": seed, "protocol": protocol, "cohort_id": cohort_id,
             "initialization_sha256": file_hash(initialization) if initialization else resume["initialization_sha256"],
             "settings": SETTINGS, "eai_grounding_coverage": coverage, "history": [],
             "next_seed": _seed(seed, DOMAINS[0], 0)}
    report_path = output / "constrained_stage2_report.json"
    if resume:
        if any(resume.get(key) != value for key, value in state.items() if key not in {"history", "next_seed"}):
            raise ValueError("constrained Stage 2 resume settings or source cohort differ from the checkpoint")
        state.update(resume)
    schedule = [(index, domain) for index in range(GENERATIONS) for domain in DOMAINS]
    history = state["history"]
    if len(history) > len(schedule) or any(
            (entry["generation"], entry["domain"], entry["seed"]) !=
            (index, domain, _seed(seed, domain, index))
            for entry, (index, domain) in zip(history, schedule)):
        raise ValueError("constrained Stage 2 checkpoint has inconsistent generation progress")
    remaining = schedule[len(history):]
    next_seed = _seed(seed, remaining[0][1], remaining[0][0]) if remaining else None
    if state["next_seed"] != next_seed:
        raise ValueError("constrained Stage 2 checkpoint has an inconsistent next seed")

    with _pilot_policy(stage2):
        if not resume:
            state["baseline"] = _evaluation(stage2, valid, evaluation_descriptors, progress=progress)
            baseline = state["baseline"]["vlabench"]
            if (baseline["successes"] < 5 or baseline["successful_task_count"] < 2
                    or baseline["positive_return_rate"] < 0.01 or baseline["ik_truncation_rate"] > 0.50):
                _atomic_json(report_path, {**state, "status": "preflight_failed"})
                progress("constrained Stage 2 preflight rejected the warmed checkpoint")
                return
            save_checkpoint(output / "constrained_stage2_progress.pt", state)

        update = es_generation if mode == "constrained-es" else reinforce_generation
        rollout = {"eai": _eai_rollout, "vlabench": _vla_rollout}
        for index, domain in remaining:
            step_seed = _seed(seed, domain, index)
            progress(f"constrained Stage 2 {mode} {domain} generation {index + 1}/{GENERATIONS}")
            with _active_head(stage2, domain) as head:
                metrics = update(head, cohorts[index][domain],
                    lambda case, draw, replay: rollout[domain](stage2, case, draw, replay),
                    step_seed, progress=progress)
            history.append({**metrics, "domain": domain, "generation": index, "seed": step_seed})
            next_step = schedule[len(history)] if len(history) < len(schedule) else None
            state["next_seed"] = _seed(seed, next_step[1], next_step[0]) if next_step else None
            save_checkpoint(output / "constrained_stage2_progress.pt", state)
        final = _evaluation(stage2, valid, evaluation_descriptors, progress=progress)

    baseline = state["baseline"]
    eai_baseline = {"goal_success": baseline["eai"]["success"], "goal_recall": baseline["eai"]["goal_recall"]}
    reasons = eai_retention_rejections(final["eai"], eai_baseline,
                                       max_success_regression=0.0, max_recall_regression=0.02)
    if final["vlabench"]["success_rate"] < baseline["vlabench"]["success_rate"]:
        reasons.append("vlabench_success_regression")
    reasons += selection_rejections(final, min_success=0.10, min_tasks=3, max_ik=0.25)
    tied = [domain for domain in DOMAINS if all(entry["tied"] for entry in history if entry["domain"] == domain)]
    reasons.extend(f"{domain}_both_generations_tied" for domain in tied)
    if not any(entry["updated"] for entry in history):
        reasons.append("no_head_update")
    report = {**state, "status": "complete", "final": final, "retention_rejections": reasons}
    if not tied:
        checkpoint = output / "constrained_stage2_candidate.pt"
        save_checkpoint(checkpoint, report)
        report.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint))
    _atomic_json(report_path, report)
    progress(f"constrained Stage 2 report={report_path}, EAI={final['eai']['successes']}/88 "
             f"VLABench={final['vlabench']['successes']}/100; source gates={reasons}")


def promotion_candidate(reports):
    """Require matched ES and PG results for each of two training seeds."""
    if len(reports) != 4 or any(report.get("status") != "complete" for report in reports):
        raise ValueError("promotion needs four completed constrained Stage 2 reports")
    for key in ("protocol", "initialization_sha256"):
        if len({report[key] for report in reports}) != 1:
            raise ValueError(f"promotion reports disagree on {key}")
    seeds = sorted({report["training_seed"] for report in reports})
    if len(seeds) != 2:
        raise ValueError("promotion needs two distinct training seeds")
    candidates = []
    improved_domains = set(DOMAINS)
    for seed in seeds:
        pair = {report["mode"]: report for report in reports if report["training_seed"] == seed}
        if set(pair) != set(MODES[1:]):
            raise ValueError(f"training seed {seed} needs both constrained optimizers")
        es, reinforce = (pair[mode] for mode in MODES[1:])
        if es["cohort_id"] != reinforce["cohort_id"]:
            raise ValueError(f"training seed {seed} used different source cohorts")
        if es["retention_rejections"]:
            raise ValueError(f"training seed {seed} failed source gates: {es['retention_rejections']}")
        wins = [(es["final"][domain]["successes"], reinforce["final"][domain]["successes"])
                for domain in DOMAINS]
        if any(es_count < pg_count for es_count, pg_count in wins) or all(
                es_count == pg_count for es_count, pg_count in wins):
            raise ValueError(f"training seed {seed} did not replicate an ES success advantage")
        improved_domains &= {domain for domain, (es_count, pg_count) in zip(DOMAINS, wins)
                             if es_count > pg_count}
        candidates.append(es)
    if not improved_domains:
        raise ValueError("ES did not repeat the improvement on either source domain")
    return candidates[0]  # the first training seed is designated before results are inspected


def promote(reports, output):
    report = promotion_candidate([json.loads(Path(path).read_text(encoding="utf-8")) for path in reports])
    checkpoint = Path(report["checkpoint"])
    if file_hash(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("promotion checkpoint digest mismatch")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    expected = {key: value for key, value in report.items() if key not in {"checkpoint", "checkpoint_sha256"}}
    if payload.get("metrics", {}).get("constrained_stage2") != expected:
        raise ValueError("promotion report disagrees with the checkpoint")
    payload["metrics"]["retention_eligible"] = True
    payload["metrics"]["constrained_stage2_promotion"] = {
        "paired_report_sha256": [file_hash(path) for path in reports],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "joint_stage2_best.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination
