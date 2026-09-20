"""Offline constrained decoding and ILP feasibility evaluation.

ILP is deliberately applied only to complete autoregressive candidates. The
model supplies a joint log-probability for each candidate, while the active
DomiKnowS domain supplies the hard feasibility tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from test_regr.EmbodiedAgentInterface.reward import evaluate_goal_satisfaction
from test_regr.VLABenchAgentInterface.graph import labels_to_plan
from test_regr.VLABenchAgentInterface.reward import score_vlabench_plan
from test_regr.VLABenchAgentInterface.world_graph import (
    materialize_plan,
    validate_plan,
    verify_plan_constraints,
)


@dataclass(frozen=True)
class LabelSequenceCandidate:
    labels: tuple[int, ...]
    log_probability: float


@dataclass(frozen=True)
class StructuredPlanCandidate:
    labels: tuple[int, ...]
    log_probability: float
    plan: tuple[Any, ...]
    dfa_valid: bool
    schema_valid: bool
    constraint_valid: bool
    constraint_score: float
    goal_success: bool | None = None
    goal_recall: float | None = None
    reward_score: float | None = None
    errors: tuple[str, ...] = ()

    @property
    def hard_valid(self) -> bool:
        return self.dfa_valid and self.schema_valid and self.constraint_valid


@dataclass(frozen=True)
class DecoderSelection:
    mode: str
    selected: StructuredPlanCandidate
    elapsed_ms: float
    solver_available: bool = True
    solver_status: str = "not_used"
    fallback_reason: str | None = None


@dataclass(frozen=True)
class DecoderComparison:
    greedy: DecoderSelection
    beam: DecoderSelection
    ilp: DecoderSelection
    candidates: tuple[StructuredPlanCandidate, ...]
    context_encoding_ms: float


class ILPSolverUnavailable(RuntimeError):
    """Raised when Gurobi cannot be imported, licensed, or used."""


@dataclass(frozen=True)
class ILPProbe:
    available: bool
    detail: str


def constrained_dfa_beam_search(
    next_logits: Callable[[Sequence[tuple[int, ...]]], torch.Tensor],
    dfa: Any,
    eos_label: int,
    max_steps: int,
    beam_width: int = 16,
) -> tuple[LabelSequenceCandidate, ...]:
    """Return complete DFA-valid paths scored by autoregressive log probability."""
    if beam_width <= 0 or max_steps <= 0:
        raise ValueError("beam_width and max_steps must be positive")
    start_state = dfa.start_state
    beams: list[tuple[tuple[int, ...], Any, float, bool]] = [
        ((), start_state, 0.0, False)
    ]
    for step_index in range(max_steps):
        live = [beam for beam in beams if not beam[3]]
        if not live:
            break
        logits = next_logits([beam[0] for beam in live])
        if logits.ndim != 2 or logits.shape[0] != len(live):
            raise ValueError("next_logits must return [number_of_prefixes, vocabulary_size]")
        expanded = [beam for beam in beams if beam[3]]
        remaining_steps = max_steps - step_index - 1
        for row, (prefix, state, score, _complete) in zip(logits, live):
            allowed = tuple(sorted(
                int(label)
                for label in dfa.allowed_tokens(state, remaining_steps=remaining_steps)
            ))
            if not allowed:
                continue
            allowed_tensor = torch.as_tensor(allowed, device=row.device, dtype=torch.long)
            log_probabilities = torch.log_softmax(row.index_select(0, allowed_tensor), dim=0)
            count = min(beam_width, len(allowed))
            values, positions = torch.topk(log_probabilities, count)
            for value, position in zip(values.tolist(), positions.tolist()):
                label = int(allowed[position])
                next_state = dfa.step(state, label)
                if next_state is None:
                    continue
                next_prefix = prefix + (label,)
                complete = label == eos_label and dfa.is_accepting(next_state)
                expanded.append((next_prefix, next_state, score + float(value), complete))
        beams = sorted(expanded, key=lambda item: (-item[2], item[0]))[:beam_width]
    complete = [beam for beam in beams if beam[3]]
    if not complete:
        raise RuntimeError("DFA beam search produced no complete accepting path")
    return tuple(
        LabelSequenceCandidate(labels=labels, log_probability=score)
        for labels, _state, score, _complete in sorted(
            complete, key=lambda item: (-item[2], item[0])
        )
    )


def _solve_binary_choice(
    scores: Sequence[float], eligible: Sequence[int], time_limit_ms: int
) -> tuple[int, str]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as error:  # pragma: no cover - installation dependent
        raise ILPSolverUnavailable(f"Gurobi import failed: {error}") from error
    environment = model = None
    try:
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()
        model = gp.Model(env=environment)
        model.Params.TimeLimit = max(float(time_limit_ms) / 1000.0, 0.001)
        choices = {index: model.addVar(vtype=GRB.BINARY) for index in eligible}
        model.addConstr(gp.quicksum(choices.values()) == 1)
        model.setObjective(
            gp.quicksum((float(scores[index]) - index * 1e-9) * choices[index] for index in eligible),
            GRB.MAXIMIZE,
        )
        model.optimize()
        if model.SolCount < 1:
            raise ILPSolverUnavailable(f"Gurobi returned no incumbent (status={model.Status})")
        selected = max(eligible, key=lambda index: float(choices[index].X))
        return selected, str(model.Status)
    except ILPSolverUnavailable:
        raise
    except Exception as error:  # pragma: no cover - license dependent
        raise ILPSolverUnavailable(f"Gurobi solve failed: {error}") from error
    finally:  # pragma: no cover - installation dependent
        for resource in (model, environment):
            if resource is not None:
                try:
                    resource.dispose()
                except Exception:
                    pass


def probe_ilp_solver() -> ILPProbe:
    try:
        _index, status = _solve_binary_choice([0.0], [0], 250)
        return ILPProbe(True, f"Gurobi available (status={status})")
    except ILPSolverUnavailable as error:
        return ILPProbe(False, str(error))


def select_candidate_with_ilp(
    candidates: Sequence[StructuredPlanCandidate],
    domain: str,
    *,
    time_limit_ms: int = 250,
    solver: Callable[[Sequence[float], Sequence[int], int], tuple[int, str]] = _solve_binary_choice,
) -> DecoderSelection:
    """Select the maximum-likelihood complete plan satisfying active constraints."""
    if domain not in {"eai", "vlabench"}:
        raise ValueError(f"unknown domain {domain!r}")
    if not candidates:
        raise ValueError("ILP selection requires at least one candidate")
    started = time.perf_counter()
    eligible = [index for index, candidate in enumerate(candidates) if candidate.hard_valid]
    if domain == "eai" and any(candidates[index].goal_success for index in eligible):
        eligible = [index for index in eligible if candidates[index].goal_success]
    if not eligible:
        return DecoderSelection(
            "ilp", candidates[0], (time.perf_counter() - started) * 1000.0,
            solver_status="infeasible", fallback_reason="no hard-valid candidate",
        )
    try:
        selected, status = solver(
            [candidate.log_probability for candidate in candidates], eligible, time_limit_ms
        )
        if selected not in eligible:
            raise ILPSolverUnavailable("ILP solver returned an ineligible candidate")
        return DecoderSelection(
            "ilp", candidates[selected], (time.perf_counter() - started) * 1000.0,
            solver_status=status,
        )
    except ILPSolverUnavailable as error:
        return DecoderSelection(
            "ilp", candidates[0], (time.perf_counter() - started) * 1000.0,
            solver_available=False, solver_status="unavailable", fallback_reason=str(error),
        )


def _score_eai_candidate(
    candidate: LabelSequenceCandidate, runtime: Any, item: Mapping[str, Any], dfa: Any
) -> StructuredPlanCandidate:
    vocabulary = runtime.eai_vocabulary
    labels = tuple(int(label) for label in candidate.labels)
    metrics = evaluate_goal_satisfaction(
        labels,
        dict(item),
        vocabulary=vocabulary,
        world_bundle=runtime.world.eai,
        reward_mode="dense",
    )
    constraint_score = metrics.get("world_constraint_score")
    constraint_valid = constraint_score is None or float(constraint_score) >= 1.0
    return StructuredPlanCandidate(
        labels=labels,
        log_probability=float(candidate.log_probability),
        plan=tuple(vocabulary.token_for_label(label) for label in labels),
        dfa_valid=bool(dfa.accepts(labels)),
        schema_valid=True,
        constraint_valid=constraint_valid,
        constraint_score=1.0 if constraint_score is None else float(constraint_score),
        goal_success=bool(metrics["is_success"]),
        goal_recall=float(metrics["recall"]),
        reward_score=float(metrics["rl_reward_score"]),
        errors=(() if metrics.get("parse_error") is None else (str(metrics["parse_error"]),)),
    )


def _score_vlabench_candidate(
    candidate: LabelSequenceCandidate, runtime: Any, item: Mapping[str, Any], dfa: Any
) -> StructuredPlanCandidate:
    labels = tuple(int(label) for label in candidate.labels)
    entity_table = item.get("entity_table", item.get("entities", ()))
    errors: list[str] = []
    plan: Sequence[Mapping[str, Any]] = ()
    constraint_score = 0.0
    try:
        plan = labels_to_plan(labels, runtime.vlabench_vocabulary, world=runtime.world.vlabench)
        validation = validate_plan(
            plan,
            entity_table=entity_table,
            skill_arguments=runtime.world.vlabench.skill_arguments,
        )
        errors.extend(validation.errors)
        if validation.valid:
            root = materialize_plan(plan, entity_table, runtime.world.vlabench)
            evaluation = verify_plan_constraints(root, runtime.world.vlabench)
            constraint_score = 1.0 if evaluation is None else float(evaluation.score)
            if constraint_score < 1.0:
                errors.append(f"DomiKnowS constraints scored {constraint_score:.6f}")
    except Exception as error:
        validation = None
        errors.append(str(error))
    return StructuredPlanCandidate(
        labels=labels,
        log_probability=float(candidate.log_probability),
        plan=tuple(plan),
        dfa_valid=bool(dfa.accepts(labels)),
        schema_valid=bool(validation and validation.valid),
        constraint_valid=constraint_score >= 1.0,
        constraint_score=constraint_score,
        errors=tuple(errors),
    )


def _planner_context(item: Mapping[str, Any], domain: str) -> Mapping[str, Any]:
    supplied = item.get("planner_context")
    if isinstance(supplied, Mapping):
        return supplied
    if domain == "eai":
        return {
            "instruction": item.get("causal_prompt_text") or item.get("instruction_text")
            or item.get("instruction", ""),
            "goal": item.get("goal") or item.get("tl_goal", ""),
        }
    return {
        "instruction": item.get("instruction", ""),
        "images": item.get("images", item.get("segmented_image_paths", ())),
        "entity_table": item.get("entities", ()),
    }


@torch.no_grad()
def compare_structured_decoders(
    planner: Any,
    runtime: Any,
    domain: str,
    item: Mapping[str, Any],
    *,
    beam_width: int = 16,
    max_steps: int | None = None,
    ilp_time_limit_ms: int = 250,
) -> DecoderComparison:
    """Compare greedy, beam, and ILP from one shared encoded observation."""
    context = _planner_context(item, domain)
    dfa = runtime.dfa_for(domain, item if domain == "eai" else context)
    limit = max_steps or (
        runtime.max_eai_steps if domain == "eai"
        else runtime.max_vlabench_operations * 5 + 1
    )
    scorer = _score_eai_candidate if domain == "eai" else _score_vlabench_candidate
    domain_planner = planner.for_domain(domain) if hasattr(planner, "for_domain") else planner
    with runtime.domain_scope(domain):
        started = time.perf_counter()
        context_vector = domain_planner.encode_context(context)
        encoding_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        greedy_labels, greedy_logprob = domain_planner.sample_labels_from_context(
            context_vector, dfa, max_steps=limit, deterministic=True
        )
        greedy_candidate = scorer(
            LabelSequenceCandidate(tuple(greedy_labels), float(greedy_logprob.detach().cpu())),
            runtime, item, dfa,
        )
        greedy = DecoderSelection(
            "greedy", greedy_candidate, encoding_ms + (time.perf_counter() - started) * 1000.0
        )

        started = time.perf_counter()
        raw_candidates = domain_planner.beam_labels_from_context(
            context_vector, dfa, max_steps=limit, beam_width=beam_width
        )
        candidates = tuple(scorer(candidate, runtime, item, dfa) for candidate in raw_candidates)
        beam_ms = (time.perf_counter() - started) * 1000.0
        beam = DecoderSelection("beam", candidates[0], encoding_ms + beam_ms)
        ilp = select_candidate_with_ilp(
            candidates, domain, time_limit_ms=ilp_time_limit_ms
        )
        ilp = DecoderSelection(
            ilp.mode,
            ilp.selected,
            encoding_ms + beam_ms + ilp.elapsed_ms,
            ilp.solver_available,
            ilp.solver_status,
            ilp.fallback_reason,
        )
    return DecoderComparison(greedy, beam, ilp, candidates, encoding_ms)


def _empty_metrics(domain: str) -> dict[str, float]:
    shared = {"examples": 0.0, "valid": 0.0, "latency_ms": 0.0}
    if domain == "eai":
        return {
            **shared,
            "exact_sequence": 0.0,
            "goal_success": 0.0,
            "goal_recall": 0.0,
            "rl_reward": 0.0,
        }
    return {
        **shared,
        "exact_graph_match": 0.0,
        "skill_with_entity_match": 0.0,
        "reference_plan_score": 0.0,
    }


def _finish_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    count = float(metrics["examples"])
    return {
        key: (value if key == "examples" else value / max(count, 1.0))
        for key, value in metrics.items()
    }


def evaluate_structured_decoders(
    planner: Any,
    runtime: Any,
    eai_examples: Sequence[Mapping[str, Any]],
    vlabench_examples: Sequence[Mapping[str, Any]],
    *,
    beam_width: int = 16,
    ilp_time_limit_ms: int = 250,
) -> dict[str, Any]:
    """Run the offline ILP feasibility gate on held-out examples."""
    probe = probe_ilp_solver()
    metrics = {
        domain: {mode: _empty_metrics(domain) for mode in ("greedy", "beam", "ilp")}
        for domain in ("eai", "vlabench")
    }
    disagreements = {"eai": 0, "vlabench": 0}
    eai_added_successes = eai_lost_successes = ilp_fallbacks = 0

    for domain, examples in (("eai", eai_examples), ("vlabench", vlabench_examples)):
        for item in examples:
            comparison = compare_structured_decoders(
                planner,
                runtime,
                domain,
                item,
                beam_width=beam_width,
                ilp_time_limit_ms=ilp_time_limit_ms,
            )
            if comparison.ilp.fallback_reason:
                ilp_fallbacks += 1
            if comparison.beam.selected.labels != comparison.ilp.selected.labels:
                disagreements[domain] += 1
            selections = {
                "greedy": comparison.greedy,
                "beam": comparison.beam,
                "ilp": comparison.ilp,
            }
            for mode, selection in selections.items():
                candidate = selection.selected
                values = metrics[domain][mode]
                values["examples"] += 1.0
                values["valid"] += float(candidate.hard_valid)
                values["latency_ms"] += float(selection.elapsed_ms)
                if domain == "eai":
                    gold = [int(value) for value in torch.as_tensor(item["target_action_labels"]).tolist()]
                    eos = int(runtime.eai_vocabulary.eos_label)
                    if eos in gold:
                        gold = gold[: gold.index(eos) + 1]
                    values["exact_sequence"] += float(list(candidate.labels) == gold)
                    values["goal_success"] += float(bool(candidate.goal_success))
                    values["goal_recall"] += float(candidate.goal_recall or 0.0)
                    values["rl_reward"] += float(candidate.reward_score or 0.0)
                else:
                    score = score_vlabench_plan(
                        list(candidate.plan),
                        item["operation_sequence"],
                        entity_table=item.get("entities", ()),
                        world_bundle=runtime.world.vlabench,
                    )
                    values["exact_graph_match"] += float(score.exact_graph_match)
                    values["skill_with_entity_match"] += float(score.skill_with_entity_match)
                    values["reference_plan_score"] += float(score.total)
            if domain == "eai":
                greedy_success = bool(comparison.greedy.selected.goal_success)
                ilp_success = bool(comparison.ilp.selected.goal_success)
                eai_added_successes += int(ilp_success and not greedy_success)
                eai_lost_successes += int(greedy_success and not ilp_success)

    finished = {
        domain: {mode: _finish_metrics(values) for mode, values in modes.items()}
        for domain, modes in metrics.items()
    }
    all_ilp_available = probe.available and ilp_fallbacks == 0
    eai_gate = all_ilp_available and eai_added_successes >= 2 and eai_lost_successes == 0
    requires_rollouts = disagreements["vlabench"] > 0
    recommendation = (
        "run paired VLABench simulator rollouts before enabling ILP in production"
        if requires_rollouts else
        "do not enable ILP by default; no distinct VLABench plans were selected"
    )
    if not eai_gate:
        recommendation = "keep DFA decoding as default; the EAI ILP benefit gate did not pass"
    return {
        "ilp_probe": asdict(probe),
        "ilp_available_for_all_examples": all_ilp_available,
        "ilp_fallbacks": ilp_fallbacks,
        "beam_width": int(beam_width),
        "ilp_time_limit_ms": int(ilp_time_limit_ms),
        "metrics": finished,
        "beam_ilp_disagreements": disagreements,
        "eai_added_successes": eai_added_successes,
        "eai_lost_successes": eai_lost_successes,
        "offline_eai_gate_passed": eai_gate,
        "requires_paired_vlabench_rollouts": requires_rollouts,
        "recommendation": recommendation,
    }


def write_evaluation_report(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target.resolve()
