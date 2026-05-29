"""Build JSON traces for the package-shipping planning visualization."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any

import torch

from domiknows.generation import explain_dfa_rejection
from domiknows.generation.learners import project_matrix
from domiknows.generation.applications.planning import planning_bundle_from_graph, planning_dfa_from_graph


TASKS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASKS_DIR.parent
for _path in (REPO_ROOT, TASKS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from Tasks.package_shipping.graph import build_graph
    from Tasks.package_shipping.learning_program import fit_graph_hmm
    from Tasks.package_shipping.planner_agent import MockPackageShippingPlannerAgent
except ImportError:  # pragma: no cover - direct task execution fallback
    from package_shipping.graph import build_graph
    from package_shipping.learning_program import fit_graph_hmm
    from package_shipping.planner_agent import MockPackageShippingPlannerAgent


SHIPPING_TASKS = ("ship_book", "ship_fragile_vase", "return_item")
DEFAULT_CANDIDATE_SOURCE = "invalid_drop_before_seal"

PLANNING_KWARGS = {
    "planned_task_name": "planned_shipping_task",
    "task_root_name": "shipping_task",
    "required_action_schema_name": "task_requires_action",
    "task_role_name": "task",
}


def build_flow(
    *,
    task: str = "ship_fragile_vase",
    candidate_source: str = DEFAULT_CANDIDATE_SOURCE,
    seed: int = 0,
    hmm_iterations: int = 20,
) -> dict[str, Any]:
    """Build a strict-JSON-friendly explanation flow for one shipping plan."""

    graph, _parts = build_graph(task)
    bundle = planning_bundle_from_graph(graph, selected_task=task, **PLANNING_KWARGS)
    dfa = planning_dfa_from_graph(bundle)
    artifacts = fit_graph_hmm(bundle, max_iter=hmm_iterations, random_seed=seed)
    planner = MockPackageShippingPlannerAgent(seed=seed)
    candidates = planner.propose(bundle, count=8)
    candidate = _select_candidate(candidates, candidate_source)

    dfa_trace = _trace_dfa(dfa, candidate.actions)
    hmm_trace = _trace_graph_hmm(artifacts.hmm, candidate.actions)
    candidate_summaries = _summarize_candidates(dfa, artifacts.hmm, candidates)
    steps = _combine_steps(bundle, candidate.actions, dfa_trace, hmm_trace)

    return {
        "title": "Package Shipping Planning Flow",
        "purpose": (
            "A planner proposes a shipping plan. The DFA enforces graph-derived hard rules; "
            "the graph-HMM scores legal phase/action flow and explains where graph support disappears."
        ),
        "graph": {
            "name": graph.name,
            "logical_constraint_count": len(getattr(graph, "_logicalConstrains", ())),
            "execution_note": (
                "The visible domain definition lives in Tasks/package_shipping/graph.py. "
                "This viewer only visualizes execution artifacts derived from it."
            ),
        },
        "task": {
            "selected": bundle.selected_task,
            "available": list(bundle.task_names),
            "required_actions": list(bundle.selected_required_actions),
            "reference_plan": list(bundle.selected_reference_plan),
            "non_terminal_limit": bundle.non_terminal_limit,
            "action_count_limits": dict(bundle.action_count_limits),
        },
        "candidate": {
            "source": candidate.source,
            "actions": list(candidate.actions),
            "accepted": dfa_trace["accepted"],
        },
        "candidate_options": candidate_summaries,
        "dfa": dfa_trace,
        "graph_hmm": hmm_trace,
        "steps": steps,
        "help": _help_text(),
    }


def _select_candidate(candidates, source: str):
    for candidate in candidates:
        if candidate.source == source:
            return candidate
    available = ", ".join(candidate.source for candidate in candidates)
    raise ValueError(f"unknown candidate source {source!r}; available sources: {available}")


def _summarize_candidates(dfa, hmm, candidates) -> list[dict[str, Any]]:
    summaries = []
    for candidate in candidates:
        score = hmm.score(candidate.actions)
        summaries.append(
            {
                "source": candidate.source,
                "actions": list(candidate.actions),
                "accepted": dfa.accepts(candidate.actions),
                "hmm_log_likelihood": _json_number(score),
            }
        )
    return summaries


def _trace_dfa(dfa, actions: tuple[str, ...]) -> dict[str, Any]:
    state = dfa.start_state
    steps = []
    blocked = False
    reason = None
    for index, action in enumerate(actions):
        allowed = sorted(str(symbol) for symbol in dfa.allowed_tokens(state))
        next_state = dfa.step(state, action)
        blocked_here = next_state is None or action not in allowed
        if blocked_here:
            blocked = True
            reason = explain_dfa_rejection(dfa, actions)
            steps.append(
                {
                    "index": index,
                    "action": action,
                    "from_state": _state_to_string(state),
                    "to_state": None if next_state is None else _state_to_string(next_state),
                    "allowed_actions": allowed,
                    "blocked": True,
                    "reason": reason,
                }
            )
            break
        steps.append(
            {
                "index": index,
                "action": action,
                "from_state": _state_to_string(state),
                "to_state": _state_to_string(next_state),
                "allowed_actions": allowed,
                "blocked": False,
                "reason": None,
            }
        )
        state = next_state

    accepted = (not blocked) and dfa.is_accepting(state)
    if not accepted and reason is None:
        reason = f"sequence ended in non-accepting state {_state_to_string(state)}"

    return {
        "start_state": _state_to_string(dfa.start_state),
        "final_state": _state_to_string(state),
        "state_count": len(dfa.states),
        "accepted": bool(accepted),
        "blocked": bool(blocked),
        "rejection_reason": None if accepted else reason,
        "steps": steps,
    }


def _trace_graph_hmm(hmm, actions: tuple[str, ...]) -> dict[str, Any]:
    hmm._require_fitted()
    encoded = hmm._encode_sequence(actions, allow_unknown=False)
    transition, emission = hmm._projected_dynamics()
    score = hmm.score(actions)
    viterbi = hmm.viterbi(actions)
    fb = hmm._forward_backward_encoded(encoded)

    steps = []
    belief = hmm.initial_.clone()
    log_likelihood = 0.0
    blocked_index = None
    for index, symbol_id in enumerate(encoded):
        action = hmm.id_to_symbol[symbol_id]
        if index == 0:
            prior = belief
        else:
            transition_t = hmm._transition_for_context(
                step=index - 1,
                prefix=tuple(actions[:index]),
                belief=belief,
                sequence=tuple(actions),
            )
            prior = belief @ transition_t
        emission_column = emission[:, symbol_id] * hmm.emission_mask_[:, symbol_id]
        unnormalized = prior * emission_column
        normalizer = float(unnormalized.sum().detach().cpu().item())
        if normalizer <= 0:
            belief = torch.zeros_like(belief)
            if blocked_index is None:
                blocked_index = index
        else:
            belief = unnormalized / normalizer
            log_likelihood += torch.log(torch.tensor(normalizer, dtype=hmm.dtype)).item()
        allowed_emitters = [
            hmm.state_names[state_index]
            for state_index in range(hmm.n_hidden_states)
            if float(hmm.emission_mask_[state_index, symbol_id].detach().cpu().item()) > 0
        ]
        steps.append(
            {
                "index": index,
                "action": str(action),
                "prior": _named_tensor(hmm.state_names, prior),
                "emission_likelihood": _named_tensor(hmm.state_names, emission_column),
                "belief": _named_tensor(hmm.state_names, belief),
                "normalizer": _json_number(normalizer),
                "log_likelihood_so_far": _json_number(log_likelihood if blocked_index is None else float("-inf")),
                "most_likely_phase": _argmax_state(hmm.state_names, belief),
                "allowed_emitting_phases": allowed_emitters,
                "support_blocked": blocked_index == index,
            }
        )

    if fb is not None:
        alpha, beta, gamma, xi, _ll = fb
        fb_summary = {
            "alpha": [_named_tensor(hmm.state_names, row) for row in alpha],
            "beta": [_named_tensor(hmm.state_names, row) for row in beta],
            "gamma": [_named_tensor(hmm.state_names, row) for row in gamma],
            "xi_shape": list(xi.shape),
        }
    else:
        fb_summary = {
            "alpha": [],
            "beta": [],
            "gamma": [],
            "xi_shape": [max(0, len(actions) - 1), hmm.n_hidden_states, hmm.n_hidden_states],
        }

    return {
        "states": list(hmm.state_names),
        "symbols": [str(symbol) for symbol in hmm.id_to_symbol],
        "log_likelihood": _json_number(score),
        "viterbi_path": list(viterbi.states),
        "viterbi_score": _json_number(viterbi.score),
        "support_blocked": blocked_index is not None or score == float("-inf"),
        "blocked_index": blocked_index,
        "transition_mask": _matrix_tensor(hmm.state_names, hmm.state_names, hmm.transition_mask_),
        "emission_mask": _matrix_tensor(hmm.state_names, tuple(str(symbol) for symbol in hmm.id_to_symbol), hmm.emission_mask_),
        "transition_probs": _matrix_tensor(hmm.state_names, hmm.state_names, transition),
        "emission_probs": _matrix_tensor(hmm.state_names, tuple(str(symbol) for symbol in hmm.id_to_symbol), emission),
        "forward_backward": fb_summary,
        "steps": steps,
    }


def _combine_steps(bundle, actions: tuple[str, ...], dfa_trace: dict[str, Any], hmm_trace: dict[str, Any]) -> list[dict[str, Any]]:
    steps = []
    required = set(bundle.selected_required_actions)
    counts = {action: 0 for action in bundle.action_count_limits}
    seen_done = False
    for index, action in enumerate(actions):
        if action in counts:
            counts[action] += 1
        if action == bundle.terminal_action:
            seen_done = True
        prefix = actions[: index + 1]
        remaining_required = sorted(required - set(prefix))
        dfa_step = dfa_trace["steps"][index] if index < len(dfa_trace["steps"]) else _after_rejection_step(index, action, dfa_trace)
        hmm_step = hmm_trace["steps"][index]
        steps.append(
            {
                "index": index,
                "action": action,
                "prefix": list(prefix),
                "dfa": dfa_step,
                "graph_hmm": hmm_step,
                "diagnostics": {
                    "remaining_required_actions": remaining_required,
                    "action_counts": dict(counts),
                    "terminal_seen": seen_done,
                },
                "explanation": _step_explanation(bundle, action, dfa_step, hmm_step, remaining_required),
            }
        )
    return steps


def _after_rejection_step(index: int, action: str, dfa_trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "action": action,
        "from_state": dfa_trace["final_state"],
        "to_state": None,
        "allowed_actions": [],
        "blocked": True,
        "reason": "DFA had already rejected an earlier action.",
    }


def _step_explanation(bundle, action: str, dfa_step: dict[str, Any], hmm_step: dict[str, Any], remaining_required: list[str]) -> str:
    if dfa_step.get("blocked") or hmm_step.get("support_blocked"):
        return (
            f"The planner proposed {action}. The hard DFA or graph-HMM support says this action "
            "does not fit the declared shipping phase/count/requirement structure at this point."
        )
    if remaining_required:
        return (
            f"The planner proposed {action}. It is legal so far; the plan still has to include "
            f"{', '.join(remaining_required)} before it can finish."
        )
    return f"The planner proposed {action}. The action is legal and all required actions are now present."


def _help_text() -> dict[str, str]:
    return {
        "dfa": "The DFA is the hard verifier derived from the graph. It remembers the current phase, counts, required actions, and whether acceptance is still reachable.",
        "graph_hmm": "The graph-HMM is a probabilistic model over hidden shipping phases. It learns likely legal phase/action flow from graph-declared reference plans.",
        "phase": "A phase is a hidden state such as item_inserted or sealed. The HMM uses phases to explain why each action is likely or impossible.",
        "belief": "Belief is the HMM's current probability distribution over hidden phases after reading the prefix so far.",
        "transition_mask": "The transition mask comes from phase_transition facts in the graph. A zero means that hidden phase jump is impossible.",
        "emission_mask": "The emission mask says which actions each phase is allowed to emit. A zero means that phase cannot explain that action.",
        "normalizer": "The normalizer is the probability mass left after applying the transition and emission masks for this step. Zero means every legal path was blocked.",
        "viterbi": "Viterbi is the single most likely hidden phase path for the whole action sequence, when one exists.",
        "required": "Required actions come from task_requires_action facts in the graph. The DFA tracks which ones have appeared.",
        "action_step": "One proposed planner action. Click it to inspect how the DFA and graph-HMM process that prefix.",
        "dfa_state": "A DFA state is execution memory: current phase, count-limited action counts, required-action bitset, and non-terminal length.",
        "allowed_actions": "Allowed actions are the next symbols that keep the DFA away from dead states and still on a path that can eventually accept.",
        "hmm_score": "The graph-HMM log-likelihood is a soft score for the whole plan under learned legal phase/action dynamics. -inf means no legal HMM path exists.",
        "belief_cell": "This number is the current HMM belief mass for one hidden shipping phase after reading the prefix so far.",
        "mask_cell": "Mask cell: 1 means this graph-declared transition or emission is possible; 0 means it is impossible and gets zero probability.",
        "candidate_option": "A proposal from the mock planner. Valid candidates are hard-DFA accepted; the graph-HMM score ranks legal plan flow.",
        "remaining_required": "These graph-declared required actions have not appeared yet in the current prefix.",
        "count_limited_action": "This action has a graph-declared maximum count. The DFA tracks the count exactly.",
    }


def _state_to_string(state: Any) -> str:
    if is_dataclass(state):
        return str(asdict(state))
    return str(state)


def _argmax_state(state_names: tuple[str, ...], values: torch.Tensor) -> str | None:
    if values.sum() <= 0:
        return None
    return state_names[int(torch.argmax(values).detach().cpu().item())]


def _named_tensor(names: tuple[str, ...], values: torch.Tensor) -> dict[str, float | str]:
    flat = values.detach().cpu().reshape(-1).tolist()
    return {str(name): _json_number(round(float(value), 6)) for name, value in zip(names, flat)}


def _matrix_tensor(rows: tuple[str, ...], columns: tuple[str, ...], values: torch.Tensor) -> dict[str, dict[str, float | str]]:
    cpu = values.detach().cpu()
    return {
        str(row): {
            str(column): _json_number(round(float(cpu[row_index, col_index].item()), 6))
            for col_index, column in enumerate(columns)
        }
        for row_index, row in enumerate(rows)
    }


def _json_number(value: float) -> float | str:
    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    if value != value:
        return "nan"
    return float(value)
