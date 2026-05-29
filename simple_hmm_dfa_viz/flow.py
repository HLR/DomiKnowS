"""Build the JSON flow that drives the simple HMM + DFA viewer."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from domiknows.generation import (
    analyze_generation_constraints,
    constraints_to_dfa_from_graph,
    explain_dfa_rejection,
    generation_bundle_from_graph,
)

try:
    from .graph import EOS_TOKEN, VOCAB, build_graph, build_two_constraint_graph
except ImportError:  # pragma: no cover - supports direct script execution
    from graph import EOS_TOKEN, VOCAB, build_graph, build_two_constraint_graph


CANDIDATES = {
    "valid": ("A", "B", "C", "A"),
    "invalid": ("A", "B", "C", "B"),
}

TWO_CONSTRAINT_CANDIDATES = {
    "valid": ("A", "B", "C", "END"),
    "two_b": ("A", "B", "C", "B", "END"),
    "missing_c": ("A", "B", "END"),
}

DEMOS = ("one", "two")


@dataclass(frozen=True)
class TinyHMM:
    """A hand-sized HMM used for explanatory tracing, not model quality."""

    states: tuple[str, ...]
    initial: tuple[float, ...]
    transition: tuple[tuple[float, ...], ...]
    emission: tuple[tuple[float, ...], ...]
    transition_mask: tuple[tuple[float, ...], ...]
    emission_mask: tuple[tuple[float, ...], ...]
    symbols: tuple[str, ...]


def build_bundle(demo: str = "one"):
    """Return ``(graph, bundle)`` for the selected declarative graph."""

    if demo == "one":
        graph, _parts = build_graph()
    elif demo == "two":
        graph, _parts = build_two_constraint_graph()
    else:
        raise ValueError(f"demo must be one of {DEMOS!r}")
    bundle = generation_bundle_from_graph(
        graph,
        vocab=VOCAB,
        eos_token=EOS_TOKEN,
        text_name="string",
        token_name="position",
        generated_token_name="generated_symbol",
        before_relation_name="precedes",
        first_role_name="earlier",
        second_role_name="later",
    )
    return graph, bundle


def build_tiny_hmm(symbols: tuple[str, ...] = VOCAB) -> TinyHMM:
    """Create a tiny masked HMM whose states explain the ``at most one B`` rule."""

    states = ("before_B", "emit_B", "after_B")
    transition_mask = (
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    emission_mask = (
        (1.0, 0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
    )
    transition = (
        (0.78, 0.22, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    emission = (
        (0.46, 0.0, 0.46, 0.08),
        (0.0, 1.0, 0.0, 0.0),
        (0.48, 0.0, 0.44, 0.08),
    )
    initial = (0.85, 0.15, 0.0)
    return TinyHMM(
        states=states,
        initial=initial,
        transition=transition,
        emission=emission,
        transition_mask=transition_mask,
        emission_mask=emission_mask,
        symbols=tuple(symbols),
    )


def build_two_constraint_hmm(symbols: tuple[str, ...] = VOCAB) -> TinyHMM:
    """Create a tiny masked HMM that explains both demo constraints.

    The hidden states track whether ``B`` has already been used and whether
    the required ``C`` has been seen.  Special ``emit_*`` states make the
    ordinary HMM transition matrix able to model flag changes in a readable
    way.
    """

    states = (
        "need_C_no_B",
        "emit_B_need_C",
        "need_C_seen_B",
        "emit_C_no_B",
        "seen_C_no_B",
        "emit_B_seen_C",
        "seen_C_seen_B",
        "emit_C_seen_B",
    )
    transition_mask = (
        (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    emission_mask = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    transition = (
        (0.55, 0.20, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.65, 0.0, 0.0, 0.0, 0.0, 0.35),
        (0.0, 0.0, 0.70, 0.0, 0.0, 0.0, 0.0, 0.30),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.15, 0.60, 0.25, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.22),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    emission = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.72, 0.0, 0.0, 0.28),
        (0.0, 1.0, 0.0, 0.0),
        (0.65, 0.0, 0.0, 0.35),
        (0.0, 0.0, 1.0, 0.0),
    )
    initial = (0.62, 0.18, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0)
    return TinyHMM(
        states=states,
        initial=initial,
        transition=transition,
        emission=emission,
        transition_mask=transition_mask,
        emission_mask=emission_mask,
        symbols=tuple(symbols),
    )


def build_flow(candidate: str = "valid", *, demo: str = "one") -> dict[str, Any]:
    """Build the complete JSON-serializable explanation flow."""

    candidate_map = CANDIDATES if demo == "one" else TWO_CONSTRAINT_CANDIDATES
    if candidate not in candidate_map:
        raise ValueError(f"candidate must be one of {sorted(candidate_map)}")

    graph, bundle = build_bundle(demo)
    analyses = analyze_generation_constraints(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")
    sequence = candidate_map[candidate]
    label_sequence = [bundle.vocabulary.label_for_token(symbol) for symbol in sequence]
    hmm = build_tiny_hmm() if demo == "one" else build_two_constraint_hmm()
    hmm_trace = trace_tiny_hmm(hmm, sequence)
    dfa_trace = trace_dfa_for_labels(dfa, bundle, label_sequence)

    steps = []
    for index, symbol in enumerate(sequence):
        dfa_step = dfa_trace["steps"][index] if index < len(dfa_trace["steps"]) else None
        hmm_step = hmm_trace["steps"][index]
        steps.append(
            {
                "index": index,
                "symbol": symbol,
                "dfa": dfa_step,
                "hmm": hmm_step,
                "explanation": _step_explanation(symbol, dfa_step, hmm_step),
            }
        )

    return {
        "title": "Simple HMM + DFA Constraint Trace",
        "audience_note": (
            "The DFA is the hard rule checker. The HMM is the probabilistic "
            "flow model that shows how hidden states and masks score the same string."
        ),
        "generator": {
            "name": "Mock limited-vocabulary generator",
            "demo": demo,
            "candidate": candidate,
            "sequence": list(sequence),
            "available_candidates": {name: list(tokens) for name, tokens in candidate_map.items()},
        },
        "vocabulary": list(VOCAB),
        "constraint": {
            "text": (
                "Token B may appear at most once."
                if demo == "one"
                else "Token B may appear at most once, and token C must appear at least once."
            ),
            "domiknows": (
                "atMostAL(generated_symbol.B('x'), 1)"
                if demo == "one"
                else "atMostAL(generated_symbol.B('x'), 1); atLeastAL(generated_symbol.C('x'), 1)"
            ),
            "discovered": [analysis.lc_name for analysis in analyses if analysis.supported],
        },
        "dfa": {
            "accepted": dfa_trace["accepted"],
            "rejection_reason": dfa_trace["rejection_reason"],
            "states": sorted((str(state) for state in dfa.states), key=str),
            "start_state": str(dfa.start_state),
            "accepting_states": sorted((str(state) for state in dfa.accepting_states), key=str),
            "dead_states": sorted((str(state) for state in dfa.dead_states), key=str),
            "transitions": _dfa_transitions_to_dict(dfa, bundle),
            "trace": dfa_trace,
        },
        "hmm": {
            "states": list(hmm.states),
            "initial": _named_vector(hmm.states, hmm.initial),
            "transition_mask": _matrix_dict(hmm.states, hmm.states, hmm.transition_mask),
            "emission_mask": _matrix_dict(hmm.states, hmm.symbols, hmm.emission_mask),
            "transition_probs": _matrix_dict(hmm.states, hmm.states, hmm.transition),
            "emission_probs": _matrix_dict(hmm.states, hmm.symbols, hmm.emission),
            "trace": hmm_trace,
        },
        "steps": steps,
    }


def trace_tiny_hmm(hmm: TinyHMM, sequence: tuple[str, ...]) -> dict[str, Any]:
    """Return alpha/belief and simple Viterbi trace for ``sequence``."""

    symbol_to_id = {symbol: index for index, symbol in enumerate(hmm.symbols)}
    alpha: tuple[float, ...] | None = None
    log_likelihood = 0.0
    blocked = False
    steps = []

    viterbi_scores = [_safe_log(value) for value in hmm.initial]
    viterbi_paths = [[state] for state in range(len(hmm.states))]

    for index, symbol in enumerate(sequence):
        symbol_id = symbol_to_id[symbol]
        prior = hmm.initial if alpha is None else _matmul_row(alpha, hmm.transition)
        emission_likelihood = tuple(row[symbol_id] for row in hmm.emission)
        unnormalized = tuple(prior[i] * emission_likelihood[i] for i in range(len(hmm.states)))
        normalizer = float(sum(unnormalized))
        if normalizer <= 0.0:
            belief = tuple(0.0 for _ in unnormalized)
            blocked = True
            log_likelihood = float("-inf")
        else:
            belief = tuple(value / normalizer for value in unnormalized)
            if not math.isinf(log_likelihood):
                log_likelihood += math.log(normalizer)
        alpha = belief

        viterbi_scores, viterbi_paths = _viterbi_step(hmm, viterbi_scores, viterbi_paths, symbol_id)
        best_state = None
        finite_indices = [idx for idx, value in enumerate(viterbi_scores) if math.isfinite(value)]
        if finite_indices:
            best_index = max(finite_indices, key=lambda idx: viterbi_scores[idx])
            best_state = hmm.states[best_index]

        steps.append(
            {
                "index": index,
                "symbol": symbol,
                "prior": _named_vector(hmm.states, prior),
                "emission_likelihood": _named_vector(hmm.states, emission_likelihood),
                "unnormalized_belief": _named_vector(hmm.states, unnormalized),
                "belief": _named_vector(hmm.states, belief),
                "normalizer": normalizer,
                "log_likelihood_so_far": _json_number(log_likelihood),
                "support_blocked": normalizer <= 0.0,
                "most_likely_state": best_state,
            }
        )

    best_path = []
    finite_indices = [idx for idx, value in enumerate(viterbi_scores) if math.isfinite(value)]
    if finite_indices and not blocked:
        best_index = max(finite_indices, key=lambda idx: viterbi_scores[idx])
        best_path = [hmm.states[state] for state in viterbi_paths[best_index]]
    return {
        "accepted_by_hmm_support": not blocked,
        "log_likelihood": _json_number(log_likelihood),
        "viterbi_path": best_path,
        "steps": steps,
    }


def trace_dfa_for_labels(dfa, bundle, labels: list[int]) -> dict[str, Any]:
    """Serialize a DFA trace while labeling symbols and states separately."""

    state = dfa.start_state
    steps = []
    blocked = False
    reason = None
    for index, label in enumerate(labels):
        allowed_labels = sorted(dfa.allowed_tokens(state))
        allowed_symbols = [bundle.vocabulary.token_for_label(item) for item in allowed_labels]
        next_state = dfa.step(state, label)
        if label not in allowed_labels:
            reason = explain_dfa_rejection(dfa, labels)
            steps.append(
                {
                    "index": index,
                    "symbol": bundle.vocabulary.token_for_label(label),
                    "from_state": str(state),
                    "to_state": None if next_state is None else str(next_state),
                    "allowed_symbols": allowed_symbols,
                    "blocked": True,
                    "reason": reason,
                }
            )
            blocked = True
            break
        steps.append(
            {
                "index": index,
                "symbol": bundle.vocabulary.token_for_label(label),
                "from_state": str(state),
                "to_state": str(next_state),
                "allowed_symbols": allowed_symbols,
                "blocked": False,
                "reason": None,
            }
        )
        state = next_state
    accepted = (not blocked) and dfa.is_accepting(state)
    if not accepted and reason is None:
        reason = f"sequence ended in non-accepting state {state}"
    return {
        "sequence": [bundle.vocabulary.token_for_label(label) for label in labels],
        "start_state": str(dfa.start_state),
        "final_state": str(state),
        "accepted": accepted,
        "blocked": blocked,
        "rejection_reason": None if accepted else reason,
        "steps": steps,
    }


def _viterbi_step(hmm: TinyHMM, previous_scores, previous_paths, symbol_id: int):
    next_scores: list[float] = []
    paths: list[list[int]] = []
    for to_state in range(len(hmm.states)):
        emission = hmm.emission[to_state][symbol_id]
        best_score = float("-inf")
        best_prev = None
        if emission > 0:
            for from_state, previous_score in enumerate(previous_scores):
                transition = hmm.transition[from_state][to_state]
                if transition > 0 and math.isfinite(previous_score):
                    score = previous_score + _safe_log(transition) + _safe_log(emission)
                    if score > best_score:
                        best_score = score
                        best_prev = from_state
        next_scores.append(best_score)
        paths.append([*previous_paths[best_prev], to_state] if best_prev is not None else [to_state])
    return next_scores, paths


def _safe_log(value: float) -> float:
    return math.log(value) if value > 0 else float("-inf")


def _json_number(value: float) -> float | str:
    """Return strict-JSON-safe numeric data for browser parsing."""

    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    if math.isnan(value):
        return "nan"
    return value


def _matmul_row(row: tuple[float, ...], matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(sum(row[i] * matrix[i][j] for i in range(len(row))) for j in range(len(matrix[0])))


def _named_vector(names: tuple[str, ...], values: tuple[float, ...]) -> dict[str, float]:
    return {name: round(float(value), 6) for name, value in zip(names, values)}


def _matrix_dict(rows: tuple[str, ...], columns: tuple[str, ...], values: tuple[tuple[float, ...], ...]) -> dict[str, dict[str, float]]:
    return {
        row: {column: round(float(values[row_index][col_index]), 6) for col_index, column in enumerate(columns)}
        for row_index, row in enumerate(rows)
    }


def _dfa_transitions_to_dict(dfa, bundle) -> list[dict[str, str]]:
    transitions = []
    for (state, label), next_state in sorted(dfa.transitions.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        if label >= bundle.vocabulary.label_count:
            continue
        transitions.append(
            {
                "from": str(state),
                "symbol": bundle.vocabulary.token_for_label(label),
                "to": str(next_state),
            }
        )
    return transitions


def _step_explanation(symbol: str, dfa_step: dict[str, Any] | None, hmm_step: dict[str, Any]) -> str:
    if dfa_step and dfa_step.get("blocked"):
        return (
            f"The generator proposed {symbol}. The DFA blocks it because the only "
            f"allowed next symbols were {dfa_step['allowed_symbols']}."
        )
    if hmm_step["support_blocked"]:
        return (
            f"The HMM assigns zero support to {symbol} from every currently possible "
            "hidden state, so the probabilistic model also treats this step as impossible."
        )
    state = hmm_step["most_likely_state"]
    return (
        f"The generator proposed {symbol}. The DFA allows it, and the HMM belief "
        f"moves mostly through hidden state {state}."
    )
