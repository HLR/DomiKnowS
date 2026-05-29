"""Build the JSON flow for the real HMM comparison viewer."""
from __future__ import annotations

from typing import Any

import torch

from domiknows.generation import (
    DiscreteHMM,
    DomiKnowSAwareHMM,
    analyze_generation_constraints,
    constraints_to_dfa_from_graph,
    domiknows_hmm_from_generation_constraints,
    explain_dfa_rejection,
    generation_bundle_from_graph,
    trace_discrete_hmm,
)
from domiknows.generation.learners import project_matrix

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
PLAIN_STATE_NAMES = ("S0", "S1", "S2")
GRAPH_STATE_NAMES = ("before_B", "emit_B", "after_B")
GRAPH_TWO_STATE_NAMES = (
    "need_C_no_B",
    "emit_B_need_C",
    "need_C_seen_B",
    "emit_C_no_B",
    "seen_C_no_B",
    "emit_B_seen_C",
    "seen_C_seen_B",
    "emit_C_seen_B",
)


def build_bundle(demo: str = "one"):
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


def build_plain_hmm() -> DiscreteHMM:
    """A permissive HMM: every state can emit every symbol with positive mass."""

    return DiscreteHMM(
        transition=[
            [0.72, 0.18, 0.10],
            [0.20, 0.20, 0.60],
            [0.22, 0.18, 0.60],
        ],
        emission=[
            [0.38, 0.24, 0.30, 0.08],
            [0.16, 0.56, 0.20, 0.08],
            [0.33, 0.16, 0.43, 0.08],
        ],
        initial=[0.76, 0.18, 0.06],
        symbols=VOCAB,
        state_names=PLAIN_STATE_NAMES,
        normalize=True,
        dtype=torch.float64,
    )


def build_domiknows_hmm() -> DomiKnowSAwareHMM:
    """A graph-aware HMM whose states/masks are compiled from the DFA."""

    graph, bundle = build_bundle("one")
    return domiknows_hmm_from_generation_constraints(
        graph,
        bundle,
        symbols=VOCAB,
        dtype=torch.float64,
    )


def build_two_constraint_domiknows_hmm() -> DomiKnowSAwareHMM:
    """A graph-aware HMM whose states/masks are compiled from both constraints."""

    graph, bundle = build_bundle("two")
    return domiknows_hmm_from_generation_constraints(
        graph,
        bundle,
        symbols=VOCAB,
        dtype=torch.float64,
    )


def build_flow(candidate: str = "valid", *, demo: str = "one") -> dict[str, Any]:
    candidate_map = CANDIDATES if demo == "one" else TWO_CONSTRAINT_CANDIDATES
    if candidate not in candidate_map:
        raise ValueError(f"candidate must be one of {sorted(candidate_map)}")

    graph, bundle = build_bundle(demo)
    analyses = analyze_generation_constraints(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")
    sequence = candidate_map[candidate]
    labels = [bundle.vocabulary.label_for_token(symbol) for symbol in sequence]

    plain_hmm = build_plain_hmm()
    graph_hmm = build_domiknows_hmm() if demo == "one" else build_two_constraint_domiknows_hmm()
    dfa_trace = _trace_dfa_for_labels(dfa, bundle, labels)
    plain_trace = trace_discrete_hmm(plain_hmm, sequence)
    graph_trace = trace_domiknows_aware_hmm(graph_hmm, sequence)

    steps = []
    for index, symbol in enumerate(sequence):
        dfa_step = dfa_trace["steps"][index] if index < len(dfa_trace["steps"]) else _missing_dfa_step(index, symbol, dfa_trace)
        plain_step = plain_trace["steps"][index]
        graph_step = graph_trace["steps"][index]
        steps.append(
            {
                "index": index,
                "symbol": symbol,
                "dfa": dfa_step,
                "discrete_hmm": plain_step,
                "domiknows_hmm": graph_step,
                "explanation": _step_explanation(symbol, dfa_step, plain_step, graph_step),
            }
        )

    return {
        "title": "Real HMM vs DomiKnowS-Aware HMM",
        "summary": [
            "DFA enforces the rule.",
            "DiscreteHMM scores the string but does not know the graph rule unless its parameters encode it.",
            "DomiKnowSAwareHMM projects probabilities through graph/constraint masks.",
        ],
        "candidate": {
            "demo": demo,
            "name": candidate,
            "sequence": list(sequence),
            "available": {name: list(tokens) for name, tokens in candidate_map.items()},
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
            "trace": dfa_trace,
        },
        "discrete_hmm": plain_trace,
        "domiknows_hmm": graph_trace,
        "steps": steps,
    }


def trace_domiknows_aware_hmm(model: DomiKnowSAwareHMM, sequence: tuple[str, ...]) -> dict[str, Any]:
    """Trace a fitted graph-aware HMM using its public score/Viterbi surface."""

    model._require_fitted()
    symbol_to_id = model.symbol_to_id
    encoded = [symbol_to_id[symbol] for symbol in sequence]
    transition = project_matrix(model.transition_, model.transition_mask_, smoothing=model.smoothing)
    emission = project_matrix(model.emission_, model.emission_mask_, smoothing=model.smoothing)
    initial = model.initial_

    alpha = None
    log_likelihood = 0.0
    blocked_index = None
    steps = []
    for index, symbol_id in enumerate(encoded):
        symbol = model.id_to_symbol[symbol_id]
        prior = initial if alpha is None else alpha @ transition
        emit = emission[:, symbol_id] * model.emission_mask_[:, symbol_id]
        unnormalized = prior * emit
        normalizer = float(unnormalized.sum().detach().cpu().item())
        if normalizer <= 0.0:
            if blocked_index is None:
                blocked_index = index
            belief = torch.zeros_like(unnormalized)
            log_likelihood = float("-inf")
        else:
            belief = unnormalized / unnormalized.sum().clamp_min(torch.finfo(model.dtype).tiny)
            if log_likelihood != float("-inf"):
                log_likelihood += float(torch.log(unnormalized.sum()).detach().cpu().item())
        alpha = belief
        steps.append(
            {
                "index": index,
                "symbol": str(symbol),
                "prior": _named_tensor(model.state_names, prior),
                "emission_likelihood": _named_tensor(model.state_names, emit),
                "unnormalized_belief": _named_tensor(model.state_names, unnormalized),
                "alpha": _named_tensor(model.state_names, belief),
                "beta": _named_tensor(model.state_names, torch.ones_like(belief) if normalizer > 0 else torch.zeros_like(belief)),
                "gamma": _named_tensor(model.state_names, belief),
                "belief": _named_tensor(model.state_names, belief),
                "normalizer": _json_number(normalizer),
                "log_likelihood_so_far": _json_number(log_likelihood),
                "support_blocked": blocked_index == index,
                "most_likely_state": _argmax_state(model.state_names, belief),
            }
        )

    score = model.score(sequence)
    viterbi = model.viterbi(sequence)
    viterbi_states = () if score == float("-inf") else viterbi.states
    viterbi_score = float("-inf") if score == float("-inf") else viterbi.score
    return {
        "states": list(model.state_names),
        "symbols": [str(symbol) for symbol in model.id_to_symbol],
        "auto_compilation": _constraint_compilation_summary(model),
        "log_likelihood": _json_number(score),
        "viterbi_path": list(viterbi_states),
        "viterbi_score": _json_number(viterbi_score),
        "support_blocked": blocked_index is not None or score == float("-inf"),
        "blocked_index": blocked_index,
        "transition_mask": _matrix_tensor(model.state_names, model.state_names, model.transition_mask_),
        "emission_mask": _matrix_tensor(model.state_names, tuple(str(symbol) for symbol in model.id_to_symbol), model.emission_mask_),
        "transition_probs": _matrix_tensor(model.state_names, model.state_names, transition),
        "emission_probs": _matrix_tensor(model.state_names, tuple(str(symbol) for symbol in model.id_to_symbol), emission),
        "steps": steps,
    }


def _trace_dfa_for_labels(dfa, bundle, labels: list[int]) -> dict[str, Any]:
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


def _constraint_compilation_summary(model: DomiKnowSAwareHMM) -> dict[str, Any] | None:
    compilation = getattr(model, "constraint_hmm_compilation", None)
    if compilation is None:
        return None
    return {
        "method": "DomiKnowS constraints -> graph-discovered DFA -> productive DFA-edge HMM states",
        "explanation": (
            "Each DomiKnowS-aware HMM state is created from one productive DFA transition "
            "(state before, emitted symbol, state after). The emission mask ties that HMM "
            "state to its symbol, and the transition mask connects it to later edge states "
            "whose DFA endpoints line up."
        ),
        "state_count": len(compilation.states),
        "constraints": [],
        "states": [
            {
                "name": state.name,
                "dfa_from": str(state.dfa_from),
                "symbol": state.symbol,
                "symbol_label": int(state.symbol_label),
                "dfa_to": str(state.dfa_to),
                "explanation": (
                    f"Before emitting {state.symbol}, the DFA is in {state.dfa_from}; "
                    f"after emitting it, the DFA moves to {state.dfa_to}."
                ),
            }
            for state in compilation.states
        ],
    }


def _missing_dfa_step(index: int, symbol: str, dfa_trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "symbol": symbol,
        "from_state": dfa_trace["final_state"],
        "to_state": None,
        "allowed_symbols": [],
        "blocked": True,
        "reason": "DFA had already rejected an earlier symbol.",
    }


def _step_explanation(symbol: str, dfa_step: dict[str, Any], plain_step: dict[str, Any], graph_step: dict[str, Any]) -> str:
    if dfa_step.get("blocked") or graph_step.get("support_blocked"):
        return (
            f"The generator proposed {symbol}. The plain DiscreteHMM can still score the symbol, "
            "but the DFA and DomiKnowS-aware HMM show the constraint boundary: the graph-aware "
            "mask removes every legal hidden path for this step."
        )
    return (
        f"The generator proposed {symbol}. The DFA allows it, the DiscreteHMM updates a soft "
        "belief, and the DomiKnowS-aware HMM follows only graph-compatible hidden paths."
    )


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
