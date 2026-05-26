"""Debug traces and DOT export for generation automata.

The helpers in this module are intentionally dependency-free.  They expose
plain dataclasses and Graphviz DOT text that can be used from tests, CLIs, or
the optional Flask viewer in :mod:`domiknows.generation.visual_server`.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .core import DFA, State, Symbol
from ..hmm import DiscreteHMM
from ..wfa import WeightedFiniteAutomaton, start_product_state, step_product_state

Labeler = Callable[[Any], str]


@dataclass(frozen=True)
class DFATraceStep:
    """One consumed symbol in a DFA debug trace."""

    index: int
    symbol: Symbol
    from_state: State
    to_state: State | None
    allowed_symbols: tuple[Symbol, ...]
    blocked: bool = False
    reason: str | None = None

    def to_dict(self, labeler: Labeler | None = None) -> dict[str, Any]:
        label = labeler or _label
        return {
            "index": self.index,
            "symbol": label(self.symbol),
            "from_state": label(self.from_state),
            "to_state": None if self.to_state is None else label(self.to_state),
            "allowed_symbols": [label(symbol) for symbol in self.allowed_symbols],
            "blocked": self.blocked,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DFATrace:
    """Full DFA trace for a concrete sequence."""

    sequence: tuple[Symbol, ...]
    start_state: State
    final_state: State
    steps: tuple[DFATraceStep, ...]
    accepted: bool
    blocked: bool = False
    rejection_reason: str | None = None

    @property
    def state_path(self) -> tuple[State, ...]:
        path = [self.start_state]
        for step in self.steps:
            if step.blocked or step.to_state is None:
                break
            path.append(step.to_state)
        return tuple(path)

    def to_dict(self, labeler: Labeler | None = None) -> dict[str, Any]:
        label = labeler or _label
        return {
            "sequence": [label(symbol) for symbol in self.sequence],
            "start_state": label(self.start_state),
            "final_state": label(self.final_state),
            "state_path": [label(state) for state in self.state_path],
            "accepted": self.accepted,
            "blocked": self.blocked,
            "rejection_reason": self.rejection_reason,
            "steps": [step.to_dict(label) for step in self.steps],
        }


@dataclass(frozen=True)
class ProductTraceStep:
    """One consumed symbol in a WFA x DFA product trace."""

    index: int
    symbol: Symbol
    from_dfa_state: State
    to_dfa_state: State | None
    from_wfa_state: tuple[float, ...]
    to_wfa_state: tuple[float, ...] | None
    from_score: float
    to_score: float | None
    allowed_symbols: tuple[Symbol, ...]
    blocked: bool = False
    reason: str | None = None

    def to_dict(self, labeler: Labeler | None = None) -> dict[str, Any]:
        label = labeler or _label
        return {
            "index": self.index,
            "symbol": label(self.symbol),
            "from_dfa_state": label(self.from_dfa_state),
            "to_dfa_state": None if self.to_dfa_state is None else label(self.to_dfa_state),
            "from_wfa_state": list(self.from_wfa_state),
            "to_wfa_state": None if self.to_wfa_state is None else list(self.to_wfa_state),
            "from_score": self.from_score,
            "to_score": self.to_score,
            "allowed_symbols": [label(symbol) for symbol in self.allowed_symbols],
            "blocked": self.blocked,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProductAutomatonTrace:
    """Full WFA x DFA trace for a concrete sequence."""

    sequence: tuple[Symbol, ...]
    start_dfa_state: State
    final_dfa_state: State
    start_score: float
    final_score: float
    steps: tuple[ProductTraceStep, ...]
    accepted: bool
    blocked: bool = False
    rejection_reason: str | None = None

    @property
    def dfa_state_path(self) -> tuple[State, ...]:
        path = [self.start_dfa_state]
        for step in self.steps:
            if step.blocked or step.to_dfa_state is None:
                break
            path.append(step.to_dfa_state)
        return tuple(path)

    @property
    def score_path(self) -> tuple[float, ...]:
        scores = [self.start_score]
        for step in self.steps:
            if step.blocked or step.to_score is None:
                break
            scores.append(step.to_score)
        return tuple(scores)

    def to_dict(self, labeler: Labeler | None = None) -> dict[str, Any]:
        label = labeler or _label
        return {
            "sequence": [label(symbol) for symbol in self.sequence],
            "start_dfa_state": label(self.start_dfa_state),
            "final_dfa_state": label(self.final_dfa_state),
            "dfa_state_path": [label(state) for state in self.dfa_state_path],
            "score_path": list(self.score_path),
            "start_score": self.start_score,
            "final_score": self.final_score,
            "accepted": self.accepted,
            "blocked": self.blocked,
            "rejection_reason": self.rejection_reason,
            "steps": [step.to_dict(label) for step in self.steps],
        }


@dataclass(frozen=True)
class ProductReachabilityGraph:
    """Small bounded reachable graph of WFA x DFA product states."""

    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    truncated: bool = False

    def to_dict(self, labeler: Labeler | None = None) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "truncated": self.truncated,
        }


def trace_discrete_hmm(
    hmm: DiscreteHMM,
    sequence: Sequence[object],
    *,
    support_threshold: float = 0.0,
) -> dict[str, Any]:
    """Trace real :class:`DiscreteHMM` factors for one observation sequence.

    The returned object is JSON-friendly and intended for educational/debug UI:
    it includes forward/backward factors from ``DiscreteHMM.forward_backward``,
    a Viterbi path, support masks, and per-step filtering diagnostics.  When a
    prefix has no positive support, the trace marks the blocked step and emits
    ``"-inf"`` for the displayed likelihood rather than non-standard JSON
    infinities.
    """

    if support_threshold < 0:
        raise ValueError("support_threshold must be non-negative")
    symbols = tuple(sequence)
    observations, lengths = hmm.encode([symbols])
    labels = observations[0, : int(lengths[0].item())].detach().cpu().tolist()
    transition = hmm.transition_with_potential()
    emission = hmm.emission_probs
    transition_support = transition > support_threshold
    emission_support = emission > support_threshold

    support_steps: list[dict[str, Any]] = []
    alpha_support = None
    blocked_index = None
    for index, label_id in enumerate(labels):
        prior = hmm.initial_probs if alpha_support is None else alpha_support @ transition
        emit = emission[:, label_id]
        unnormalized = prior * emit
        normalizer = float(unnormalized.sum().detach().cpu().item())
        if normalizer <= support_threshold and blocked_index is None:
            blocked_index = index
            belief = torch.zeros_like(unnormalized)
        else:
            belief = unnormalized / unnormalized.sum().clamp_min(torch.finfo(hmm.dtype).tiny)
        alpha_support = belief
        support_steps.append(
            {
                "index": index,
                "symbol": _label(hmm.symbols[label_id]),
                "prior": _named_tensor(hmm.state_names, prior),
                "emission_likelihood": _named_tensor(hmm.state_names, emit),
                "unnormalized_belief": _named_tensor(hmm.state_names, unnormalized),
                "filtering_belief": _named_tensor(hmm.state_names, belief),
                "normalizer": _json_number(normalizer),
                "support_blocked": blocked_index == index,
            }
        )

    factors = hmm.forward_backward(observations, lengths)
    viterbi_paths, viterbi_scores = hmm.viterbi(observations, lengths)
    viterbi_ids = viterbi_paths[0, : int(lengths[0].item())].detach().cpu().tolist()
    viterbi_names = [hmm.state_names[int(state)] for state in viterbi_ids]
    log_likelihood = float(factors.log_likelihood[0].detach().cpu().item())
    if blocked_index is not None:
        log_likelihood_value: float | str = "-inf"
    else:
        log_likelihood_value = _json_number(log_likelihood)

    steps = []
    for index, support in enumerate(support_steps):
        steps.append(
            {
                **support,
                "alpha": _named_tensor(hmm.state_names, factors.alpha[0, index]),
                "beta": _named_tensor(hmm.state_names, factors.beta[0, index]),
                "gamma": _named_tensor(hmm.state_names, factors.gamma[0, index]),
                "scale": _json_number(float(factors.scales[0, index].detach().cpu().item())),
                "viterbi_state": viterbi_names[index],
            }
        )

    xi = factors.xi[0].detach().cpu()
    return {
        "symbols": [_label(symbol) for symbol in symbols],
        "states": list(hmm.state_names),
        "log_likelihood": log_likelihood_value,
        "viterbi_path": viterbi_names,
        "viterbi_score": _json_number(float(viterbi_scores[0].detach().cpu().item())),
        "support_blocked": blocked_index is not None,
        "blocked_index": blocked_index,
        "transition_mask": _matrix_tensor(hmm.state_names, hmm.state_names, transition_support.to(dtype=hmm.dtype)),
        "emission_mask": _matrix_tensor(hmm.state_names, tuple(_label(symbol) for symbol in hmm.symbols), emission_support.to(dtype=hmm.dtype)),
        "transition_probs": _matrix_tensor(hmm.state_names, hmm.state_names, transition),
        "emission_probs": _matrix_tensor(hmm.state_names, tuple(_label(symbol) for symbol in hmm.symbols), emission),
        "xi": [
            _matrix_tensor(hmm.state_names, hmm.state_names, xi_step)
            for xi_step in xi
        ],
        "steps": steps,
    }


def trace_dfa(dfa: DFA, sequence: Sequence[Symbol], *, remaining_steps: int | None = None) -> DFATrace:
    """Trace DFA movement over *sequence*.

    A step is marked blocked when the symbol is not in
    ``dfa.allowed_tokens(current_state, remaining_steps=...)``.  This mirrors
    the constrained decoder's view of the automaton.
    """

    symbols = tuple(sequence)
    state = dfa.start_state
    steps: list[DFATraceStep] = []
    blocked = False
    reason = None

    for index, symbol in enumerate(symbols):
        budget = None if remaining_steps is None else max(remaining_steps - index, 0)
        allowed = tuple(_sorted_values(dfa.allowed_tokens(state, remaining_steps=budget)))
        next_state = dfa.step(state, symbol)
        if symbol not in allowed:
            reason = _blocked_reason(dfa, state, symbol, budget, next_state)
            steps.append(
                DFATraceStep(
                    index=index,
                    symbol=symbol,
                    from_state=state,
                    to_state=next_state,
                    allowed_symbols=allowed,
                    blocked=True,
                    reason=reason,
                )
            )
            blocked = True
            break
        steps.append(
            DFATraceStep(
                index=index,
                symbol=symbol,
                from_state=state,
                to_state=next_state,
                allowed_symbols=allowed,
            )
        )
        state = next_state

    accepted = (not blocked) and dfa.is_accepting(state)
    if not accepted and reason is None:
        reason = f"sequence ended in non-accepting state {_label(state)}"
    return DFATrace(
        sequence=symbols,
        start_state=dfa.start_state,
        final_state=state,
        steps=tuple(steps),
        accepted=accepted,
        blocked=blocked,
        rejection_reason=None if accepted else reason,
    )


def explain_dfa_rejection(dfa: DFA, sequence: Sequence[Symbol], *, remaining_steps: int | None = None) -> str | None:
    """Return a human-readable rejection reason, or ``None`` if accepted."""

    return trace_dfa(dfa, sequence, remaining_steps=remaining_steps).rejection_reason


def trace_product_automaton(
    wfa: WeightedFiniteAutomaton,
    dfa: DFA,
    sequence: Sequence[Symbol],
) -> ProductAutomatonTrace:
    """Trace synchronous WFA x DFA traversal for *sequence*."""

    symbols = tuple(sequence)
    state = start_product_state(wfa, dfa)
    steps: list[ProductTraceStep] = []
    blocked = False
    reason = None

    for index, symbol in enumerate(symbols):
        allowed = tuple(_sorted_values(set(wfa.symbols) & dfa.allowed_tokens(state.dfa_state)))
        before_wfa = _tensor_to_tuple(state.wfa_state)
        next_state = step_product_state(wfa, dfa, state, symbol) if symbol in wfa.symbols else None
        if symbol not in allowed or next_state is None:
            reason = _blocked_reason(dfa, state.dfa_state, symbol, None, dfa.step(state.dfa_state, symbol))
            steps.append(
                ProductTraceStep(
                    index=index,
                    symbol=symbol,
                    from_dfa_state=state.dfa_state,
                    to_dfa_state=None if next_state is None else next_state.dfa_state,
                    from_wfa_state=before_wfa,
                    to_wfa_state=None if next_state is None else _tensor_to_tuple(next_state.wfa_state),
                    from_score=float(state.score),
                    to_score=None if next_state is None else float(next_state.score),
                    allowed_symbols=allowed,
                    blocked=True,
                    reason=reason,
                )
            )
            blocked = True
            break
        steps.append(
            ProductTraceStep(
                index=index,
                symbol=symbol,
                from_dfa_state=state.dfa_state,
                to_dfa_state=next_state.dfa_state,
                from_wfa_state=before_wfa,
                to_wfa_state=_tensor_to_tuple(next_state.wfa_state),
                from_score=float(state.score),
                to_score=float(next_state.score),
                allowed_symbols=allowed,
            )
        )
        state = next_state

    accepted = (not blocked) and dfa.is_accepting(state.dfa_state)
    if not accepted and reason is None:
        reason = f"sequence ended in non-accepting DFA state {_label(state.dfa_state)}"
    return ProductAutomatonTrace(
        sequence=symbols,
        start_dfa_state=dfa.start_state,
        final_dfa_state=state.dfa_state,
        start_score=float(start_product_state(wfa, dfa).score),
        final_score=float(state.score),
        steps=tuple(steps),
        accepted=accepted,
        blocked=blocked,
        rejection_reason=None if accepted else reason,
    )


def dfa_to_dot(
    dfa: DFA,
    *,
    labeler: Labeler | None = None,
    highlight_path: Sequence[State] | DFATrace | None = None,
    max_states: int | None = None,
    title: str | None = None,
) -> str:
    """Export a DFA to Graphviz DOT text."""

    label = labeler or _label
    path_states = _highlight_states(highlight_path)
    state_limit = len(dfa.states) if max_states is None else max_states
    states = list(_sorted_values(dfa.states))[:state_limit]
    shown = set(states)
    truncated = len(dfa.states) > len(states)

    lines = ["digraph DFA {", "  rankdir=LR;", "  node [shape=circle];"]
    if title:
        lines.append(f"  label={_dot_quote(title)};")
        lines.append("  labelloc=t;")
    lines.append("  __start__ [shape=point,label=\"\"];")
    for state in states:
        attrs = {"label": label(state)}
        if state in dfa.accepting_states:
            attrs["shape"] = "doublecircle"
        if state in dfa.dead_states:
            attrs["style"] = "filled"
            attrs["fillcolor"] = "#f5d0d0"
        if state in path_states:
            attrs["color"] = "#2563eb"
            attrs["penwidth"] = "2"
        lines.append(f"  {_node_id(state)} [{_dot_attrs(attrs)}];")
    lines.append(f"  __start__ -> {_node_id(dfa.start_state)};")

    grouped: dict[tuple[State, State], list[Symbol]] = {}
    for (from_state, symbol), to_state in dfa.transitions.items():
        if from_state in shown and to_state in shown:
            grouped.setdefault((from_state, to_state), []).append(symbol)
    for (from_state, to_state), symbols in sorted(grouped.items(), key=lambda item: (_label(item[0][0]), _label(item[0][1]))):
        attrs = {"label": ", ".join(label(symbol) for symbol in _sorted_values(symbols))}
        if from_state in path_states and to_state in path_states:
            attrs["color"] = "#2563eb"
            attrs["penwidth"] = "2"
        lines.append(f"  {_node_id(from_state)} -> {_node_id(to_state)} [{_dot_attrs(attrs)}];")

    if truncated:
        lines.append(f"  __truncated__ [shape=note,label={_dot_quote('truncated: showing first ' + str(len(states)) + ' states')}];")
    lines.append("}")
    return "\n".join(lines)


def product_trace_to_dot(
    trace: ProductAutomatonTrace,
    *,
    labeler: Labeler | None = None,
    title: str | None = None,
) -> str:
    """Export a concrete WFA x DFA product trace path to DOT text."""

    label = labeler or _label
    path_states = trace.dfa_state_path
    scores = trace.score_path
    lines = ["digraph ProductTrace {", "  rankdir=LR;", "  node [shape=box];"]
    if title:
        lines.append(f"  label={_dot_quote(title)};")
        lines.append("  labelloc=t;")
    for idx, state in enumerate(path_states):
        score = scores[idx] if idx < len(scores) else trace.final_score
        attrs = {"label": f"{idx}: q={label(state)}\\nscore={score:.6g}"}
        if idx == len(path_states) - 1 and trace.accepted:
            attrs["shape"] = "doubleoctagon"
        lines.append(f"  p{idx} [{_dot_attrs(attrs)}];")
    for idx, step in enumerate(trace.steps):
        if step.blocked:
            lines.append(f"  blocked{idx} [shape=octagon,color=\"#dc2626\",label={_dot_quote(step.reason or 'blocked')}];")
            lines.append(f"  p{idx} -> blocked{idx} [color=\"#dc2626\",label={_dot_quote(label(step.symbol))}];")
            break
        lines.append(f"  p{idx} -> p{idx + 1} [label={_dot_quote(label(step.symbol))}];")
    lines.append("}")
    return "\n".join(lines)


def reachable_product_graph(
    wfa: WeightedFiniteAutomaton,
    dfa: DFA,
    *,
    max_depth: int = 3,
    max_states: int = 100,
    labeler: Labeler | None = None,
) -> ProductReachabilityGraph:
    """Explore a bounded WFA x DFA product graph for small debug views."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if max_states < 1:
        raise ValueError("max_states must be at least 1")
    label = labeler or _label
    start = start_product_state(wfa, dfa)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    queue = deque([((), start, 0)])
    seen = {start.dfa_state}
    truncated = False

    while queue:
        prefix, state, depth = queue.popleft()
        node_id = _product_node_id(prefix)
        nodes.append(
            {
                "id": node_id,
                "prefix": [label(symbol) for symbol in prefix],
                "dfa_state": label(state.dfa_state),
                "score": float(state.score),
                "accepting": dfa.is_accepting(state.dfa_state),
            }
        )
        if len(nodes) >= max_states:
            truncated = bool(queue) or depth < max_depth
            break
        if depth >= max_depth:
            continue
        for symbol in _sorted_values(set(wfa.symbols) & dfa.allowed_tokens(state.dfa_state)):
            nxt = step_product_state(wfa, dfa, state, symbol)
            if nxt is None:
                continue
            next_prefix = tuple(prefix) + (symbol,)
            edges.append(
                {
                    "from": node_id,
                    "to": _product_node_id(next_prefix),
                    "symbol": label(symbol),
                    "score": float(nxt.score),
                }
            )
            if nxt.dfa_state not in seen or len(next_prefix) <= max_depth:
                seen.add(nxt.dfa_state)
                queue.append((next_prefix, nxt, depth + 1))
    return ProductReachabilityGraph(tuple(nodes), tuple(edges), truncated=truncated)


def _blocked_reason(dfa: DFA, state: State, symbol: Symbol, remaining_steps: int | None, next_state: State | None) -> str:
    if symbol not in dfa.alphabet:
        return f"symbol {_label(symbol)} is not in the DFA alphabet"
    if next_state is None:
        return f"no transition from {_label(state)} on {_label(symbol)}"
    if next_state in dfa.dead_states:
        return f"{_label(symbol)} reaches dead state {_label(next_state)}"
    if remaining_steps is not None:
        return f"{_label(symbol)} cannot reach acceptance within remaining step budget {remaining_steps}"
    return f"{_label(symbol)} is not allowed from {_label(state)}"


def _highlight_states(highlight_path: Sequence[State] | DFATrace | None) -> set[State]:
    if highlight_path is None:
        return set()
    if isinstance(highlight_path, DFATrace):
        return set(highlight_path.state_path)
    return set(highlight_path)


def _tensor_to_tuple(value: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu().reshape(-1).tolist())


def _json_number(value: float) -> float | str:
    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    if value != value:
        return "nan"
    return float(value)


def _named_tensor(names: Sequence[str], values: torch.Tensor) -> dict[str, float | str]:
    flat = values.detach().cpu().reshape(-1).tolist()
    return {str(name): _json_number(round(float(value), 6)) for name, value in zip(names, flat)}


def _matrix_tensor(rows: Sequence[str], columns: Sequence[str], values: torch.Tensor) -> dict[str, dict[str, float | str]]:
    cpu = values.detach().cpu()
    return {
        str(row): {
            str(column): _json_number(round(float(cpu[row_index, col_index].item()), 6))
            for col_index, column in enumerate(columns)
        }
        for row_index, row in enumerate(rows)
    }


def _sorted_values(values):
    return sorted(values, key=_label)


def _label(value: Any) -> str:
    return str(value)


def _node_id(value: Any) -> str:
    safe = repr(value).encode("utf8").hex()
    return f"n_{safe}"


def _product_node_id(prefix: Sequence[Symbol]) -> str:
    if not prefix:
        return "p_start"
    return "p_" + "_".join(repr(symbol).encode("utf8").hex() for symbol in prefix)


def _dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _dot_attrs(attrs: dict[str, str]) -> str:
    return ",".join(f"{key}={_dot_quote(value)}" for key, value in attrs.items())
