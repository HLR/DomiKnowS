"""Compile generation DFA constraints into graph-HMM hidden support.

The regular generation fragment already has an exact DFA compiler.  This
module reuses that DFA as the memory template for a DomiKnowS-aware HMM:
hidden states become productive DFA edges, emissions are tied to the edge
symbol, and HMM transitions connect edges whose DFA endpoints line up.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Sequence

import torch

from domiknows.generation.graph_discovery import constraints_to_dfa_from_graph, discover_generation_constraints

from .graph_hmm import DomiKnowSAwareHMM


@dataclass(frozen=True)
class ConstraintHMMState:
    """One automatically generated HMM state backed by a DFA transition."""

    name: str
    dfa_from: Any
    symbol: str
    symbol_label: int
    dfa_to: Any


@dataclass(frozen=True)
class ConstraintHMMCompilation:
    """Artifacts produced by compiling a generation DFA into HMM support."""

    dfa: Any
    constraints: tuple[Any, ...]
    states: tuple[ConstraintHMMState, ...]
    transition_mask: torch.Tensor
    emission_mask: torch.Tensor
    initial: torch.Tensor
    transition: torch.Tensor
    emission: torch.Tensor
    symbols: tuple[str, ...]


def compile_generation_constraints_to_hmm_support(
    graph,
    bundle,
    *,
    symbols: Sequence[str] | None = None,
    eos_token: str | None = None,
    include_other: bool = False,
    state_name_fn: Callable[[Any, str, Any], str] | None = None,
    on_unsupported: str = "error",
    device=None,
    dtype: torch.dtype = torch.float64,
) -> ConstraintHMMCompilation:
    """Compile graph-discovered DFA support into HMM masks and parameters.

    The construction is exact for the DFA language under the usual generation
    EOS convention: if ``eos_token`` is supplied, EOS-emitting edge states are
    only created when the DFA successor is accepting, and they have no outgoing
    HMM transitions.
    """

    constraints = tuple(discover_generation_constraints(graph, bundle, on_unsupported=on_unsupported))
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported=on_unsupported)
    vocab = bundle.vocabulary
    token_symbols = tuple(symbols or vocab.tokens)
    if include_other and vocab.other_token not in token_symbols:
        token_symbols = token_symbols + (vocab.other_token,)
    if eos_token is None:
        eos_token = getattr(vocab, "eos_token", None)

    label_by_symbol = {symbol: vocab.label_for_token(symbol) for symbol in token_symbols}
    component_specs = _component_specs_from_constraints(constraints)
    if state_name_fn is None:
        state_name_fn = lambda q0, symbol, q1: _edge_state_name(q0, symbol, q1, component_specs)

    states: list[ConstraintHMMState] = []
    seen_names: set[str] = set()
    for dfa_state in sorted(dfa.states, key=str):
        if dfa_state in dfa.dead_states or not dfa.can_reach_accepting(dfa_state):
            continue
        for symbol in token_symbols:
            label = label_by_symbol[symbol]
            next_state = dfa.step(dfa_state, label)
            if next_state is None or next_state in dfa.dead_states:
                continue
            if not dfa.can_reach_accepting(next_state):
                continue
            if eos_token is not None and symbol == eos_token and not dfa.is_accepting(next_state):
                continue
            name = _unique_name(str(state_name_fn(dfa_state, symbol, next_state)), seen_names)
            states.append(
                ConstraintHMMState(
                    name=name,
                    dfa_from=dfa_state,
                    symbol=str(symbol),
                    symbol_label=label,
                    dfa_to=next_state,
                )
            )

    if not states:
        raise ValueError("generation constraints produced no productive HMM edge states")

    state_count = len(states)
    symbol_count = len(token_symbols)
    symbol_index = {symbol: index for index, symbol in enumerate(token_symbols)}
    transition_mask = torch.zeros((state_count, state_count), dtype=dtype, device=device)
    emission_mask = torch.zeros((state_count, symbol_count), dtype=dtype, device=device)
    initial = torch.zeros(state_count, dtype=dtype, device=device)

    for index, state in enumerate(states):
        if state.dfa_from == dfa.start_state:
            initial[index] = 1.0
        emission_mask[index, symbol_index[state.symbol]] = 1.0

    for left_index, left in enumerate(states):
        if eos_token is not None and left.symbol == eos_token:
            continue
        for right_index, right in enumerate(states):
            if left.dfa_to == right.dfa_from:
                transition_mask[left_index, right_index] = 1.0

    transition = transition_mask.clone()
    emission = emission_mask.clone()
    return ConstraintHMMCompilation(
        dfa=dfa,
        constraints=constraints,
        states=tuple(states),
        transition_mask=transition_mask,
        emission_mask=emission_mask,
        initial=initial,
        transition=transition,
        emission=emission,
        symbols=token_symbols,
    )


def domiknows_hmm_from_generation_constraints(
    graph,
    bundle,
    *,
    symbols: Sequence[str] | None = None,
    eos_token: str | None = None,
    include_other: bool = False,
    state_name_fn: Callable[[Any, str, Any], str] | None = None,
    on_unsupported: str = "error",
    smoothing: float = 1e-6,
    device=None,
    dtype: torch.dtype = torch.float64,
    random_seed: int = 0,
) -> DomiKnowSAwareHMM:
    """Build an initialized :class:`DomiKnowSAwareHMM` from generation constraints."""

    compilation = compile_generation_constraints_to_hmm_support(
        graph,
        bundle,
        symbols=symbols,
        eos_token=eos_token,
        include_other=include_other,
        state_name_fn=state_name_fn,
        on_unsupported=on_unsupported,
        device=device,
        dtype=dtype,
    )
    model = DomiKnowSAwareHMM(
        graph=graph,
        n_hidden_states=len(compilation.states),
        transition_mask=compilation.transition_mask,
        emission_mask=compilation.emission_mask,
        symbols=compilation.symbols,
        state_names=tuple(state.name for state in compilation.states),
        smoothing=smoothing,
        device=device,
        dtype=dtype,
        random_seed=random_seed,
    )
    init_sequence = _initialization_sequence(compilation)
    model.fit(
        [init_sequence],
        max_iter=0,
        init={
            "initial": compilation.initial,
            "transition": compilation.transition,
            "emission": compilation.emission,
        },
    )
    model.constraint_hmm_compilation = compilation
    model.constraint_report.add_applied(
        f"compiled {len(compilation.states)} DFA edge state(s) from generation constraints"
    )
    return model


def _initialization_sequence(compilation: ConstraintHMMCompilation) -> tuple[str, ...]:
    for symbol_index, symbol in enumerate(compilation.symbols):
        if (compilation.emission_mask[:, symbol_index] > 0).any():
            return (symbol,)
    raise ValueError("compiled HMM has no globally emittable symbol")


def _component_specs_from_constraints(constraints: Sequence[Any]) -> tuple[dict[str, str], ...]:
    specs: list[dict[str, str]] = []
    for constraint in constraints:
        name = str(getattr(constraint, "name", constraint))
        token = _quoted_token(name)
        lower = name.lower()
        if "at least" in lower:
            specs.append({"kind": "at_least", "token": token or "token"})
        elif "at most" in lower:
            specs.append({"kind": "at_most", "token": token or "token"})
        else:
            specs.append({"kind": "state", "token": token or f"q{len(specs)}"})
    return tuple(specs)


def _quoted_token(text: str) -> str | None:
    match = re.search(r"'([^']+)'", text)
    if match:
        return _safe_part(match.group(1))
    return None


def _edge_state_name(dfa_from: Any, symbol: str, dfa_to: Any, component_specs: tuple[dict[str, str], ...]) -> str:
    source = _dfa_state_name(dfa_from, component_specs)
    target = _dfa_state_name(dfa_to, component_specs)
    symbol_part = _safe_part(symbol)
    if source == target:
        return f"{source}__emit_{symbol_part}"
    return f"{source}__emit_{symbol_part}__to_{target}"


def _dfa_state_name(state: Any, component_specs: tuple[dict[str, str], ...]) -> str:
    if isinstance(state, tuple) and component_specs and len(state) == len(component_specs):
        parts = []
        for index, spec in sorted(enumerate(component_specs), key=lambda item: 0 if item[1]["kind"] == "at_least" else 1):
            value = state[index]
            token = _safe_part(spec["token"])
            if spec["kind"] == "at_least":
                parts.append(f"seen_{token}" if int(value) >= 1 else f"need_{token}")
            elif spec["kind"] == "at_most":
                if int(value) <= 0:
                    parts.append(f"no_{token}")
                elif int(value) == 1:
                    parts.append(f"seen_{token}")
                else:
                    parts.append(f"too_many_{token}")
            else:
                parts.append(f"{token}_{_safe_part(value)}")
        return "_".join(parts)
    return "q_" + _safe_part(state)


def _safe_part(value: Any) -> str:
    text = str(value)
    text = text.replace("<", "").replace(">", "")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text or "state"


def _unique_name(base: str, seen: set[str]) -> str:
    name = base
    suffix = 2
    while name in seen:
        name = f"{base}_{suffix}"
        suffix += 1
    seen.add(name)
    return name
