"""Runtime DFA composition utilities.

These helpers keep graph-derived DFA compilation focused on stable global
constraints while cheap runtime overlays provide per-request constraints.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping

from .core import State, Symbol
from .vocabulary import TokenVocabulary


StepFn = Callable[[State, Symbol], State | None]


@dataclass(frozen=True)
class RuntimeDFAOverlay:
    """Finite runtime state machine layered over a base DFA.

    The overlay constrains the same emitted symbols as the base DFA but owns
    only its small local state. ``ComposedRuntimeDFA`` synchronizes one or more
    overlays with the base DFA and exposes the standard DFA surface.
    """

    states: frozenset[State]
    alphabet: frozenset[Symbol]
    start_state: State
    step_fn: StepFn
    accepting_states: frozenset[State]
    dead_states: frozenset[State] = frozenset()
    name: str = "runtime_overlay"

    def __post_init__(self):
        if self.start_state not in self.states:
            raise ValueError("overlay start_state must be in states")
        if not self.accepting_states <= self.states:
            raise ValueError("overlay accepting_states must be a subset of states")
        if not self.dead_states <= self.states:
            raise ValueError("overlay dead_states must be a subset of states")

    def step(self, state: State, symbol: Symbol) -> State | None:
        nxt = self.step_fn(state, symbol)
        if nxt is not None and nxt not in self.states:
            raise ValueError(f"overlay {self.name!r} returned unknown state {nxt!r}")
        return nxt

    def is_accepting(self, state: State) -> bool:
        return state in self.accepting_states

    def accepts(self, sequence: Iterable[Symbol]) -> bool:
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)


class ComposedRuntimeDFA:
    """A base DFA intersected with finite runtime overlays.

    States are tuples ``(base_state, overlay_state_0, ...)``. The object is
    intentionally DFA-like instead of subclassing ``DFA`` so it can wrap any
    object that exposes the common DFA methods.
    """

    def __init__(self, base_dfa, overlays: Iterable[RuntimeDFAOverlay] = ()):
        self.base_dfa = base_dfa
        self.overlays = tuple(overlays)
        self.alphabet = frozenset(base_dfa.alphabet)
        for overlay in self.overlays:
            extra = set(overlay.alphabet) - set(self.alphabet)
            if extra:
                raise ValueError(
                    f"overlay {overlay.name!r} uses symbols outside the base DFA alphabet: {sorted(extra)!r}"
                )

        overlay_state_sets = [overlay.states for overlay in self.overlays]
        if overlay_state_sets:
            self.states = frozenset(
                (base_state, *overlay_states)
                for base_state in base_dfa.states
                for overlay_states in product(*overlay_state_sets)
            )
        else:
            self.states = frozenset((base_state,) for base_state in base_dfa.states)
        self.start_state = (base_dfa.start_state, *(overlay.start_state for overlay in self.overlays))
        self.accepting_states = frozenset(
            state for state in self.states if self.is_accepting(state)
        )
        base_dead = set(getattr(base_dfa, "dead_states", frozenset()))
        self.dead_states = frozenset(
            state
            for state in self.states
            if state[0] in base_dead
            or any(part in overlay.dead_states for part, overlay in zip(state[1:], self.overlays))
        )
        self._allowed_cache: dict[tuple[State, int | None], frozenset[Symbol]] = {}
        self._reachability_cache: dict[tuple[State, int | None], bool] = {}

    def step(self, state: State, symbol: Symbol) -> State | None:
        parts = tuple(state)
        if len(parts) != len(self.overlays) + 1:
            return None
        next_base = self.base_dfa.step(parts[0], symbol)
        if next_base is None:
            return None
        next_parts = [next_base]
        for overlay, overlay_state in zip(self.overlays, parts[1:]):
            nxt = overlay.step(overlay_state, symbol)
            if nxt is None:
                return None
            next_parts.append(nxt)
        return tuple(next_parts)

    def is_accepting(self, state: State) -> bool:
        parts = tuple(state)
        if len(parts) != len(self.overlays) + 1:
            return False
        return self.base_dfa.is_accepting(parts[0]) and all(
            overlay.is_accepting(part) for overlay, part in zip(self.overlays, parts[1:])
        )

    def accepts(self, sequence: Iterable[Symbol]) -> bool:
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def can_reach_accepting(self, state: State, max_steps: int | None = None) -> bool:
        key = (state, None if max_steps is None else int(max_steps))
        cached = self._reachability_cache.get(key)
        if cached is not None:
            return cached
        if state in self.dead_states:
            self._reachability_cache[key] = False
            return False
        if max_steps is not None and int(max_steps) < 0:
            self._reachability_cache[key] = False
            return False
        if self.is_accepting(state):
            self._reachability_cache[key] = True
            return True

        queue = deque([(state, 0)])
        seen = {state}
        limit = None if max_steps is None else int(max_steps)
        while queue:
            current, depth = queue.popleft()
            if limit is not None and depth >= limit:
                continue
            for symbol in self.alphabet:
                nxt = self.step(current, symbol)
                if nxt is None or nxt in seen or nxt in self.dead_states:
                    continue
                if self.is_accepting(nxt):
                    self._reachability_cache[key] = True
                    return True
                seen.add(nxt)
                queue.append((nxt, depth + 1))
        self._reachability_cache[key] = False
        return False

    def allowed_tokens(self, state: State, remaining_steps: int | None = None) -> set[Symbol]:
        remaining_key = None if remaining_steps is None else int(remaining_steps)
        key = (state, remaining_key)
        cached = self._allowed_cache.get(key)
        if cached is not None:
            return set(cached)
        if remaining_key is not None and remaining_key <= 0:
            self._allowed_cache[key] = frozenset()
            return set()

        parts = tuple(state)
        if len(parts) != len(self.overlays) + 1:
            self._allowed_cache[key] = frozenset()
            return set()

        try:
            base_allowed = set(self.base_dfa.allowed_tokens(parts[0], remaining_steps=remaining_steps))
        except TypeError:
            base_allowed = set(self.base_dfa.allowed_tokens(parts[0]))
        base_allowed &= set(self.alphabet)

        allowed = set()
        for symbol in base_allowed:
            nxt = self.step(state, symbol)
            if nxt is None:
                continue
            if self.can_reach_accepting(nxt, None if remaining_key is None else remaining_key - 1):
                allowed.add(symbol)
        self._allowed_cache[key] = frozenset(allowed)
        return allowed

    def __getattr__(self, name):
        return getattr(self.base_dfa, name)


def compose_runtime_dfa(base_dfa, overlays: Iterable[RuntimeDFAOverlay]) -> ComposedRuntimeDFA:
    """Compose a base DFA with finite runtime overlays."""
    return ComposedRuntimeDFA(base_dfa, overlays)


def _label_for(value: int | str, vocabulary: TokenVocabulary | None) -> int:
    if isinstance(value, str):
        if vocabulary is None:
            raise ValueError("token names require a TokenVocabulary")
        return int(vocabulary.label_for_token(value))
    return int(value)


def _label_set(values: Iterable[int | str], vocabulary: TokenVocabulary | None) -> frozenset[int]:
    return frozenset(_label_for(value, vocabulary) for value in values)


def _overlay_alphabet(
    vocabulary: TokenVocabulary | None,
    alphabet: Iterable[int] | None,
    labels: Iterable[int],
) -> frozenset[int]:
    if vocabulary is not None:
        return frozenset(int(label) for label in vocabulary.alphabet)
    if alphabet is not None:
        return frozenset(int(label) for label in alphabet)
    return frozenset(int(label) for label in labels)


def token_class_sequence_overlay(
    first_tokens: Iterable[int | str],
    second_tokens: Iterable[int | str],
    stop_token: int | str,
    *,
    vocabulary: TokenVocabulary | None = None,
    alphabet: Iterable[int] | None = None,
    name: str = "token_class_sequence",
) -> RuntimeDFAOverlay:
    """Build an overlay for ``first second (first second)* stop*`` patterns.

    The state after a complete ``first second`` pair is accepting, so callers
    can use max-step stop policies without requiring the stop token.
    """
    first_labels = _label_set(first_tokens, vocabulary)
    second_labels = _label_set(second_tokens, vocabulary)
    stop_label = _label_for(stop_token, vocabulary)
    labels = set(first_labels) | set(second_labels) | {stop_label}
    overlay_alphabet = _overlay_alphabet(vocabulary, alphabet, labels)
    states = frozenset({"start", "want_second", "between", "stopped"})

    def step(state, symbol):
        symbol = int(symbol)
        if state == "start":
            return "want_second" if symbol in first_labels else None
        if state == "want_second":
            return "between" if symbol in second_labels else None
        if state == "between":
            if symbol in first_labels:
                return "want_second"
            if symbol == stop_label:
                return "stopped"
            return None
        if state == "stopped":
            return "stopped" if symbol == stop_label else None
        return None

    return RuntimeDFAOverlay(
        states=states,
        alphabet=overlay_alphabet,
        start_state="start",
        step_fn=step,
        accepting_states=frozenset({"between", "stopped"}),
        name=name,
    )


def token_set_sequence_overlay(
    tokens: Iterable[int | str],
    stop_token: int | str,
    *,
    vocabulary: TokenVocabulary | None = None,
    alphabet: Iterable[int] | None = None,
    allow_empty: bool = False,
    name: str = "token_set_sequence",
) -> RuntimeDFAOverlay:
    """Build an overlay for ``token* stop*`` with optional non-empty prefix."""
    token_labels = _label_set(tokens, vocabulary)
    stop_label = _label_for(stop_token, vocabulary)
    labels = set(token_labels) | {stop_label}
    overlay_alphabet = _overlay_alphabet(vocabulary, alphabet, labels)
    states = frozenset({"start", "emitted", "stopped"})

    def step(state, symbol):
        symbol = int(symbol)
        if state == "start":
            if symbol in token_labels:
                return "emitted"
            if allow_empty and symbol == stop_label:
                return "stopped"
            return None
        if state == "emitted":
            if symbol in token_labels:
                return "emitted"
            if symbol == stop_label:
                return "stopped"
            return None
        if state == "stopped":
            return "stopped" if symbol == stop_label else None
        return None

    accepting = {"emitted", "stopped"}
    if allow_empty:
        accepting.add("start")
    return RuntimeDFAOverlay(
        states=states,
        alphabet=overlay_alphabet,
        start_state="start",
        step_fn=step,
        accepting_states=frozenset(accepting),
        name=name,
    )


def pending_token_allowed_set_overlay(
    trigger_to_allowed: Mapping[int | str, Iterable[int | str]],
    *,
    vocabulary: TokenVocabulary | None = None,
    alphabet: Iterable[int] | None = None,
    name: str = "pending_token_allowed_set",
) -> RuntimeDFAOverlay:
    """Build an overlay for one-step ``trigger -> allowed next token`` rules."""
    normalized: dict[int, frozenset[int]] = {}
    labels = set()
    for trigger, allowed_values in trigger_to_allowed.items():
        trigger_label = _label_for(trigger, vocabulary)
        allowed = _label_set(allowed_values, vocabulary)
        if not allowed:
            continue
        normalized[trigger_label] = allowed
        labels.add(trigger_label)
        labels.update(allowed)

    overlay_alphabet = _overlay_alphabet(vocabulary, alphabet, labels)
    open_state = "__open__"
    states = frozenset({open_state, *sorted(normalized)})

    def step(state, symbol):
        symbol = int(symbol)
        if state != open_state and symbol not in normalized.get(int(state), frozenset()):
            return None
        return symbol if symbol in normalized else open_state

    return RuntimeDFAOverlay(
        states=states,
        alphabet=overlay_alphabet,
        start_state=open_state,
        step_fn=step,
        accepting_states=frozenset({open_state}),
        name=name,
    )


__all__ = [
    "ComposedRuntimeDFA",
    "RuntimeDFAOverlay",
    "compose_runtime_dfa",
    "pending_token_allowed_set_overlay",
    "token_class_sequence_overlay",
    "token_set_sequence_overlay",
]
