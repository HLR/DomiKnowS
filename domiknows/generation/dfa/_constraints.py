"""DFA builders for DomiKnowS-guided text generation.

Each public function in this module constructs a :class:`~.dfa.DFA` that
accepts exactly the label sequences satisfying one regular constraint over
the vocabulary's compact label alphabet.  DFAs can be intersected, unioned,
and complemented with :func:`~.core.product_dfa`, :func:`~.core.union_dfa`,
and :func:`~.core.complement_dfa`.

Available builders
------------------
- :func:`eos_closure_dfa` — once EOS is produced, all later tokens must also be EOS.
- :func:`max_non_eos_dfa` — caps the total number of non-EOS tokens.
- :func:`required_token_dfa` — requires a token to appear at least *n* times.
- :func:`forbidden_token_dfa` — forbids a specific token.
- :func:`ordered_tokens_dfa` — enforces an appearance ordering over a list of tokens.
- :func:`conditional_max_non_eos_dfa` — caps non-EOS count only when a trigger token appears.
- :func:`token_set_count_dfa` — counts tokens matching a finite token set or its complement.
- :func:`after_token_allowed_dfa` — after a trigger token, only an allowed token set may appear.
"""
from __future__ import annotations

from typing import Iterable

from .core import DFA
from .vocabulary import TokenVocabulary


def _complete_transitions(states, alphabet, fn):
    """Build a complete DFA transition table from a step function."""
    return {(state, symbol): fn(state, symbol) for state in states for symbol in alphabet}


def eos_closure_dfa(vocabulary: TokenVocabulary) -> DFA:
    """Build a DFA that rejects any non-EOS token following an EOS token.

    States:
    - ``"open"`` — no EOS seen yet; any token allowed.
    - ``"eos"``  — EOS has been generated; only EOS is allowed.
    - ``"dead"`` — a non-EOS token appeared after an EOS (constraint violated).
    """
    alphabet = frozenset(vocabulary.alphabet)
    eos = vocabulary.eos_label
    states = frozenset({"open", "eos", "dead"})

    def step(state, symbol):
        if state == "dead":
            return "dead"
        if state == "eos":
            return "eos" if symbol == eos else "dead"
        return "eos" if symbol == eos else "open"

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state="open",
        accepting_states=frozenset({"open", "eos"}),
        dead_states=frozenset({"dead"}),
    )


def max_non_eos_dfa(vocabulary: TokenVocabulary, max_count: int) -> DFA:
    """Build a DFA capping the number of non-EOS tokens at *max_count*.

    DFA states are integers ``0 … max_count+1`` where state *i* means *i*
    non-EOS tokens have been seen.  State ``max_count+1`` is a dead state.
    """
    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    alphabet = frozenset(vocabulary.alphabet)
    eos = vocabulary.eos_label
    dead = max_count + 1
    states = frozenset(range(dead + 1))

    def step(state, symbol):
        if state == dead:
            return dead
        if symbol == eos:
            return state
        return min(state + 1, dead)

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state=0,
        accepting_states=frozenset(range(max_count + 1)),
        dead_states=frozenset({dead}),
    )


def required_token_dfa(vocabulary: TokenVocabulary, token: str, min_count: int = 1) -> DFA:
    """Build a DFA that accepts only sequences containing *token* at least
    *min_count* times.

    DFA states are integers ``0 … min_count``; only state ``min_count`` is
    accepting.
    """
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    target = vocabulary.label_for_token(token)
    alphabet = frozenset(vocabulary.alphabet)
    states = frozenset(range(min_count + 1))

    def step(state, symbol):
        if symbol == target:
            return min(state + 1, min_count)
        return state

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state=0,
        accepting_states=frozenset({min_count}),
    )


def forbidden_token_dfa(vocabulary: TokenVocabulary, token: str) -> DFA:
    """Build a two-state DFA that rejects any sequence containing *token*."""
    target = vocabulary.label_for_token(token)
    alphabet = frozenset(vocabulary.alphabet)
    states = frozenset({"ok", "dead"})

    def step(state, symbol):
        if state == "dead" or symbol == target:
            return "dead"
        return "ok"

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state="ok",
        accepting_states=frozenset({"ok"}),
        dead_states=frozenset({"dead"}),
    )


def ordered_tokens_dfa(vocabulary: TokenVocabulary, tokens: Iterable[str]) -> DFA:
    """Build a DFA requiring *tokens* to appear in the given relative order.

    Each token must appear at some point after the previous one (not
    necessarily adjacent).  Only the final state (all matched) is accepting.
    """
    targets = tuple(vocabulary.label_for_token(token) for token in tokens)
    if not targets:
        raise ValueError("tokens must not be empty")
    alphabet = frozenset(vocabulary.alphabet)
    states = frozenset(range(len(targets) + 1))

    def step(state, symbol):
        if state < len(targets) and symbol == targets[state]:
            return state + 1
        return state

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state=0,
        accepting_states=frozenset({len(targets)}),
    )


def conditional_max_non_eos_dfa(
    vocabulary: TokenVocabulary,
    token: str,
    max_count: int,
) -> DFA:
    """Build a DFA that enforces ``at most max_count non-EOS tokens`` only
    when *token* appears in the sequence.

    Each state is a tuple ``(seen: bool, count: int)`` plus a single dead
    state ``("dead", max_count+1)``.
    """
    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    token_label = vocabulary.label_for_token(token)
    eos_label = vocabulary.eos_label
    alphabet = frozenset(vocabulary.alphabet)
    dead = ("dead", max_count + 1)
    states = {dead}
    for seen in (False, True):
        for count in range(max_count + 2):
            states.add((seen, count))

    def step(state, symbol):
        if state == dead:
            return dead
        seen, count = state
        seen = seen or symbol == token_label
        if symbol != eos_label:
            count += 1
        if seen and count > max_count:
            return dead
        return (seen, count)

    accepting = {state for state in states if state != dead}
    return DFA(
        states=frozenset(states),
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state=(False, 0),
        accepting_states=frozenset(accepting),
        dead_states=frozenset({dead}),
    )


def token_set_count_dfa(
    vocabulary: TokenVocabulary,
    tokens: Iterable[str],
    *,
    min_count: int | None = None,
    max_count: int | None = None,
    exact_count: int | None = None,
    negated: bool = False,
) -> DFA:
    """Build a counting DFA for a token-set predicate.

    The counted predicate is ``symbol in tokens`` by default, or
    ``symbol not in tokens`` when *negated* is ``True``.  Any combination of
    *min_count* and *max_count* may be supplied; *exact_count* sets both
    bounds to the same value.
    """
    tokens = tuple(tokens)
    if exact_count is not None:
        if min_count is not None or max_count is not None:
            raise ValueError("exact_count cannot be combined with min_count or max_count")
        min_count = exact_count
        max_count = exact_count
    if min_count is None and max_count is None:
        raise ValueError("at least one count bound is required")
    if min_count is not None and min_count < 0:
        raise ValueError("min_count must be non-negative")
    if max_count is not None and max_count < 0:
        raise ValueError("max_count must be non-negative")
    if min_count is not None and max_count is not None and min_count > max_count:
        raise ValueError("min_count cannot exceed max_count")

    token_labels = frozenset(vocabulary.label_for_token(token) for token in tokens)
    alphabet = frozenset(vocabulary.alphabet)
    floor_count = min_count or 0
    if max_count is None:
        max_state = floor_count
        dead = None
    else:
        max_state = max_count + 1
        dead = max_state
    states = frozenset(range(max_state + 1))

    def matches(symbol):
        in_set = symbol in token_labels
        return not in_set if negated else in_set

    def step(state, symbol):
        if dead is not None and state == dead:
            return dead
        if not matches(symbol):
            return state
        if max_count is None:
            return min(state + 1, floor_count)
        return min(state + 1, dead)

    accepting = {
        state
        for state in states
        if state >= floor_count and (max_count is None or state <= max_count)
    }
    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state=0,
        accepting_states=frozenset(accepting),
        dead_states=frozenset({dead}) if dead is not None else frozenset(),
    )


def after_token_allowed_dfa(
    vocabulary: TokenVocabulary,
    trigger_tokens: Iterable[str],
    allowed_tokens: Iterable[str],
) -> DFA:
    """Build a three-state DFA that, after any trigger token appears, only
    permits tokens from *allowed_tokens*.
    """
    triggers_tuple = tuple(trigger_tokens)
    allowed_tuple = tuple(allowed_tokens)
    if not triggers_tuple:
        raise ValueError("trigger_tokens must not be empty")
    if not allowed_tuple:
        raise ValueError("allowed_tokens must not be empty")

    triggers = frozenset(vocabulary.label_for_token(token) for token in triggers_tuple)
    allowed = frozenset(vocabulary.label_for_token(token) for token in allowed_tuple)
    alphabet = frozenset(vocabulary.alphabet)
    states = frozenset({"open", "after", "dead"})

    def step(state, symbol):
        if state == "dead":
            return "dead"
        if state == "after":
            return "after" if symbol in allowed else "dead"
        return "after" if symbol in triggers else "open"

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=_complete_transitions(states, alphabet, step),
        start_state="open",
        accepting_states=frozenset({"open", "after"}),
        dead_states=frozenset({"dead"}),
    )


def accept_all_dfa(vocabulary: TokenVocabulary) -> DFA:
    """Return a trivial single-state DFA that accepts every sequence."""
    alphabet = frozenset(vocabulary.alphabet)
    return DFA(
        states=frozenset({"ok"}),
        alphabet=alphabet,
        transitions={("ok", symbol): "ok" for symbol in alphabet},
        start_state="ok",
        accepting_states=frozenset({"ok"}),
    )


def empty_dfa(vocabulary: TokenVocabulary) -> DFA:
    """Return a trivial single-state DFA that rejects every sequence.

    Used by the LC normalizer when a constraint tree constant-folds to a
    contradiction (e.g. ``andL(A, notL(A))``).  The single state is marked as
    dead so reachability queries short-circuit immediately, and decoders can
    detect that no extension is feasible.
    """
    alphabet = frozenset(vocabulary.alphabet)
    return DFA(
        states=frozenset({"reject"}),
        alphabet=alphabet,
        transitions={("reject", symbol): "reject" for symbol in alphabet},
        start_state="reject",
        accepting_states=frozenset(),
        dead_states=frozenset({"reject"}),
    )
