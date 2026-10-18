"""Reusable bounded-language audit against an application supplied oracle.

Enumerate syntax with step(), never semantic allowed_tokens(), so rejected
counterexamples cannot disappear from the comparison. Terminal symbols end a
candidate; repeated padding after termination is deliberately outside the audit.
"""
from dataclasses import dataclass


def finite_language_dfa(sequences):
    """Build syntax from supplied complete paths, independently of a monitor."""
    from domiknows.generation.dfa.core import DFA
    states, alphabet, transitions, accepting = {()}, set(), {}, set()
    for sequence in sequences:
        prefix = ()
        for symbol in sequence:
            successor = prefix + (symbol,)
            states.add(successor)
            alphabet.add(symbol)
            transitions[prefix, symbol] = successor
            prefix = successor
        accepting.add(prefix)
    return DFA(frozenset(states), frozenset(alphabet), transitions, (), frozenset(accepting))


@dataclass(frozen=True)
class LanguageAudit:
    candidates: int
    accepted: int
    rejected: int
    checked_prefixes: int
    checked_masks: int


def assert_bounded_equivalence(grammar, monitor, oracle, *, max_steps, terminal_symbols):
    """Compare complete paths and exact-budget next-token masks to an oracle."""
    assert max_steps >= 0
    terminals = frozenset(terminal_symbols)
    pending = [((), grammar.start_state)]
    prefixes = set()
    accepted = []
    candidates = 0
    while pending:
        sequence, state = pending.pop()
        if grammar.is_accepting(state):
            expected = bool(oracle(sequence))
            actual = bool(monitor.accepts(sequence))
            assert actual == expected, ("acceptance mismatch", sequence, expected, actual)
            candidates += 1
            if expected:
                accepted.append(sequence)
        if len(sequence) == max_steps or (sequence and sequence[-1] in terminals):
            continue
        prefixes.add(sequence)
        for symbol in grammar.alphabet:
            successor = grammar.step(state, symbol)
            if successor is not None:
                pending.append((sequence + (symbol,), successor))

    expected_next = {}
    for sequence in accepted:
        for index, symbol in enumerate(sequence):
            costs = expected_next.setdefault(sequence[:index], {})
            cost = len(sequence) - index - 1
            costs[symbol] = min(costs.get(symbol, cost), cost)
    checked = masks = 0
    for prefix in prefixes:
        state = monitor.start_state
        for symbol in prefix:
            state = monitor.step(state, symbol)
            if state is None:
                break
        if state is None:
            assert prefix not in expected_next, ("accepted completion has rejected prefix", prefix)
            continue
        costs = expected_next.get(prefix, {})
        for budget in range(max_steps - len(prefix)):
            expected = {symbol for symbol, cost in costs.items() if cost <= budget}
            actual = set(monitor.allowed_tokens(state, remaining_steps=budget))
            assert actual == expected, ("mask mismatch", prefix, budget, expected, actual)
            masks += 1
        checked += 1
    return LanguageAudit(candidates, len(accepted), candidates - len(accepted), checked, masks)
