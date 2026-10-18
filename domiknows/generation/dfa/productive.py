"""Exact bounded reachability for lazy DFA-like runtime policies."""
from __future__ import annotations

from operator import index


class SemanticSearchLimitExceeded(RuntimeError):
    """A reachability query exhausted its computational budget, not its language."""


class BoundedReachability:
    """Reusable search over pure transition, acceptance and candidate callbacks.

The token budget counts additional transitions, including termination symbols.
Results are memoized per state and budget. A separate expansion cap raises an
explicit error; hitting it never returns an approximate feasibility answer.
"""

    def __init__(self, step, is_accepting, candidates, *, max_state_expansions=10000):
        self.step = step
        self.is_accepting = is_accepting
        self.candidates = candidates
        self.max_state_expansions = index(max_state_expansions)
        if self.max_state_expansions <= 0:
            raise ValueError("max_state_expansions must be positive")
        self._reachable = {}

    def _successors(self, state):
        for symbol in sorted(self.candidates(state), key=lambda item: (type(item).__name__, repr(item))):
            successor = self.step(state, symbol)
            if successor is not None:
                yield symbol, successor

    def _search(self, state, budget, work):
        root = (state, budget)
        stack = [root]
        iterators, pending = {}, {}
        while stack:
            key = stack[-1]
            if key in self._reachable:
                stack.pop()
                iterators.pop(key, None)
                continue
            current, remaining = key
            if key not in iterators:
                work[0] += 1
                if work[0] > self.max_state_expansions:
                    raise SemanticSearchLimitExceeded(
                        f"bounded semantic reachability exceeded {self.max_state_expansions} state expansions; "
                        "feasibility is unknown (no hard constraint was relaxed)"
                    )
                if remaining < 0:
                    self._reachable[key] = False
                    continue
                if self.is_accepting(current):
                    self._reachable[key] = True
                    continue
                if remaining == 0:
                    self._reachable[key] = False
                    continue
                iterators[key] = self._successors(current)
            child = pending.pop(key, None)
            if child is not None and self._reachable[child]:
                self._reachable[key] = True
                continue
            successor = next(iterators[key], None)
            if successor is None:
                self._reachable[key] = False
                continue
            child = (successor[1], remaining - 1)
            if child in self._reachable:
                if self._reachable[child]:
                    self._reachable[key] = True
            else:
                pending[key] = child
                stack.append(child)
        return self._reachable[root]

    def can_reach_accepting(self, state, max_steps):
        if max_steps is None:
            raise ValueError("semantic reachability requires a finite max_steps budget")
        if state is None:
            return False
        return self._search(state, index(max_steps), [0])

    def allowed_tokens(self, state, remaining_steps=None):
        if state is None:
            return set()
        if remaining_steps is None:
            return {symbol for symbol, _ in self._successors(state)}
        budget = index(remaining_steps)
        if budget < 0:
            return set()
        work = [0]
        # remaining_steps is the budget AFTER consuming the candidate, so do
        # not subtract it a second time. EOS can be legal with zero remaining.
        return {symbol for symbol, successor in self._successors(state)
                if self._search(successor, budget, work)}


class ProductiveDFA:
    """Add bounded completion checks outside a fully composed runtime policy.

This outer view includes restrictions introduced by wrappers above a semantic
monitor. step and acceptance retain the underlying policy's exact language.
"""

    def __init__(self, base_dfa, *, max_state_expansions=10000):
        self.base_dfa = base_dfa
        self.start_state = base_dfa.start_state
        self.alphabet = base_dfa.alphabet
        self._reachability = BoundedReachability(
            self.step, self.is_accepting, lambda state: base_dfa.allowed_tokens(state),
            max_state_expansions=max_state_expansions,
        )

    def step(self, state, symbol):
        return None if state is None else self.base_dfa.step(state, symbol)

    def is_accepting(self, state):
        return state is not None and self.base_dfa.is_accepting(state)

    def allowed_tokens(self, state, remaining_steps=None):
        return self._reachability.allowed_tokens(state, remaining_steps)

    def can_reach_accepting(self, state, max_steps=None):
        return self._reachability.can_reach_accepting(state, max_steps)

    def accepts(self, sequence):
        state = self.start_state
        for symbol in sequence:
            state = self.step(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)
