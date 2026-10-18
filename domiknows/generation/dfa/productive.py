"""Exact bounded reachability for lazy DFA-like runtime policies."""
from __future__ import annotations

from collections import deque
from operator import index


class SemanticSearchLimitExceeded(RuntimeError):
    """A reachability query exhausted its computational budget, not its language."""


class BoundedReachability:
    """Reusable search over pure transition, acceptance and candidate callbacks.

The token budget counts additional transitions, including termination symbols.
Proven witness lengths and exhausted budgets are reused across queries. A small
depth-first probe preserves cheap deep witnesses; breadth-first fallback avoids
starvation behind long branches and repeated cycle expansion. Both stages share
the query's expansion cap; exhaustion never returns an approximate answer.
"""

    def __init__(self, step, is_accepting, candidates, *, max_state_expansions=10000):
        self.step = step
        self.is_accepting = is_accepting
        self.candidates = candidates
        self.max_state_expansions = index(max_state_expansions)
        if self.max_state_expansions <= 0:
            raise ValueError("max_state_expansions must be positive")
        self._reachable = {}
        self._witness_lengths = {}
        self._failed_budgets = {}
        self._successor_cache = {}

    def _successors(self, state):
        cached = self._successor_cache.get(state)
        if cached is not None:
            yield from cached
            return
        successors = []
        for symbol in sorted(self.candidates(state), key=lambda item: (type(item).__name__, repr(item))):
            successor = self.step(state, symbol)
            if successor is not None:
                successors.append((symbol, successor))
                yield symbol, successor
        # Publish only a complete enumeration; an interrupted search must not
        # turn a partial successor list into a permanent token restriction.
        self._successor_cache[state] = tuple(successors)

    def _known(self, state, budget):
        if budget < 0:
            return False
        witness = self._witness_lengths.get(state)
        if witness is not None and witness <= budget:
            return True
        failed = self._failed_budgets.get(state)
        if failed is not None and failed >= budget:
            return False
        return self._reachable.get((state, budget))

    def _remember_witness(self, state, distance, parents):
        while True:
            self._witness_lengths[state] = min(self._witness_lengths.get(state, distance), distance)
            if state not in parents:
                break
            state = parents[state]
            distance += 1

    def _probe(self, state, budget, work):
        """Try a cheap depth-first witness before allocating a broad frontier."""
        limit = min(256, self.max_state_expansions // 4)
        stop = min(work[0] + limit, self.max_state_expansions)
        pending = [(state, budget, None)]
        best, parents = {}, {}
        while pending:
            if work[0] >= stop:
                return None
            current, remaining, parent = pending.pop()
            if best.get(current, -1) >= remaining:
                continue
            best[current] = remaining
            if parent is not None:
                parents[current] = parent
            known = self._known(current, remaining)
            if known is True:
                self._remember_witness(current, self._witness_lengths[current], parents)
                return True
            if known is False:
                continue
            work[0] += 1
            if self.is_accepting(current):
                self._remember_witness(current, 0, parents)
                return True
            if remaining:
                pending.extend((successor, remaining - 1, current)
                               for _, successor in reversed(tuple(self._successors(current))))
        for current, covered in best.items():
            self._failed_budgets[current] = max(self._failed_budgets.get(current, -1), covered)
        return False

    def _search(self, state, budget, work):
        known = self._known(state, budget)
        if known is not None:
            return known
        probed = self._probe(state, budget, work)
        if probed is not None:
            self._reachable[state, budget] = probed
            return probed
        # Breadth-first discovery visits a state with the greatest remaining
        # budget first. A later, longer route to that same state cannot expose
        # a new completion, so cycles never need to be unrolled to the horizon.
        depths = {state: 0}
        parents = {}
        queue = deque([state])
        while queue:
            current = queue.popleft()
            remaining = budget - depths[current]
            known = self._known(current, remaining)
            if known is True:
                self._remember_witness(current, self._witness_lengths[current], parents)
                self._reachable[state, budget] = True
                return True
            if known is False:
                continue
            work[0] += 1
            if work[0] > self.max_state_expansions:
                raise SemanticSearchLimitExceeded(
                    f"bounded semantic reachability exceeded {self.max_state_expansions} state expansions; "
                    "feasibility is unknown (no hard constraint was relaxed)"
                )
            if self.is_accepting(current):
                self._remember_witness(current, 0, parents)
                self._reachable[state, budget] = True
                return True
            if remaining == 0:
                continue
            for _, successor in self._successors(current):
                if successor not in depths:
                    depths[successor] = depths[current] + 1
                    parents[successor] = current
                    queue.append(successor)
        # Only a fully exhausted search proves infeasibility. Each discovered
        # state was covered for at least budget-depth additional transitions.
        for current, depth in depths.items():
            covered = budget - depth
            self._failed_budgets[current] = max(self._failed_budgets.get(current, -1), covered)
        self._reachable[state, budget] = False
        return False

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
