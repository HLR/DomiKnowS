"""Exact bounded reachability for lazy DFA-like runtime policies."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from operator import index
from types import MappingProxyType
from typing import Mapping


class SemanticSearchLimitExceeded(RuntimeError):
    """A reachability query exhausted its computational budget, not its language."""


@dataclass(frozen=True)
class DFACompletionBounds:
    """Shortest completion distances in an explicit finite relaxation.

    Filters may remove only transitions or accepting states forbidden by the
    concrete policy. Project a concrete state to this DFA before querying its
    bound. None proves no completion; a finite distance is a lower bound, never
    a proof that a concrete completion exists.
    """

    distances: Mapping

    def __call__(self, state):
        return self.distances.get(state)

    @classmethod
    def from_dfa(cls, dfa, *, transition_allowed=None, accepting_allowed=None):
        reverse = {}
        for (source, symbol), target in dfa.transitions.items():
            if transition_allowed is None or transition_allowed(source, symbol, target):
                reverse.setdefault(target, set()).add(source)
        distances = {state: 0 for state in dfa.accepting_states
                     if accepting_allowed is None or accepting_allowed(state)}
        pending = deque(distances)
        while pending:
            target = pending.popleft()
            for source in reverse.get(target, ()):
                if source not in distances:
                    distances[source] = distances[target] + 1
                    pending.append(source)
        return cls(MappingProxyType(distances))


class _Siblings:
    """Unvisited successors of one state on the probe's stack."""

    __slots__ = ("iterator", "remaining", "parent")

    def __init__(self, iterator, remaining, parent):
        self.iterator = iterator
        self.remaining = remaining
        self.parent = parent


class _LazySuccessors:
    """The successors of one state, stepped one candidate at a time, in the deterministic candidate order.

    Candidates are ordered once; each is stepped only when a search reaches it, and every outcome is kept, so a
    depth-first probe that finds a witness through the first few candidates never steps the rest. Iterators
    resume from the shared progress, which makes the partial list safe to keep: it is a prefix of the full
    enumeration, never a restriction of it.
    """

    __slots__ = ("_owner", "_state", "_symbols", "_next", "items", "symbol_of")

    def __init__(self, owner, state):
        self._owner = owner
        self._state = state
        self._symbols = None
        self._next = 0
        self.items = []
        self.symbol_of = {}  # successor -> the first symbol that produced it

    def __iter__(self):
        position = 0
        while True:
            if position < len(self.items):
                yield self.items[position]
                position += 1
                continue
            if self._symbols is None:
                self._symbols = sorted(self._owner.candidates(self._state), key=self._owner._order_key)
            if self._next >= len(self._symbols):
                return
            symbol = self._symbols[self._next]
            successor = self._owner.step(self._state, symbol)
            self._next += 1  # only after the step returned: an interrupted step is retried, never skipped
            if successor is not None:
                self.items.append((symbol, successor))
                self.symbol_of.setdefault(successor, symbol)


class BoundedReachability:
    """Reusable search over pure transition, acceptance and candidate callbacks.

The token budget counts additional transitions, including termination symbols.
Proven witness lengths and exhausted budgets are reused across queries. A small
depth-first probe preserves cheap deep witnesses; breadth-first fallback avoids
starvation behind long branches and repeated cycle expansion. Both stages share
the query's expansion cap; exhaustion never returns an approximate answer.
"""

    def __init__(self, step, is_accepting, candidates, *, max_state_expansions=10000,
                 completion_bound=None):
        self.step = step
        self.is_accepting = is_accepting
        self.candidates = candidates
        self.completion_bound = completion_bound
        self.max_state_expansions = index(max_state_expansions)
        if self.max_state_expansions <= 0:
            raise ValueError("max_state_expansions must be positive")
        self._reachable = {}
        self._witness_lengths = {}
        self._failed_budgets = {}
        self._successor_cache = {}
        # History heuristic: symbols that took part in proven witnesses are tried first (ties keep the
        # deterministic type-name/repr order). It only reorders a search whose answer is exact either way.
        self._symbol_score = {}
        self._symbol_text = {}

    def _order_key(self, symbol):
        text = self._symbol_text.get(symbol)
        if text is None:
            text = self._symbol_text[symbol] = (type(symbol).__name__, repr(symbol))
        return (-self._symbol_score.get(symbol, 0), *text)

    def _lazy(self, state):
        lazy = self._successor_cache.get(state)
        if lazy is None:
            lazy = self._successor_cache[state] = _LazySuccessors(self, state)
        return lazy

    def _successors(self, state):
        yield from self._lazy(state)

    def _known(self, state, budget):
        if budget < 0:
            return False
        if self.completion_bound is not None:
            distance = self.completion_bound(state)
            if distance is None:
                return False
            distance = index(distance)
            if distance < 0:
                raise ValueError("completion bounds must be nonnegative or None")
            if distance > budget:
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
            parent = parents[state]
            lazy = self._successor_cache.get(parent)
            symbol = None if lazy is None else lazy.symbol_of.get(state)
            if symbol is not None:
                self._symbol_score[symbol] = self._symbol_score.get(symbol, 0) + 1
            state = parent
            distance += 1

    def _completion_rank(self, successor):
        distance = self.completion_bound(successor[1])
        return (distance is None, 0 if distance is None else index(distance))

    def _probe(self, state, budget, work):
        """Try a cheap depth-first witness before allocating a broad frontier."""
        limit = min(256, self.max_state_expansions // 4)
        stop = min(work[0] + limit, self.max_state_expansions)
        pending = [(state, budget, None)]
        best, parents = {}, {}
        while pending:
            if work[0] >= stop:
                return None
            entry = pending.pop()
            if type(entry) is _Siblings:
                # The unvisited siblings of a state, stepped lazily: they cost nothing unless the search
                # backtracks to them, and are visited in exactly the order eager pushing would give.
                following = next(entry.iterator, None)
                if following is None:
                    continue
                pending.append(entry)
                current, remaining, parent = following[1], entry.remaining, entry.parent
            else:
                current, remaining, parent = entry
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
                if self.completion_bound is None:
                    pending.append(_Siblings(iter(self._lazy(current)), remaining - 1, current))
                else:
                    successors = sorted(tuple(self._successors(current)), key=self._completion_rank)
                    pending.extend((successor, remaining - 1, current)
                                   for _, successor in reversed(successors))
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
        # A successor is tested for acceptance (or a known verdict) when it is discovered, not when it reaches
        # the front of the queue: a witness one layer down is found without expanding the rest of this layer.
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
                if successor in depths:
                    continue
                depths[successor] = depths[current] + 1
                parents[successor] = current
                known = self._known(successor, remaining - 1)
                if known is True:
                    self._remember_witness(successor, self._witness_lengths[successor], parents)
                    self._reachable[state, budget] = True
                    return True
                if known is False:
                    continue
                if self.is_accepting(successor):
                    self._remember_witness(successor, 0, parents)
                    self._reachable[state, budget] = True
                    return True
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

    def __init__(self, base_dfa, *, max_state_expansions=10000, completion_bound=None):
        self.base_dfa = base_dfa
        self.start_state = base_dfa.start_state
        self.alphabet = base_dfa.alphabet
        self._reachability = BoundedReachability(
            self.step, self.is_accepting, lambda state: base_dfa.allowed_tokens(state),
            max_state_expansions=max_state_expansions,
            completion_bound=completion_bound,
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
