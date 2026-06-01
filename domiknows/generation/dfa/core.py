"""Deterministic Finite Automaton (DFA) primitives.

Provides:
- ``DFA``: immutable DFA with convenience methods for stepping, acceptance
  testing, reachability queries, and allowed-token enumeration.
- ``product_dfa``: intersect an arbitrary number of DFAs over a shared
  alphabet via the standard product-construction algorithm.
- ``union_dfa``: take the union of an arbitrary number of DFAs over a shared
  alphabet via a product construction with accepting-if-any semantics.
- ``complement_dfa``: complete a DFA and flip accepting states.
- ``minimize_dfa``: collapse equivalent states via partition refinement.

Type aliases ``State`` and ``Symbol`` are both ``Hashable``; any hashable
Python object can serve as a state identifier or an alphabet symbol.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Iterable

# Type aliases — any hashable value is valid as a state or symbol.
State = Hashable
Symbol = Hashable


@dataclass(frozen=True)
class DFA:
    """An immutable deterministic finite automaton used for token constraints.

    Attributes:
        states: Complete set of states in the automaton.
        alphabet: Set of symbols the automaton can consume.
        transitions: Mapping of ``(state, symbol) -> next_state``.  Pairs that
            are absent represent implicit rejection (no transition defined).
        start_state: The state the automaton begins in; must be in ``states``.
        accepting_states: Subset of ``states`` that constitute accepting
            (final) states.
        dead_states: Optional subset of ``states`` known to be *sink* states
            from which no accepting state is reachable.  Used as a fast-reject
            hint in reachability queries.  Defaults to the empty set.
    """

    states: frozenset[State]
    alphabet: frozenset[Symbol]
    transitions: dict[tuple[State, Symbol], State]
    start_state: State
    accepting_states: frozenset[State]
    dead_states: frozenset[State] = frozenset()

    def __post_init__(self):
        """Validate internal consistency of the DFA on construction."""
        if self.start_state not in self.states:
            raise ValueError("start_state must be in states")
        if not self.accepting_states <= self.states:
            raise ValueError("accepting_states must be a subset of states")
        if not self.dead_states <= self.states:
            raise ValueError("dead_states must be a subset of states")

    def step(self, state: State, symbol: Symbol) -> State | None:
        """Return the successor state for (*state*, *symbol*), or ``None`` if undefined."""
        return self.transitions.get((state, symbol))

    def is_accepting(self, state: State) -> bool:
        """Return ``True`` if *state* is an accepting (final) state."""
        return state in self.accepting_states

    def accepts(self, sequence: Iterable[Symbol]) -> bool:
        """Return ``True`` if the DFA accepts the full *sequence*.

        Runs from ``start_state``, consuming symbols one by one.  Returns
        ``False`` immediately if any transition is undefined.
        """
        state = self.start_state
        for symbol in sequence:
            next_state = self.step(state, symbol)
            if next_state is None:
                # Undefined transition — implicit rejection.
                return False
            state = next_state
        return self.is_accepting(state)

    def can_reach_accepting(self, state: State, max_steps: int | None = None) -> bool:
        """Return ``True`` if an accepting state is reachable from *state*.

        Uses a breadth-first search (BFS) over the transition graph.  States
        in ``dead_states`` are treated as sinks and never expanded.

        Args:
            state: The starting state for the reachability query.
            max_steps: Maximum number of transitions to follow.  ``None``
                means unbounded search.

        Returns:
            ``True`` if there exists a path of length ≤ *max_steps* (or any
            length when unbounded) from *state* to some accepting state.
        """
        # Fast-reject: known dead states and negative budgets.
        if state in self.dead_states:
            return False
        if max_steps is not None and max_steps < 0:
            return False
        # Fast-accept: *state* itself is accepting.
        if self.is_accepting(state):
            return True
        # Breadth-first search (BFS) with optional depth limit.
        queue = deque([(state, 0)])
        seen = {state}
        while queue:
            current, depth = queue.popleft()
            if max_steps is not None and depth >= max_steps:
                continue
            for symbol in self.alphabet:
                nxt = self.step(current, symbol)
                if nxt is None or nxt in seen or nxt in self.dead_states:
                    continue
                if self.is_accepting(nxt):
                    return True
                seen.add(nxt)
                queue.append((nxt, depth + 1))
        return False

    def allowed_tokens(self, state: State, remaining_steps: int | None = None) -> set[Symbol]:
        """Return the set of symbols that keep the DFA on a productive path.

        A symbol is *allowed* when consuming it leads to a successor state
        from which an accepting state is still reachable within the remaining
        step budget.

        Args:
            state: The current DFA state.
            remaining_steps: How many further transitions are permitted after
                the one being considered.  ``None`` means no limit.

        Returns:
            A subset of ``self.alphabet`` containing every symbol worth
            emitting from *state* given the step budget.
        """
        allowed = set()
        for symbol in self.alphabet:
            nxt = self.step(state, symbol)
            if nxt is None:
                # No transition defined for this symbol — skip it.
                continue
            if remaining_steps is None:
                # Unbounded: any non-dead successor is acceptable.
                if nxt not in self.dead_states:
                    allowed.add(symbol)
            elif self.can_reach_accepting(nxt, remaining_steps - 1):
                # Bounded: only include the symbol if we can still reach
                # an accepting state within the remaining budget.
                allowed.add(symbol)
        return allowed


def product_dfa(dfas: Iterable[DFA]) -> DFA:
    """Intersect multiple DFAs by taking their synchronous product.

    Constructs a new DFA whose states are tuples of component states, one
    per input DFA.  A product state is:

    - **accepting** when *all* component states are accepting (intersection
      semantics).
    - **dead** when *any* component state is in its DFA's ``dead_states``
      set.

    Transitions that are undefined in any component DFA simply have no
    corresponding entry in the product transition table (implicit rejection).

    The construction is on-demand (BFS from the start tuple) so only
    reachable states are materialised.

    Args:
        dfas: One or more :class:`DFA` instances, all sharing the same
            alphabet.

    Returns:
        A new :class:`DFA` representing the intersection of all inputs.

    Raises:
        ValueError: If *dfas* is empty or the alphabets differ.
    """
    dfa_list = list(dfas)
    if not dfa_list:
        raise ValueError("at least one DFA is required")

    alphabet = dfa_list[0].alphabet
    if any(dfa.alphabet != alphabet for dfa in dfa_list):
        raise ValueError("all DFAs must have the same alphabet")

    # Each product state is a tuple of component states (one per input DFA).
    start = tuple(dfa.start_state for dfa in dfa_list)
    states = {start}
    transitions: dict[tuple[State, Symbol], State] = {}
    accepting = set()
    dead = set()
    queue = deque([start])

    while queue:
        state = queue.popleft()

        # A product state accepts iff every component is accepting.
        if all(dfa.is_accepting(part) for dfa, part in zip(dfa_list, state)):
            accepting.add(state)

        # A product state is dead if any component is dead.
        if any(part in dfa.dead_states for dfa, part in zip(dfa_list, state)):
            dead.add(state)

        for symbol in alphabet:
            next_parts = []
            for dfa, part in zip(dfa_list, state):
                nxt = dfa.step(part, symbol)
                if nxt is None:
                    # At least one component has no transition — skip symbol.
                    break
                next_parts.append(nxt)
            else:
                # All components have a defined transition for this symbol.
                nxt_state = tuple(next_parts)
                transitions[(state, symbol)] = nxt_state
                if nxt_state not in states:
                    states.add(nxt_state)
                    queue.append(nxt_state)

    return DFA(
        states=frozenset(states),
        alphabet=alphabet,
        transitions=transitions,
        start_state=start,
        accepting_states=frozenset(accepting),
        dead_states=frozenset(dead),
    )


def union_dfa(dfas: Iterable[DFA]) -> DFA:
    """Union multiple DFAs by taking their synchronous product.

    Constructs a new DFA whose states are tuples of component states, one per
    input DFA.  A component state may be ``None`` after that branch has no
    transition for the consumed prefix.  A product state is:

    - **accepting** when *any* live component state is accepting (union
      semantics).
    - **dead** when every component is either ``None`` or in its DFA's
      ``dead_states`` set.

    Args:
        dfas: One or more :class:`DFA` instances, all sharing the same
            alphabet.

    Returns:
        A new :class:`DFA` representing the union of all inputs.

    Raises:
        ValueError: If *dfas* is empty or the alphabets differ.
    """
    dfa_list = list(dfas)
    if not dfa_list:
        raise ValueError("at least one DFA is required")

    alphabet = dfa_list[0].alphabet
    if any(dfa.alphabet != alphabet for dfa in dfa_list):
        raise ValueError("all DFAs must have the same alphabet")

    start = tuple(dfa.start_state for dfa in dfa_list)
    states = {start}
    transitions: dict[tuple[State, Symbol], State] = {}
    accepting = set()
    dead = set()
    queue = deque([start])

    while queue:
        state = queue.popleft()

        if any(part is not None and dfa.is_accepting(part) for dfa, part in zip(dfa_list, state)):
            accepting.add(state)

        if all(part is None or part in dfa.dead_states for dfa, part in zip(dfa_list, state)):
            dead.add(state)

        for symbol in alphabet:
            next_parts = []
            for dfa, part in zip(dfa_list, state):
                if part is None:
                    next_parts.append(None)
                else:
                    next_parts.append(dfa.step(part, symbol))
            if all(part is None for part in next_parts):
                continue
            nxt_state = tuple(next_parts)
            transitions[(state, symbol)] = nxt_state
            if nxt_state not in states:
                states.add(nxt_state)
                queue.append(nxt_state)

    return DFA(
        states=frozenset(states),
        alphabet=alphabet,
        transitions=transitions,
        start_state=start,
        accepting_states=frozenset(accepting),
        dead_states=frozenset(dead),
    )


def minimize_dfa(dfa: DFA) -> DFA:
    """Return an equivalent DFA with the minimum number of states.

    Uses Hopcroft-style partition refinement:

    "Hopcroft-style" refers to the classic DFA minimization approach
    introduced by John E. Hopcroft (1971), where states are repeatedly split
    into equivalence classes based on how transitions lead into already-known
    classes.  The original algorithm is known for an efficient
    :math:`O(n \log n)` refinement strategy (for fixed alphabet size), and
    this implementation follows that same partition-refinement idea in a
    straightforward, readability-first form.

    1. Complete the input by funnelling missing transitions to a fresh
       rejecting sink (same trick :func:`complement_dfa` uses).
    2. Discard states that are unreachable from ``start_state``.
    3. Partition states by ``(accepting, dead)`` and refine until the
       transition signature on the alphabet stabilises.
    4. Reindex equivalence classes to integers 0..k for a compact output.

    The returned DFA preserves language exactly, including the ``dead_states``
    set: any equivalence class whose members were all dead in the input stays
    dead.

    Args:
        dfa: Any DFA over a finite alphabet.

    Returns:
        A new :class:`DFA` whose ``states`` are integers and whose transition
        table is total over its alphabet (every reachable equivalence class
        has a defined outgoing transition for every symbol).
    """
    alphabet = dfa.alphabet
    sink = ("__minimize_sink__",)
    while sink in dfa.states:
        sink = (sink,)

    # Step 1: complete the transition table with the sink for any missing edge.
    all_states = set(dfa.states)
    transitions: dict[tuple[State, Symbol], State] = dict(dfa.transitions)
    sink_used = False
    for state in dfa.states:
        for symbol in alphabet:
            if (state, symbol) not in transitions:
                transitions[(state, symbol)] = sink
                sink_used = True
    if sink_used:
        all_states.add(sink)
        for symbol in alphabet:
            transitions[(sink, symbol)] = sink

    # Step 2: breadth-first search (BFS) from start_state and keep only reachable states.
    reachable: set[State] = {dfa.start_state}
    queue = deque([dfa.start_state])
    while queue:
        state = queue.popleft()
        for symbol in alphabet:
            nxt = transitions.get((state, symbol))
            if nxt is not None and nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)

    accepting = set(dfa.accepting_states) & reachable
    dead = set(dfa.dead_states) & reachable
    if sink in reachable:
        dead.add(sink)

    # Step 3: initial partition by (accepting, dead).
    def _signature(state):
        return (state in accepting, state in dead)

    partitions: dict[tuple[bool, bool], set[State]] = {}
    for state in reachable:
        partitions.setdefault(_signature(state), set()).add(state)
    blocks = [frozenset(block) for block in partitions.values()]

    # Refine: split blocks until each block's members agree on the destination
    # block under every symbol.
    changed = True
    while changed:
        changed = False
        block_of: dict[State, int] = {}
        for index, block in enumerate(blocks):
            for state in block:
                block_of[state] = index
        new_blocks: list[frozenset[State]] = []
        for block in blocks:
            buckets: dict[tuple, set[State]] = {}
            for state in block:
                signature = tuple(
                    block_of.get(transitions.get((state, symbol)), -1)
                    for symbol in sorted(alphabet, key=repr)
                )
                buckets.setdefault(signature, set()).add(state)
            if len(buckets) == 1:
                new_blocks.append(block)
            else:
                new_blocks.extend(frozenset(bucket) for bucket in buckets.values())
                changed = True
        blocks = new_blocks

    # Step 4: reindex blocks to integers 0..k; the block containing start_state
    # becomes the new start state.
    block_of_final: dict[State, int] = {}
    for index, block in enumerate(blocks):
        for state in block:
            block_of_final[state] = index

    start_block = block_of_final[dfa.start_state]
    new_states = frozenset(range(len(blocks)))
    new_transitions: dict[tuple[State, Symbol], State] = {}
    for index, block in enumerate(blocks):
        # Every member of a block agrees on the destination block per symbol,
        # so picking any representative suffices.
        representative = next(iter(block))
        for symbol in alphabet:
            destination = transitions.get((representative, symbol))
            if destination is None:
                continue
            new_transitions[(index, symbol)] = block_of_final[destination]

    new_accepting = frozenset(
        index for index, block in enumerate(blocks)
        if any(state in accepting for state in block)
    )
    new_dead = frozenset(
        index for index, block in enumerate(blocks)
        if all(state in dead for state in block)
    )

    return DFA(
        states=new_states,
        alphabet=alphabet,
        transitions=new_transitions,
        start_state=start_block,
        accepting_states=new_accepting,
        dead_states=new_dead,
    )


def complement_dfa(dfa: DFA) -> DFA:
    """Return the language complement of *dfa* over its existing alphabet.

    Undefined transitions are first completed with a fresh rejecting sink state
    so complementing partial DFAs is exact over ``dfa.alphabet``.

    Args:
        dfa: The automaton to complement.

    Returns:
        A complete DFA that accepts exactly the strings rejected by *dfa*.
    """
    sink = ("__complement_sink__",)
    while sink in dfa.states:
        sink = (sink,)

    states = set(dfa.states)
    transitions = dict(dfa.transitions)
    needs_sink = False
    for state in dfa.states:
        for symbol in dfa.alphabet:
            if (state, symbol) not in transitions:
                transitions[(state, symbol)] = sink
                needs_sink = True
    if needs_sink:
        states.add(sink)
        for symbol in dfa.alphabet:
            transitions[(sink, symbol)] = sink

    accepting = states - set(dfa.accepting_states)
    return DFA(
        states=frozenset(states),
        alphabet=dfa.alphabet,
        transitions=transitions,
        start_state=dfa.start_state,
        accepting_states=frozenset(accepting),
        dead_states=frozenset(),
    )
