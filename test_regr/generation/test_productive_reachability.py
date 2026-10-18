"""Search order must affect cost, never the bounded language or token masks."""
from itertools import product
from random import Random

import pytest

from domiknows.generation import DFA, DFACompletionBounds
from domiknows.generation.dfa.productive import BoundedReachability, SemanticSearchLimitExceeded


def test_short_witness_is_found_beside_an_exponentially_large_first_branch():
    def step(state, symbol):
        if state == "root":
            return "tree" if symbol == 0 else "done"
        return (state, symbol)

    search = BoundedReachability(step, lambda s: s == "done", lambda s: (0, 1))
    assert search.can_reach_accepting("root", 80)
    # The old depth-first search exhausts 10,000 expansions in the tree.
    assert search.can_reach_accepting("root", 1)
    assert not search.can_reach_accepting("root", 0)


def test_fast_witness_probe_avoids_breadth_explosion_for_deeper_easy_completions():
    calls = []
    def step(state, symbol):
        calls.append((state, symbol))
        return state + (symbol,)
    search = BoundedReachability(step, lambda s: len(s) == 6, lambda s: range(20))
    assert search.can_reach_accepting((), 80)
    assert len(calls) <= 120


def test_cycle_is_exhausted_once_instead_of_unrolled_to_the_token_horizon():
    calls = []
    def step(state, symbol):
        calls.append((state, symbol))
        return state
    search = BoundedReachability(step, lambda s: False, lambda s: (0,), max_state_expansions=2)
    assert not search.can_reach_accepting(0, 100000)
    assert calls == [(0, 0)]
    assert not search.can_reach_accepting(0, 80)


def test_exhaustion_does_not_cache_unknown_and_completed_negative_results_are_monotone():
    search = BoundedReachability(lambda s, _: s + 1, lambda s: s == 8,
                                 lambda s: (0,), max_state_expansions=3)
    with pytest.raises(SemanticSearchLimitExceeded):
        search.can_reach_accepting(0, 80)
    assert not search.can_reach_accepting(0, 1)
    assert search.can_reach_accepting(6, 2)
    search.max_state_expansions = 30
    assert search.can_reach_accepting(0, 80)
    assert not search.can_reach_accepting(0, 7)
    assert search.can_reach_accepting(0, 8)


def test_token_mask_candidates_share_one_expansion_cap_including_witness_probes():
    def step(state, symbol):
        return (symbol, 0) if state == "root" else (state[0], state[1] + 1)
    search = BoundedReachability(step, lambda s: s != "root" and s[1] == 3,
                                 lambda s: range(8) if s == "root" else (0,),
                                 max_state_expansions=20)
    with pytest.raises(SemanticSearchLimitExceeded):
        search.allowed_tokens("root", 80)
    search.max_state_expansions = 100
    assert search.allowed_tokens("root", 80) == set(range(8))


def test_masks_and_cached_answers_match_independent_word_enumeration():
    rng = Random(42)
    for _ in range(12):
        transitions = {(s, a): rng.choice((None, 0, 1, 2, 3))
                       for s in range(4) for a in (0, 1)}
        accepting = {s for s in range(4) if rng.choice((False, True))}
        def step(s, a):
            return transitions[s, a]
        def oracle(state, budget):
            for length in range(budget + 1):
                for word in product((0, 1), repeat=length):
                    current = state
                    for symbol in word:
                        if current is None:
                            break
                        current = step(current, symbol)
                    if current in accepting:
                        return True
            return False
        search = BoundedReachability(step, accepting.__contains__, lambda s: (0, 1))
        # Mixed budget order exercises upper/lower bound caches in both directions.
        for budget in (5, 0, 3, 1, 4, 2):
            for state in range(4):
                assert search.can_reach_accepting(state, budget) == oracle(state, budget)
                expected = {a for a in (0, 1) if step(state, a) is not None
                            and oracle(step(state, a), budget)}
                assert search.allowed_tokens(state, budget) == expected


def test_relaxed_completion_bounds_prune_impossible_paths_without_proving_concrete_acceptance():
    grammar = DFA(frozenset(range(4)), frozenset({0, 1}),
                  {(0, 0): 1, (1, 0): 2, (0, 1): 3, (3, 1): 3}, 0, frozenset({2, 3}))
    bounds = DFACompletionBounds.from_dfa(
        grammar, transition_allowed=lambda s, a, t: a != 1,
        accepting_allowed=lambda s: s == 2,
    )
    assert dict(bounds.distances) == {2: 0, 1: 1, 0: 2}
    with pytest.raises(TypeError):
        bounds.distances[0] = 99
    calls = []
    def step(s, a):
        calls.append((s, a))
        return grammar.step(s, a) if a == 0 else None
    search = BoundedReachability(step, lambda s: False, lambda s: (0, 1),
                                 completion_bound=bounds, max_state_expansions=3)
    assert not search.can_reach_accepting(0, 1)
    assert not search.can_reach_accepting(3, 80)
    assert not calls
    # A finite relaxed bound does not override the concrete acceptance test.
    assert not search.can_reach_accepting(0, 2)


def test_completion_bounds_guide_probe_toward_a_valid_short_witness():
    grammar = DFA(frozenset(range(5)), frozenset({0, 1}),
                  {(0, 0): 1, (1, 0): 2, (2, 0): 3, (0, 1): 4},
                  0, frozenset({3, 4}))
    bounds = DFACompletionBounds.from_dfa(grammar)
    visits = []
    def accepting(s):
        visits.append(s)
        return grammar.is_accepting(s)
    search = BoundedReachability(grammar.step, accepting, lambda s: grammar.allowed_tokens(s),
                                 completion_bound=bounds)
    assert search.can_reach_accepting(0, 80)
    assert visits == [0, 4]
