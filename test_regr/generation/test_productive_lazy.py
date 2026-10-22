"""Lazy successor stepping keeps bounded reachability exact and avoids stepping siblings it never needs."""
import random
from collections import deque

import pytest

from domiknows.generation.dfa.productive import BoundedReachability


def random_graph(rng, nodes=14, symbols=5):
    edges = {n: {s: rng.randrange(nodes) for s in range(symbols) if rng.random() < 0.45} for n in range(nodes)}
    accepting = {n for n in range(nodes) if rng.random() < 0.12}
    return edges, accepting


def brute_force(edges, accepting, start, budget):
    seen, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        if node in accepting:
            return True
        if seen[node] == budget:
            continue
        for target in edges[node].values():
            if target not in seen:
                seen[target] = seen[node] + 1
                queue.append(target)
    return False


@pytest.mark.parametrize("seed", range(12))
def test_allowed_tokens_and_reachability_match_brute_force(seed):
    rng = random.Random(seed)
    edges, accepting = random_graph(rng)
    search = BoundedReachability(lambda n, s: edges[n].get(s), accepting.__contains__, lambda n: edges[n])
    for start in range(len(edges)):
        for budget in (0, 1, 2, 4, 8):
            assert search.can_reach_accepting(start, budget) == brute_force(edges, accepting, start, budget)
            expected = {s for s, target in edges[start].items() if brute_force(edges, accepting, target, budget)}
            assert search.allowed_tokens(start, budget) == expected


def test_probe_does_not_step_siblings_after_it_found_a_witness():
    stepped = []

    def step(state, symbol):
        stepped.append((state, symbol))
        return "goal" if symbol == "a" else "dead"

    search = BoundedReachability(step, lambda s: s == "goal", lambda s: ("a", "b", "c", "d") if s == "root" else ())
    assert search.can_reach_accepting("root", 3)
    assert stepped == [("root", "a")]


def test_an_interrupted_enumeration_resumes_without_skipping_a_candidate():
    calls = []

    def step(state, symbol):
        calls.append(symbol)
        if symbol == 1 and calls.count(1) == 1:
            raise RuntimeError("interrupted")
        return symbol

    search = BoundedReachability(step, lambda s: False, lambda s: (0, 1, 2) if s == "root" else ())
    with pytest.raises(RuntimeError):
        search.allowed_tokens("root", 2)
    assert search.allowed_tokens("root", 2) == set()  # no candidate reaches an accepting state
    assert sorted(set(calls)) == [0, 1, 2]
    assert [s for s, _ in search._lazy("root").items] == [0, 1, 2]
