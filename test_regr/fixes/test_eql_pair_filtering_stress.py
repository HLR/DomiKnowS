"""
Stress tests for issue #372 — ``eqL`` auto-wrap + kwargs forwarding.

10,000 iterations per test by default.  Override via
``DOMIKNOWS_EQL_STRESS_ITERS``.  A 100k sweep is gated behind
``DOMIKNOWS_RUN_HEAVY=1``.
"""
import os
import random
import string

import pytest

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import eqL


ITERS = int(os.environ.get("DOMIKNOWS_EQL_STRESS_ITERS", "10000"))
SEED = int(os.environ.get("DOMIKNOWS_EQL_STRESS_SEED", "20260421"))


def _rand_hashable_scalar(rng):
    kind = rng.choice(['str', 'int', 'bool'])
    if kind == 'str':
        n = rng.randint(1, 6)
        return ''.join(rng.choice(string.ascii_letters) for _ in range(n))
    if kind == 'int':
        return rng.randint(-100, 100)
    return rng.choice([True, False])


def _rand_container(rng):
    size = rng.randint(1, 4)
    items = [_rand_hashable_scalar(rng) for _ in range(size)]
    kind = rng.choice(['set', 'frozenset', 'list', 'tuple'])
    if kind == 'set':
        return set(items)
    if kind == 'frozenset':
        return frozenset(items)
    if kind == 'list':
        return list(items)
    return tuple(items)


def test_stress_scalar_always_wrapped():
    rng = random.Random(SEED)
    failures = []

    with Graph('issue_372_stress_scalar'):
        c = Concept(name='c_stress_scalar')
        for i in range(ITERS):
            value = _rand_hashable_scalar(rng)
            lc = eqL(c, 'attr', value)
            if not isinstance(lc.e[2], set):
                failures.append(f"iter={i} value={value!r} got {type(lc.e[2]).__name__}")
            elif value not in lc.e[2] or len(lc.e[2]) != 1:
                failures.append(f"iter={i} value={value!r} set={lc.e[2]}")
            if len(failures) >= 5:
                break

    assert not failures, f"{len(failures)} failures. First 5: {failures[:5]}"


def test_stress_container_coerced_to_set():
    rng = random.Random(SEED + 1)
    failures = []

    with Graph('issue_372_stress_container'):
        c = Concept(name='c_stress_container')
        for i in range(ITERS):
            container = _rand_container(rng)
            lc = eqL(c, 'attr', container)
            if not isinstance(lc.e[2], set):
                failures.append(f"iter={i} got {type(lc.e[2]).__name__} want set")
            elif lc.e[2] != set(container):
                failures.append(f"iter={i} contents mismatch {lc.e[2]} vs {set(container)}")
            if len(failures) >= 5:
                break

    assert not failures, f"{len(failures)} failures. First 5: {failures[:5]}"


def test_stress_kwargs_forwarded():
    rng = random.Random(SEED + 2)
    failures = []

    with Graph('issue_372_stress_kwargs'):
        c = Concept(name='c_stress_kwargs')
        for i in range(ITERS):
            active = rng.choice([True, False])
            sample = rng.choice([True, False])
            p = rng.randint(0, 100)

            lc = eqL(c, 'attr', 'v', active=active,
                     sampleEntries=sample, p=p)

            if lc.active is not active:
                failures.append(f"iter={i} active want {active} got {lc.active}")
            if lc.sampleEntries is not sample:
                failures.append(f"iter={i} sampleEntries mismatch")
            if lc.p != p:
                failures.append(f"iter={i} p want {p} got {lc.p}")
            if len(failures) >= 5:
                break

    assert not failures, f"{len(failures)} failures. First 5: {failures[:5]}"


def test_stress_two_arg_form_uses_instanceid():
    rng = random.Random(SEED + 3)
    failures = []

    with Graph('issue_372_stress_two_arg'):
        c = Concept(name='c_stress_two_arg')
        for i in range(ITERS):
            val = ''.join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 8)))
            lc = eqL(c, val)
            if lc.e[1] != 'instanceID':
                failures.append(f"iter={i} attr want instanceID got {lc.e[1]!r}")
            if lc.e[2] != {val}:
                failures.append(f"iter={i} value want {{{val!r}}} got {lc.e[2]!r}")
            if len(failures) >= 5:
                break

    assert not failures, f"{len(failures)} failures. First 5: {failures[:5]}"


@pytest.mark.skipif(
    os.environ.get("DOMIKNOWS_RUN_HEAVY", "0") != "1",
    reason="Heavy 100k sweep. Set DOMIKNOWS_RUN_HEAVY=1 to enable.",
)
def test_stress_one_lakh_combined():
    rng = random.Random(SEED + 4)
    total = 100_000
    failures = 0

    with Graph('issue_372_stress_100k'):
        c = Concept(name='c_stress_100k')
        for i in range(total):
            if rng.random() < 0.5:
                value = _rand_hashable_scalar(rng)
                lc = eqL(c, 'attr', value,
                         active=rng.choice([True, False]),
                         p=rng.randint(0, 100))
                if not isinstance(lc.e[2], set) or value not in lc.e[2]:
                    failures += 1
            else:
                container = _rand_container(rng)
                lc = eqL(c, 'attr', container)
                if not isinstance(lc.e[2], set) or lc.e[2] != set(container):
                    failures += 1
            if failures >= 5:
                break

    assert failures == 0, f"{failures} failures in {total} iterations"
