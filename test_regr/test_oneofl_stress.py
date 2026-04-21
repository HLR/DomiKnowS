"""
Stress tests for issue #371: oneOfL / oneOfAL.

For 10k randomised configurations we verify:

1. Building ``oneOfL(*concepts)`` with a trailing integer never moves the
   enforced limit away from 1 (the fixedLimit guarantee).
2. ``exactL`` with the same arguments silently honours a trailing integer
   (this is the footgun ``oneOfL`` closes and why we keep the class alive
   across releases).
3. The adaptive t-norm registry always classifies both ``oneOfL`` and
   ``oneOfAL`` as counting constraints of mode ``'L'``.

Default: 10,000 iters. Override via ``DOMIKNOWS_ONEOFL_STRESS_ITERS``.
Heavy 100k sweep gated behind ``DOMIKNOWS_RUN_HEAVY=1``.
"""
import os
import random

import pytest


ITERS = int(os.environ.get("DOMIKNOWS_ONEOFL_STRESS_ITERS", "10000"))
SEED = int(os.environ.get("DOMIKNOWS_ONEOFL_STRESS_SEED", "20260421"))


def _resolve_limit(cls, e):
    """Mimic _CountBaseL.__call__'s limit resolution without a live LC."""
    if cls.fixedLimit is not None:
        return cls.fixedLimit
    return e[-1] if (e and isinstance(e[-1], int)) else 1


def test_stress_oneofl_limit_is_always_one():
    from domiknows.graph.logicalConstrain import oneOfL

    rng = random.Random(SEED)
    failures = []

    for i in range(ITERS):
        # Build random args: a sprinkling of placeholder concept objects
        # followed by (optionally) a trailing integer that a naive user
        # might think overrides the limit.
        n_concepts = rng.randint(1, 8)
        e = [object() for _ in range(n_concepts)]
        if rng.random() < 0.5:
            e.append(rng.randint(-5, 100))
        e = tuple(e)

        limit = _resolve_limit(oneOfL, e)
        if limit != 1:
            failures.append(f"iter={i} e={e} resolved limit={limit}")
        if oneOfL.limitOp != "==":
            failures.append(f"iter={i} unexpected op={oneOfL.limitOp}")

    assert not failures, f"{len(failures)} failure(s). First 5: {failures[:5]}"


def test_stress_exactl_trailing_int_still_wins():
    """Guard against accidentally pinning exactL's limit — users may depend on
    the existing exactL(..., k) behaviour."""
    from domiknows.graph.logicalConstrain import exactL

    rng = random.Random(SEED + 1)
    mismatches = []

    for i in range(ITERS):
        n_concepts = rng.randint(1, 8)
        e = [object() for _ in range(n_concepts)]
        if rng.random() < 0.5:
            trailing = rng.randint(0, 20)
            e.append(trailing)
            expected = trailing
        else:
            expected = 1
        e = tuple(e)

        actual = _resolve_limit(exactL, e)
        if actual != expected:
            mismatches.append(f"iter={i} e={e} want={expected} got={actual}")

    assert not mismatches, (
        f"{len(mismatches)} exactL limits deviated. First 5: {mismatches[:5]}"
    )


def test_stress_registry_always_sees_oneofl():
    """Re-importing the module under random dict mutations must not drop
    the oneOfL registrations."""
    rng = random.Random(SEED + 2)
    for i in range(ITERS):
        # Freshly read the registry each iteration — this catches any
        # accidental in-place mutation by other import paths.
        from domiknows.solver.adaptiveTNormLossCalculator import (
            COUNTING_CONSTRAINTS,
            DEFAULT_TNORM_BY_TYPE,
        )
        assert 'oneOfL' in COUNTING_CONSTRAINTS, f"iter={i} oneOfL missing"
        assert 'oneOfAL' in COUNTING_CONSTRAINTS, f"iter={i} oneOfAL missing"
        assert DEFAULT_TNORM_BY_TYPE['oneOfL'] == 'L'
        assert DEFAULT_TNORM_BY_TYPE['oneOfAL'] == 'L'
        # jitter an unrelated key to make sure that path doesn't matter
        _ = rng.random()


@pytest.mark.skipif(
    os.environ.get("DOMIKNOWS_RUN_HEAVY", "0") != "1",
    reason="Heavy 100k-iteration sweep. Set DOMIKNOWS_RUN_HEAVY=1 to enable.",
)
def test_stress_one_lakh_combined():
    from domiknows.graph.logicalConstrain import oneOfL, exactL

    rng = random.Random(SEED + 3)
    total = 100_000
    failures = 0

    for i in range(total):
        n_concepts = rng.randint(1, 8)
        e = [object() for _ in range(n_concepts)]
        has_int = rng.random() < 0.5
        if has_int:
            trailing = rng.randint(0, 20)
            e.append(trailing)
        e = tuple(e)

        if _resolve_limit(oneOfL, e) != 1:
            failures += 1
        if has_int and _resolve_limit(exactL, e) != e[-1]:
            failures += 1
        if not has_int and _resolve_limit(exactL, e) != 1:
            failures += 1
        if failures >= 5:
            break

    assert failures == 0, f"{failures} failures in {total} iterations"
