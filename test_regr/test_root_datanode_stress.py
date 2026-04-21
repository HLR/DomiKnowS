"""
Stress / property-based tests for issue #406.

We build *many* randomized builder states and verify the invariants that the
fix is supposed to guarantee, no matter what ordering the caller used:

1. DETERMINISM
   Given the same set of root DataNodes, ``findRootDataNode`` returns the
   same concept name regardless of the input list's order.

2. NO CONSTRAINT PICKED
   As long as at least one non-constraint root exists, the selected root
   must never be the ``constraint`` concept.

3. EXPLICIT OVERRIDE ALWAYS WINS
   When ``primaryRootConcept`` names a concept that is present, the
   returned DN is always of that concept, regardless of ordering.

4. __updateRootDataNodeList TAIL INVARIANT
   After ``__updateRootDataNodeList`` has been called with an arbitrary
   permutation of DNs, any ``constraint`` DN is at the tail of the list
   and all non-constraint DNs come before it.

The default scale is 10,000 iterations. Bump ``DOMIKNOWS_STRESS_ITERS`` in
the environment to run more (e.g. ``DOMIKNOWS_STRESS_ITERS=100000`` for a
full lakh). The test is self-contained and doesn't need gurobipy.
"""
import os
import random
import string
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domiknows.graph import Concept, Graph
from domiknows.graph.dataNode import DataNode, DataNodeBuilder


# Controls how many random scenarios each test runs. Keep the default modest
# so the suite stays fast; override via env var for heavier stress runs.
ITERS = int(os.environ.get("DOMIKNOWS_STRESS_ITERS", "10000"))
SEED = int(os.environ.get("DOMIKNOWS_STRESS_SEED", "20260421"))


def _rand_name(rng, used):
    """Generate a short unique concept name."""
    while True:
        name = ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
        if name not in used and name != 'constraint':
            used.add(name)
            return name


def _build_scenario(rng, graph_idx):
    """
    Build one random builder + DN list.

    Returns ``(builder, dns, concept_names)`` where ``concept_names`` is the
    list of the ontologyNode names in the same order as ``dns``.
    """
    n_real = rng.randint(1, 6)
    has_constraint = rng.random() < 0.6  # most scenarios include constraint

    used = set()
    names = [_rand_name(rng, used) for _ in range(n_real)]

    with Graph(name=f'stress_g_{graph_idx}') as g:
        concepts = {name: Concept(name=name) for name in names}

    builder = DataNodeBuilder({"graph": g})
    dns = []
    dn_names = []

    next_id = 0
    for name in names:
        dn = DataNode(
            myBuilder=builder,
            instanceID=next_id,
            instanceValue="",
            ontologyNode=concepts[name],
        )
        dns.append(dn)
        dn_names.append(name)
        next_id += 1

    if has_constraint:
        # The graph auto-creates a ``constraint`` concept on __enter__; reuse it.
        constraint_concept = g.findConcept("constraint")
        n_constraint = rng.randint(1, 3)
        for _ in range(n_constraint):
            dn = DataNode(
                myBuilder=builder,
                instanceID=next_id,
                instanceValue="",
                ontologyNode=constraint_concept,
            )
            dns.append(dn)
            dn_names.append('constraint')
            next_id += 1

    # Shuffle so the builder never sees a canonical ordering.
    combined = list(zip(dns, dn_names))
    rng.shuffle(combined)
    dns, dn_names = [list(t) for t in zip(*combined)]

    dict.__setitem__(builder, 'dataNode', list(dns))
    return builder, dns, dn_names


# ---------------------------------------------------------------------------


def test_stress_determinism_and_constraint_exclusion():
    """Core invariant sweep — runs ITERS randomized scenarios."""
    rng = random.Random(SEED)
    failures = []

    for i in range(ITERS):
        builder, dns, names = _build_scenario(rng, i)

        # Pick 1: in the shuffled order the builder already has.
        first = builder.findRootDataNode(list(dns))

        # Pick 2: reversed.
        second = builder.findRootDataNode(list(reversed(dns)))

        # Pick 3: another independent shuffle.
        another = list(dns)
        rng.shuffle(another)
        third = builder.findRootDataNode(another)

        picks = [first, second, third]

        # ---- invariant 1: determinism ----
        pick_names = {p.ontologyNode.name for p in picks}
        if len(pick_names) != 1:
            failures.append(
                f"iter={i} non-deterministic selection: {pick_names} for inputs {names}"
            )
            continue

        # ---- invariant 2: constraint never picked if a real root exists ----
        has_real = any(n != 'constraint' for n in names)
        if has_real and first.ontologyNode.name == 'constraint':
            failures.append(
                f"iter={i} picked constraint despite real roots present: {names}"
            )

    assert not failures, f"{len(failures)} failure(s) out of {ITERS}. First 5: {failures[:5]}"


def test_stress_explicit_primary_root_always_honored():
    """If ``primaryRootConcept`` names a present concept, we must get it back."""
    rng = random.Random(SEED + 1)
    failures = []

    for i in range(ITERS):
        builder, dns, names = _build_scenario(rng, i)

        real_names = [n for n in set(names) if n != 'constraint']
        if not real_names:
            continue

        target = rng.choice(real_names)
        picked = builder.findRootDataNode(list(dns), primaryRootConcept=target)

        if picked.ontologyNode.name != target:
            failures.append(
                f"iter={i} asked for {target!r} got {picked.ontologyNode.name!r}; inputs={names}"
            )

    assert not failures, f"{len(failures)} failure(s) out of {ITERS}. First 5: {failures[:5]}"


def test_stress_update_root_list_puts_constraint_last():
    """After __updateRootDataNodeList, constraint DNs must be at the tail."""
    rng = random.Random(SEED + 2)
    failures = []

    for i in range(ITERS):
        builder, dns, names = _build_scenario(rng, i)
        # Reset and let the builder's own method re-sort the list.
        dict.__setitem__(builder, 'dataNode', [])
        shuffled = list(dns)
        rng.shuffle(shuffled)
        builder._DataNodeBuilder__updateRootDataNodeList(shuffled)

        roots = dict.__getitem__(builder, 'dataNode')
        root_names = [dn.ontologyNode.name for dn in roots]

        if 'constraint' in root_names and any(n != 'constraint' for n in root_names):
            # All constraints must form a contiguous suffix.
            first_constraint = root_names.index('constraint')
            tail = root_names[first_constraint:]
            if any(n != 'constraint' for n in tail):
                failures.append(
                    f"iter={i} constraint not in tail: {root_names}"
                )

    assert not failures, f"{len(failures)} failure(s) out of {ITERS}. First 5: {failures[:5]}"


# ---------------------------------------------------------------------------
# Heavy mode: only runs when explicitly asked for, so normal `pytest` stays
# fast. Invoke with:  pytest -m heavy  (after registering the marker below)
# or simply run with env var DOMIKNOWS_RUN_HEAVY=1.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("DOMIKNOWS_RUN_HEAVY", "0") != "1",
    reason="Heavy 100k-iteration sweep. Set DOMIKNOWS_RUN_HEAVY=1 to enable.",
)
def test_stress_one_lakh_combined():
    """
    Full 1,00,000 iteration sweep combining all invariants.

    This is the "lakh" mode — disabled by default because it takes a while.
    Enable with:  $env:DOMIKNOWS_RUN_HEAVY=1
    """
    rng = random.Random(SEED + 3)
    total = 100_000
    failures = []

    for i in range(total):
        builder, dns, names = _build_scenario(rng, i)

        a = builder.findRootDataNode(list(dns))
        b = builder.findRootDataNode(list(reversed(dns)))
        if a.ontologyNode.name != b.ontologyNode.name:
            failures.append(f"iter={i} non-det: {a.ontologyNode.name} vs {b.ontologyNode.name}")

        if any(n != 'constraint' for n in names) and a.ontologyNode.name == 'constraint':
            failures.append(f"iter={i} picked constraint when real roots existed")

        real = [n for n in set(names) if n != 'constraint']
        if real:
            tgt = real[i % len(real)]
            picked = builder.findRootDataNode(list(dns), primaryRootConcept=tgt)
            if picked.ontologyNode.name != tgt:
                failures.append(f"iter={i} explicit mismatch: wanted {tgt} got {picked.ontologyNode.name}")

        if failures and len(failures) >= 10:
            break

    assert not failures, f"{len(failures)} failure(s) out of {total}. First 5: {failures[:5]}"
