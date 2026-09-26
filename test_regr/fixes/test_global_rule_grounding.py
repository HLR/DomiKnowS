"""Graph-level (global) rules over relation pairs and triples must be grounded exactly.

Runs global_rules_repro.py (synthetic scenes, hand-set 0/1 relation values,
brute-force truth per grounding).  Covered fixes:
  inverse     equivalenceL(left('a','b'), right('b','a')): operands over the same
              variables in a different order were paired row by row, i.e.
              right(a, b) was evaluated (expandToJointGrounding compared sets)
  trans_nand  nandL(left(a,b), left(b,c), notL(left(a,c))): a head n-ary
              connective with three or more operands returned its truth value
              instead of its loss (createLogicalConstrains dropped onlyConstrains)
  trans_if    ifL(andL(left(a,b), left(b,c)), left(a,c)): the nested andL was
              quantified down to b and could not be aligned (size mismatch)
  xor_distinct ifL(distinct, xorL(...)) is exact under 'P' and 'L'; Goedel
              implication counts a near-0 premise with a smaller consequent as
              violated, which is why InferenceModel has global_constraint_tnorm.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).with_name("global_rules_repro.py")


def _run(rule, tnorm, compiled):
    env = dict(os.environ, TNORM=tnorm, COMPILED="1" if compiled else "0")
    out = subprocess.run([sys.executable, str(REPRO), rule], capture_output=True, text=True,
                         cwd=str(REPRO.parent), timeout=900, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith(rule)]
    assert lines, out.stdout[-2000:] + out.stderr[-2000:]
    return lines


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tnorm", ["G", "P", "L"])
@pytest.mark.parametrize("rule", ["inverse", "xor_flat", "trans_nand"])
def test_rule_counts_match_brute_force(rule, tnorm, compiled):
    for line in _run(rule, tnorm, compiled):
        assert " OK " in line, line


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tnorm", ["P", "L"])
def test_implication_with_nested_consequent(tnorm, compiled):
    for line in _run("xor_distinct", tnorm, compiled):
        assert " OK " in line, line


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tnorm", ["P", "L"])
def test_implication_over_nested_conjunction(tnorm, compiled):
    # ifL(andL(r(a,b), r(b,c)), r(a,c)): the nested andL used to be quantified
    # down to its shared variable b (existential reading), so it could not be
    # aligned with r(a,c) (size mismatch).  Under a plain connective it now
    # keeps the (a, b, c) table.
    for line in _run("trans_if", tnorm, compiled):
        assert " OK " in line, line
