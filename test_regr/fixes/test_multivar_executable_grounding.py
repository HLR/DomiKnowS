"""Multi-variable executable formulas must be grounded jointly.

Standalone reproduction: synthetic scene, hand-set probabilities, brute-force
truth. Two-variable ``existsL(andL(A(a), B(b), left(a, b)))`` always worked;
the three-variable chain ``existsL(andL(A(a), B(b), C(c), left(a, b),
left(b, c)))`` used to be verified wrong (scene ``s1`` is True, the framework
said False) and the loss path raised a tensor size mismatch, because operands
enumerated over different variable tuples were never joined.
``LogicalConstraintConstructor.expandToJointGrounding`` fixes that; scene
``s2`` is the discriminating case where the per-relation existential
approximation says True but the exact answer is False.
"""
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

REPRO = Path(__file__).with_name("multivar_executable_repro.py")


def _run(mode, spec):
    out = subprocess.run([sys.executable, str(REPRO), mode, spec], capture_output=True, text=True,
                         cwd=str(REPRO.parent), timeout=900)
    lines = [l for l in out.stdout.splitlines() if l.startswith(mode)]
    assert lines, out.stdout[-2000:] + out.stderr[-2000:]
    return lines[-1]


@pytest.mark.parametrize("spec", ["s1/q2", "s1/q2neg", "s2/q2", "s1/q2,s1/q2neg,s2/q2"])
def test_two_variable_formulas_verify_exactly(spec):
    line = _run("verify", spec)
    assert "ERR" not in line and "evaluate_condition=100%" in line, line


def test_two_variable_formulas_train():
    assert "trained" in _run("train", "s1/q2,s2/q2")


@pytest.mark.parametrize("spec", ["s1/q3", "s2/q3", "s1/q3r", "s2/q3r", "s1/q3s", "s3/q3s", "s1/q2,s1/q3,s2/q3,s2/q2,s3/q3s"])
def test_three_variable_chain_verifies_exactly(spec):
    line = _run("verify", spec)
    assert "ERR" not in line and "evaluate_condition=100%" in line, line


@pytest.mark.parametrize("spec", ["s1/q2or", "s1/q2orneg", "s2/q2or", "s1/q2and", "s2/q2and",
                                  "s1/q3or", "s2/q3or", "s1/q2or,s1/q2orneg,s2/q3or",
                                  "s1/q3and", "s2/q3and", "s1/q3orand", "s2/q3orand", "s1/q3orandb",
                                  "s3/q3orandb", "s4/q4orand", "s2/q4orand", "s4/q4or2", "s2/q4or2",
                                  "s4/q4orand,s1/q3orandb,s2/q3and"])
def test_relation_inside_nested_connective_verifies_exactly(spec):
    # The outer unaries are paths over a relation declared only inside the
    # nested orL; they used to get no candidates and every such formula was
    # verified False.  q3and/q3orand*/q4*: a unary on an enclosing variable
    # inside the nested andL (its binding, shared-variable alignment and the
    # path projection to one variable were each missing).
    line = _run("verify", spec)
    assert "ERR" not in line and "evaluate_condition=100%" in line, line


def test_relation_inside_nested_connective_trains():
    assert "trained" in _run("train", "s1/q2or,s2/q3or,s1/q2and,s4/q4orand,s1/q3orandb")


def test_three_variable_chain_trains():
    assert "trained" in _run("train", "s1/q3,s2/q3,s1/q3r,s3/q3s")


def test_three_variable_chain_loss_follows_joint_truth():
    # calculateSingleLcLoss is label-agnostic: satisfied -> low, violated -> high.
    def value(spec):
        return float(_run("loss", spec).rsplit(", ", 1)[1].rstrip(")]"))
    satisfied, violated = value("s1/q2"), value("s1/q2neg")
    assert satisfied < violated
    assert abs(value("s1/q3") - satisfied) < 1e-3   # True chain
    assert abs(value("s2/q3") - violated) < 1e-3    # False chain (shared-binding case)


@pytest.mark.parametrize("spec", ["s1/m2", "s1/m3", "s2/m3", "s1/m2,s1/m3,s2/m3"])
def test_entity_selection_reduces_joint_table_to_answer_variable(spec):
    # miotaL over a nested andL: the selection distribution must have one value
    # per object and pick exactly the objects that can play the first variable.
    line = _run("select", spec)
    assert "ERR" not in line and "evaluate_condition=100%" in line, line
    for item in line.split("[('")[1:]:
        truth = item.split("', ", 1)[1].split("], [")[0] + "]"
        pred = "[" + item.split("], [")[1].split(")")[0]
        assert truth == pred, line


def test_entity_selection_trains():
    assert "trained" in _run("train", "s1/m2,s1/m3,s2/m3")
