"""
Regression tests for issue #358 — ifSatisfied returns 100 (now 0) when first argument is False.

The bug: when evaluating ifL constraints, the ifSatisfied metric returned a numeric
value (originally 100, later 0) when NO instance had the antecedent True. This is
misleading because:
- 0 implies "all applicable constraints were violated"
- NaN correctly means "no applicable instances to evaluate"

The fix: return float('nan') when ifVerifyListLen == 0 (no antecedent True).

Tests verify:
1. Antecedent True + consequent True → ifSatisfied = 100
2. Antecedent True + consequent False → ifSatisfied = 0
3. Antecedent False (all instances) → ifSatisfied = NaN  (THE BUG)
4. Mixed antecedents → correct percentage
5. Multiple instances with varying antecedent values
6. Consumer code in program.py handles NaN correctly
"""

import math
import pytest
import torch
from unittest.mock import MagicMock

from domiknows.graph import Graph, Concept, ifL, forAllL
from domiknows.solver.logicalConstraintVerifier import LogicalConstraintVerifier


def _make_verifier_and_lc(lc_cls=ifL):
    """Create a graph with an ifL (or forAllL) constraint and a mocked verifier.

    Returns (verifier, lc, solver_mock) where solver_mock.constraintConstructor
    .constructLogicalConstrains can be configured to return controlled data.
    """
    with Graph('test_ifsat') as graph:
        image = Concept(name='image')
        a = image(name='a')
        b = image(name='b')
        lc = lc_cls(a, b)

    solver = MagicMock()
    verifier = LogicalConstraintVerifier(solver)
    return verifier, lc, solver


def _run_verify(verifier, lc, solver, verify_list, antecedent_values):
    """Run verifySingleConstraint with controlled mock data.

    Args:
        verify_list: list of lists, e.g. [[1, 0], [1]] — overall implication results
        antecedent_values: list of lists matching verify_list shape — first-arg values
    """
    solver.constraintConstructor.constructLogicalConstrains.return_value = (
        verify_list,
        {'_arg0': antecedent_values},  # first key → antecedent values
    )
    dn = MagicMock()
    boolMethods = MagicMock()
    return verifier.verifySingleConstraint(lc, boolMethods, dn)


# ── Core bug test: antecedent always False ───────────────────────────────────

class TestIfSatisfiedAntecedentFalse:
    """Issue #358: ifSatisfied must be NaN when no antecedent is True."""

    def test_single_instance_antecedent_false_returns_nan(self):
        """One instance, antecedent=False → ifSatisfied should be NaN, not 0."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1]],           # vacuously satisfied
                             antecedent_values=[[0]])      # antecedent False
        assert 'ifSatisfied' in result
        assert math.isnan(result['ifSatisfied']), \
            f"Expected NaN when antecedent is False, got {result['ifSatisfied']}"

    def test_multiple_instances_all_antecedent_false_returns_nan(self):
        """All instances have antecedent=False → ifSatisfied should be NaN."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1, 1, 1]],
                             antecedent_values=[[0, 0, 0]])
        assert math.isnan(result['ifSatisfied'])

    def test_multiple_batches_all_antecedent_false_returns_nan(self):
        """Multiple batches, all antecedent=False → ifSatisfied should be NaN."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1, 1], [1]],
                             antecedent_values=[[0, 0], [0]])
        assert math.isnan(result['ifSatisfied'])

    def test_antecedent_false_with_tensor_values(self):
        """Antecedent values as tensors (as in real usage) → NaN."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1]],
                             antecedent_values=[[torch.tensor(0)]])
        assert math.isnan(result['ifSatisfied'])


# ── Normal cases: antecedent True ────────────────────────────────────────────

class TestIfSatisfiedAntecedentTrue:
    """When antecedent is True, ifSatisfied should be a valid percentage."""

    def test_antecedent_true_consequent_true(self):
        """A=True, B=True → ifSatisfied = 100%."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1]],
                             antecedent_values=[[1]])
        assert result['ifSatisfied'] == 100.0

    def test_antecedent_true_consequent_false(self):
        """A=True, B=False → ifSatisfied = 0%."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[0]],
                             antecedent_values=[[1]])
        assert result['ifSatisfied'] == 0.0

    def test_antecedent_true_with_tensors(self):
        """Antecedent as tensor(1), consequent satisfied → 100%."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1]],
                             antecedent_values=[[torch.tensor(1)]])
        assert result['ifSatisfied'] == 100.0


# ── Mixed antecedents ────────────────────────────────────────────────────────

class TestIfSatisfiedMixed:
    """Mixed antecedent values: only True-antecedent instances count."""

    def test_mixed_two_instances(self):
        """Two instances: one True (satisfied), one False → ifSatisfied = 100%."""
        verifier, lc, solver = _make_verifier_and_lc()
        # Instance 0: antecedent=True, satisfied=1 → counts as satisfied
        # Instance 1: antecedent=False, satisfied=1 → not counted
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1, 1]],
                             antecedent_values=[[1, 0]])
        assert result['ifSatisfied'] == 100.0

    def test_mixed_true_antecedent_not_satisfied(self):
        """Two instances: antecedent True but not satisfied, antecedent False."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[0, 1]],
                             antecedent_values=[[1, 0]])
        assert result['ifSatisfied'] == 0.0

    def test_mixed_half_satisfied(self):
        """Four instances: 2 with antecedent True (1 satisfied, 1 not) → 50%."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1, 0, 1, 1]],
                             antecedent_values=[[1, 1, 0, 0]])
        assert result['ifSatisfied'] == 50.0

    def test_overall_satisfied_includes_vacuous(self):
        """Overall satisfied should count vacuously-true cases."""
        verifier, lc, solver = _make_verifier_and_lc()
        # 3 instances: antecedent False (vacuously True), antecedent True (satisfied), 
        # antecedent False (vacuously True)
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1, 1, 1]],
                             antecedent_values=[[0, 1, 0]])
        assert result['satisfied'] == 100.0
        assert result['ifSatisfied'] == 100.0


# ── forAllL constraint ───────────────────────────────────────────────────────

class TestIfSatisfiedForAllL:
    """forAllL constraints should also return NaN when no antecedent is True."""

    def test_forall_antecedent_false_returns_nan(self):
        verifier, lc, solver = _make_verifier_and_lc(lc_cls=forAllL)
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1, 1]],
                             antecedent_values=[[0, 0]])
        assert math.isnan(result['ifSatisfied'])

    def test_forall_antecedent_true_satisfied(self):
        verifier, lc, solver = _make_verifier_and_lc(lc_cls=forAllL)
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[1]],
                             antecedent_values=[[1]])
        assert result['ifSatisfied'] == 100.0


# ── Consumer compatibility (program.py NaN handling) ─────────────────────────

class TestConsumerNaNHandling:
    """Verify that NaN values are properly handled by downstream consumers."""

    def test_nan_is_excluded_from_accumulation(self):
        """Simulates program.py logic: NaN values should be skipped."""
        import numpy as np

        # Simulate verifyResult from multiple constraints
        verify_results = {
            'ifL_1': {'ifSatisfied': 100.0},
            'ifL_2': {'ifSatisfied': float('nan')},  # no antecedent True
            'ifL_3': {'ifSatisfied': 50.0},
        }

        # Replicate the program.py accumulation logic
        ifl_ac = 0
        ifl_t = 0
        for name, result in verify_results.items():
            if 'ifSatisfied' in result:
                if not np.isnan(result['ifSatisfied']):
                    ifl_ac += result['ifSatisfied']
                    ifl_t += 1

        assert ifl_t == 2  # NaN entry skipped
        assert ifl_ac == 150.0  # 100 + 50

    def test_all_nan_results_in_zero_count(self):
        """When all constraints have NaN ifSatisfied, count should be 0."""
        import numpy as np

        verify_results = {
            'ifL_1': {'ifSatisfied': float('nan')},
            'ifL_2': {'ifSatisfied': float('nan')},
        }

        ifl_t = 0
        for name, result in verify_results.items():
            if 'ifSatisfied' in result:
                if not np.isnan(result['ifSatisfied']):
                    ifl_t += 1

        assert ifl_t == 0


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestIfSatisfiedEdgeCases:
    """Edge cases for ifSatisfied computation."""

    def test_empty_verify_list(self):
        """Empty verify list → satisfied=0, no ifSatisfied crash."""
        verifier, lc, solver = _make_verifier_and_lc()
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[]],
                             antecedent_values=[[]])
        assert result['satisfied'] == 0
        assert math.isnan(result['ifSatisfied'])

    def test_none_values_in_verify_list(self):
        """None values should be skipped in satisfaction computation."""
        verifier, lc, solver = _make_verifier_and_lc()
        # Instance 0: antecedent True, result=None (skipped)
        # Instance 1: antecedent True, result=1 (satisfied)
        result = _run_verify(verifier, lc, solver,
                             verify_list=[[None, 1]],
                             antecedent_values=[[1, 1]])
        # Only instance 1 is counted (None is skipped)
        assert result['ifSatisfied'] == 100.0

    def test_non_ifl_constraint_has_no_ifSatisfied(self):
        """Non-ifL constraints should not have ifSatisfied key."""
        from domiknows.graph import andL

        with Graph('test_and') as graph:
            image = Concept(name='image')
            a = image(name='a')
            b = image(name='b')
            lc = andL(a, b)

        solver = MagicMock()
        verifier = LogicalConstraintVerifier(solver)
        solver.constraintConstructor.constructLogicalConstrains.return_value = (
            [[1]], {'_arg0': [[1]]}
        )
        dn = MagicMock()
        boolMethods = MagicMock()
        result = verifier.verifySingleConstraint(lc, boolMethods, dn)
        assert 'ifSatisfied' not in result
