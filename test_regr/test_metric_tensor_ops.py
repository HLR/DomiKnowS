"""
Tests for tensor-based metric calculations (issue #315).

Verifies that the optimized tensor operations in metric.py produce the same
results as the old Python-loop approach, and benchmarks the performance gain.
"""
import time
import pytest
import torch
import numpy as np

from domiknows.program.metric import (
    _aggregate_cm_value,
    CMWithLogitsMetric,
    PRF1Tracker,
    MacroAverageTracker,
    MetricTracker,
)
from domiknows.utils import wrap_batch


# ---------------------------------------------------------------------------
# Helpers: old (pre-#315) implementations for correctness comparison
# ---------------------------------------------------------------------------

def _old_aggregate_and_compute_prf1(values):
    """Original PRF1 binary-class logic using Python sum() and isinstance checks."""
    CM = wrap_batch(values)

    if isinstance(CM['TP'], list):
        tp = sum(CM['TP'])
    else:
        tp = CM['TP'].sum().float()
    if isinstance(CM['FP'], list):
        fp = sum(CM['FP'])
    else:
        fp = CM['FP'].sum().float()
    if isinstance(CM['FN'], list):
        fn = sum(CM['FN'])
    else:
        fn = CM['FN'].sum().float()
    if isinstance(CM['TN'], list):
        tn = sum(CM['TN'])
    else:
        tn = CM['TN'].sum().float()

    if not torch.is_tensor(tp):
        tp = torch.tensor(tp)
    if not torch.is_tensor(fp):
        fp = torch.tensor(fp)
    if not torch.is_tensor(fn):
        fn = torch.tensor(fn)
    if not torch.is_tensor(tn):
        tn = torch.tensor(tn)

    if tp:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    else:
        p = torch.zeros_like(tp)
        r = torch.zeros_like(tp)
        f1 = torch.zeros_like(tp)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else torch.zeros_like(tp)
    return {'P': p, 'R': r, 'F1': f1, 'accuracy': accuracy}


def _new_aggregate_and_compute_prf1(values):
    """New PRF1 binary-class logic using _aggregate_cm_value (tensor ops)."""
    CM = wrap_batch(values)
    tp = _aggregate_cm_value(CM['TP'])
    fp = _aggregate_cm_value(CM['FP'])
    fn = _aggregate_cm_value(CM['FN'])
    tn = _aggregate_cm_value(CM['TN'])

    if tp:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    else:
        p = torch.zeros_like(tp)
        r = torch.zeros_like(tp)
        f1 = torch.zeros_like(tp)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else torch.zeros_like(tp)
    return {'P': p, 'R': r, 'F1': f1, 'accuracy': accuracy}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cm_batch(n, device='cpu'):
    """Generate n confusion-matrix dicts with random tensor values."""
    batch = []
    for _ in range(n):
        tp = torch.randint(0, 50, (1,), device=device).float().squeeze()
        fp = torch.randint(0, 50, (1,), device=device).float().squeeze()
        fn = torch.randint(0, 50, (1,), device=device).float().squeeze()
        tn = torch.randint(0, 50, (1,), device=device).float().squeeze()
        batch.append({'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn})
    return batch


def _make_cm_batch_numpy(n):
    """Generate n confusion-matrix dicts with numpy int values (DatanodeCMMetric style)."""
    batch = []
    for _ in range(n):
        batch.append({
            'TP': np.random.randint(0, 50),
            'FP': np.random.randint(0, 50),
            'TN': np.random.randint(0, 50),
            'FN': np.random.randint(0, 50),
        })
    return batch


# ---------------------------------------------------------------------------
# Tests: _aggregate_cm_value
# ---------------------------------------------------------------------------

class TestAggregateCMValue:
    def test_tensor_input(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _aggregate_cm_value(t)
        assert torch.isclose(result, torch.tensor(6.0))
        assert result.dtype == torch.float32

    def test_list_of_tensors(self):
        vals = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = _aggregate_cm_value(vals)
        assert torch.isclose(result, torch.tensor(6.0))

    def test_list_of_ints(self):
        result = _aggregate_cm_value([1, 2, 3])
        assert torch.isclose(result, torch.tensor(6.0))

    def test_list_of_numpy(self):
        result = _aggregate_cm_value([np.int64(5), np.int64(10)])
        assert torch.isclose(result, torch.tensor(15.0))

    def test_scalar_int(self):
        result = _aggregate_cm_value(7)
        assert torch.isclose(result, torch.tensor(7.0))

    def test_empty_tensor(self):
        result = _aggregate_cm_value(torch.tensor([]))
        assert torch.isclose(result, torch.tensor(0.0))

    def test_single_element_list(self):
        result = _aggregate_cm_value([torch.tensor(42.0)])
        assert torch.isclose(result, torch.tensor(42.0))


# ---------------------------------------------------------------------------
# Tests: PRF1Tracker correctness (old vs new give same results)
# ---------------------------------------------------------------------------

class TestPRF1TrackerCorrectness:
    def test_tensor_batch_same_results(self):
        torch.manual_seed(42)
        values = _make_cm_batch(100)
        old = _old_aggregate_and_compute_prf1(values)
        new = _new_aggregate_and_compute_prf1(values)
        for key in ('P', 'R', 'F1', 'accuracy'):
            assert torch.isclose(old[key].float(), new[key].float(), atol=1e-6), \
                f"Mismatch on {key}: old={old[key]}, new={new[key]}"

    def test_numpy_batch_same_results(self):
        np.random.seed(42)
        values = _make_cm_batch_numpy(100)
        old = _old_aggregate_and_compute_prf1(values)
        new = _new_aggregate_and_compute_prf1(values)
        for key in ('P', 'R', 'F1', 'accuracy'):
            assert torch.isclose(old[key].float(), new[key].float(), atol=1e-6), \
                f"Mismatch on {key}: old={old[key]}, new={new[key]}"

    def test_all_zeros(self):
        """Edge case: all confusion matrix values are zero."""
        values = [{'TP': torch.tensor(0.), 'FP': torch.tensor(0.),
                    'TN': torch.tensor(0.), 'FN': torch.tensor(0.)}]
        result = _new_aggregate_and_compute_prf1(values)
        assert result['P'].item() == 0.0
        assert result['R'].item() == 0.0
        assert result['F1'].item() == 0.0
        assert result['accuracy'].item() == 0.0

    def test_perfect_predictions(self):
        """All predictions correct -> P=R=F1=accuracy=1."""
        values = [{'TP': torch.tensor(50.), 'FP': torch.tensor(0.),
                    'TN': torch.tensor(50.), 'FN': torch.tensor(0.)}]
        result = _new_aggregate_and_compute_prf1(values)
        assert torch.isclose(result['P'], torch.tensor(1.0))
        assert torch.isclose(result['R'], torch.tensor(1.0))
        assert torch.isclose(result['F1'], torch.tensor(1.0))
        assert torch.isclose(result['accuracy'], torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Tests: MetricTracker._to_python_scalars
# ---------------------------------------------------------------------------

class TestToScalars:
    def test_flat_dict(self):
        d = {'a': torch.tensor(1.5), 'b': torch.tensor(2.5)}
        result = MetricTracker._to_python_scalars(d)
        assert result == {'a': 1.5, 'b': 2.5}
        assert isinstance(result['a'], float)

    def test_nested_dict(self):
        d = {'outer': {'inner': torch.tensor(3.0), 'plain': 42}}
        result = MetricTracker._to_python_scalars(d)
        assert result == {'outer': {'inner': 3.0, 'plain': 42}}

    def test_non_tensor_passthrough(self):
        assert MetricTracker._to_python_scalars('hello') == 'hello'
        assert MetricTracker._to_python_scalars(42) == 42


# ---------------------------------------------------------------------------
# Tests: MacroAverageTracker with torch.as_tensor
# ---------------------------------------------------------------------------

class TestMacroAverageTracker:
    def test_tensor_input(self):
        tracker = MacroAverageTracker(metric=torch.nn.Identity())
        result = tracker.forward(torch.tensor([1.0, 2.0, 3.0]))
        assert torch.isclose(result, torch.tensor(2.0))

    def test_dict_input(self):
        tracker = MacroAverageTracker(metric=torch.nn.Identity())
        result = tracker.forward({'a': torch.tensor([1.0, 3.0]), 'b': torch.tensor([2.0, 4.0])})
        assert torch.isclose(result['a'], torch.tensor(2.0))
        assert torch.isclose(result['b'], torch.tensor(3.0))

    def test_list_input(self):
        tracker = MacroAverageTracker(metric=torch.nn.Identity())
        result = tracker.forward([1.0, 2.0, 3.0])
        assert torch.isclose(result, torch.tensor(2.0))


# ---------------------------------------------------------------------------
# Benchmark: old vs new approach (marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_prf1_tensor_ops_no_regression():
    """Verify new tensor ops have no significant performance regression vs old Python loops."""
    torch.manual_seed(0)
    values = _make_cm_batch(1000)

    # Warm up
    _old_aggregate_and_compute_prf1(values)
    _new_aggregate_and_compute_prf1(values)

    n_runs = 50

    start = time.perf_counter()
    for _ in range(n_runs):
        _old_aggregate_and_compute_prf1(values)
    old_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_runs):
        _new_aggregate_and_compute_prf1(values)
    new_time = time.perf_counter() - start

    print(f"\nBenchmark (n=1000 batches, {n_runs} runs):")
    print(f"  Old (Python sum):    {old_time:.4f}s")
    print(f"  New (tensor ops):    {new_time:.4f}s")
    print(f"  Ratio:               {old_time / new_time:.2f}x")

    # For the common tensor path (CMWithLogitsMetric), both approaches call
    # .sum().float() so performance is equivalent. The new approach adds a
    # thin _aggregate_cm_value wrapper, which should not cause meaningful
    # regression (<2x slower at worst on any platform).
    assert new_time < old_time * 2.0, \
        f"New approach unexpectedly slower: {new_time:.4f}s vs {old_time:.4f}s"
