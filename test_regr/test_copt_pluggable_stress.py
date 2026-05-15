"""
Stress tests for issue #422 — pluggable constraint optimizer.

For 10,000 random configurations we verify:

1. ``_make_copt`` always produces an optimizer of the requested class with
   the requested learning rate.
2. Extra keyword arguments passed through ``copt_kwargs`` land on the
   optimizer's param_group.
3. Passing ``'lr'`` inside ``copt_kwargs`` is always rejected at
   ``LossProgram.__init__`` time.

Default: 10,000 iters per test. Override via
``DOMIKNOWS_COPT_STRESS_ITERS``.
"""
import os
import random

import pytest
import torch

from domiknows.program.lossprogram import InferenceProgram, LossProgram


ITERS = int(os.environ.get("DOMIKNOWS_COPT_STRESS_ITERS", "10000"))
SEED = int(os.environ.get("DOMIKNOWS_COPT_STRESS_SEED", "20260421"))


class _DummyCModel(torch.nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(n))


def _bare_program(copt_class, copt_kwargs):
    prog = object.__new__(LossProgram)
    prog.cmodel = _DummyCModel()
    prog.copt = None
    prog.copt_class = copt_class
    prog.copt_kwargs = dict(copt_kwargs) if copt_kwargs else {}
    return prog


# Candidate optimizer classes with an associated "extra kwarg" we can probe.
_OPTIMIZER_CHOICES = [
    (torch.optim.Adam, {}, None),
    (torch.optim.Adam, {'betas': (0.9, 0.999)}, ('betas', (0.9, 0.999))),
    (torch.optim.Adam, {'weight_decay': 0.01}, ('weight_decay', 0.01)),
    (torch.optim.SGD, {}, None),
    (torch.optim.SGD, {'momentum': 0.9}, ('momentum', 0.9)),
    (torch.optim.SGD, {'weight_decay': 0.0005}, ('weight_decay', 0.0005)),
    (torch.optim.RMSprop, {'alpha': 0.95}, ('alpha', 0.95)),
    (torch.optim.AdamW, {'weight_decay': 0.1}, ('weight_decay', 0.1)),
]


def test_stress_make_copt_class_and_lr():
    rng = random.Random(SEED)
    failures = []

    for i in range(ITERS):
        cls, extra, _ = rng.choice(_OPTIMIZER_CHOICES)
        lr = rng.uniform(1e-5, 1.0)

        prog = _bare_program(copt_class=cls, copt_kwargs=extra)
        copt = prog._make_copt(lr=lr)

        if not isinstance(copt, cls):
            failures.append(f"iter={i} wrong class: want {cls.__name__} got {type(copt).__name__}")
            continue
        if abs(copt.param_groups[0]['lr'] - lr) > 1e-9:
            failures.append(f"iter={i} wrong lr: want {lr} got {copt.param_groups[0]['lr']}")

    assert not failures, f"{len(failures)} failure(s). First 5: {failures[:5]}"


def test_stress_copt_kwargs_threaded_through():
    rng = random.Random(SEED + 1)
    failures = 0
    checked = 0

    # Restrict to the entries that have an "extra kwarg" to inspect.
    with_extras = [c for c in _OPTIMIZER_CHOICES if c[2] is not None]

    for i in range(ITERS):
        cls, extra, (key, expected) = rng.choice(with_extras)
        prog = _bare_program(copt_class=cls, copt_kwargs=extra)
        copt = prog._make_copt(lr=rng.uniform(1e-4, 0.5))

        actual = copt.param_groups[0].get(key)
        if actual != expected:
            failures += 1
            if failures <= 5:
                print(f"iter={i} {cls.__name__} kwarg {key}: want {expected} got {actual}")
        checked += 1

    assert failures == 0, f"{failures}/{checked} configurations mismatched"


def test_stress_lr_in_copt_kwargs_always_rejected():
    """Any random lr value smuggled into copt_kwargs must fail fast."""
    rng = random.Random(SEED + 2)
    failures = []

    for i in range(ITERS):
        lr_value = rng.uniform(-1.0, 1.0)
        # Occasionally add extra innocuous kwargs alongside lr.
        extras = {'lr': lr_value}
        if rng.random() < 0.5:
            extras['weight_decay'] = rng.uniform(0, 0.1)

        try:
            InferenceProgram(graph=None, Model=None, copt_kwargs=extras)
        except ValueError as e:
            if 'copt_kwargs' not in str(e):
                failures.append(f"iter={i} wrong ValueError message: {e}")
        except Exception as e:
            failures.append(f"iter={i} unexpected {type(e).__name__}: {e}")
        else:
            failures.append(f"iter={i} no error raised for extras={extras}")

    assert not failures, f"{len(failures)} failure(s). First 5: {failures[:5]}"


# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("DOMIKNOWS_RUN_HEAVY", "0") != "1",
    reason="Heavy 100k-iteration sweep. Set DOMIKNOWS_RUN_HEAVY=1 to enable.",
)
def test_stress_one_lakh_combined():
    rng = random.Random(SEED + 3)
    total = 100_000
    failures = 0

    for i in range(total):
        cls, extra, probe = rng.choice(_OPTIMIZER_CHOICES)
        lr = rng.uniform(1e-5, 1.0)
        prog = _bare_program(copt_class=cls, copt_kwargs=extra)
        copt = prog._make_copt(lr=lr)

        if not isinstance(copt, cls):
            failures += 1
        elif abs(copt.param_groups[0]['lr'] - lr) > 1e-9:
            failures += 1
        elif probe is not None:
            key, expected = probe
            if copt.param_groups[0].get(key) != expected:
                failures += 1

        if failures >= 5:
            break

    assert failures == 0, f"{failures} failures in {total} iterations"
