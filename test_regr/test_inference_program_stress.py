"""
Stress tests for issue #424.

The fix adds new hyperparameters to ``InferenceProgram`` and delegates the
Primal-Dual training loop to ``GumbelPrimalDualProgram``. These tests run
thousands of randomized configurations to make sure:

1. VALIDATION
   ``training_style`` is rejected when it's not ``'simple'`` or
   ``'primal_dual'`` — for any random string we throw at it.

2. SESSION SHAPE
   ``_init_session()`` always returns the correct keys for the chosen
   style, with ``c_update_freq`` matching whatever ``_c_freq`` was set
   to.

3. GUMBEL TEMPERATURE MATH
   For any random (initial_temp, final_temp, anneal_start, anneal_epochs,
   current_epoch) the returned temperature is within the expected range
   and matches the linear formula when annealing is active.

4. TRAIN_EPOCH DISPATCH
   ``'primal_dual'`` always routes to ``GumbelPrimalDualProgram.train_epoch``;
   ``'simple'`` always routes to ``_train_epoch_simple``. Tested across
   thousands of random configurations.

Default scale: 10,000 iterations per test. Override via
``DOMIKNOWS_INFER_STRESS_ITERS``.
"""
import os
import random
import string

import pytest

from domiknows.program.lossprogram import (
    GumbelPrimalDualProgram,
    InferenceProgram,
)


ITERS = int(os.environ.get("DOMIKNOWS_INFER_STRESS_ITERS", "10000"))
SEED = int(os.environ.get("DOMIKNOWS_INFER_STRESS_SEED", "20260421"))


def _bare_program(training_style, c_freq=None):
    """Build an InferenceProgram shell without touching graph/Model."""
    prog = object.__new__(InferenceProgram)
    prog.training_style = training_style
    if c_freq is not None:
        prog._c_freq = c_freq
    return prog


# ---------------------------------------------------------------------------


def test_stress_training_style_validation():
    """Any string not in {'simple', 'primal_dual'} must be rejected."""
    rng = random.Random(SEED)
    valid = {'simple', 'primal_dual'}
    failures = []

    for i in range(ITERS):
        n = rng.randint(1, 15)
        candidate = ''.join(rng.choices(string.ascii_letters + '_-' + string.digits, k=n))

        if candidate in valid:
            # Skip the rare collision; not what this test is about.
            continue

        try:
            InferenceProgram(graph=None, Model=None, training_style=candidate)
        except ValueError as e:
            if 'training_style' not in str(e):
                failures.append(f"iter={i} wrong ValueError message for {candidate!r}: {e}")
        except Exception as e:
            failures.append(f"iter={i} unexpected {type(e).__name__} for {candidate!r}: {e}")
        else:
            failures.append(f"iter={i} no error raised for invalid style {candidate!r}")

    assert not failures, f"{len(failures)} failure(s). First 5: {failures[:5]}"


def test_stress_init_session_shape():
    """_init_session keys/values match the chosen style in every run."""
    rng = random.Random(SEED + 1)
    failures = []

    for i in range(ITERS):
        if rng.random() < 0.5:
            # 'simple' style — always {'iter': 0}
            prog = _bare_program('simple')
            session = prog._init_session()
            if session != {'iter': 0}:
                failures.append(f"iter={i} simple style got {session}")
        else:
            # 'primal_dual' with a random c_freq (or None)
            c_freq = None if rng.random() < 0.2 else rng.randint(1, 500)
            prog = _bare_program('primal_dual', c_freq=c_freq)
            session = prog._init_session()
            expected_freq = c_freq if c_freq is not None else 10
            expected = {
                'iter': 0,
                'c_update_iter': 0,
                'c_update_freq': expected_freq,
                'c_update': 0,
            }
            if session != expected:
                failures.append(f"iter={i} PD c_freq={c_freq} got {session}")

    assert not failures, f"{len(failures)} failure(s). First 5: {failures[:5]}"


def test_stress_gumbel_temperature_schedule():
    """Linear-anneal math must hold for random schedules."""
    rng = random.Random(SEED + 2)
    failures = []

    for i in range(ITERS):
        prog = object.__new__(InferenceProgram)

        use_gumbel = rng.random() < 0.9
        initial = rng.uniform(0.2, 5.0)
        final = rng.uniform(0.01, initial)   # final <= initial
        anneal_start = rng.randint(0, 20)
        # Sometimes None to exercise "stay at final after warmup" branch
        anneal_epochs = None if rng.random() < 0.1 else rng.randint(1, 100)

        prog._init_gumbel(
            use_gumbel=use_gumbel,
            initial_temp=initial,
            final_temp=final,
            anneal_start_epoch=anneal_start,
            anneal_epochs=anneal_epochs,
            hard_gumbel=bool(rng.getrandbits(1)),
        )

        # Pick a random current epoch anywhere in a reasonable range.
        prog.current_epoch = rng.randint(0, 200)
        temp = prog.get_temperature()

        if not use_gumbel:
            if temp != 1.0:
                failures.append(f"iter={i} disabled but temp={temp}")
            continue

        # Enabled — verify the math:
        lo, hi = min(initial, final), max(initial, final)
        if not (lo - 1e-9 <= temp <= hi + 1e-9):
            failures.append(
                f"iter={i} temp {temp} outside [{lo},{hi}] "
                f"(init={initial}, final={final}, start={anneal_start}, "
                f"epochs={anneal_epochs}, current={prog.current_epoch})"
            )
            continue

        if prog.current_epoch < anneal_start:
            if abs(temp - initial) > 1e-9:
                failures.append(f"iter={i} pre-warmup temp should be {initial}, got {temp}")
        elif anneal_epochs is None:
            if abs(temp - final) > 1e-9:
                failures.append(f"iter={i} no anneal schedule should return final={final}, got {temp}")
        else:
            progress = min(1.0, max(0.0, (prog.current_epoch - anneal_start) / anneal_epochs))
            expected = initial - (initial - final) * progress
            if abs(temp - expected) > 1e-6:
                failures.append(
                    f"iter={i} expected {expected} got {temp} "
                    f"(progress={progress})"
                )

    assert not failures, f"{len(failures)} failure(s). First 5: {failures[:5]}"


def test_stress_train_epoch_dispatch(monkeypatch):
    """``train_epoch`` must route to the correct underlying loop every time."""
    rng = random.Random(SEED + 3)

    pd_calls = []
    simple_calls = []

    def fake_pd_train_epoch(self, dataset, **kwargs):
        pd_calls.append(kwargs.get('_tag'))
        yield 'pd'

    def fake_simple(self, dataset, **kwargs):
        simple_calls.append(kwargs.get('_tag'))
        yield 'simple'

    monkeypatch.setattr(GumbelPrimalDualProgram, 'train_epoch', fake_pd_train_epoch)
    monkeypatch.setattr(InferenceProgram, '_train_epoch_simple', fake_simple)

    failures = []

    for i in range(ITERS):
        style = 'simple' if rng.random() < 0.5 else 'primal_dual'
        prog = _bare_program(style)

        tag = f"t{i}"
        out = list(prog.train_epoch(dataset=['x'], _tag=tag))

        if style == 'primal_dual':
            if out != ['pd'] or tag not in pd_calls:
                failures.append(f"iter={i} PD dispatch wrong: out={out}")
        else:
            if out != ['simple'] or tag not in simple_calls:
                failures.append(f"iter={i} simple dispatch wrong: out={out}")

    assert not failures, f"{len(failures)} failure(s). First 5: {failures[:5]}"
    # Sanity: both branches were exercised.
    assert pd_calls and simple_calls, (
        f"one branch never tested: pd={len(pd_calls)} simple={len(simple_calls)}"
    )


# ---------------------------------------------------------------------------
# Heavy mode — run with DOMIKNOWS_RUN_HEAVY=1 to do a 100k sweep.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("DOMIKNOWS_RUN_HEAVY", "0") != "1",
    reason="Heavy 100k-iteration sweep. Set DOMIKNOWS_RUN_HEAVY=1 to enable.",
)
def test_stress_one_lakh_combined(monkeypatch):
    """100,000 combined configurations covering all invariants."""
    rng = random.Random(SEED + 4)
    total = 100_000

    def fake_pd(self, dataset, **kwargs):
        yield 'pd'

    def fake_simple(self, dataset, **kwargs):
        yield 'simple'

    monkeypatch.setattr(GumbelPrimalDualProgram, 'train_epoch', fake_pd)
    monkeypatch.setattr(InferenceProgram, '_train_epoch_simple', fake_simple)

    failures = []

    for i in range(total):
        style = 'simple' if rng.random() < 0.5 else 'primal_dual'
        c_freq = rng.randint(1, 100)
        prog = _bare_program(style, c_freq=c_freq)

        # session shape
        session = prog._init_session()
        if style == 'simple' and session != {'iter': 0}:
            failures.append(f"iter={i} simple session wrong")
        if style == 'primal_dual' and session.get('c_update_freq') != c_freq:
            failures.append(f"iter={i} PD c_update_freq mismatch")

        # Gumbel math (enabled)
        prog._init_gumbel(
            use_gumbel=True,
            initial_temp=rng.uniform(0.5, 3.0),
            final_temp=rng.uniform(0.01, 0.5),
            anneal_start_epoch=rng.randint(0, 5),
            anneal_epochs=rng.randint(1, 50),
            hard_gumbel=False,
        )
        prog.current_epoch = rng.randint(0, 100)
        t = prog.get_temperature()
        if not (prog.final_temp - 1e-9 <= t <= prog.initial_temp + 1e-9):
            failures.append(f"iter={i} temp out of range: {t}")

        # dispatch
        out = list(prog.train_epoch(dataset=['x']))
        expected = 'pd' if style == 'primal_dual' else 'simple'
        if out != [expected]:
            failures.append(f"iter={i} dispatch wrong: {out}")

        if len(failures) >= 10:
            break

    assert not failures, f"{len(failures)} failures in {total} iters. First 5: {failures[:5]}"
