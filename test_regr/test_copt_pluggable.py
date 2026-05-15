"""
Regression tests for issue #422: let users pick the constraint optimizer.

Before the fix, the `copt` used inside `PrimalDualProgram`,
`InferenceProgram` and `SampleLossProgram` was hardcoded to
`torch.optim.Adam`. Now `LossProgram` accepts `copt_class` and
`copt_kwargs`, threading the choice through a small `_make_copt` helper.

The tests poke the helper directly with a dummy model so we don't need a
full training stack (which would require gurobipy and real data).
"""
import pytest
import torch

from domiknows.program.lossprogram import InferenceProgram, LossProgram


class _DummyCModel(torch.nn.Module):
    """Minimal stand-in for the constraint model."""
    def __init__(self, n=3):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(n))


def _bare_program(copt_class=None, copt_kwargs=None, params=True):
    """Build a LossProgram-like shell wired with a real cmodel but no graph."""
    prog = object.__new__(LossProgram)
    prog.cmodel = _DummyCModel() if params else torch.nn.Module()
    prog.copt = None
    prog.copt_class = copt_class if copt_class is not None else torch.optim.Adam
    prog.copt_kwargs = dict(copt_kwargs) if copt_kwargs else {}
    return prog


# ---------------------------------------------------------------------------
# _make_copt default behaviour
# ---------------------------------------------------------------------------

class TestMakeCoptDefaults:
    def test_default_is_adam(self):
        prog = _bare_program()
        copt = prog._make_copt(lr=0.05)
        assert isinstance(copt, torch.optim.Adam)
        assert copt.param_groups[0]['lr'] == pytest.approx(0.05)

    def test_none_when_no_parameters(self):
        prog = _bare_program(params=False)
        assert prog._make_copt(lr=0.05) is None

    def test_respects_custom_lr(self):
        prog = _bare_program()
        copt = prog._make_copt(lr=0.123)
        assert copt.param_groups[0]['lr'] == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# Pluggable optimizer class
# ---------------------------------------------------------------------------

class TestMakeCoptClassSwap:
    def test_swap_to_sgd(self):
        prog = _bare_program(copt_class=torch.optim.SGD)
        copt = prog._make_copt(lr=0.01)
        assert isinstance(copt, torch.optim.SGD)
        assert copt.param_groups[0]['lr'] == pytest.approx(0.01)

    def test_sgd_with_momentum_kwarg(self):
        prog = _bare_program(copt_class=torch.optim.SGD,
                             copt_kwargs={'momentum': 0.9})
        copt = prog._make_copt(lr=0.01)
        assert isinstance(copt, torch.optim.SGD)
        assert copt.param_groups[0]['momentum'] == pytest.approx(0.9)

    def test_rmsprop_is_accepted(self):
        prog = _bare_program(copt_class=torch.optim.RMSprop,
                             copt_kwargs={'alpha': 0.95})
        copt = prog._make_copt(lr=0.001)
        assert isinstance(copt, torch.optim.RMSprop)
        assert copt.param_groups[0]['alpha'] == pytest.approx(0.95)

    def test_custom_factory_class(self):
        """Any callable returning an Optimizer should be accepted."""
        class LoggingAdam(torch.optim.Adam):
            pass

        prog = _bare_program(copt_class=LoggingAdam)
        copt = prog._make_copt(lr=0.01)
        assert isinstance(copt, LoggingAdam)


# ---------------------------------------------------------------------------
# Validation on __init__
# ---------------------------------------------------------------------------

class TestLossProgramInitValidation:
    def test_lr_in_copt_kwargs_is_rejected(self):
        # Construction with lr in copt_kwargs is a clear misuse — it would
        # silently fight with c_lr at train() time.
        with pytest.raises(ValueError, match="copt_kwargs"):
            InferenceProgram(graph=None, Model=None,
                             copt_kwargs={'lr': 0.1})


# ---------------------------------------------------------------------------
# Stepping works end-to-end with the swapped optimizer
# ---------------------------------------------------------------------------

class TestSwappedOptimizerTrains:
    def test_sgd_step_updates_params(self):
        """One optimizer step should move parameters in the expected direction."""
        prog = _bare_program(copt_class=torch.optim.SGD)
        copt = prog._make_copt(lr=0.1)

        # Handcrafted loss so the gradient is exactly [1., 1., 1.]
        loss = prog.cmodel.w.sum()
        loss.backward()

        before = prog.cmodel.w.detach().clone()
        copt.step()
        after = prog.cmodel.w.detach().clone()

        # w -= lr * grad  =>  after - before == -0.1 for each coord
        assert torch.allclose(after - before, torch.full_like(before, -0.1))
