"""
Regression tests for issue #424:
``InferenceProgram`` gains the ``training_style`` hyperparameter (plus Gumbel
annealing knobs) so it can run the Primal-Dual algorithm — and optionally
Gumbel-Softmax — without subclassing / swapping its base class.

These tests focus on the *public surface* of the change: hyperparameter
validation, session shape, class hierarchy, and backward compatibility.
They deliberately avoid running a full training loop (which would need a
real graph + Model + gurobipy) — the loop itself is already delegated to
``GumbelPrimalDualProgram.train_epoch`` which is covered by existing PMD
tests.
"""
import inspect

import pytest
import torch

from domiknows.program.lossprogram import (
    GumbelPrimalDualProgram,
    GumbelTemperatureMixin,
    InferenceProgram,
    LossProgram,
    PrimalDualProgram,
)


# ---------------------------------------------------------------------------
# Hierarchy / MRO
# ---------------------------------------------------------------------------

class TestInferenceProgramHierarchy:
    def test_inference_program_mixes_in_gumbel_temperature(self):
        # After the fix, Gumbel capabilities are available directly on
        # InferenceProgram without having to switch classes.
        assert issubclass(InferenceProgram, GumbelTemperatureMixin)
        assert issubclass(InferenceProgram, LossProgram)

    def test_primal_dual_program_hierarchy_unchanged(self):
        # We should not have accidentally altered PrimalDualProgram.
        assert issubclass(PrimalDualProgram, LossProgram)
        assert issubclass(GumbelPrimalDualProgram, PrimalDualProgram)
        assert issubclass(GumbelPrimalDualProgram, GumbelTemperatureMixin)

    def test_gumbel_methods_exposed(self):
        for name in ('_init_gumbel', 'get_temperature',
                     '_auto_set_anneal_epochs', '_call_cmodel_with_gumbel'):
            assert hasattr(InferenceProgram, name), f"missing {name}"


# ---------------------------------------------------------------------------
# Signature / hyperparameter validation
# ---------------------------------------------------------------------------

class TestInferenceProgramSignature:
    def test_new_hyperparameters_in_signature(self):
        sig = inspect.signature(InferenceProgram.__init__)
        params = sig.parameters
        for name in ('training_style', 'use_gumbel',
                     'initial_temp', 'final_temp',
                     'anneal_start_epoch', 'anneal_epochs', 'hard_gumbel',
                     'include_global_constraint_loss',
                     'global_constraint_loss_weight',
                     'executable_constraint_loss_weight'):
            assert name in params, f"InferenceProgram.__init__ missing {name}"

    def test_defaults_preserve_backward_compat(self):
        sig = inspect.signature(InferenceProgram.__init__)
        assert sig.parameters['training_style'].default == 'default'
        assert sig.parameters['use_gumbel'].default is False
        assert sig.parameters['include_global_constraint_loss'].default is False
        # If these defaults change, existing users will see different behaviour.

    def test_primal_dual_style_defaults_global_loss_off(self, monkeypatch):
        captured = {}

        def fake_init(self, graph, Model, CModel=None, beta=1, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(LossProgram, '__init__', fake_init)
        InferenceProgram(graph=None, Model=None, training_style='primal_dual')

        assert captured['include_global_constraint_loss'] is False

    def test_default_style_defaults_global_loss_off(self, monkeypatch):
        captured = {}

        def fake_init(self, graph, Model, CModel=None, beta=1, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(LossProgram, '__init__', fake_init)
        InferenceProgram(graph=None, Model=None, training_style='default')

        assert captured['include_global_constraint_loss'] is False

    def test_simple_style_alias_maps_to_default(self, monkeypatch):
        def fake_init(self, graph, Model, CModel=None, beta=1, **kwargs):
            pass

        monkeypatch.setattr(LossProgram, '__init__', fake_init)
        program = InferenceProgram(graph=None, Model=None, training_style='simple')

        assert program.training_style == 'default'

    def test_explicit_global_loss_true_enables_global_loss(self, monkeypatch):
        captured = {}

        def fake_init(self, graph, Model, CModel=None, beta=1, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(LossProgram, '__init__', fake_init)
        InferenceProgram(
            graph=None,
            Model=None,
            training_style='primal_dual',
            include_global_constraint_loss=True,
        )

        assert captured['include_global_constraint_loss'] is True

    def test_invalid_training_style_rejected_before_construction(self):
        # The validation runs before the heavy super().__init__, so we can
        # trigger it without a real graph/Model.
        with pytest.raises(ValueError, match="training_style"):
            InferenceProgram(
                graph=None, Model=None, training_style='bogus',
            )


# ---------------------------------------------------------------------------
# _init_session shape depends on training_style
# ---------------------------------------------------------------------------

def _bare_program(training_style='default', c_freq=None):
    """Build a stripped-down InferenceProgram without touching graph/Model.

    We only need the attributes ``_init_session`` looks at.
    """
    prog = object.__new__(InferenceProgram)
    prog.training_style = training_style
    if c_freq is not None:
        prog._c_freq = c_freq
    return prog


class TestInitSessionShape:
    def test_default_style_session(self):
        prog = _bare_program('default')
        session = prog._init_session()
        assert session == {'iter': 0}

    def test_primal_dual_style_session_defaults(self):
        prog = _bare_program('primal_dual')
        session = prog._init_session()
        # When c_freq hasn't been set by train(), default of 10 is used.
        assert session == {
            'iter': 0,
            'c_update_iter': 0,
            'c_update_freq': 10,
            'c_update': 0,
        }

    def test_primal_dual_session_uses_configured_c_freq(self):
        prog = _bare_program('primal_dual', c_freq=25)
        session = prog._init_session()
        assert session['c_update_freq'] == 25
        assert session['iter'] == 0
        assert session['c_update_iter'] == 0
        assert session['c_update'] == 0


# ---------------------------------------------------------------------------
# Gumbel init wiring
# ---------------------------------------------------------------------------

class TestGumbelInit:
    def test_init_gumbel_stores_schedule(self):
        prog = object.__new__(InferenceProgram)
        prog._init_gumbel(
            use_gumbel=True,
            initial_temp=2.0,
            final_temp=0.5,
            anneal_start_epoch=1,
            anneal_epochs=10,
            hard_gumbel=True,
        )
        assert prog.use_gumbel is True
        assert prog.initial_temp == 2.0
        assert prog.final_temp == 0.5
        assert prog.anneal_start_epoch == 1
        assert prog.anneal_epochs == 10
        assert prog.hard_gumbel is True
        assert prog.current_epoch == 0
        assert prog.current_temp == 2.0

    def test_get_temperature_linear_annealing(self):
        prog = object.__new__(InferenceProgram)
        prog._init_gumbel(
            use_gumbel=True, initial_temp=1.0, final_temp=0.0,
            anneal_start_epoch=0, anneal_epochs=10, hard_gumbel=False,
        )
        # start
        assert prog.get_temperature() == pytest.approx(1.0)
        # halfway
        prog.current_epoch = 5
        assert prog.get_temperature() == pytest.approx(0.5)
        # end
        prog.current_epoch = 10
        assert prog.get_temperature() == pytest.approx(0.0)
        # past end — still clamped to final_temp
        prog.current_epoch = 50
        assert prog.get_temperature() == pytest.approx(0.0)

    def test_get_temperature_noop_when_disabled(self):
        prog = object.__new__(InferenceProgram)
        prog._init_gumbel(
            use_gumbel=False, initial_temp=1.0, final_temp=0.1,
            anneal_start_epoch=0, anneal_epochs=10, hard_gumbel=False,
        )
        prog.current_epoch = 5
        assert prog.get_temperature() == 1.0

    def test_auto_set_anneal_epochs_only_when_unspecified(self):
        prog = object.__new__(InferenceProgram)
        prog._init_gumbel(
            use_gumbel=True, initial_temp=1.0, final_temp=0.1,
            anneal_start_epoch=0, anneal_epochs=None, hard_gumbel=False,
        )
        prog._auto_set_anneal_epochs(num_epochs=20)
        assert prog.anneal_epochs == 20

        # Should not overwrite an explicit value.
        prog._auto_set_anneal_epochs(num_epochs=5)
        assert prog.anneal_epochs == 20

    def test_train_uses_train_epoch_num_for_gumbel_annealing(self, monkeypatch):
        class EmptyCModel:
            def parameters(self):
                return []

        def fake_train(self, training_set, valid_set=None, test_set=None, **kwargs):
            return kwargs

        monkeypatch.setattr(LossProgram, 'train', fake_train)

        prog = object.__new__(InferenceProgram)
        prog.training_style = 'default'
        prog.cmodel = EmptyCModel()
        prog._make_copt = lambda _lr: None
        prog._init_gumbel(
            use_gumbel=True, initial_temp=1.0, final_temp=0.1,
            anneal_start_epoch=0, anneal_epochs=None, hard_gumbel=False,
        )

        result = InferenceProgram.train(
            prog,
            training_set=[],
            train_epoch_num=9,
        )

        assert prog.anneal_epochs == 9
        assert result['train_epoch_num'] == 9


# ---------------------------------------------------------------------------
# train_epoch dispatch — we patch the PD delegate and verify routing.
# ---------------------------------------------------------------------------

class TestTrainEpochDispatch:
    def test_primal_dual_style_without_gumbel_delegates_to_basic_pd(self, monkeypatch):
        calls = []

        def fake_basic_pd_train_epoch(self, dataset, **kwargs):
            calls.append(('basic_pd', dataset, kwargs))
            yield  # must be a generator

        def fake_gumbel_pd_train_epoch(self, dataset, **kwargs):
            calls.append(('gumbel_pd', dataset, kwargs))
            yield  # must be a generator

        monkeypatch.setattr(
            PrimalDualProgram, 'train_epoch', fake_basic_pd_train_epoch,
        )
        monkeypatch.setattr(
            GumbelPrimalDualProgram, 'train_epoch', fake_gumbel_pd_train_epoch,
        )

        prog = object.__new__(InferenceProgram)
        prog.training_style = 'primal_dual'
        prog.use_gumbel = False

        gen = prog.train_epoch(dataset=['x'], c_session={'iter': 0}, batch_size=1)
        list(gen)  # drain

        assert len(calls) == 1
        tag, dataset, kwargs = calls[0]
        assert tag == 'basic_pd'
        assert dataset == ['x']
        assert kwargs['c_session'] == {'iter': 0}
        assert kwargs['batch_size'] == 1

    def test_primal_dual_style_with_gumbel_delegates_to_gumbel_pd(self, monkeypatch):
        calls = []

        def fake_basic_pd_train_epoch(self, dataset, **kwargs):
            calls.append(('basic_pd', dataset, kwargs))
            yield

        def fake_gumbel_pd_train_epoch(self, dataset, **kwargs):
            calls.append(('gumbel_pd', dataset, kwargs))
            yield

        monkeypatch.setattr(
            PrimalDualProgram, 'train_epoch', fake_basic_pd_train_epoch,
        )
        monkeypatch.setattr(
            GumbelPrimalDualProgram, 'train_epoch', fake_gumbel_pd_train_epoch,
        )

        prog = object.__new__(InferenceProgram)
        prog.training_style = 'primal_dual'
        prog.use_gumbel = True

        gen = prog.train_epoch(dataset=['x'], c_session={'iter': 0}, batch_size=1)
        list(gen)

        assert len(calls) == 1
        tag, dataset, kwargs = calls[0]
        assert tag == 'gumbel_pd'
        assert dataset == ['x']
        assert kwargs['c_session'] == {'iter': 0}
        assert kwargs['batch_size'] == 1

    def test_default_style_does_not_call_pd(self, monkeypatch):
        called = []

        def fake_pd_train_epoch(self, dataset, **kwargs):
            called.append(True)
            yield

        monkeypatch.setattr(
            GumbelPrimalDualProgram, 'train_epoch', fake_pd_train_epoch,
        )

        # Replace the default helper with a tiny stub so we don't need a
        # real model. We only care that PD was NOT chosen.
        def fake_simple(self, dataset, **kwargs):
            yield 'simple-result'

        monkeypatch.setattr(InferenceProgram, '_train_epoch_simple', fake_simple)

        prog = object.__new__(InferenceProgram)
        prog.training_style = 'default'

        out = list(prog.train_epoch(dataset=['x']))
        assert out == ['simple-result']
        assert called == []

    def test_default_epoch_yields_once_per_data_item(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def mode(self, _mode):
                pass

            def train(self):
                pass

            def reset(self):
                pass

            def parameters(self):
                return []

            def __call__(self, data):
                self.calls.append(data)
                return torch.tensor(1.0), f"metric-{data}", f"datanode-{data}", f"builder-{data}"

        class FakeCModel:
            def train(self):
                pass

            def reset(self):
                pass

            def parameters(self):
                return []

            def __call__(self, _builder):
                return torch.tensor(0.5), None, None

        prog = object.__new__(InferenceProgram)
        prog.model = FakeModel()
        prog.cmodel = FakeCModel()
        prog.opt = None
        prog.copt = None
        prog.beta = 2.0
        prog.use_gumbel = False
        prog.current_epoch = 0

        session = {'iter': 0}
        out = list(prog._train_epoch_simple([1, 2], c_session=session))

        assert len(out) == 2
        assert [row[1] for row in out] == ["metric-1", "metric-2"]
        assert [row[2] for row in out] == ["datanode-1", "datanode-2"]
        assert prog.current_epoch == 1
