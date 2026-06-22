from types import SimpleNamespace

from domiknows.program.lossprogram import LossProgram, PrimalDualProgram
from domiknows.program.program import LearningBasedProgram


def test_persistent_c_session_reuses_iter_across_train_calls(monkeypatch):
    calls = []

    def fake_train(self, training_set, **kwargs):
        session = kwargs["c_session"]
        calls.append(session)
        session["iter"] += len(training_set)
        return session["iter"]

    monkeypatch.setattr(LearningBasedProgram, "train", fake_train)

    program = object.__new__(LossProgram)
    program._init_session = lambda: {"iter": 0}

    assert LossProgram.train(program, [1, 2], persist_c_session=True) == 2
    assert LossProgram.train(program, [3, 4, 5], persist_c_session=True) == 5

    assert calls[0] is calls[1]
    assert program._persistent_c_session["iter"] == 5


def test_non_persistent_c_session_starts_fresh_each_train_call(monkeypatch):
    calls = []

    def fake_train(self, training_set, **kwargs):
        session = kwargs["c_session"]
        calls.append(session)
        session["iter"] += len(training_set)
        return session["iter"]

    monkeypatch.setattr(LearningBasedProgram, "train", fake_train)

    program = object.__new__(LossProgram)
    program._init_session = lambda: {"iter": 0}

    assert LossProgram.train(program, [1, 2]) == 2
    assert LossProgram.train(program, [3, 4, 5]) == 3

    assert calls[0] is not calls[1]
    assert calls[0]["iter"] == 2
    assert calls[1]["iter"] == 3


def test_reset_persistent_c_session_restarts_iter(monkeypatch):
    def fake_train(self, training_set, **kwargs):
        session = kwargs["c_session"]
        session["iter"] += len(training_set)
        return session["iter"]

    monkeypatch.setattr(LearningBasedProgram, "train", fake_train)

    program = object.__new__(LossProgram)
    program._init_session = lambda: {"iter": 0}

    assert LossProgram.train(program, [1, 2], persist_c_session=True) == 2
    assert LossProgram.train(program, [3], persist_c_session=True, reset_c_session=True) == 1
    assert program._persistent_c_session["iter"] == 1


def test_primal_dual_train_forwards_persistent_session_flags(monkeypatch):
    calls = []

    def fake_loss_train(self, training_set, **kwargs):
        calls.append((training_set, kwargs))
        return "trained"

    monkeypatch.setattr(LossProgram, "train", fake_loss_train)

    program = object.__new__(PrimalDualProgram)
    program._make_copt = lambda lr: None

    result = PrimalDualProgram.train(
        program,
        [1],
        persist_c_session=True,
        reset_c_session=True,
    )

    assert result == "trained"
    assert calls[0][0] == [1]
    assert calls[0][1]["persist_c_session"] is True
    assert calls[0][1]["reset_c_session"] is True


def test_persistent_call_epoch_uses_cumulative_tqdm_initial_and_total(monkeypatch):
    tqdm_calls = []
    epoch_kwargs = []

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    def fake_epoch(dataset, **kwargs):
        epoch_kwargs.append(kwargs)
        yield from dataset

    monkeypatch.setattr("domiknows.program.lossprogram.tqdm", fake_tqdm)

    program = object.__new__(LossProgram)
    program.epoch = 1
    program.model = SimpleNamespace(loss=None, metric=None)
    program.cmodel = SimpleNamespace(loss=None)

    LossProgram.call_epoch(
        program,
        "Training",
        [1, 2, 3, 4],
        fake_epoch,
        c_session={"iter": 4},
        _persist_c_session_tqdm=True,
    )

    assert tqdm_calls == [
        {"total": 8, "desc": "Epoch 1 Training", "initial": 4}
    ]
    assert epoch_kwargs == [{"c_session": {"iter": 4, "_epoch": 1}}]


def test_persistent_call_epoch_uses_cumulative_epoch_description(monkeypatch):
    tqdm_calls = []

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    def fake_epoch(dataset, **kwargs):
        session = kwargs["c_session"]
        session["iter"] += len(dataset)
        yield from dataset

    monkeypatch.setattr("domiknows.program.lossprogram.tqdm", fake_tqdm)

    program = object.__new__(LossProgram)
    program.epoch = 1
    program.model = SimpleNamespace(loss=None, metric=None)
    program.cmodel = SimpleNamespace(loss=None)
    c_session = {"iter": 0}

    LossProgram.call_epoch(
        program,
        "Training",
        [1, 2, 3, 4],
        fake_epoch,
        c_session=c_session,
        _persist_c_session_tqdm=True,
    )
    LossProgram.call_epoch(
        program,
        "Training",
        [5, 6, 7, 8],
        fake_epoch,
        c_session=c_session,
        _persist_c_session_tqdm=True,
    )

    assert tqdm_calls == [
        {"total": 4, "desc": "Epoch 1 Training", "initial": 0},
        {"total": 8, "desc": "Epoch 2 Training", "initial": 4},
    ]


def test_non_persistent_call_epoch_uses_local_tqdm_total(monkeypatch):
    tqdm_calls = []

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    def fake_epoch(dataset, **kwargs):
        yield from dataset

    monkeypatch.setattr("domiknows.program.lossprogram.tqdm", fake_tqdm)

    program = object.__new__(LossProgram)
    program.epoch = 1
    program.model = SimpleNamespace(loss=None, metric=None)
    program.cmodel = SimpleNamespace(loss=None)

    LossProgram.call_epoch(
        program,
        "Training",
        [1, 2, 3, 4],
        fake_epoch,
        c_session={"iter": 4},
    )

    assert tqdm_calls == [
        {"total": 4, "desc": "Epoch 1 Training"}
    ]
