from collections import OrderedDict

import pytest
import torch

from domiknows.program.model.lossModel import InferenceModel, LossModel


class _Recorder(dict):
    def __missing__(self, key):
        values = []
        self[key] = values
        return values.append


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


class _Constraint:
    active = True

    def strEs(self):
        return "fake"


class _Builder:
    def __init__(self, datanode):
        self.datanode = datanode
        self.created = False

    def createBatchRootDN(self):
        self.created = True

    def getDataNode(self, device="cpu"):
        self.device = device
        return self.datanode


class _DataNode:
    current_dtype = torch.float32

    def __init__(self, labels=None, global_loss=None):
        self.labels = labels or {}
        self.global_loss = global_loss or {}
        self.single_calls = []
        self.global_calls = []

    def getExecutableConstraintLabels(self):
        return self.labels

    def _prepareLcLossContext(self, **kwargs):
        self.context_kwargs = kwargs
        return {"context": True}

    def getExecutableConstraintLabel(self, lc_name):
        return self.labels[f"{lc_name}/label"]

    def calculateSingleLcLoss(self, lc_name, **kwargs):
        self.single_calls.append((lc_name, kwargs))
        return {
            "loss": torch.tensor(1.0),
            "conversionSigmoid": torch.tensor([0.8], dtype=torch.float32),
        }

    def calculateLcLoss(self, **kwargs):
        self.global_calls.append(kwargs)
        return self.global_loss


def _model(
    *,
    include_global=False,
    global_weight=1.0,
    executable_weight=1.0,
):
    model = object.__new__(InferenceModel)
    torch.nn.Module.__init__(model)
    model.build = True
    model.device = "cpu"
    model.use_gumbel = False
    model.temperature = 1.0
    model.hard_gumbel = False
    model.tnorm = "P"
    model.counting_tnorm = None
    model.sample = False
    model.sampleSize = 0
    model.sampleGlobalLoss = False
    model.include_global_constraint_loss = include_global
    model.global_constraint_loss_weight = global_weight
    model.executable_constraint_loss_weight = executable_weight
    model.loss_func = torch.nn.BCELoss()
    model.loss = _Recorder()
    model.pos_weight = 1.0
    model._diag_budget = 0
    model._diag_step = 0
    model.inferenceLogger = _Logger()
    model.constr = OrderedDict([
        ("ELC0", _Constraint()),
        ("GLC0", _Constraint()),
    ])
    model.get_lmbd_calls = []

    def get_lmbd(key):
        model.get_lmbd_calls.append(key)
        return torch.tensor(2.0)

    model.get_lmbd = get_lmbd
    return model


def test_combines_executable_bce_and_raw_global_loss():
    model = _model(include_global=True, executable_weight=3.0, global_weight=5.0)
    datanode = _DataNode(
        labels={"ELC0/label": torch.tensor([1.0])},
        global_loss={"GLC0": {"lossTensor": torch.tensor([2.0, float("nan"), -1.0])}},
    )

    loss, returned_datanode, returned_builder = model.forward(_Builder(datanode))

    executable = torch.nn.functional.binary_cross_entropy(
        torch.tensor([0.8]), torch.tensor([1.0])
    )
    expected = 3.0 * executable + 5.0 * 2.0
    assert loss.item() == pytest.approx(expected.item())
    assert returned_datanode is datanode
    assert returned_builder.datanode is datanode
    assert model.get_lmbd_calls == []
    assert datanode.single_calls[0][0] == "ELC0"
    assert len(datanode.global_calls) == 1
    assert datanode.global_calls[0]["sampleGlobalLoss"] is False


def test_global_loss_does_not_use_lambda():
    model = _model(include_global=True)
    datanode = _DataNode(
        labels={"ELC0/label": torch.tensor([1.0])},
        global_loss={"GLC0": {"lossTensor": torch.tensor([2.0])}},
    )

    loss, *_ = model.forward(_Builder(datanode))

    executable = torch.nn.functional.binary_cross_entropy(
        torch.tensor([0.8]), torch.tensor([1.0])
    )
    assert loss.item() == pytest.approx((executable + 2.0).item())
    assert model.get_lmbd_calls == []


def test_global_only_loss_works_when_enabled():
    model = _model(include_global=True)
    datanode = _DataNode(global_loss={"GLC0": {"lossTensor": torch.tensor([2.0])}})

    loss, *_ = model.forward(_Builder(datanode))

    assert loss.item() == pytest.approx(2.0)
    assert datanode.single_calls == []
    assert model.get_lmbd_calls == []


def test_no_executable_labels_still_errors_when_global_loss_disabled():
    model = _model(include_global=False)

    with pytest.raises(ValueError, match="No active executable constraint labels"):
        model.forward(_Builder(_DataNode()))


def test_lagrange_multipliers_are_nonnegative_and_projected():
    model = object.__new__(LossModel)
    torch.nn.Module.__init__(model)
    model.lmbd = torch.nn.Parameter(torch.tensor([-3.0, 7.0]))
    model.lmbd_p = torch.tensor([2.0, 4.0])
    model.lmbd_index = {"low": 0, "high": 1}

    assert model.get_lmbd("low").item() == pytest.approx(0.0)
    assert model.get_lmbd("high").item() == pytest.approx(4.0)

    model.project_lmbd_()

    assert model.lmbd.detach().tolist() == pytest.approx([0.0, 4.0])
