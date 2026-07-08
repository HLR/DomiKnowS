from collections import OrderedDict

import pytest
import torch

from domiknows.graph.logicalConstrain import queryL, sumL
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.model.lossModel import InferenceModel


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


class _Constraint:
    active = True

    def __init__(self, inner_lc=None):
        self.innerLC = inner_lc

    def strEs(self):
        return "fake executable"


class _Builder:
    def __init__(self, datanode):
        self.datanode = datanode

    def createBatchRootDN(self):
        pass

    def getDataNode(self, device="cpu"):
        return self.datanode


class _DataNode:
    current_dtype = torch.float32

    def __init__(self, label, loss_dict):
        self.label = label
        self.loss_dict = loss_dict
        self.infer_local_calls = []
        self.infer_gumbel_calls = []

    def inferLocal(self, *args, **kwargs):
        self.infer_local_calls.append((args, kwargs))

    def inferGumbelLocal(self, *args, **kwargs):
        self.infer_gumbel_calls.append((args, kwargs))

    def getExecutableConstraintLabels(self):
        return {"ELC0/label": self.label}

    def _prepareLcLossContext(self, **_kwargs):
        return {}

    def getExecutableConstraintLabel(self, _lc_name):
        return self.label

    def calculateSingleLcLoss(self, _lc_name, **_kwargs):
        return self.loss_dict


def _inference_model(constraint, loss_func):
    model = object.__new__(InferenceModel)
    torch.nn.Module.__init__(model)
    model.build = True
    model.device = "cpu"
    model.use_gumbel = False
    model.temperature = 1.0
    model.hard_gumbel = False
    model.tnorm = "P"
    model.counting_tnorm = None
    model.include_global_constraint_loss = False
    model.global_constraint_loss_weight = 1.0
    model.executable_constraint_loss_weight = 1.0
    model.loss_func = loss_func
    model.loss = {}
    model.pos_weight = 1.0
    model._diag_budget = 0
    model._diag_step = 0
    model.inferenceLogger = _Logger()
    model.constr = OrderedDict([("ELC0", constraint)])
    return model


@pytest.mark.parametrize("use_gumbel", [False, True])
def test_boolean_executable_constraints_work_with_and_without_gumbel(use_gumbel):
    datanode = _DataNode(
        label=torch.tensor([1.0]),
        loss_dict={
            "loss": torch.tensor(0.2),
            "conversionSigmoid": torch.tensor([0.8], dtype=torch.float32),
        },
    )
    model = _inference_model(_Constraint(), torch.nn.BCELoss())

    loss, *_ = model.forward(
        _Builder(datanode),
        use_gumbel=use_gumbel,
        temperature=0.7,
        hard_gumbel=True,
    )

    expected = torch.nn.functional.binary_cross_entropy(
        torch.tensor([0.8]), torch.tensor([1.0])
    )
    assert loss.item() == pytest.approx(expected.item())
    assert bool(datanode.infer_gumbel_calls) is use_gumbel


@pytest.mark.parametrize("use_gumbel", [False, True])
def test_suml_executable_constraints_work_with_wrappers_and_gumbel(use_gumbel):
    datanode = _DataNode(
        label=torch.tensor([3.0]),
        loss_dict={
            "loss": torch.tensor(0.25),
            "conversionSigmoid": torch.tensor([0.75], dtype=torch.float32),
            "expectedCount": torch.tensor(0.25),
        },
    )
    model = _inference_model(
        _Constraint(inner_lc=object.__new__(sumL)),
        torch.nn.BCELoss(),
    )

    loss, *_ = model.forward(
        _Builder(datanode),
        use_gumbel=use_gumbel,
        temperature=0.5,
        hard_gumbel=False,
    )

    expected = torch.nn.functional.binary_cross_entropy(
        torch.tensor([0.75]), torch.tensor([1.0])
    )
    assert loss.item() == pytest.approx(expected.item())
    assert bool(datanode.infer_gumbel_calls) is use_gumbel


@pytest.mark.parametrize("use_gumbel", [False, True])
def test_queryl_executable_constraints_work_with_multiclass_loss_and_gumbel(use_gumbel):
    query_distribution = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float32)
    datanode = _DataNode(
        label=torch.tensor([1]),
        loss_dict={
            "loss": torch.tensor(0.3),
            "conversionSigmoid": query_distribution,
            "queryDistribution": query_distribution,
        },
    )
    model = _inference_model(
        _Constraint(inner_lc=object.__new__(queryL)),
        NBCrossEntropyLoss(),
    )

    loss, *_ = model.forward(
        _Builder(datanode),
        use_gumbel=use_gumbel,
        temperature=0.9,
        hard_gumbel=True,
    )

    expected = torch.nn.functional.cross_entropy(
        query_distribution.reshape(1, -1), torch.tensor([1])
    )
    assert loss.item() == pytest.approx(expected.item())
    assert bool(datanode.infer_gumbel_calls) is use_gumbel
