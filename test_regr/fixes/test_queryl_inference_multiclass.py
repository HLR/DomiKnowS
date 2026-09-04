from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import sys

import pytest
import torch

from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.model.lossModel import InferenceModel

TASK_DIR = Path(__file__).resolve().parents[2] / "Tasks" / "clevr_inference_vs_gumbel"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

import main as clevr_main


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _args(tnorm="G"):
    return SimpleNamespace(
        epochs=1,
        train_items=4,
        eval_items=2,
        device=_device(),
        lr=1e-2,
        tnorm=tnorm,
        seed=0,
        gumbel_temp_start=1.0,
        gumbel_temp_end=0.3,
        hard_gumbel=False,
    )


def _built_program(tnorm="G"):
    args = _args(tnorm=tnorm)
    items = clevr_main.load_items()
    return clevr_main.build_program(
        "InferenceProgram",
        clevr_main.InferenceProgram,
        items,
        args,
        _device(),
    )


def _first_query_loss_dict(built):
    _mloss, _metric, _datanode, builder = built.program.model(built.train_dataset[0])
    datanode = builder.getDataNode(device=_device())
    labels = datanode.getExecutableConstraintLabels()
    context = datanode._prepareLcLossContext(
        tnorm=built.program.cmodel.tnorm,
        counting_tnorm=built.program.cmodel.counting_tnorm,
    )
    for lc_name in built.program.cmodel.constr:
        if f"{lc_name}/label" in labels:
            return datanode.calculateSingleLcLoss(
                lc_name,
                tnorm=built.program.cmodel.tnorm,
                counting_tnorm=built.program.cmodel.counting_tnorm,
                _context=context,
            )
    raise AssertionError("No active query constraint found")


def test_query_l_executable_returns_query_distribution():
    built = _built_program(tnorm="L")

    loss_dict = _first_query_loss_dict(built)

    distribution = loss_dict["queryDistribution"]
    assert distribution.shape == torch.Size([2])
    assert loss_dict["conversionSigmoid"].shape == torch.Size([2])
    assert torch.isclose(distribution.sum(), torch.tensor(1.0), atol=1e-5)


def test_clevr_task_enables_global_constraint_loss_by_default():
    built = _built_program(tnorm="L")

    assert built.program.cmodel.include_global_constraint_loss is True
    assert len(getattr(built.program.graph, "logicalConstrains", {})) > 0


def test_evaluate_condition_scores_query_l_as_multiclass_accuracy():
    built = _built_program(tnorm="L")

    results = built.program.evaluate_condition(
        built.eval_dataset,
        device=_device(),
        return_dict=True,
    )

    assert results["query_total"] == len(built.eval_dataset)
    assert results["boolean_total"] == 0
    assert results["query_accuracy"] is not None
    assert results["accuracy"] == pytest.approx(results["query_accuracy"])


def test_evaluate_condition_can_score_query_l_with_gumbel_samples():
    built = _built_program(tnorm="L")

    results = built.program.evaluate_condition(
        built.eval_dataset,
        device=_device(),
        return_dict=True,
        use_gumbel=True,
        temperature=1.0,
        hard_gumbel=False,
    )

    assert results["evaluation_mode"] == "gumbel"
    assert results["query_total"] == len(built.eval_dataset)
    assert results["query_accuracy"] is not None


def test_inference_model_backprops_direct_query_label():
    built = _built_program(tnorm="L")
    built.program.cmodel.loss_func = NBCrossEntropyLoss()

    _mloss, _metric, _datanode, builder = built.program.model(built.train_dataset[0])
    loss, *_ = built.program.cmodel(builder)
    loss.backward()

    grad_norm = 0.0
    for param in built.program.model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().norm().cpu())
    assert grad_norm > 0.0


def test_godel_query_distribution_is_hard_with_gradient():
    built = _built_program(tnorm="G")

    loss_dict = _first_query_loss_dict(built)
    distribution = loss_dict["queryDistribution"]

    assert sorted(distribution.detach().cpu().tolist()) == pytest.approx([0.0, 1.0])
    grad_norm = 0.0
    for data in built.train_dataset:
        built.program.model.zero_grad(set_to_none=True)
        built.program.cmodel.zero_grad(set_to_none=True)
        _mloss, _metric, _datanode, builder = built.program.model(data)
        loss, *_ = built.program.cmodel(builder)
        loss.backward()
        grad_norm = 0.0
        for param in built.program.model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().norm().cpu())
        if grad_norm > 0:
            break
    assert grad_norm > 0.0


def test_binary_executable_bce_behavior_still_works():
    class Recorder(dict):
        def __missing__(self, key):
            values = []
            self[key] = values
            return values.append

    class Constraint:
        active = True

        def strEs(self):
            return "fake"

    class Builder:
        def __init__(self, datanode):
            self.datanode = datanode

        def createBatchRootDN(self):
            pass

        def getDataNode(self, device="cpu"):
            return self.datanode

    class DataNode:
        current_dtype = torch.float32

        def getExecutableConstraintLabels(self):
            return {"ELC0/label": torch.tensor([1.0])}

        def _prepareLcLossContext(self, **_kwargs):
            return {}

        def getExecutableConstraintLabel(self, _lc_name):
            return torch.tensor([1.0])

        def calculateSingleLcLoss(self, _lc_name, **_kwargs):
            return {
                "loss": torch.tensor(1.0),
                "conversionSigmoid": torch.tensor([0.8], dtype=torch.float32),
            }

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
    model.loss_func = torch.nn.BCELoss()
    model.loss = Recorder()
    model.pos_weight = 1.0
    model._diag_budget = 0
    model._diag_step = 0
    model.inferenceLogger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
    model.constr = OrderedDict([("ELC0", Constraint())])

    loss, *_ = model.forward(Builder(DataNode()))

    assert loss.item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy(
            torch.tensor([0.8]), torch.tensor([1.0])
        ).item()
    )
