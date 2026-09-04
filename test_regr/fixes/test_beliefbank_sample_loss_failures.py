from types import SimpleNamespace

import torch

from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor
from domiknows.solver.sampleLossCalculator import SampleLossCalculator


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_add_loss_to_vdns_ignores_none_entries():
    constructor = LogicalConstraintConstructor(_Logger())
    constructor.current_device = "cpu"
    constructor.current_dtype = torch.float32

    result = constructor.addLossTovDns(
        True,
        [[torch.tensor(0.25, requires_grad=True)], [None]],
    )

    assert len(result) == 1
    assert torch.is_tensor(result[0][0])
    assert result[0][0].shape == (1,)
    assert result[0][0].requires_grad


def test_sample_loss_tolerates_ragged_sample_info(monkeypatch):
    sample_size = 4
    lc = SimpleNamespace(
        headLC=True,
        active=True,
        lcName="LC0",
        sampleEntries=False,
        strEs=lambda: "ragged sampleInfo",
    )
    graph = SimpleNamespace(logicalConstrains={"LC0": lc})
    loss_tensor = [
        [torch.zeros(sample_size)],
        [torch.ones(sample_size)],
    ]
    sample_info = {
        "x": [[("px", torch.ones(sample_size), "x")]],
        "s": [],
        "V1": [
            [("p0", torch.ones(sample_size), "v0")],
            [("p1", torch.ones(sample_size), "v1")],
        ],
    }
    constructor = SimpleNamespace(
        current_device="cpu",
        current_dtype=torch.float32,
        myGraph=None,
        constructLogicalConstrains=lambda *args, **kwargs: (
            loss_tensor,
            sample_info,
            None,
            None,
        ),
    )
    logger = _Logger()
    logger.handlers = []
    solver = SimpleNamespace(
        current_device="cpu",
        myLcLossSampleBooleanMethods=SimpleNamespace(),
        myLogger=logger,
        myLoggerTime=logger,
        myGraph=[graph],
        constraintConstructor=constructor,
    )
    dn = SimpleNamespace(
        current_device="cpu",
        current_dtype=torch.float32,
        setActiveExecutableLCs=lambda: None,
    )

    monkeypatch.setattr(
        SampleLossCalculator,
        "_calculateSampleLossForVariable",
        lambda self, *args: (
            torch.ones(args[-2]),
            args[-2],
        ),
    )

    result = SampleLossCalculator(solver).calculateSampleLoss(
        dn,
        sample_size,
        sampleGlobalLoss=False,
        conceptsRelations=(),
    )

    assert "LC0" in result
    assert len(result["LC0"]["lossTensor"]) == 1
    assert torch.equal(result["LC0"]["lossTensor"][0], torch.ones(sample_size))
