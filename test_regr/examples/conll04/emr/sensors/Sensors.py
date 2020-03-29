from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor
from typing import Any
import torch


class DummyEdgeStoW(TorchEdgeSensor):
    def forward(self,) -> Any:
        return ["word1", "word2", "word3", "word4"]


class DummyWordEmb(TorchSensor):
    def forward(self,) -> Any:
        dummy = torch.randn(len(self.inputs[0]), 2048)
        return dummy