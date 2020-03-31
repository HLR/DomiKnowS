from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor
from typing import Any
import torch


class TestSensor(TorchSensor):
    def __init__(self, *pres, output=None, edges=None, label=False, expected_inputs=None, expected_outputs=None, **kwargs):
        super().__init__(*pres, output=output, edges=edges, label=label)
        self._expected_inputs = expected_inputs
        self._expected_outputs = expected_outputs

    @property
    def expected_inputs(self):
        return self._expected_inputs

    @property
    def expected_outputs(self):
        return self._expected_outputs

    def forward(self,) -> Any:
        assert self.inputs == self.expected_inputs
        return self.expected_outputs


class DummyEdgeStoW(TorchEdgeSensor):
    def forward(self,) -> Any:
        return ["John", "works", "for", "IBM"]


class DummyWordEmb(TestSensor): pass

class DummyFullyConnectedLearner(TestSensor): pass
