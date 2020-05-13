from typing import Any
import torch

from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor


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
        if self.expected_inputs is not None:
            assert self.inputs == self.expected_inputs
        return self.expected_outputs


class TestEdgeSensor(TorchEdgeSensor):
    def __init__(self, *pres, mode="forward", keyword="default", expected_inputs=None, expected_outputs=None, edges=None, **kwargs):
        super().__init__(*pres, mode=mode, keyword=keyword, edges=edges)
        self._expected_inputs = expected_inputs
        self._expected_outputs = expected_outputs

    @property
    def expected_inputs(self):
        return self._expected_inputs

    @property
    def expected_outputs(self):
        return self._expected_outputs

    def forward(self,) -> Any:
        if self.expected_inputs is not None:
            assert self.inputs == self.expected_inputs
        return self.expected_outputs
