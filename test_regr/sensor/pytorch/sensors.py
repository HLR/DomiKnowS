from typing import Any
import torch

from regr.sensor.pytorch.sensors import FunctionalSensor, TorchEdgeSensor


class TestSensor(FunctionalSensor):
    def __init__(self, *pres, edges=None, label=False, expected_inputs=None, expected_outputs=None, device='auto', **kwargs):
        super().__init__(*pres, edges=edges, label=label, device=device)
        self._expected_inputs = expected_inputs
        self._expected_outputs = expected_outputs

    @property
    def expected_inputs(self):
        return self._expected_inputs

    @property
    def expected_outputs(self):
        return self._expected_outputs

    def forward(self, *inputs) -> Any:
        if self.expected_inputs is not None:
            assert tuple(inputs) == tuple(self.expected_inputs)
        return self.expected_outputs


class TestEdgeSensor(TorchEdgeSensor, TestSensor):
    def __init__(self, *pres, to, mode="forward", edges=None, label=False, expected_inputs=None, expected_outputs=None, device='auto', **kwargs):
        super().__init__(*pres, to=to, mode=mode, edges=edges, label=label, device=device)
        self._expected_inputs = expected_inputs
        self._expected_outputs = expected_outputs
