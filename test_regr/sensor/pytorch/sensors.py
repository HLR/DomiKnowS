from typing import Any
import torch

from regr.sensor.pytorch.sensors import ConstantSensor, ConstantEdgeSensor


class TestSensor(ConstantSensor):
    def __init__(self, *pres, edges=None, label=False, expected_inputs=None, expected_outputs=None, device='auto', **kwargs):
        super().__init__(*pres, data=expected_outputs, edges=edges, label=label, device=device)
        self._expected_inputs = expected_inputs

    @property
    def expected_inputs(self):
        return self._expected_inputs

    @property
    def expected_outputs(self):
        return self.data

    def forward(self, *inputs) -> Any:
        if self.expected_inputs is not None:
            assert len(inputs) == len(self.expected_inputs)
            for input, expected_input in  zip(inputs, self.expected_inputs):
                if isinstance(input, torch.Tensor):
                    assert (input == expected_input).all()
                else:
                    assert input == expected_input
        return super().forward(*inputs)


class TestEdgeSensor(ConstantEdgeSensor):
    def __init__(self, *pres, to, mode="forward", edges=None, label=False, expected_inputs=None, expected_outputs=None, device='auto', **kwargs):
        super().__init__(*pres, data=expected_outputs, to=to, mode=mode, edges=edges, label=label, device=device)
        self._expected_inputs = expected_inputs

    @property
    def expected_inputs(self):
        return self._expected_inputs

    @property
    def expected_outputs(self):
        return self.data

    def forward(self, *inputs) -> Any:
        if self.expected_inputs is not None:
            assert tuple(inputs) == tuple(self.expected_inputs)
        return super().forward(*inputs)
