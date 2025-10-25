from typing import Any
import torch
import numpy as np

from domiknows.sensor.pytorch.sensors import FunctionalSensor, ConstantSensor, JointSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.learners import TorchLearner


class BaseTestSensor(JointSensor, ConstantSensor):
    def __init__(self, *args, expected_inputs=None, expected_outputs=None, **kwargs):
        super().__init__(*args, data=expected_outputs, **kwargs)
        self._expected_inputs = expected_inputs

    @property
    def expected_inputs(self):
        return self._expected_inputs

    @property
    def expected_outputs(self):
        return self.data

    def evaluate(self, inputs, expected_inputs):
        assert len(inputs) == len(expected_inputs)
        for input, expected_input in zip(inputs, expected_inputs):
            if (isinstance(input, (torch.Tensor, np.ndarray, np.generic)) or
                isinstance(expected_input, (torch.Tensor, np.ndarray, np.generic))):
                try:
                    if isinstance(input, torch.Tensor):
                        input = input.detach().clone().to(self.device)
                    else:
                        input = torch.tensor(input, device=self.device)

                    if isinstance(expected_input, torch.Tensor):
                        expected_input = expected_input.detach().clone().to(self.device)
                    else:
                        expected_input = torch.tensor(expected_input, device=self.device)
                except:
                    pass
                
                # Use torch.equal for exact tensor comparison, with fallback
                if isinstance(input, torch.Tensor) and isinstance(expected_input, torch.Tensor):
                    assert torch.equal(input, expected_input) or (input == expected_input).all(), \
                        f"Tensor mismatch: shapes {input.shape} vs {expected_input.shape}"
                else:
                    assert (input == expected_input).all()
            else:
                assert input == expected_input

    def forward(self, *inputs) -> Any:
        if self.expected_inputs is not None:
            self.evaluate(inputs, self.expected_inputs)
        return super().forward(*inputs)


class TestSensor(BaseTestSensor):
    pass


class TestLearner(TestSensor, TorchLearner):
    pass


class TestEdgeSensor(BaseTestSensor, EdgeSensor):
    pass