from typing import Any
import torch

from regr.sensor.pytorch.sensors import TorchSensor, ConstantSensor, ConstantEdgeSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.pytorch.learners import TorchLearner


class BaseTestSensor(JointSensor, TorchSensor):
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
                try:
                    if isinstance(input, torch.Tensor):
                        if not isinstance(expected_input, torch.Tensor):
                            expected_input = torch.tensor(expected_input, device=self.device)
                        assert (input == expected_input).all()
                        continue
                except:
                    pass
                assert input == expected_input
        return super().forward(*inputs)


class TestSensor(BaseTestSensor, ConstantSensor):
    def __init__(self, *pres, edges=None, label=False, expected_inputs=None, expected_outputs=None, as_tensor=True, device='auto'):
        super().__init__(*pres, data=expected_outputs, edges=edges, label=label, as_tensor=True, device=device)
        self._expected_inputs = expected_inputs


class TestLearner(TestSensor, TorchLearner):
    pass


class TestEdgeSensor(BaseTestSensor, EdgeSensor, ConstantSensor):
    def __init__(self, *pres, relation, mode='forward', expected_inputs=None, expected_outputs=None, **kwargs):
        super().__init__(*pres, relation=relation, mode=mode, data=expected_outputs, **kwargs)
        self._expected_inputs = expected_inputs


class TestTorchEdgeSensor(BaseTestSensor, ConstantEdgeSensor):
    def __init__(self, *pres, to, mode="forward", edges=None, label=False, expected_inputs=None, expected_outputs=None, device='auto'):
        super().__init__(*pres, data=expected_outputs, to=to, mode=mode, edges=edges, label=label, device=device)
        self._expected_inputs = expected_inputs
