from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor, ReaderSensor
from regr.sensor.pytorch.learners import TorchLearner
from typing import Any
import torch

#  --- City
class DummyCityLearner(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        # Initially all cities are firestation cities
        result[:, 1] = 1
        return result
