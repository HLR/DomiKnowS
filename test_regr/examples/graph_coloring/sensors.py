from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor, ReaderSensor
from regr.sensor.pytorch.learners import TorchLearner
from typing import Any
import torch

#  --- City
class DummyCityLearner(TorchLearner):  # Learn Fire station classification for City
    def forward(self,) -> Any:
        result = torch.zeros(len(self.inputs[0]), 2)
        
        for t in result: # Initially all cities are firestation cities
            t[1] = 1
            t[0] = 0

        return result
