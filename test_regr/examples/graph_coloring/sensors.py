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

class DummyCityLabelSensor(TorchSensor): # Get Truth for Fire station classification
    def __init__(self, *pres, label=True):
        super().__init__(*pres, label=label)

    def forward(self,) -> Any:
        return []

# --- CityLink

class DummyCityLinkEdgeSensor(TorchEdgeSensor): # Get CityLink to city edge
    def forward(self,) -> Any:
        return self.inputs[0]
