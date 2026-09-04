from domiknows.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor, ReaderSensor
from domiknows.sensor.pytorch.learners import TorchLearner
from typing import Any
import torch

#  --- City
class DummyCityLearner(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        # Initially all cities are firestation cities
        result[:, 1] = 1
        return result

class MainFirestationLearner(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        # Initially no cities are main firestations
        result[:, 0] = 1
        return result

class AncillaryFirestationLearner(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        # Initially no cities are ancillary firestations
        result[:, 0] = 1
        return result

class EmergencyServiceLearner(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        # Initially some cities have emergency services
        result[:, 0] = 0.7
        result[:, 1] = 0.3
        return result

class GroceryShopLearner(TorchLearner):
    def forward(self, x):
        result = torch.zeros(len(x), 2)
        # Initially some cities have grocery shops
        result[:, 0] = 0.6
        result[:, 1] = 0.4
        return result