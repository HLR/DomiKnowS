from Graphs.Sensors.mainSensors import CallingSensor
import abc
from typing import Any

class CallingLearner(CallingSensor):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def parameters(self) -> Any: pass