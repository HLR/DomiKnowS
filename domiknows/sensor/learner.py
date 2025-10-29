import abc
from typing import Any
from . import Sensor

class Learner(Sensor):
    """
    Represents the bare parent learner that, like Sensors, can update and propagate context of a datanode based on the given data and create new properties, but additionally has parameters updated during training.
    
    Inherits from:
    - Sensor: A bare parent sensor class that manages the datanode context and creates new properties.
    """
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def parameters(self) -> Any: pass
