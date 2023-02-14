import abc
from typing import Any
from . import Sensor

class Learner(Sensor):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def parameters(self) -> Any: pass
