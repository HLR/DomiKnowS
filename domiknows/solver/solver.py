import abc
from typing import Tuple, Any

class Solver(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimize(self, min=True) -> Tuple[Any, Any]: pass

    def min(self) -> Any:
        return self.optimize()[0]

    def argmin(self) -> Any:
        return self.optimize()[1]

    def max(self) -> Any:
        return self.optimize(min=False)[0]

    def argmax(self) -> Any:
        return self.optimize(min=False)[1]
