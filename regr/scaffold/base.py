from .. import Graph, Concept
import abc
from typing import Callable


class Scaffold(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def assign(
        self,
        concept: Concept,
        prop: str,
        *args: list,
        **kwargs: dict
    ) -> None:
        pass
    # TODO: remove this interface and inject to property assignmnet somehow

    @abc.abstractmethod
    def build(
        self,
        graph: Graph,
        *args: list,
        **kwargs: dict
    ) -> Callable:
        pass

    @abc.abstractmethod
    def get_loss(
        self,
        graph: Graph,
        *args: list,
        **kwargs: dict
    ) -> Callable:
        pass
