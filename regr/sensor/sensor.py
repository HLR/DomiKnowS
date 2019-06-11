from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.fullname in context:
            # context cached results. override if forced recalc is needed
            return context
        context = self.update_context(context)
        return context

    def update_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return context
