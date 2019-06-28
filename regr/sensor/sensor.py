from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            context = self.update_context(context)
        except:
            print('Error during updating context with sensor {}'.format(self.fullname))
            raise
        return context

    def update_context(
        self,
        context: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if not force and self.fullname in context:
            # context cached results by sensor name. override if forced recalc is needed
            val = context[self.fullname]
        else:
            val = self.forward(context)
        if val is not None:
            context[self.fullname] = val
            context[self.sup.fullname] = val # override state under property name
        return context

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return None
