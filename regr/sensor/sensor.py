from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.fullname in context:
            # context cached results by sensor name. override if forced recalc is needed
            return context
        try:
            context = self.update_context(context)
        except:
            print('Error during updating context with sensor {}'.format(self.fullname))
            raise
        return context

    def update_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        fwval = self.forward(context)
        if fwval is not None:
            context[self.fullname] = fwval
            context[self.sup.fullname] = fwval # override state under property name
        return context

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        return None
