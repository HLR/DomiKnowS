from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        context: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        try:
            self.update_context(context, force)
        except:
            print('Error during updating context with sensor {}'.format(self.fullname))
            raise
        return context[self.fullname]

    def update_context(
        self,
        context: Dict[str, Any],
        force=False,
        propagate=True
    ) -> Dict[str, Any]:
        if not force and (self.fullname in context):
            # context cached results by sensor name. override if forced recalc is needed
            val = context[self.fullname]
        else:
            val = self.forward(context)
            context[self.fullname] = val
        self.propagate_context(context, self, force, propagate)

    def propagate_context(self, context, node, force=False, propagate=True):
        if propagate and node.sup is not None and (node.sup.fullname not in context or force):
            context[node.sup.fullname] = context[self.fullname]
            self.propagate_context(context, node.sup, force, propagate)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        raise NotImplementedError
