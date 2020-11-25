from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Any:
        try:
            self.update_context(data_item, force)
        except:
            print('Error during updating data_item with sensor {}'.format(self))
            raise
        return data_item[self]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ):
        if not force and (self in data_item):
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self]
        else:
            val = self.forward(data_item)
            data_item[self] = val
        self.propagate_context(data_item, self, force)

    def propagate_context(self, data_item, node, force=False):
        if node.sup is not None and (node.sup not in data_item or force):
            data_item[node.sup] = data_item[self]
            self.propagate_context(data_item, node.sup, force)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        raise NotImplementedError
