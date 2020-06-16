from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        try:
            self.update_context(data_item, force)
        except:
            print('Error during updating data_item with sensor {}'.format(self.fullname))
            raise
        return data_item[self.fullname]

    def update_context(
        self,
        data_item: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if not force and (self.fullname in data_item):
            # data_item cached results by sensor name. override if forced recalc is needed
            val = data_item[self.fullname]
        else:
            val = self.forward(data_item)
            data_item[self.fullname] = val
        self.propagate_context(data_item, self, force)

    def propagate_context(self, data_item, node, force=False):
        if node.sup is not None and (node.sup.fullname not in data_item or force):
            data_item[node.sup.fullname] = data_item[self.fullname]
            self.propagate_context(data_item, node.sup, force)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        raise NotImplementedError
