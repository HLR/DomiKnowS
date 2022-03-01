from typing import Dict, NoReturn, Any
import abc
from ..graph.base import BaseGraphTreeNode


class Sensor(BaseGraphTreeNode):
    def __call__(
        self,
        data_item: Dict[str, Any],
        force=False,
        sensor_name="None"
    ) -> Any:
        try:
            self.update_context(data_item, force)
        except Exception as ex:
            print('Error {} during updating data item {} with sensor {}'.format(ex, data_item, sensor_name))
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
        self.propagate_context(data_item, force)

    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        raise NotImplementedError

    @property
    def prop(self):
        if self.sup is None:
            raise ValueError('{} must be used with with property assignment.'.format(type(self)))
        return self.sup

    @property
    def concept(self):
        if self.prop.sup is None:
            raise ValueError('{} must be used with with concept[property] assignment.'.format(type(self)))
        return self.prop.sup