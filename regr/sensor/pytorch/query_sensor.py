from typing import Dict, Any

from ...graph import DataNode
from .sensors import TorchSensor, Sensor

class QuerySensor(TorchSensor):
    def __init__(self, *pres, output=None, edges=None, label=False, query=None):
        super().__init__(*pres, output=output, edges=edges, label=label)
        if callable(query):
            self.selector = query
        else:
            self.selector = None

    def __call__(
            self,
            context: DataNode
    ) -> Dict[str, Any]:
        try:
            self.update_pre_context(context)
        except:
            print('Error during updating pre context with sensor {}'.format(self.fullname))
            raise
        self.context_helper = context
        try:
            context = self.update_context(context)
        except:
            print('Error during updating context with sensor {}'.format(self.fullname))
            raise

        if self.output:
            return context[self.sup.sup[self.output].fullname]

        try:
            return context[self.fullname]
        except KeyError:
            return context[self.sup.sup['raw'].fullname]

    def update_context(
            self,
            context: DataNode,
            force=False
    ) -> Dict[str, Any]:
        if not force and self.fullname in context:
            # context cached results by sensor name. override if forced recalc is needed
            val = context[self.fullname]
        else:
            self.define_inputs()
            val = self.forward()

        if val is not None:
            context[self.fullname] = val
            if not self.label:
                context[self.sup.fullname] = val  # override state under property name
        else:
            context[self.fullname] = None
            if not self.label:
                context[self.sup.fullname] = None

        if self.output:
            context[self.fullname] = self.fetch_value(self.output)
            context[self.sup.fullname] = self.fetch_value(self.output)

        return context

    def update_pre_context(
            self,
            context: DataNode
    ) -> Any:
        for edge in self.edges:
            for _, sensor in edge.find(Sensor):
                sensor(context=context)
        for pre in self.pres:
            for _, sensor in self.sup.sup[pre].find(Sensor):
                sensor(context=context)
