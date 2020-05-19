from regr.sensor.sensor import Sensor
from regr.graph.dataNode import DataNode
from regr.graph import Graph

from typing import Dict, Any
import torch


class TorchSensor(Sensor):

    def __init__(self, *pres, output=None, edges=None, label=False, query=None):
        super().__init__()
        if not edges:
            edges = []
        self.pres = pres
        self.output = output
        self.context_helper = None
        self.inputs = []
        self.edges = edges
        self.label = label
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

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
        except:
            return context[self.sup.sup['raw'].fullname]

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

    def fetch_value(self, pre, selector=None):
        if selector:
            try:
                return self.context_helper[list(self.sup.sup[pre].find(selector))[0][1].fullname]
            except:
                print("The key you are trying to access to with a selector doesn't exist")
                raise
            pass
        else:
            return self.context_helper[self.sup.sup[pre].fullname]

    def define_inputs(self):
        self.inputs = []
        for pre in self.pres:
            self.inputs.append(self.fetch_value(pre))

    def forward(self, ) -> Any:
        return None