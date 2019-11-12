from regr.sensor.sensor import Sensor
from typing import Dict, Any
import torch
from ...graph.base import BaseGraphTreeNode


class TorchSensor(BaseGraphTreeNode):

    def __init__(self, *pres, output=None, edge=None):
        super(TorchSensor).__init__()
        self.pres = pres
        self.output = output
        self.context_helper= None
        self.inputs = []
        self.edge = edge
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def __call__(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            context = self.update_pre_context(context)
        except:
            print('Error during updating pre context with sensor {}'.format(self.fullname))
            raise
        self.context_helper = context
        try:
            context = self.update_context(context)
        except:
            print('Error during updating context with sensor {}'.format(self.fullname))
            raise
        return context[self.fullname]

    def update_context(
        self,
        context: Dict[str, Any],
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
            context[self.sup.fullname] = val # override state under property name
        return context

    def update_pre_context(
        self,
        context: Dict[str, Any]
    ) -> Any:
        for pre in self.pres:
            for _, sensor in self.sup.sup[pre].find(Sensor):
                sensor(context=context)
        if self.edge:
            for _, sensor in self.edge.find(Sensor):
                sensor(context=context)
        if self.output:
            return context[self.output.fullname]

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
        for pre in self.pres:
            self.inputs.append(self.fetch_value(pre))

    def forward(self,) -> Any:
        return None


class ReaderSensor(TorchSensor):
    def __init__(self, *pres, keyword):
        super(ReaderSensor).__init__(*pres)
        self.data = None
        self.keyword = keyword

    def fill_data(self, data):
        self.data = data

    def forward(
        self,
    ) -> Any:
        if self.data:
            try:
                return self.data[self.keyword]
            except:
                print("the key you requested from the reader doesn't exist")
                raise
        else:
            print("there is no data to operate on")
            raise


class NominalSensor(TorchSensor):
    def __init__(self, *pres, vocab=None):
        super(NominalSensor).__init__(*pres)
        self.vocab = vocab

    def complete_vocab(self):
        if not self.vocab:
            self.vocab = []
        value = self.forward()
        if value not in self.vocab:
            self.vocab.append(value)

    def one_hot_encoder(self, value):
        output = torch.zeros(len(self.vocab))
        output[self.vocab.index(value)] = 1
        return output

    def update_context(
        self,
        context: Dict[str, Any],
        force=False
    ) -> Dict[str, Any]:
        if not force and self.fullname in context:
            # context cached results by sensor name. override if forced recalc is needed
            val = context[self.fullname]
        else:
            val = self.forward()
            val = self.one_hot_encoder(val)
        if val is not None:
            context[self.fullname] = val
            context[self.sup.fullname] = val # override state under property name
        return context


class TorchEdgeSensor(TorchSensor):
    def __init__(self, *pres, mode="forward"):
        super(TorchEdgeSensor).__init__(*pres)
        self.mode = mode
        self.edge = self.sup.sup
        if mode == "forward":
            self.src = self.edge.src
            self.dst = self.edge.dst
        elif mode == "backward":
            self.src = self.edge.dst
            self.dst = self.edge.src
        else:
            print("the mode passed to the edge sensor is not right")
            raise

    def __call__(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            context = self.update_pre_context(context)
        except:
            print('Error during updating pre context with sensor {}'.format(self.fullname))
            raise
        self.context_helper = context
        try:
            context = self.update_context(context)
        except:
            print('Error during updating context with sensor {}'.format(self.fullname))
            raise
        return context[self.dst[self.pres[0]].fullname]

    def update_context(
        self,
        context: Dict[str, Any],
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
            context[self.dst[self.pres[0]].fullname] = val # override state under property name
        return context

    def update_pre_context(
        self,
        context: Dict[str, Any]
    ) -> Any:
        for pre in self.pres:
            for _, sensor in self.src[pre].find(Sensor):
                sensor(context=context)
        if self.output:
            return context[self.output.fullname]

    def fetch_value(self, pre, selector=None):
        if selector:
            try:
                return self.context_helper[list(self.src[pre].find(selector))[0][1].fullname]
            except:
                print("The key you are trying to access to with a selector doesn't exist")
                raise
            pass
        else:
            return self.context_helper[self.src[pre].fullname]


class TorchEdgeReaderSensor(TorchEdgeSensor):
    def __init__(self, *pres, keyword, mode="forward"):
        super(ReaderSensor).__init__(*pres, mode=mode)
        self.data = None
        self.keyword = keyword

    def fill_data(self, data):
        self.data = data

    def forward(
            self,
    ) -> Any:
        if self.data:
            try:
                return self.data[self.keyword]
            except:
                print("the key you requested from the reader doesn't exist")
                raise
        else:
            print("there is no data to operate on")
            raise