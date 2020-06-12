from typing import Dict, Any
from itertools import product
import torch

from ...graph import DataNode, DataNodeBuilder, Concept, Property
from .sensors import TorchSensor, Sensor


class QuerySensor0(TorchSensor):
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


class FunctionalSensor(TorchSensor):
    def __init__(self, *pres, output=None, edges=None, label=False, forward=None):
        super().__init__(*pres, output=output, edges=edges, label=label)
        self.forward_ = forward

    def update_pre_context(
        self,
        context: Dict[str, Any]
    ) -> Any:
        for edge in self.edges:
            for _, sensor in edge.find(Sensor):
                sensor(context=context)
        for pre in self.pres:
            if isinstance(pre, str):
                if self.sup is None:
                    raise ValueError('{} must be used with with property assignment.'.format(type(self)))
                for _, sensor in self.sup.sup[pre].find(Sensor):
                    sensor(context=context)
            elif isinstance(pre, (Property, Sensor)):
                pre(context)

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
            val = self.forward_wrap()
            
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
        if isinstance(pre, str):
            return super().fetch_value(pre, selector)
        elif isinstance(pre, (Property, Sensor)):
            return self.context_helper[pre.fullname]
        return pre

    def forward_wrap(self):
        return self.forward(*self.inputs)

    def forward(self, *inputs):
        if self.forward_ is not None:
            return self.forward_(*inputs)
        raise NotImplementedError

class QuerySensor(FunctionalSensor):
    @property
    def builder(self):
        builder = self.context_helper
        if not isinstance(builder, DataNodeBuilder):
            raise TypeError('{} should work with DataNodeBuilder context.'.format(type(self)))
        return builder

    @property
    def concept(self):
        prop = self.sup
        if prop is None:
            raise ValueError('{} must be assigned to property'.format(type(self)))
        concept = prop.sup
        return concept

    def define_inputs(self):
        super().define_inputs()
        if self.inputs is None:
            self.inputs = []

        root = self.builder.getDataNode()
        datanodes = root.findDatanodes(select=self.concept)

        self.inputs.insert(0, datanodes)


class DataNodeSensor(QuerySensor):
    def forward_wrap(self):
        datanodes = self.inputs[0]

        return [self.forward(datanode, *self.inputs[1:]) for datanode in datanodes]


class CandidateSensor(QuerySensor):
    @property
    def args(self):
        return [rel.dst for rel in self.concept.has_a()]

    def update_pre_context(
        self,
        context: Dict[str, Any]
    ) -> Any:
        super().update_pre_context(context)
        for concept in self.args:
            concept['index'](context)  # call index property to make sure it is constructed
    
    def define_inputs(self):
        super().define_inputs()
        args = []
        for concept in self.args:
            root = self.builder.getDataNode()
            datanodes = root.findDatanodes(select=concept)
            args.append(datanodes)
        self.inputs = self.inputs[:1] + args + self.inputs[1:]

    def forward_wrap(self):
        # current existing datanodes (if any)
        datanodes = self.inputs[0]
        # args
        args = self.inputs[1:len(self.args)+1]
        # functional inputs
        inputs = self.inputs[len(self.args)+1:]

        arg_lists = []
        dims = []
        for arg_list in args:
            arg_lists.append(enumerate(arg_list))
            dims.append(len(arg_list))

        output = torch.zeros(dims, dtype=torch.uint8)
        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            output[(*index,)] = self.forward(datanodes, index, *arg_list, *inputs)
        return output


class InstantiateSensor(TorchSensor):
    def __call__(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            self.update_pre_context(context)
        except:
            print('Error during updating pre context with sensor {}'.format(self.fullname))
            raise
        try:
            return context[self.fullname]
        except KeyError:
            return context[self.sup.sup['index'].fullname]


class CandidateReaderSensor(CandidateSensor):
    def __init__(self, *pres, output=None, edges=None, label=False, forward=None, keyword=None):
        super().__init__(*pres, output=output, edges=edges, label=label, forward=forward)
        self.data = None
        self.keyword = keyword
        if keyword is None:
            raise ValueError('{} "keyword" must be assign.'.format(type(self)))

    def fill_data(self, data):
        self.data = data[self.keyword]

    def forward_wrap(self):
        # current existing datanodes (if any)
        datanodes = self.inputs[0]
        # args
        args = self.inputs[1:len(self.args)+1]
        # functional inputs
        inputs = self.inputs[len(self.args)+1:]

        arg_lists = []
        dims = []
        for arg_list in args:
            arg_lists.append(enumerate(arg_list))
            dims.append(len(arg_list))

        output = torch.zeros(dims, dtype=torch.uint8)
        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            output[(*index,)] = self.forward(self.data, datanodes, index, *arg_list, *inputs)
        return output
