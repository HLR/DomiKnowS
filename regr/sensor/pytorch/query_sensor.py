from typing import Dict, Any
from itertools import product
import torch

from ...graph import DataNode, DataNodeBuilder, Concept, Property
from .sensors import TorchSensor, FunctionalSensor, Sensor


class QuerySensor(FunctionalSensor):
    @property
    def builder(self):
        builder = self.context_helper
        if not isinstance(builder, DataNodeBuilder):
            raise TypeError('{} should work with DataNodeBuilder.'.format(type(self)))
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
        data_item: Dict[str, Any]
    ) -> Any:
        super().update_pre_context(data_item)
        for concept in self.args:
            concept['index'](data_item)  # call index property to make sure it is constructed
    
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
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            self.update_pre_context(data_item)
        except:
            print('Error during updating pre with sensor {}'.format(self.fullname))
            raise
        try:
            return data_item[self.fullname]
        except KeyError:
            return data_item[self.sup.sup['index'].fullname]


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

        if self.data is None and self.keyword in self.context_helper:
            self.data = self.context_helper[self.keyword]
            
        output = torch.zeros(dims, dtype=torch.uint8, names=('CandidateIdxOne','CandidateIdxTwo'))
        for arg_enum in product(*arg_lists):
            index, arg_list = zip(*arg_enum)
            output[(*index,)] = self.forward(self.data, datanodes, index, *arg_list, *inputs)
        return output
