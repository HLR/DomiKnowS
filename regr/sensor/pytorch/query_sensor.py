from typing import Dict, Any
import torch

from .sensors import FunctionalSensor, FunctionalReaderSensor


class QuerySensor(FunctionalSensor):
    def __init__(self, *pres, **kwargs):
        super().__init__(*pres, **kwargs)
        self.kwinputs = {}

    @property
    def builder(self):
        builder = self.context_helper
        from ...graph import DataNodeBuilder

        if not isinstance(builder, DataNodeBuilder):
            raise TypeError(f'{type(self)} should work with DataNodeBuilder.'
                            'For example, set `build` option to `True` when running the program')
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
        datanodes = self.builder.getDataNode(device=self.device).findDatanodes(select=self.concept)
        self.kwinputs['datanodes'] = datanodes

    def forward_wrap(self):
        value = self.forward(*self.inputs, **self.kwinputs)
        if isinstance(value, torch.Tensor) and value.device is not self.device:
            value = value.to(device=self.device)
        return value


class DataNodeSensor(QuerySensor):
    def forward_wrap(self):
        from ...graph import Property
        datanodes = self.kwinputs['datanodes']
        assert len(self.inputs) == len(self.pres)
        inputs = []
        for input, pre in zip(self.inputs, self.pres):
            if isinstance(pre, str):
                try:
                    pre = self.concept[pre]
                except KeyError:
                    pass
            if isinstance(pre, Property) and pre.sup == self.concept:
                assert len(input) == len(datanodes)
                inputs.append(input)
            else:
                # otherwise, repeat the input
                inputs.append([input] * len(datanodes))

        value = [self.forward(*input, datanode=datanode) for datanode, *input in zip(datanodes, *inputs)]

        try:
            return torch.tensor(value, device=self.device)
        except (TypeError, RuntimeError, ValueError):
            return value


class DataNodeReaderSensor(DataNodeSensor, FunctionalReaderSensor):
    pass
