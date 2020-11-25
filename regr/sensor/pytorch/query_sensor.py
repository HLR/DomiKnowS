from typing import Dict, Any

from .sensors import TorchSensor, FunctionalSensor


class QuerySensor(FunctionalSensor):
    @property
    def builder(self):
        builder = self.context_helper
        from ...graph import DataNodeBuilder

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

        datanodes = self.builder.findDataNodesInBuilder(select=self.concept)
        self.inputs.insert(0, datanodes)


class DataNodeSensor(QuerySensor):
    def forward_wrap(self):
        from ...graph import Property
        datanodes = self.inputs[0]
        assert len(self.inputs[1:]) == len(self.pres)
        inputs = []
        for input, pre in zip(self.inputs[1:], self.pres):
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

        return [self.forward(datanode, *input) for datanode, *input in zip(datanodes, *inputs)]


class InstantiateSensor(TorchSensor):
    def __call__(
        self,
        data_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            self.update_pre_context(data_item)
        except:
            print('Error during updating pre with sensor {}'.format(self))
            raise
        try:
            return data_item[self]
        except KeyError:
            return data_item[self.sup.sup['index']]

