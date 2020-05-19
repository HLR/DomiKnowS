from itertools import combinations

import torch

from regr.graph import Property, DataNodeBuilder
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor
from regr.sensor.pytorch.learners import TorchLearner


def all_properties(node):
    if isinstance(node, Property):
        return node
    return None


class TorchModel(torch.nn.Module):
    def __init__(self, graph, loss=None, metric=None):
        super().__init__()
        self.graph = graph
        self.loss = loss
        self.metric = metric

        for node in self.graph.traversal_apply(all_properties):
            for _, sensor in node.find(TorchLearner):
                self.add_module(sensor.fullname, sensor.model)

        self.poi = {prop: (output_sensor, target_sensor) for prop, output_sensor, target_sensor in self.find_poi()}

        self.graph.poi = self.poi

    def reset(self):
        if self.loss is not None:
            self.loss.reset()
        if self.metric is not None:
            self.metric.reset()

    def move(self, value, device=None):
        parameters = list(self.parameters())
        if parameters:
            device = device or next(self.parameters()).device
        else:
            device = device
        if isinstance(value, torch.Tensor):
            return value.to(device)
        elif isinstance(value, list):
            return [self.move(v, device) for v in value]
        elif isinstance(value, tuple):
            return (self.move(v, device) for v in value)
        elif isinstance(value, dict):
            return {k: self.move(v, device) for k, v in value.items()}
        else:
            return value

    def find_poi(self):
        for prop in self.graph.traversal_apply(all_properties):
            for (_, sensor1), (_, sensor2) in combinations(prop.find(TorchSensor), r=2):
                if sensor1.label:
                    target_sensor = sensor1
                    output_sensor = sensor2
                elif sensor2.label:
                    target_sensor = sensor2
                    output_sensor = sensor1
                else:
                    # TODO: should different learners get closer?
                    continue
                if output_sensor.label:
                    # two targets, skip
                    continue
                yield prop, output_sensor, target_sensor

    def forward(self, data):
        data = self.move(data)

        def all_properties(node):
            if isinstance(node, Property):
                return node
        for prop in self.graph.traversal_apply(all_properties):
            for _, sensor in prop.find(ReaderSensor):
                sensor.fill_data(data)
        data.update({"graph": self.graph, 'READER': 1})
        context = DataNodeBuilder(data)
        datanode = context.getDataNode()
        return datanode


class PoiModel(TorchModel):
    def poi_loss(self, data, prop, output_sensor, target_sensor):
        if not self.loss:
            return 0
        logit = output_sensor(data)
        mask = output_sensor.mask(data)
        labels = target_sensor(data)

        local_loss = self.loss[output_sensor, target_sensor](logit, labels, mask)
        return local_loss

    def poi_metric(self, data, prop, output_sensor, target_sensor):
        if not self.metric:
            return None
        mask = output_sensor.mask(data)
        labels = target_sensor(data)
        inference = prop(data)

        local_metric = self.metric[output_sensor, target_sensor](inference, labels, mask)
        return local_metric

    def forward(self, data, inference=True):
        data = self.move(data)
        loss = 0
        metric = {}

        def all_properties(node):
            if isinstance(node, Property):
                return node
        for prop in self.graph.traversal_apply(all_properties):
            for _, sensor in prop.find(ReaderSensor):
                sensor.fill_data(data)
        data.update({"graph": self.graph, 'READER': 1})
        context = DataNodeBuilder(data)

        for prop, (output_sensor, target_sensor) in self.poi.items():
            # make sure the sensors are evaluated
            output = output_sensor(context)
            target = target_sensor(context)
            # calculated any loss or metric
            loss += self.poi_loss(context, prop, output_sensor, target_sensor)
            metric[output_sensor, target_sensor] = self.poi_metric(context, prop, output_sensor, target_sensor)

        datanode = context.getDataNode()
        return loss, metric, datanode

# class IMLModel(PoiModel):
#     def poi_loss(self, data, prop, output_sensor, target_sensor):
#         logit = output_sensor(data)
#         mask = output_sensor.mask(data)
#         labels = target_sensor(data)
#         inference = prop(data)

#         if self.loss:
#             local_loss = self.loss[output_sensor, target_sensor](logit, inference, labels, mask)
#             return local_loss
