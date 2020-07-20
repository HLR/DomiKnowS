from itertools import combinations

import torch

from regr.graph import Property, DataNodeBuilder
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor
from regr.sensor.pytorch.learners import TorchLearner

from .base import Mode


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
        self.mode_ = Mode.TRAIN

        for node in self.graph.traversal_apply(all_properties):
            for _, sensor in node.find(TorchLearner):
                self.add_module(sensor.fullname, sensor.model)

        self.poi = {prop: (output_sensor, target_sensor) for prop, output_sensor, target_sensor in self.find_poi()}

        self.graph.poi = self.poi

    def mode(self, mode):
        if mode in (Mode.TEST, Mode.POPULATE):
            self.eval()
        if mode == Mode.TRAIN:
            self.train()
        self.mode_ = mode

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

    def forward(self, data_item):
        data_item = self.move(data_item)

        def all_properties(node):
            if isinstance(node, Property):
                return node
        for prop in self.graph.traversal_apply(all_properties):
            for _, sensor in prop.find(ReaderSensor):
                sensor.fill_data(data_item)
        data_item.update({"graph": self.graph, 'READER': 1})
        builder = DataNodeBuilder(data_item)
        datanode = builder.getDataNode()
        return datanode


class PoiModel(TorchModel):
    def poi_loss(self, data_item, prop, output_sensor, target_sensor):
        if not self.loss:
            return 0
        logit = output_sensor(data_item)
        # mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)

        local_loss = self.loss[output_sensor, target_sensor](logit, labels)
        return local_loss

    def poi_metric(self, data_item, prop, output_sensor, target_sensor):
        if not self.metric:
            return None
        # mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)
        inference = prop(data_item)

        local_metric = self.metric[output_sensor, target_sensor](inference, labels)
        return local_metric

    def forward(self, data_item, inference=True):
        data_item = self.move(data_item)
        loss = 0
        metric = {}

        def all_properties(node):
            if isinstance(node, Property):
                return node
        for prop in self.graph.traversal_apply(all_properties):
            for _, sensor in prop.find(ReaderSensor):
                sensor.fill_data(data_item)
        data_item.update({"graph": self.graph, 'READER': 1})
        builder = DataNodeBuilder(data_item)

        for prop, (output_sensor, target_sensor) in self.poi.items():
            # make sure the sensors are evaluated
            output = output_sensor(builder)
            target = target_sensor(builder)
            if self.mode_ not in {Mode.POPULATE,}:
                # calculated any loss or metric
                if self.loss:
                    loss += self.poi_loss(builder, prop, output_sensor, target_sensor)
                if self.metric:
                    metric[output_sensor, target_sensor] = self.poi_metric(builder, prop, output_sensor, target_sensor)

        datanode = builder.getDataNode()
        return loss, metric, datanode
