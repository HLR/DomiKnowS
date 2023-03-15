from itertools import combinations

import torch

from ...graph import Property
from ...sensor import Sensor, Learner
from ...sensor.torch.sensor import TorchSensor
from ...sensor.torch.learner import ModuleLearner

from .base import Mode


def all_properties(node):
    if isinstance(node, Property):
        return node
    return None


class BaseModel(torch.nn.Module):
    def __init__(self, graph, loss=None, metric=None, device='auto'):
        super().__init__()
        self.graph = graph
        self.loss = loss
        self.metric = metric
        self.mode_ = Mode.TRAIN
        self.device = device

    def mode(self, mode):
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

    def forward(self, data_item):
        data_item = self.move(data_item)
        return data_item


class TorchModel(BaseModel):
    BaseSensor = TorchSensor
    BaseLearner = ModuleLearner

    def __init__(self, graph, loss=None, metric=None, device='auto'):
        super().__init__(graph, loss=loss, metric=metric, device=device)
        self.poi = {prop: (output_sensor, target_sensor) for prop, output_sensor, target_sensor in self.find_poi()}
        self.graph.poi = self.poi
        for node in self.graph.traversal_apply(all_properties):
            for sensor in node.find(self.BaseLearner):
                self.add_module(sensor.fullname, sensor.module)

    def find_poi(self):
        for prop in self.graph.traversal_apply(all_properties):
            for sensor1, sensor2 in combinations(prop.find(self.BaseSensor), r=2):
                if sensor1.target:
                    target_sensor = sensor1
                    output_sensor = sensor2
                elif sensor2.target:
                    target_sensor = sensor2
                    output_sensor = sensor1
                else:
                    # TODO: should different learners get closer?
                    continue
                if output_sensor.target:
                    # two targets, skip
                    continue
                yield prop, output_sensor, target_sensor


class PoiModel(TorchModel):
    def poi_loss(self, data_item, prop, output_sensor, target_sensor):
        if not self.loss:
            return 0
        logit = output_sensor(data_item)
        mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)

        local_loss = self.loss[output_sensor, target_sensor](logit, labels, mask)
        return local_loss

    def poi_metric(self, data_item, prop, output_sensor, target_sensor):
        if not self.metric:
            return None
        mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)
        inference = prop(data_item)

        local_metric = self.metric[output_sensor, target_sensor](inference, labels, mask)
        return local_metric

    def forward(self, data_item, inference=True):
        data_item = super().forward(data_item)
        loss = 0
        metric = {}

        for prop, (output_sensor, target_sensor) in self.poi.items():
            # make sure the sensors are evaluated
            output = output_sensor(data_item)
            target = target_sensor(data_item)
            # calculated any loss or metric
            loss += self.poi_loss(data_item, prop, output_sensor, target_sensor)
            metric[output_sensor, target_sensor] = self.poi_metric(data_item, prop, output_sensor, target_sensor)

        return loss, metric, data_item
