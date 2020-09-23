from itertools import combinations

import torch

from ....graph import Property
from ....sensor import Sensor, Learner
from ....sensor.torch.sensor import TorchSensor
from ....sensor.torch.learner import ModuleLearner
from ....program.metric import MetricKey

from ..base import Mode


class POIKey(MetricKey):
    def __str__(self):
        return self.pred.sup.prop_name.name


def all_properties(node):
    if isinstance(node, Property):
        return node
    return None


class BaseModel(torch.nn.Module):
    def __init__(self, graph, loss=None, metric=None):
        super().__init__()
        self.graph = graph
        self.loss = loss
        self.metric = metric
        self._mode = Mode.TRAIN

    def mode(self, mode=None):
        if mode is None:
            return self._mode
        if mode in (Mode.TEST, Mode.POPULATE):
            self.eval()
        if mode == Mode.TRAIN:
            self.train()
        self._mode = mode

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
            return list(self.move(v, device) for v in value)
        elif isinstance(value, tuple):
            return tuple(self.move(v, device) for v in value)
        elif isinstance(value, dict):
            return dict({k: self.move(v, device) for k, v in value.items()})
        else:
            return value

    def forward(self, data_item):
        data_item = self.move(data_item)
        return data_item


class TorchModel(BaseModel):
    BaseSensor = TorchSensor
    BaseLearner = ModuleLearner

    def __init__(self, graph, loss=None, metric=None):
        super().__init__(graph, loss=loss, metric=metric)
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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for node in self.graph.traversal_apply(all_properties):
            for sensor in node.find(self.BaseSensor):
                sensor.to(*args, **kwargs)


class PoiModel(TorchModel):
    def poi_loss(self, data_item, prop, output_sensor, target_sensor):
        if not self.loss:
            return 0
        logit = output_sensor(data_item)
        mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)

        local_loss = self.loss[POIKey(output_sensor, target_sensor)](logit, labels, mask)
        return local_loss

    def poi_metric(self, data_item, prop, output_sensor, target_sensor):
        if not self.metric:
            return None
        mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)
        inference = prop(data_item)

        local_metric = self.metric[POIKey(output_sensor, target_sensor)](inference, labels, mask)
        return local_metric

    def forward(self, data_item):
        data_item = super().forward(data_item)
        loss = 0
        metric = {}

        for prop, (output_sensor, target_sensor) in self.poi.items():
            # make sure the sensors are evaluated
            if 'index' in prop.sup:
                prop.sup['index'](data_item)
            output = output_sensor(data_item)
            target = target_sensor(data_item)
            # calculated any loss or metric
            loss += self.poi_loss(data_item, prop, output_sensor, target_sensor)
            metric[output_sensor, target_sensor] = self.poi_metric(data_item, prop, output_sensor, target_sensor)

        return loss, metric, data_item


class SolverModel(PoiModel):
    def __init__(self, graph, loss=None, metric=None, train_inference=False, test_inference=True, Solver=None):
        super().__init__(graph, loss, metric)
        self.train_inference = train_inference
        self.test_inference = test_inference
        if Solver:
            self.solver = Solver(self.graph)
        else:
            self.solver = None

    def inference(self, data_item):
        data_item = self.solver.inferSelection(data_item, list(self.poi))
        return data_item

    def forward(self, data_item, inference=None):
        data_item = self.move(data_item)
        if inference is None:
            inference = (
                (self.mode() is Mode.TRAIN and self.train_inference) or
                (self.mode() is Mode.TEST or self.mode() is Mode.POPULATE and self.test_inference))
        if inference:
            data_item = self.inference(data_item)
        return super().forward(data_item)