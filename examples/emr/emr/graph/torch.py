from itertools import combinations

import torch

from regr.graph import Property

from ..sensor.sensor import TorchSensor
from ..sensor.learner import ModuleLearner


def all_properties(node):
    if isinstance(node, Property):
        return node
    return None


class TorchModel(torch.nn.Module):
    def __init__(self, graph, loss=None, metric=None, solver_fn=None):
        super().__init__()
        self.graph = graph
        self.loss = loss
        self.metric = metric

        for node in self.graph.traversal_apply(all_properties):
            for _, sensor in node.find(ModuleLearner):
                self.add_module(sensor.fullname, sensor.module)

        self.poi = {prop: (output_sensor, target_sensor) for prop, output_sensor, target_sensor in self.find_poi()}

        self.solver = solver_fn(self.graph)
        self.graph.poi = self.poi

    def move(self, value, device=None):
        device = device or next(self.parameters()).device
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

    def forward(self, context, inference=True):
        context = self.move(context)
        loss = 0
        metric = {}

        if inference:
            context = self.inference(context)

        for prop, (output_sensor, target_sensor) in self.poi.items():
            logit = output_sensor(context)
            mask = output_sensor.mask(context)
            inference = prop(context)
            labels = target_sensor(context)

            if self.loss:
                local_loss = self.loss[output_sensor, target_sensor](logit, labels, mask)
                loss += local_loss
            if self.metric:
                local_metric = self.metric[output_sensor, target_sensor](logit, labels, mask)
                metric[output_sensor, target_sensor] = local_metric

        return loss, metric, context

    def inference(self, context):
        context = self.solver.inferSelection(context, list(self.poi))
        return context


class PoiModel(TorchModel):
    def poi_loss(self, context, prop, output_sensor, target_sensor):
        logit = output_sensor(context)
        mask = output_sensor.mask(context)
        labels = target_sensor(context)

        if self.loss:
            local_loss = self.loss[output_sensor, target_sensor](logit, labels, mask)
            return local_loss

    def poi_metric(self, context, prop, output_sensor, target_sensor):
        mask = output_sensor.mask(context)
        labels = target_sensor(context)
        inference = prop(context)

        if self.metric:
            local_metric = self.metric[output_sensor, target_sensor](inference, labels, mask)
            return local_metric

    def forward(self, context, inference=True):
        context = self.move(context)
        loss = 0
        metric = {}

        if inference:
            context = self.inference(context)

        for prop, (output_sensor, target_sensor) in self.poi.items():
            loss += self.poi_loss(context, prop, output_sensor, target_sensor)
            metric[output_sensor, target_sensor] = self.poi_metric(context, prop, output_sensor, target_sensor)

        return loss, metric, context

class IMLModel(PoiModel):
    def poi_loss(self, context, prop, output_sensor, target_sensor):
        logit = output_sensor(context)
        mask = output_sensor.mask(context)
        labels = target_sensor(context)
        inference = prop(context)

        if self.loss:
            local_loss = self.loss[output_sensor, target_sensor](logit, inference, labels, mask)
            return local_loss
