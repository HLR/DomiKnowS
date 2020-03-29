import abc
from itertools import combinations

import torch

from regr.graph.property import Property
from emr.sensor.learner import TorchSensor, ModuleLearner


class TorchModel(torch.nn.Module):
    def __init__(self, graph, loss, metric=None):
        super().__init__()
        self.graph = graph
        self.loss = loss
        self.metric = metric

        def func(node):
            if isinstance(node, Property):
                return node
            return None
        for node in self.graph.traversal_apply(func):
            for _, sensor in node.find(ModuleLearner):
                self.add_module(sensor.fullname, sensor.module)

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
            raise NotImplementedError('%s is not supported. Can only move list, dict of tensors.', type(value))

    def forward(self, data):
        data = self.move(data)
        loss = 0
        metric = {}
        def all_properties(node):
            if isinstance(node, Property):
                return node
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
                logit = output_sensor(data)
                logit = logit.squeeze()
                mask = output_sensor.mask(data)
                labels = target_sensor(data)
                labels = labels.float()
                if self.loss:
                    local_loss = self.loss[output_sensor, target_sensor](logit, labels, mask)
                    loss += local_loss
                if self.metric:
                    local_metric = self.metric[output_sensor, target_sensor](logit, labels, mask)
                    metric[output_sensor, target_sensor] = local_metric
        return loss, metric, data


class LearningBasedProgram():
    def __init__(self, graph, **config):
        self.graph = graph
        self.model = TorchModel(graph, **config)

    def train(self, dataset, opt):
        self.model.train()
        for data in dataset:
            opt.zero_grad()
            loss, metric, output = self.model(data)
            loss.backward()
            opt.step()
            yield loss, metric, output


    def test(self, dataset):
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            for data in dataset:
                loss, metric, output = self.model(data)
                yield loss, metric, output


    def eval_many(self, dataset):
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            for data in dataset:
                _, _, output = self.model(data)
                yield output


    def eval_one(self, data):
        # TODO: extend one sample data to 1-batch data
        self.model.eval()
        self.model.loss.reset()
        self.model.metric.reset()
        with torch.no_grad():
            _, _, output = self.model(data)
            return output
