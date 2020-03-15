from itertools import combinations

import torch

from regr.graph.property import Property
from emr.sensor.learner import TorchSensor, ModuleLearner


class TorchModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.metric_func = None
        
        def func(node):
            if isinstance(node, Property):
                return node
        for node in self.graph.traversal_apply(func):
            for _, sensor in node.find(ModuleLearner):
                self.add_module(sensor.fullname, sensor.module)

    def forward(self, data):
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
                    continue
                if output_sensor.target:
                    continue
                logit = output_sensor(data)
                logit = logit.squeeze()
                labels = target_sensor(data)
                labels = labels.float()
                if self.loss_func:
                    local_loss = self.loss_func(logit, labels)
                    loss += local_loss
                if self.metric_func:
                    local_metric = self.metric_func(logit, labels)
                    metric[output_sensor, target_sensor] = local_metric
        return loss, metric, data


def train(model, dataset, opt):
    model.train()
    for data in dataset:
        opt.zero_grad()
        loss, metric, output = model(data)
        loss.backward()
        opt.step()
        yield loss, metric, output


def test(model, dataset):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            loss, metric, output = model(data)
            yield loss, metric, output


def eval_many(model, dataset):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            _, _, output = model(data)
            yield output


def eval_one(model, data):
    model.eval()
    with torch.no_grad():
        _, _, output = model(data)
        return output
