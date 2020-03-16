from itertools import combinations

import torch
from torch.nn import functional as F

from regr.graph.property import Property
from emr.sensor.learner import TorchSensor, ModuleLearner


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


class CMWithLogitsMetric(torch.nn.Module):
    def forward(self, input, target, weight=None):
        if weight is None:
            weight = torch.ones_like(input, device='cpu', dtype=torch.bool)
        preds = (input > 0).clone().detach().cpu().bool()
        labels = target.clone().detach().cpu().bool()
        tp = (preds * labels * weight).sum()
        fp = (preds * (~labels) * weight).sum()
        tn = ((~preds) * (~labels) * weight).sum()
        fn = ((~preds) * labels * weight).sum()
        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


class PRF1WithLogitsMetric(CMWithLogitsMetric):
    def forward(self, input, target, weight=None):
        CM = super().forward(input, target, weight)
        tp = CM['TP'].float()
        fp = CM['FP'].float()
        fn = CM['FN'].float()
        if CM['TP']:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
        else:
            p = torch.zeros_like(tp)
            r = torch.zeros_like(tp)
            f1 = torch.zeros_like(tp)
        return {'P': p, 'R': r, 'F1': f1}


class TorchModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.loss_func = BCEWithLogitsLoss()
        self.metric_func = PRF1WithLogitsMetric()

        def func(node):
            if isinstance(node, Property):
                return node
            return None
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
                mask = output_sensor.mask(data)
                labels = target_sensor(data)
                labels = labels.float()
                if self.loss_func:
                    local_loss = self.loss_func(logit, labels, mask)
                    loss += local_loss
                if self.metric_func:
                    local_metric = self.metric_func(logit, labels, mask)
                    metric[output_sensor, target_sensor] = local_metric
        return loss, metric, data


def dict_zip(*dicts, fillvalue=None):  # https://codereview.stackexchange.com/a/160584
    all_keys = {k for d in dicts for k in d.keys()}
    return {k: [d.get(k, fillvalue) for d in dicts] for k in all_keys}


def wrap_batch(values, fillvalue=0):
    if isinstance(values, (list, tuple)):
        if isinstance(values[0], dict):
            values = dict_zip(*values, fillvalue=fillvalue)
            values = {k: wrap_batch(v, fillvalue=fillvalue) for k, v in values.items()}
        elif isinstance(values[0], torch.Tensor):
            values = torch.stack(values)
    return values


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
