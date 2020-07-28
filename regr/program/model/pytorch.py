from itertools import combinations
import warnings

import torch
import torch.nn.functional as F

from regr.graph import Property, DataNodeBuilder
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, TorchEdgeReaderSensor
from regr.sensor.pytorch.learners import TorchLearner

from .base import Mode


def all_properties(node):
    if isinstance(node, Property):
        return node
    return None


class TorchModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.mode_ = Mode.TRAIN

        for node in self.graph.traversal_apply(all_properties):
            for sensor in node.find(TorchLearner):
                self.add_module(sensor.fullname, sensor.model)


    def mode(self, mode):
        if mode in (Mode.TEST, Mode.POPULATE):
            self.eval()
        if mode == Mode.TRAIN:
            self.train()
        self.mode_ = mode

    def reset(self):
        pass

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

        for prop in self.graph.traversal_apply(all_properties):
            for sensor in prop.find(lambda s: isinstance(s, (ReaderSensor, TorchEdgeReaderSensor))):
                sensor.fill_data(data_item)
        data_item.update({"graph": self.graph, 'READER': 0})
        builder = DataNodeBuilder(data_item)
        *out, = self.populate(builder)
        datanode = builder.getDataNode()
        return (*out, datanode)

    def populate(self):
        raise NotImplementedError


def model_helper(Model, *args, **kwargs):
    return lambda graph: Model(graph, *args, **kwargs)


class PoiModel(TorchModel):
    def __init__(self, graph, poi=None, loss=None, metric=None):
        super().__init__(graph)
        if poi is None:
            self.poi = [(prop, [output_sensor, target_sensor]) for prop, output_sensor, target_sensor in self.find_poi()]
        else:
            self.poi = poi
        # self.graph.poi = self.poi
        self.loss = loss
        self.metric = metric

    def find_poi(self):
        for prop in self.graph.traversal_apply(all_properties):
            for sensor1, sensor2 in combinations(prop.find(TorchSensor), r=2):
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

    def reset(self):
        if self.loss is not None:
            self.loss.reset()
        if self.metric is not None:
            self.metric.reset()

    def poi_loss(self, data_item, prop, sensors):
        if not self.loss:
            return 0
        outs = [sensor(data_item) for sensor in sensors]
        local_loss = self.loss[(*sensors,)](*outs)
        return local_loss

    def poi_metric(self, data_item, prop, sensors):
        if not self.metric:
            return None
        outs = [sensor(data_item) for sensor in sensors]
        local_metric = self.metric[(*sensors,)](*outs)
        return local_metric

    def populate(self, builder):
        loss = 0
        metric = {}
        for prop, sensors in self.poi:
            # make sure the sensors are evaluated
            for sensor in sensors:
                sensor(builder)
            if self.mode_ not in {Mode.POPULATE,}:
                # calculated any loss or metric
                if self.loss:
                    loss += self.poi_loss(builder, prop, sensors)
                if self.metric:
                    metric[(*sensors,)] = self.poi_metric(builder, prop, sensors)
        return loss, metric


class SolverModel(PoiModel):
    def __init__(self, graph, poi=None, loss=None, metric=None, Solver=None):
        super().__init__(graph, poi=poi, loss=loss, metric=metric)
        if Solver:
            self.solver = Solver(self.graph)
        else:
            self.solver = None
        self.inference_with = []

    def inference(self, builder):
        for prop, (output_sensor, target_sensor) in self.poi:
            # make sure the sensors are evaluated
            output = output_sensor(builder)
            target = target_sensor(builder)
        # data_item = self.solver.inferSelection(builder, list(self.poi))
        datanode = builder.getDataNode()
        # trigger inference
        datanode.inferILPConstrains(*self.inference_with, fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(), epsilon=None)
        return builder

    def populate(self, builder):
        data_item = self.inference(builder)
        return super().populate(builder)


class IMLModel(SolverModel):
    def poi_loss(self, data_item, prop, sensors):
        output_sensor, target_sensor = sensors
        logit = output_sensor(data_item)
        # mask = output_sensor.mask(data_item)
        labels = target_sensor(data_item)

        builder = data_item
        datanode = builder.getDataNode()
        concept = prop.sup
        values = []
        try:
            for cdn in datanode.findDatanodes(select=concept):
                value = cdn.getAttribute(f'<{prop.name}>/ILP')
                values.append(torch.cat((1-value, value), dim=-1))
            inference = torch.stack(values)
        except TypeError:
            message = (f'Failed to get inference result for {prop}. '
                       'Is it included in the inference (with `inference_with` attribute)? '
                       'Continue with predicted value.')
            warnings.warn(message)
            inference = logit.softmax(dim=-1).detach()

        if self.loss:
            local_loss = self.loss[output_sensor, target_sensor](logit, inference, labels)
            return local_loss
