from itertools import combinations, product
import warnings

import torch
import torch.nn.functional as F

from regr.graph import Property, Concept, DataNodeBuilder
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, TorchEdgeReaderSensor
from regr.sensor.pytorch.learners import TorchLearner

from .base import Mode
from ..tracker import MacroAverageTracker
from ..metric import MetricTracker


class TorchModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.mode(Mode.TRAIN)

        for learner in self.graph.get_sensors(TorchLearner):
            self.add_module(learner.fullname, learner.model)

    def mode(self, mode=None):
        if mode is None:
            return self._mode
        if mode in (Mode.TEST, Mode.POPULATE):
            self.eval()
        if mode == Mode.TRAIN:
            self.train()
        self._mode = mode

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

        for sensor in self.graph.get_sensors(ReaderSensor):
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
            self.poi = self.default_poi()
        else:
            properties = []
            for item in poi:
                if isinstance(item, Property):
                    properties.append(item)
                elif isinstance(item, Concept):
                    for prop in item.values():
                        properties.append(prop)
                else:
                    raise ValueError(f'Unexpected type of POI item {type(item)}: Property or Concept expected.')
            self.poi = properties
        self.loss = loss
        self.metric = metric

    def default_poi(self):
        poi = []
        for prop in self.graph.get_properties():
            if len(list(prop.find(TorchSensor))) > 1:
                poi.append(prop)
        return poi

    def find_sensors(self, prop):
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
            yield output_sensor, target_sensor

    def reset(self):
        if isinstance(self.loss, MetricTracker):
            self.loss.reset()
        if isinstance(self.metric, MetricTracker):
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
        for prop in self.poi:
            # make sure the sensors are evaluated
            for sensor in prop.find(TorchSensor):
                    sensor(builder)
            for sensors in self.find_sensors(prop):
                if self.mode() not in {Mode.POPULATE,}:
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
        for prop in self.poi:
            for output_sensor, target_sensor in self.find_sensors(prop):
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

class PoiModelToWorkWithLearnerWithLoss(TorchModel):
    def __init__(self, graph, poi=None):
        super().__init__(graph)
        if poi is not None:
            self.poi = poi
        else:
            self.poi = self.default_poi()
        self.loss_tracker = MacroAverageTracker()
        self.metric_tracker = None

    def reset(self):
        if self.loss_tracker is not None:
            self.loss_tracker.reset()
        if self.metric_tracker is not None:
            self.metric_tracker.reset()

    def default_poi(self):
        poi = []
        for prop in self.graph.get_properties():
            if len(list(prop.find(TorchSensor))) > 1:
                poi.append(prop)
        return poi

    def populate(self, builder):
        losses = {}
        metrics = {}
        for prop in self.poi:
            targets = []
            predictors = []
            for sensor in prop.find(TorchSensor):
                sensor(builder)
                if sensor.label:
                    targets.append(sensor)
                else:
                    predictors.append(sensor)
            for predictor in predictors:
                # TODO: any loss or metric or genaral function apply to just prediction?
                pass
            for target, predictor in product(targets, predictors):
                if predictor._loss is not None:
                    losses[predictor, target] = predictor.loss(builder, target)
                if predictor._metric is not None:
                    metrics[predictor, target] = predictor.metric(builder, target)

        self.loss_tracker.append(losses)
        # self.metrics_tracker.append(metrics)

        loss = sum(losses.values())
        return loss, metrics

    @property
    def loss(self):
        return self.loss_tracker

    @property
    def metric(self):
        # return self.metrics_tracker
        pass
