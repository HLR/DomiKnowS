from itertools import combinations, product
import hashlib
import pickle
from typing import Iterable

import torch

from domiknows.graph import Property, Concept, DataNodeBuilder
from domiknows.sensor.pytorch.sensors import TorchSensor, ReaderSensor, CacheSensor
from domiknows.sensor.pytorch.learners import TorchLearner

from .base import Mode
from ..tracker import MacroAverageTracker
from ..metric import MetricTracker


class TorchModel(torch.nn.Module):
    def __init__(self, graph, device='auto'):
        super().__init__()
        self.graph = graph
        self.mode(Mode.TRAIN)
        self.build = True
        self.device = device

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
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:  # no parameters
                pass
        if isinstance(value, torch.Tensor):
            return value.to(device)
        elif isinstance(value, list):
            return [self.move(v, device) for v in value]
        elif isinstance(value, tuple):
            return tuple(self.move(v, device) for v in value)
        elif isinstance(value, dict) and not isinstance(value, Concept):
            return {k: self.move(v, device) for k, v in value.items()}
        else:
            return value

    def data_hash(self, data_item):
        # NB: hash() is not consistent over different runs, don't use it!
        # try:
        #     return str(hash(data_item))
        # except TypeError as te:
        #     hash_message = te.args
        #     pass  # fall back to 'id'
        try:
            return data_item['id']
        except KeyError as ke:
            key_message = ke.args
            pass  # fall back to pickle dump and hashlib
        try:
            return hashlib.sha1(pickle.dumps(data_item)).hexdigest()
        except TypeError as te:
            dump_message = te.args
            raise ValueError(f'To enable cache for {self}, data item must either contain a identifier key "id" or picklable (might be slow): \n{key_message}\n{dump_message}')

    def forward(self, data_item, build=None):
        if build is None:
            build = self.build
        data_hash = None
        if not isinstance(data_item, dict):
            if isinstance(data_item, Iterable):
                data_item = dict((k, v) for k, v in enumerate(data_item))
            else:
                data_item = {None: data_item}
        data_item = self.move(data_item)

        for sensor in self.graph.get_sensors(CacheSensor, lambda s: s.cache):
            data_hash = data_hash or self.data_hash(data_item)
            sensor.fill_hash(data_hash)
        for sensor in self.graph.get_sensors(ReaderSensor):
            sensor.fill_data(data_item)
        if build:
            data_item.update({"graph": self.graph, 'READER': 0})
            builder = DataNodeBuilder(data_item)
            if (builder.needsBatchRootDN()):
                builder.addBatchRootDN()
            *out, = self.populate(builder)
            datanode = builder.getDataNode(context="build", device=self.device)
            return (*out, datanode, builder)
        else:
            *out, = self.populate(data_item)
            return (*out,)

    def populate(self):
        raise NotImplementedError


def model_helper(Model, *args, **kwargs):
    return lambda graph: Model(graph, *args, **kwargs)


class PoiModel(TorchModel):
    def __init__(self, graph, poi=None, loss=None, metric=None, device='auto'):
        super().__init__(graph, device)
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
        if metric is None:
            self.metric = None
        elif isinstance(metric, dict):
            self.metric = metric
        else:
            self.metric = {'': metric}

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
        if self.metric is not None:
            for metric in self.metric.values():
                if isinstance(metric, MetricTracker):
                    metric.reset()

    def poi_loss(self, data_item, _, sensors):
        if not self.loss:
            return 0
        outs = [sensor(data_item) for sensor in sensors]
        if len(outs[0]) == 0:
            return None
        local_loss = self.loss[(*sensors,)](*outs)
        
        if local_loss != local_loss:
            raise Exception("Calculated local_loss is nan") 
        
        return local_loss

    def poi_metric(self, data_item, prop, sensors):
        if not self.metric:
            return None
        outs = [sensor(data_item) for sensor in sensors]
        if len(outs[0]) == 0:
            return None
        local_metric = {}
        for key, metric in self.metric.items():
            local_metric[key] = metric[(*sensors,)](*outs, data_item=data_item, prop=prop)
        if len(local_metric) == 1:
            local_metric = list(local_metric.values())[0]
            
        return local_metric

    def populate(self, builder, run=True):
        loss = 0
        metric = {}
        
        for prop in self.poi:
            # make sure the sensors are evaluated
            if run:
                for sensor in prop.find(TorchSensor):
                    sensor(builder)
                    
            for sensors in self.find_sensors(prop):
                if self.mode() not in {Mode.POPULATE,}:
                    # calculated any loss or metric
                    if self.loss is not None:
                        local_loss = self.poi_loss(builder, prop, sensors)
                        if local_loss is not None:
                            loss += local_loss
                    if self.metric:
                        local_metric = self.poi_metric(builder, prop, sensors)
                        if local_metric is not None:
                            metric[(*sensors,)] = local_metric
        
        return loss, metric

class SolverModel(PoiModel):
    def __init__(self, graph, poi=None, loss=None, metric=None, inferTypes=None, inference_with = None, device='auto'):
        super().__init__(graph, poi=poi, loss=loss, metric=metric, device=device)
        
        if inferTypes == None:
            self.inferTypes = ['ILP']
        else:
            self.inferTypes = inferTypes
            
        if inference_with == None:
            self.inference_with = []
        else:
            self.inference_with = inference_with

    def inference(self, builder):
        for prop in self.poi:
            for sensor in prop.find(TorchSensor):
                sensor(builder)
#             for output_sensor, target_sensor in self.find_sensors(prop):
#             # make sure the sensors are evaluated
#                 output = output_sensor(builder)
#                 target = target_sensor(builder)
#         print("Done with the computation")

        # Check if this is batch
        if (builder.needsBatchRootDN()):
            builder.addBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        # trigger inference
#         fun=lambda val: torch.tensor(val, dtype=float).softmax(dim=-1).detach().cpu().numpy().tolist()
        for infertype in self.inferTypes:
            {
                'ILP': lambda :datanode.inferILPResults(*self.inference_with, fun=None, epsilon=None),
                'local/argmax': lambda :datanode.inferLocal(),
                'local/softmax': lambda :datanode.inferLocal(),
                'argmax': lambda :datanode.infer(),
                'softmax': lambda :datanode.infer(),
            }[infertype]()
#         print("Done with the inference")
        return builder

    def populate(self, builder, run=True):
        data_item = self.inference(builder)
        return super().populate(builder, run=False)
    

class PoiModelToWorkWithLearnerWithLoss(TorchModel):
    def __init__(self, graph, poi=None, device='auto'):
        super().__init__(graph, device=device)
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
                # TODO: any loss or metric or general function apply to just prediction?
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
    
    
class PoiModelDictLoss(PoiModel):
    def __init__(self, graph, poi=None, loss=None, metric=None, dictloss=None, device='auto'):
        self.loss_tracker = MacroAverageTracker()
        if dictloss:
            super().__init__(graph, poi=poi, loss=self.loss_tracker, metric=metric, device=device)
        else:
            super().__init__(graph, poi=poi, loss=loss, metric=metric, device=device)
        
        self.metric_tracker = None
        self.losses = dict()
        self.dictloss = dictloss
        
    
    def reset(self):
        if self.loss_tracker is not None:
            self.loss_tracker.reset()
        if self.metric_tracker is not None:
            self.metric_tracker.reset()
            
        if isinstance(self.loss, MetricTracker):
            self.loss.reset()
        if self.metric is not None:
            for metric in self.metric.values():
                if isinstance(metric, MetricTracker):
                    metric.reset()
            
    def poi_loss(self, data_item, prop, sensors):
        if self.dictloss:
            if self.loss is None:
                return 0
    #         outs = [sensor(data_item) for sensor in sensors]
            target = None
            pred = None
            for sensor in sensors:
                if sensor.label:
                    target = sensor
                else:
                    pred = sensor
            if target == None or pred == None or pred(data_item).shape[0] == 0:
                return None
            if not str(prop.name) in self.dictloss and not "default" in self.dictloss:
                return None
            elif not str(prop.name) in self.dictloss:
                self.losses[pred, target] = self.dictloss["default"](data_item, prop, pred(data_item), target(data_item))
                if torch.isnan(self.losses[pred, target]).any() :
                    print("here it is")
                return self.losses[pred, target]
            else:
                self.losses[pred, target] = self.dictloss[str(prop.name)](data_item, prop, pred(data_item), target(data_item))
                return self.losses[pred, target]
        else:
            super().poi_loss(data_item, prop, sensors)
    
    
    def populate(self, builder, run=True):
        if self.dictloss:
            loss, metric = super().populate(builder, run=True)
            self.loss_tracker.append(self.losses)
#             print(loss, metric)
            return loss, metric
        else:
            return super().populate(builder, run=True)
        
    
    
# class PoiModelDictLoss(TorchModel):
#     def __init__(self, graph, poi=None, loss=None, metric=None):
#         super().__init__(graph)
#         if poi is None:
#             self.poi = self.default_poi()
#         else:
#             properties = []
#             for item in poi:
#                 if isinstance(item, Property):
#                     properties.append(item)
#                 elif isinstance(item, Concept):
#                     for prop in item.values():
#                         properties.append(prop)
#                 else:
#                     raise ValueError(f'Unexpected type of POI item {type(item)}: Property or Concept expected.')
#             self.poi = properties
            
#         self.loss = loss
#         if metric is None:
#             self.metric = None
#         elif isinstance(metric, dict):
#             self.metric = metric
#         else:
#             self.metric = {'': metric}
            
#         self.loss_tracker = MacroAverageTracker()
#         self.metric_tracker = None

#     def reset(self):
#         if self.loss_tracker is not None:
#             self.loss_tracker.reset()
#         if self.metric_tracker is not None:
#             self.metric_tracker.reset()

#     def default_poi(self):
#         poi = []
#         for prop in self.graph.get_properties():
#             if len(list(prop.find(TorchSensor))) > 1:
#                 poi.append(prop)
#         return poi

#     def populate(self, builder, run=True):
#         losses = {}
#         metrics = {}
        
#         for prop in self.poi:
#             # make sure the sensors are evaluated
#             if run:
#                 for sensor in prop.find(TorchSensor):
#                     sensor(builder)
                            
#             targets = []
#             predictors = []
#             for sensor in prop.find(TorchSensor):
#                 if self.mode() not in {Mode.POPULATE,}:
#                     if sensor.label:
#                         targets.append(sensor)
#                     else:
#                         predictors.append(sensor)
#             for predictor in predictors:
#                 # TODO: any loss or metric or general function apply to just prediction?
#                 pass
#             for target, predictor in product(targets, predictors):
# #                 print(predictor, predictor(builder))
#                 if str(predictor.prop.name) in self.loss.keys():
#                     losses[predictor, target] = self.loss[str(predictor.prop.name)](builder, predictor.prop, predictor(builder), target(builder))
#                 if predictor._metric is not None:
#                     metrics[predictor, target] = predictor.metric(builder, target)

#         loss = sum(losses.values())
#         return loss, metrics
#     @property
#     def loss(self):
#         return self.loss_tracker

#     @property
#     def metric(self):
#         # return self.metrics_tracker
#         pass
    
    
class SolverModelDictLoss(PoiModelDictLoss):
    def __init__(self, graph, poi=None, loss=None, metric=None, dictloss=None, inferTypes=['ILP'], device='auto'):
        super().__init__(graph, poi=poi, loss=loss, metric=metric, dictloss=dictloss, device=device)
        self.inference_with = []
        self.inferTypes = inferTypes

    def inference(self, builder):
        for prop in self.poi:
            for sensor in prop.find(TorchSensor):
                sensor(builder)
#             for output_sensor, target_sensor in self.find_sensors(prop):
#             # make sure the sensors are evaluated
#                 output = output_sensor(builder)
#                 target = target_sensor(builder)
#         print("Done with the computation")

        # Check if this is batch
        if (builder.needsBatchRootDN()):
            builder.addBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        # trigger inference
#         fun=lambda val: torch.tensor(val, dtype=float).softmax(dim=-1).detach().cpu().numpy().tolist()
        for infertype in self.inferTypes:
            {
                'ILP': lambda :datanode.inferILPResults(*self.inference_with, fun=None, epsilon=None),
                'local/argmax': lambda :datanode.inferLocal(),
                'local/softmax': lambda :datanode.inferLocal(),
                'argmax': lambda :datanode.infer(),
                'softmax': lambda :datanode.infer(),
            }[infertype]()
#         print("Done with the inference")
        return builder

    def populate(self, builder, run=True):
        data_item = self.inference(builder)
        return super().populate(builder, run=False)

from .iml import IMLModel
from .ilpu import ILPUModel