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
from domiknows.utils import setDnSkeletonMode, getDnSkeletonMode


class TorchModel(torch.nn.Module):

    def __init__(self, graph, device='auto', ignore_modules=False):
        """
        The function initializes an object with a graph, device, and build flag.
        
        :param graph: The `graph` parameter is an object that represents the computational graph. It
        contains the nodes and edges that define the operations and data flow in the graph
        :param device: The `device` parameter specifies the device on which the graph will be executed.
        It can be set to `'auto'` to automatically select the device based on the availability of GPUs.
        Alternatively, it can be set to a specific device, such as `'cpu'` or `'cuda:0', defaults to auto
        (optional)
        :param ignore_modules: The `ignore_modules` parameter is a boolean flag that determines whether
        or not to ignore any modules in the graph. If set to `True`, any modules present in the graph
        will be ignored during training. If set to `False`, all modules in the graph will be considered
        during training, defaults to False (optional)
        """
        super().__init__()
        self.graph = graph
        self.mode(Mode.TRAIN)
        self.build = True
        self.device = device

        if not ignore_modules: ### added for the inference only models which do not update the initial parameters
            for learner in self.graph.get_sensors(TorchLearner):
                self.add_module(learner.fullname, learner.model)

    def mode(self, mode=None):
        """
        The mode function sets the mode of the object and performs certain actions based on the mode.
        
        :param mode: The `mode` parameter is used to specify the mode of operation for the code. It can
        take one of the following values:
        :return: The method is returning the value of the `_mode` attribute.
        """
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
        """
        The `forward` function takes a data item, performs various operations on it based on the graph
        structure, and returns the output.
        
        :param data_item: The `data_item` parameter is the input data that will be passed through the
        forward pass of the neural network. It can be either a dictionary or an iterable object. If it
        is an iterable, it will be converted into a dictionary where the keys are the indices of the
        elements in the iterable
        :param build: The `build` parameter is a boolean flag that indicates whether the method should
        build a data node or not. If `build` is `True`, the method will build a data node and return the
        loss, metric, data node, and builder. If `build` is `False`, the method
        :return: If the `build` parameter is `True`, the function returns a tuple containing `(loss,
        metric, datanode, builder)`. 
        If the `build` parameter is `False`, the function returns a tuple containing the values returned
        by the `populate` method.
        """
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
            # build datanode with tensor dictionary for attributes when not ILP or GBI inference needed
            if getDnSkeletonMode():
                if not hasattr(self, 'inferTypes') or ('ILP' not in self.inferTypes and 'GBI' not in self.inferTypes):
                    setDnSkeletonMode(True, full=True)
            builder = DataNodeBuilder(data_item)
            out = self.populate(builder)
            
            if len(out) == 2:
                builder.createBatchRootDN()
                datanode = builder.getDataNode(context="build", device=self.device)
                loss = out[0]
                metric = out[1]
            else:
                datanode, loss, metric = out
                
            return (loss, metric, datanode, builder)
        else:
            *out, = self.populate(data_item)
            return (*out,)

    def populate(self):
        raise NotImplementedError


def model_helper(Model, *args, **kwargs):
    return lambda graph: Model(graph, *args, **kwargs)


class PoiModel(TorchModel):
    def __init__(self, graph, poi=None, loss=None, metric=None, device='auto', ignore_modules=False):
        """
        The function initializes an object with various parameters for graph processing.
        
        :param graph: The graph parameter is used to specify the graph structure. It represents the
        connections between nodes in the graph
        :param poi: The "poi" parameter stands for "point of interest". It is used to specify a specific
        node or set of nodes in the graph that you are interested in. This can be useful when you want to
        perform operations or calculations only on a subset of nodes in the graph. If no point of interest
        :param loss: The `loss` parameter is used to specify the loss function that will be used during
        training. It is typically a function that measures the difference between the predicted output and
        the true output. The choice of loss function depends on the specific task and the type of data being
        used. Some common loss functions include
        :param metric: The `metric` parameter is used to specify the evaluation metric to be used during
        training and evaluation of the model. It is typically a function that takes the predicted values and
        the ground truth values as inputs and returns a scalar value representing the performance of the
        model. Examples of common evaluation metrics include accuracy,
        :param device: The `device` parameter specifies the device on which the computation will be
        performed. It can take the following values:, defaults to auto (optional)
        :param ignore_modules: The `ignore_modules` parameter is a boolean flag that determines whether to
        ignore any modules in the graph. If set to `True`, any modules present in the graph will be ignored
        during the initialization process. If set to `False` (default), all modules in the graph will be
        considered during initialization, defaults to False (optional)
        """
        super().__init__(graph, device, ignore_modules=ignore_modules)
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
        """
        The function `default_poi` returns a list of properties from a graph that have more than one
        instance of the `TorchSensor` class.
        :return: a list of properties that have more than one instance of the `TorchSensor` class in the
        graph.
        """
        poi = []
        for prop in self.graph.get_properties():
            if len(list(prop.find(TorchSensor))) > 1:
                poi.append(prop)
        return poi

    def find_sensors(self, prop):
        """
        The function `find_sensors` finds pairs of sensors in a given property that have one sensor
        labeled as the target and the other sensor as the output.
        
        :param prop: The parameter "prop" is expected to be an object that has a method called "find"
        which takes a class name as an argument and returns a list of objects of that class. In this
        case, it is being used to find objects of the class "TorchSensor"
        """
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
        """
        The function calculates the loss for a given data item using a set of sensors.
        
        :param data_item: The `data_item` parameter represents a single data item that is being
        processed. It could be any type of data, depending on the context of your code
        :param _: The underscore "_" is a convention in Python to indicate that a variable is not going
        to be used in the code. It is often used as a placeholder for a variable that needs to be
        present for the function signature but is not actually used within the function. In this case,
        it seems that the variable
        :param sensors: The `sensors` parameter is a list of sensor functions. These sensor functions
        take a `data_item` as input and return some output
        :return: the calculated local_loss.
        """
        if not self.loss:
            return 0
        
        outs = [sensor(data_item) for sensor in sensors]
        
        if len(outs[0]) == 0:
            return None
        
        selfLoss = self.loss[(*sensors,)]
        local_loss = selfLoss(*outs)
        
        if local_loss != local_loss:
            raise Exception("Calculated local_loss is nan") 
        
        return local_loss

    def poi_metric(self, data_item, prop, sensors, datanode=None):
        """
        The `poi_metric` function calculates a local metric based on the given data item, property,
        sensors, and optional datanode.
        
        :param data_item: The `data_item` parameter is a single data item that is being processed. It
        could be any type of data, depending on the context of your code
        :param prop: The "prop" parameter is a property or attribute of the data item that is being
        evaluated
        :param sensors: The `sensors` parameter is a list of sensor functions. These sensor functions
        take a `data_item` as input and return a value. The `sensors` list contains multiple sensor
        functions that will be called to collect data for the metric calculation
        :param datanode: The `datanode` parameter is an optional argument that represents the data node
        for which the metric is being calculated. It is used as an input to the metric function to
        provide additional context or information for the calculation
        :return: the local_metric value.
        """
        if not self.metric:
            return None
        outs = [sensor(data_item) for sensor in sensors]
        if len(outs[0]) == 0:
            return None
        local_metric = {}
        for key, metric in self.metric.items():
            local_metric[key] = metric[(*sensors,)](*outs, data_item=datanode, prop=prop)
        if len(local_metric) == 1:
            local_metric = list(local_metric.values())[0]
            
        return local_metric

    def populate(self, builder, datanode = None, run=True):
        """
        The `populate` function evaluates sensors, calculates loss and metrics, and returns the total
        loss and metric values.
        
        :param builder: The `builder` parameter is an object that is used to construct and populate data
        nodes in a data structure. It is likely an instance of a class that provides methods for
        creating and manipulating data nodes
        :param datanode: The `datanode` parameter is an optional argument that represents a data node in
        the builder. It is used to store the metric values calculated during the population process. If
        `datanode` is not provided, a new batch root data node is created and assigned to `datanode`
        :param run: The `run` parameter is a boolean flag that determines whether the sensors should be
        evaluated or not. If `run` is `True`, the sensors will be evaluated by calling their `__call__`
        method. If `run` is `False`, the sensors will not be evaluated, defaults to True (optional)
        :return: two values: `loss` and `metric`.
        """
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
                        if datanode is None:
                            builder.createBatchRootDN()
                            datanode = builder.getDataNode()
            
                        local_metric = self.poi_metric(builder, prop, sensors, datanode=datanode)
                        if local_metric is not None:
                            metric[(*sensors,)] = local_metric
        
        return loss, metric

class SolverModel(PoiModel):
    def __init__(self, graph, poi=None, loss=None, metric=None, inferTypes=None, inference_with = None, probKey = ("local" , "softmax"), device='auto', probAcc=None, ignore_modules=False):
        """
        The function initializes an object with various parameters for graph-based inference.
        
        :param graph: The graph parameter is used to specify the graph structure that the model will be
        trained on. It can be a graph object or a graph file
        :param poi: The "poi" parameter stands for "point of interest". It is used to specify a specific
        node or set of nodes in the graph that you want to focus on or perform operations on. If you don't
        specify any value for "poi", it will default to None, meaning that the entire graph
        :param loss: The `loss` parameter is used to specify the loss function to be used during training.
        It is typically a function that measures the difference between the predicted output and the true
        output. The choice of loss function depends on the specific task and the type of data being used.
        Some common loss functions include mean
        :param metric: The `metric` parameter is used to specify the evaluation metric to be used during
        training and inference. It is typically a function that takes the predicted values and the ground
        truth values as inputs and returns a scalar value representing the performance of the model. Common
        examples of evaluation metrics include accuracy, precision, recall
        :param inferTypes: The `inferTypes` parameter is used to specify the types of nodes in the graph
        that should be inferred. It is a list of strings, where each string represents a type of node. Only
        nodes with these types will be inferred during the graph inference process
        :param inference_with: The "inference_with" parameter is used to specify the type of inference to be
        performed. It determines how the model will make predictions or inferences based on the input data.
        The possible values for this parameter can vary depending on the specific implementation or
        framework being used. Some common options include "classification
        :param probKey: The `probKey` parameter is a tuple that specifies the type of probability
        distribution to use for inference. The first element of the tuple specifies the type of distribution
        to use for local inference, and the second element specifies the type of distribution to use for
        softmax inference
        :param device: The `device` parameter specifies the device on which the computations will be
        performed. It can take the following values:, defaults to auto (optional)
        :param probAcc: The `probAcc` parameter is the values of accuracies used inside the new ILP variation:
        :param ignore_modules: The `ignore_modules` parameter is a boolean flag that determines whether to
        ignore certain modules during the inference process. If set to `True`, the inference process will
        not consider these modules. If set to `False`, all modules will be considered during inference,
        defaults to False (optional)
        """
        super().__init__(graph, poi=poi, loss=loss, metric=metric, device=device, ignore_modules=ignore_modules)
        
        if inferTypes == None:
            self.inferTypes = ['ILP']
        else:
            self.inferTypes = inferTypes
            
        if inference_with == None:
            self.inference_with = []
        else:
            self.inference_with = inference_with
            
        self.probKey = probKey
        self.probAcc = probAcc

    def inference(self, builder):
        """
        The `inference` function takes a builder object, iterates over a list of properties, and performs
        inference using different types of models based on the `inferTypes` list.
        
        :param builder: The `builder` parameter is an object that is used to construct a computation
        graph. It is typically used to define the inputs, operations, and outputs of a neural network
        model
        :return: the `datanode` object.
        """
        # import time
        # start = time.time()
        for i, prop in enumerate(self.poi):
            for sensor in prop.find(TorchSensor):
                sensor(builder)
#             for output_sensor, target_sensor in self.find_sensors(prop):
#             # make sure the sensors are evaluated
#                 output = output_sensor(builder)
#                 target = target_sensor(builder)
#         print("Done with the computation")

        # Check if this is batch
        # end = time.time()
        # print("Time taken for computation: ", end-start)
        # start = time.time()
        builder.createBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        # end = time.time()
        # print("Time taken for creating datanode in its builder: ", end-start)
        # trigger inference
#         fun=lambda val: torch.tensor(val, dtype=float).softmax(dim=-1).detach().cpu().numpy().tolist()
        # start = time.time()
        if datanode:
            for infertype in self.inferTypes:
                # sub_start = time.time()
                {
                    'local/argmax': lambda :datanode.inferLocal(),
                    'local/softmax': lambda :datanode.inferLocal(),
                    'argmax': lambda :datanode.infer(),
                    'softmax': lambda :datanode.infer(),
                    'ILP': lambda :datanode.inferILPResults(*self.inference_with, key=self.probKey, fun=None, epsilon=None, Acc=self.probAcc),
                    'GBI': lambda :datanode.inferGBIResults(*self.inference_with, model=self),
                }[infertype]()
                # sub_end = time.time()
                # print("Time taken for inference of type ", infertype, " : ", sub_end-sub_start)
    #         print("Done with the inference")
        # end = time.time()
        # print("Time taken for inference after datanode creation: ", end-start)
        return datanode

    def populate(self, builder, run=True):
        """
        The `populate` function takes a `builder` object, performs inference on it, and then calls the
        `populate` method of the superclass with the resulting `datanode`, returning the `datanode`,
        `lose`, and `metric` values.
        
        :param builder: The "builder" parameter is an object that is used to build or construct the data
        node. It is likely an instance of a class that has methods for creating and manipulating data
        nodes
        :param run: The "run" parameter is a boolean flag that determines whether to run the population
        process immediately after populating the data node. If set to True, the population process will
        be executed; if set to False, the population process will be skipped, defaults to True
        (optional)
        :return: three values: datanode, lose, and metric.
        """
        datanode = self.inference(builder)
        lose, metric = super().populate(builder, datanode = datanode, run=False)
        
        return datanode, lose, metric
    

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
                    print("The loss has resulted in a Nan value")
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
        builder.createBatchRootDN()
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