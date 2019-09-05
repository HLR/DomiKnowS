import os
from typing import Dict, List, Tuple, Iterable
from collections import OrderedDict
from torch import Tensor
from torch.nn import Module
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask

from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from .metrics import Epoch, DatasetType

from .. import Graph
from ..property import Property
from ...utils import prod, get_prop_result
from ...sensor.allennlp import AllenNlpLearner
from ...sensor.allennlp.base import ModuleSensor
from ...sensor.allennlp.sensor import LabelMaskSensor
from .utils import sequence_cross_entropy_with_logits


DEBUG_TRAINING = 'REGR_DEBUG' in os.environ and os.environ['REGR_DEBUG']


DataInstance = Dict[str, Tensor]


def update_metrics(
    graph: Graph,
    data: DataInstance,
    metrics: List[Tuple[str, Property, callable]]
) -> DataInstance:
    label_masks = {}
    for name, sensor in graph.get_sensors(LabelMaskSensor):
        mask = data[sensor.fullname]
        label_masks[mask.shape] = mask
    for metric_name, class_index, prop, metric in metrics:
        label, pred, mask = get_prop_result(prop, data)
        label_mask = label_masks.get(mask.shape)
        if label_mask is not None:
            mask = mask * label_mask
        metric(pred, label, mask)
    return data


class GraphModel(Model):
    def __init__(
        self,
        graph: Graph,
        vocab: Vocabulary,
        balance_factor: float = 0.5,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.,
        inference_interval: int = 10,
        inference_training_set: bool = False
    ) -> None:
        super().__init__(vocab)
        self.inference_interval = inference_interval
        self.inference_training_set = inference_training_set
        self.meta = []
        self.metrics = []
        self.metrics_inferenced = []

        self.balance_factor = balance_factor
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

        self.graph = graph

        self.meta.append(('epoch', Epoch()))
        self.meta.append(('type', DatasetType()))

        whole_metrics = {
            'Accuracy': CategoricalAccuracy,
        }
        class_metrics = {
            ('P', 'R', 'F1'): F1Measure,
        }

        for prop in self.graph.poi:
            for metric_name, MetricClass in whole_metrics.items():
                self.metrics.append((metric_name, None, prop, MetricClass()))
                self.metrics_inferenced.append((metric_name, None, prop, MetricClass()))
            for metric_name, MetricClass in class_metrics.items():
                for name, sensor in prop.find(AllenNlpLearner):
                    class_num = prod(sensor.output_dim)
                    if class_num == 2:
                        self.metrics.append((metric_name, None, prop, MetricClass(1)))
                        self.metrics_inferenced.append((metric_name, None, prop, MetricClass(1)))
                    else:
                        for class_index in range(class_num):
                            self.metrics.append((metric_name, class_index, prop, MetricClass(class_index)))
                            self.metrics_inferenced.append((metric_name, class_index, prop, MetricClass(class_index)))

        #i = 0  # TODO: this looks too bad
        for prop in self.graph.poi:
            for name, sensor in prop.find(ModuleSensor, lambda s: s.module is not None):
                self.add_module(name, sensor.module)
                #i += 1

    def _need_inference(
        self,
        data: DataInstance
    ) -> bool:
        #import pdb; pdb.set_trace()
        if DEBUG_TRAINING:
            return True
        dataset_type_key = 'dataset_type' # specify in regr.graph.allennlp.base.AllenNlpGraph
        if (not self.inference_training_set and
            dataset_type_key in data and
            all(dataset_type == 'train' for dataset_type in data[dataset_type_key])):
            return False
        if (dataset_type_key in data and
            all(dataset_type == 'test' for dataset_type in data[dataset_type_key])):
            return True
        epoch_key = 'epoch_num'  # TODO: this key... is from Allennlp doc
        if epoch_key not in data:
            return True  # no epoch record, then always inference
        epoch = min(data[epoch_key])
        need = ((epoch + 1) % self.inference_interval) == 0  # inference every 10 epoch
        return need

    def _update_metrics(
        self,
        data: DataInstance
    ) -> DataInstance:
        for metric_name, metric in self.meta:
            metric(data)
        data = update_metrics(self.graph, data, self.metrics)
        if self._need_inference(data):
            #import pdb; pdb.set_trace()
            data = self._inference(data)
            data = update_metrics(self.graph, data, self.metrics_inferenced)
        return data

    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        metrics = OrderedDict()

        def add(metric_name, class_index, suffix, prop, metric):
            try:
                out = metric.get_metric(reset)
            except RuntimeError:
                # in case inferenced ones are not called
                # RuntimeError("You never call this metric before.")
                # then pass
                return

            if isinstance(out, Iterable):
                for i, (metric_name_item, out_item) in enumerate(sorted(zip(metric_name, out))):
                    key = prop.sup.name
                    if class_index is not None:
                        class_name = self.vocab.get_token_from_index(class_index, namespace='labels')
                        key += '({})'.format(class_name)
                    if suffix is not None:
                        key += '{}'.format(suffix)
                    key += '-{}'.format(metric_name_item)
                    if i == 0:
                        key = '\n' + key
                    metrics[key] = out_item
            else:
                key = prop.sup.name
                if class_index is not None:
                    class_name = self.vocab.get_token_from_index(class_index, namespace='labels')
                    key += '({})'.format(class_name)
                if suffix is not None:
                    key += '{}'.format(suffix)
                key += '-{}'.format(metric_name)
                key = '\n' + key
                metrics[key] = out

        for metric_name, class_index, prop, metric in self.metrics:
            add(metric_name, class_index, None, prop, metric)
        for metric_name, class_index, prop, metric in self.metrics_inferenced:
            add(metric_name, class_index, '_i', prop, metric)

        max_len = max(len(metric_name.lstrip()) for metric_name in metrics)
        pretty_metrics = OrderedDict()
        for metric_name, metric in sorted(metrics.items(), key=lambda x: x[0].strip()):
            lspace_len = len(metric_name) - len(metric_name.lstrip())
            lspace = metric_name[:lspace_len]
            name = metric_name[lspace_len:]
            pretty_name = '{}{:>{max_len}}'.format(lspace, name, max_len=max_len)
            pretty_metrics[pretty_name] = metric

        for i, (metric_name, metric) in enumerate(self.meta):
            pretty_metrics['{}[ {} ]'.format('\n' if not i else '', metric_name)] = metric.get_metric(reset)
        return pretty_metrics

    def _update_loss(self, data):
        data['loss'] = self.loss_func(data)
        return data

    def forward(
        self,
        **data: DataInstance
    ) -> DataInstance:
        # make sure every node needed are calculated
        for prop in self.graph.poi:
            for name, sensor in prop.items():
                sensor(data)

        data = self._update_loss(data)
        data = self._update_metrics(data)

        return data

    def _inference(
        self,
        data: DataInstance
    ) -> DataInstance:
        #data = inference(self.graph, self.graph.solver, data, self.vocab)
        data = self.graph.solver.inferSelection(self.graph, data, self.vocab)
        
        return data

    def loss_func(
        self,
        data: DataInstance
    ) -> DataInstance:
        loss = 0
        label_masks = {}
        for name, sensor in self.graph.get_sensors(LabelMaskSensor):
            mask = data[sensor.fullname]
            label_masks[mask.shape] = mask
        for prop in self.graph.poi:
            # label (b, l)
            # pred  (b, l, c)
            # mask  (b, l)
            label, pred, mask = get_prop_result(prop, data)
            label = label.clone().detach()
            mask = mask.clone().float()
            label_mask = label_masks.get(mask.shape)
            if label_mask is not None:
                mask = mask * label_mask
            #import pdb; pdb.set_trace()

            # class balance weighted
            num_token = float(prod(pred.shape[:-1]))
            num_classes = pred.shape[-1]
            balance_bias = num_token / self.balance_factor

            alpha = [((num_token + balance_bias) / ((label == class_index).sum().float() + balance_bias))
                     for class_index in range(num_classes)]
            #import pdb; pdb.set_trace()
            loss += sequence_cross_entropy_with_logits(
                pred, label, mask,
                label_smoothing=self.label_smoothing,
                gamma=self.focal_gamma,
                alpha=alpha
            )

        return loss
