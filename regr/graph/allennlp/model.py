import os
from typing import Dict, Tuple, Iterable
from torch import Tensor
from torch.nn import Module
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask

from .. import Graph
from ...sensor.allennlp import AllenNlpLearner
from ...sensor.allennlp.base import ModuleSensor
from ...solver.allennlp.inference import inference


DEBUG_TRAINING = 'REGR_DEBUG' in os.environ and os.environ['REGR_DEBUG']


DataInstance = Dict[str, Tensor]


def get_prop_result(prop, data):
    vals = []
    mask = None
    for name, sensor in prop.items():
        sensor(data)
        if hasattr(sensor, 'get_mask'):
            if mask is None:
                mask = sensor.get_mask(data)
            else:
                assert mask == sensor.get_mask(data)
        tensor = data[sensor.fullname]
        vals.append(tensor)
    label = vals[0]  # TODO: from readersensor
    pred = vals[1]  # TODO: from learner
    return label, pred, mask


class BaseModel(Model):
    def __init__(self, vocab: Vocabulary, inference_interval: int=10) -> None:
        super().__init__(vocab)
        self.inference_interval = inference_interval
        self.meta = {}
        self.metrics = {}
        self.metrics_inferenced = {}

    def _update_metrics_base(
        self,
        data: DataInstance,
        metrics: Dict[str, Tuple[callable, Tuple[Tuple[Module, callable], float]]]
    ) -> DataInstance:
        for metric_name, (metric, prop) in metrics.items():
            label, pred, mask = get_prop_result(prop, data)

            from .metrics import Auc, AP, PRAuc
            if isinstance(metric, (Auc, AP, PRAuc)):  # FIXME: bad to have cases here!
                # AUC has problem using GPU
                metric(pred.select(dim=-1,index=1).reshape(-1).cpu(),  # (b,l,c) -> (b,l) -> (b*l)
                       label.reshape(-1).cpu(),  # (b,l) -> (b*l）
                       mask)
            else:
                metric(pred, label, mask)
        return data

    def _need_inference(
        self,
        data: DataInstance
    ) -> bool:
        epoch_key = 'epoch_num'  # TODO: this key... is from Allennlp doc
        if epoch_key not in data:
            return True  # no epoch record, then always inference
        epoch = min(data[epoch_key])
        need = ((epoch + 1) % self.inference_interval) == 0  # inference every 10 epoch
        return need or DEBUG_TRAINING

    def _update_metrics(
        self,
        data: DataInstance
    ) -> DataInstance:
        for metric_name, metric in self.meta.items():
            metric(data)
        data = self._update_metrics_base(data, self.metrics)
        if self._need_inference(data):
            #import pdb; pdb.set_trace()
            data = self._inference(data)
            data = self._update_metrics_base(data, self.metrics_inferenced)
        return data

    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        from collections import OrderedDict

        metrics = {}

        def add(metric_name, metric):
            try:
                out = metric.get_metric(reset)
            except RuntimeError:
                # in case inferenced ones are not called
                # RuntimeError("You never call this metric before.")
                # then pass
                return

            import numpy as np
            import numbers
            if isinstance(out, Iterable):
                for i, out_item in enumerate(out):
                    metrics['{}[{}]'.format(metric_name, i)] = out_item
            else:
                metrics[metric_name] = out

        for metric_name, metric in self.meta.items():
            add(metric_name, metric)
        for metric_name, (metric, _) in self.metrics.items():
            add(metric_name, metric)
        for metric_name, (metric, _) in self.metrics_inferenced.items():
            add(metric_name + '_i', metric)

        metrics = OrderedDict(sorted(metrics.items()))
        return metrics

    def _update_loss(self, data):
        if hasattr(self, 'loss_func') and self.loss_func is not None:
            data['loss'] = self.loss_func(data)
        return data

    def forward(
        self,
        **data: DataInstance
    ) -> DataInstance:
        data = self._update_loss(data)
        data = self._update_metrics(data)
        import pdb; pdb.set_trace()

        return data

    def _inference(
        self,
        data: DataInstance
    ) -> DataInstance:
        # pass through
        return data


class GraphModel(BaseModel):
    def __init__(
        self,
        graph: Graph,
        vocab: Vocabulary,
        inference_interval: int = 10
    ) -> None:
        BaseModel.__init__(self, vocab, inference_interval)

        self.graph = graph

        from allennlp.training.metrics import CategoricalAccuracy, F1Measure
        from .metrics import Epoch, Auc, AP, PRAuc, Precision

        def F1MeasureProxy(): return F1Measure(1)

        def PrecisionProxy(): return Precision(1)

        self.meta['epoch'] = Epoch()
        metrics = {
            #'Accuracy': CategoricalAccuracy,
            #'Precision': PrecisionProxy,
            'P/R/F1': F1MeasureProxy,
            #'ROC-AUC': Auc,
            #'PR-AUC': PRAuc,
            #'AP': AP,
        }

        for prop in self.graph.get_multiassign():
            # if concept == graph.organization: # just don't print too much
            #    continue
            for metric_name, MetricClass in metrics.items():
                fullname = '\n{}-{}'.format(prop.fullname, metric_name)
                shortname = '\n{}-{}'.format(prop.sup.name, metric_name)
                name = shortname
                self.metrics[name] = (MetricClass(), prop)
                self.metrics_inferenced[name] = (MetricClass(), prop)

        i = 0  # TODO: this looks too bad
        for prop in self.graph.get_multiassign():
            for name, sensor in prop.find(ModuleSensor, lambda s: s.module is not None):
                self.add_module(str(i), sensor.module)
                i += 1

    def forward(
        self,
        **data: DataInstance
    ) -> DataInstance:
        # make sure every node needed are calculated
        for prop in self.graph.get_multiassign():
            for name, sensor in prop.items():
                sensor(data)

        return BaseModel.forward(self, **data)

    def _inference(
        self,
        data: DataInstance
    ) -> DataInstance:
        # print(data['global/application/other[label]-1'])
        data = inference(self.graph, self.graph.solver, data, self.vocab)
        # print(data['global/application/other[label]-1'])
        return data

    def loss_func(
        self,
        data: DataInstance
    ) -> DataInstance:
        from .utils import sequence_cross_entropy_with_logits
        loss = 0
        for prop in self.graph.get_multiassign():
            label, pred, mask = get_prop_result(prop, data)
            mask = mask.clone().float()
            # class balance weighted
            target = (label > 0)
            pos = (target).sum().float()
            neg = (1 - target).sum().float()
            # alpha=(pos+neg)/pos for focal loss, but it cause divide by zero
            loss += sequence_cross_entropy_with_logits(
                pred, label, mask,
                label_smoothing=0.1, # (0.05, 0.95)
                gamma=2.0,
                alpha=neg/(pos+neg)
            )

        return loss
