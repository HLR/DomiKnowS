from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F

from ..utils import wrap_batch


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
            weight = torch.ones_like(input, dtype=torch.bool)
        preds = (input > 0).clone().detach().bool()
        labels = target.clone().detach().bool()
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


class MetricTracker(torch.nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.list = []
        self.dict = defaultdict(list)

    def reset(self):
        self.list.clear()
        self.dict.clear()

    def __call__(self, *args, **kwargs) -> Any:
        value = self.metric(*args, **kwargs)
        self.list.append(value)
        return value

    def __call_dict__(self, keys, *args, **kwargs) -> Any:
        value = self.metric(*args, **kwargs)
        self.dict[keys].append(value)
        return value

    def __getitem__(self, keys):
        return lambda *args, **kwargs: self.__call_dict__(keys, *args, **kwargs)

    def value(self, reset=False):
        if self.list and self.dict:
            raise RuntimeError('%s cannot be used as list-like and dict-like the same time.', str(type(self)))
        if self.list:
            value = wrap_batch(self.list)
            value = super().__call__(value)
        elif self.dict:
            #value = wrap_batch(self.dict)
            #value = super().__call__(value)
            func = super().__call__
            value = {k: func(v) for k, v in self.dict.items()}
        else:
            value = None
        if reset:
            self.reset()
        return value


class MacroAverageTracker(MetricTracker):
    def forward(self, values):
        def func(value):
            return value.clone().detach().mean()
        def apply(value):
            if isinstance(value, dict):
                return {k: apply(v) for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                return func(value)
            else:
                return apply(torch.tensor(value))
        retval = apply(values)
        return retval

class PRF1Tracker(MetricTracker):
    def __init__(self):
        super().__init__(CMWithLogitsMetric())

    def forward(self, values):
        CM = wrap_batch(values)
        tp = CM['TP'].sum().float()
        fp = CM['FP'].sum().float()
        fn = CM['FN'].sum().float()
        if tp:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
        else:
            p = torch.zeros_like(tp)
            r = torch.zeros_like(tp)
            f1 = torch.zeros_like(tp)
        return {'P': p, 'R': r, 'F1': f1}
