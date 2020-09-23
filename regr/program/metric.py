from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F

from ..utils import wrap_batch, value, FormatPrinter
from ..base import AutoNamed


class BinaryCMWithLogitsMetric(torch.nn.Module):
    def forward(self, input, target, weight=None, dim=None):
        if weight is None:
            weight = torch.tensor(1, device=input.device)
        weight = weight.long()
        preds = (input > 0).clone().detach().to(dtype=weight.dtype)
        labels = target.clone().detach().to(dtype=weight.dtype, device=input.device)
        assert (0 <= labels).all() and (labels <= 1).all()
        tp = (preds * labels * weight).sum()
        fp = (preds * (1 - labels) * weight).sum()
        tn = ((1 - preds) * (1 - labels) * weight).sum()
        fn = ((1 - preds) * labels * weight).sum()
        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


class CMWithLogitsMetric(BinaryCMWithLogitsMetric):
    def forward(self, input, target, weight=None):
        num_classes = input.shape[-1]
        input = input.view(-1, num_classes)
        target = F.one_hot(target.view(-1), num_classes=num_classes)
        return super().forward(input, target, weight)


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
        return {'P': value(p), 'R': value(r), 'F1': value(f1)}


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

    def kprint(self, k):
        if (
            isinstance(k, tuple) and
            len(k) == 2 and
            isinstance(k[0], AutoNamed) and 
            isinstance(k[1], AutoNamed)):
            return k[0].sup.name.name
        else:
            return k

    def value(self, reset=False):
        if self.list and self.dict:
            raise RuntimeError('{} cannot be used as list-like and dict-like the same time.'.format(type(self)))
        if self.list:
            value = wrap_batch(self.list)
            value = super().__call__(value)
        elif self.dict:
            #value = wrap_batch(self.dict)
            #value = super().__call__(value)
            func = super().__call__
            value = {self.kprint(k): func(v) for k, v in self.dict.items()}
        else:
            value = None
        if reset:
            self.reset()
        return value

    printer = FormatPrinter({float: "%.4f"})

    def __str__(self):
        return self.printer.pformat(self.value())


class MacroAverageTracker(MetricTracker):
    def forward(self, values):
        def func(x):
            return value(x.clone().detach().mean())
        def apply(x):
            if isinstance(x, dict):
                return {k: apply(v) for k, v in x.items()}
            elif isinstance(x, torch.Tensor):
                return func(x)
            else:
                return apply(torch.tensor(x))
        retval = apply(values)
        return retval


class ValueTracker(MetricTracker):
    def forward(self, values):
        return values


class BinaryPRF1Tracker(MetricTracker):
    def __init__(self):
        super().__init__(BinaryCMWithLogitsMetric())

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
        return {'P': value(p), 'R': value(r), 'F1': value(f1)}


class PRF1Tracker(BinaryPRF1Tracker):
    def __init__(self):
        super(BinaryPRF1Tracker, self).__init__(CMWithLogitsMetric())


class MetricKey():
    def __init__(self, pred, target):
        self.pred = pred
        self.target = target

    def __eq__(self, other):
        return self.pred == other.pred and self.target == other.target

    def __hash__(self):
        return hash((self.pred, self.target))

    def __str__(self):
        return '{}:{}'.format(self.pred, self.target)

    def __repr__(self):
        return str(self)
