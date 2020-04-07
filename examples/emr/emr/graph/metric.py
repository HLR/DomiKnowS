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


class BCEFocalLoss(BCEWithLogitsLoss):
    def __init__(self, weight=None, pos_weight=None, reduction='mean', alpha=1, gamma=2, with_logits=True):
        super().__init__(weight=weight, pos_weight=pos_weight, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits

    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight
        if self.with_logits:
            bce = F.binary_cross_entropy_with_logits(input, target, weight,
                                                     pos_weight=self.pos_weight,
                                                     reduction='none')
        else:
            # TODO: update weight based on pos_weight
            bce = F.binary_cross_entropy(input, target, weight, reduction='none')

        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt)**self.gamma * bce

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum

        raise ValueError('Unknown reduction method "{}"'.format(self.reduction))

class BCEWithLogitsFocalLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction='mean', alpha=0.5, gamma=2):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight or 1
        logp = torch.nn.functional.logsigmoid(input)
        lognp = logp - input  # log(1-1/(1+exp(-x))) = log(exp(-x)/(1+exp(-x))) = log(exp(-x)) + log(1/(1+exp(-x)))
        p = torch.exp(logp)
        # FL(p_t) = - alpha_t * (1 - p_t) ** gamma  * log(p_t)
        loss = - self.alpha * (1 - p)**self.gamma * target * logp
        loss += - (1 - self.alpha) * p**self.gamma * (1 - target) * lognp
        loss *= weight

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum

        raise ValueError('Unknown reduction method "{}"'.format(self.reduction))

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
            raise RuntimeError('{} cannot be used as list-like and dict-like the same time.'.format(type(self)))
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
