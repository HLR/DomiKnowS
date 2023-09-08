from collections import defaultdict
from typing import Any
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from ..base import AutoNamed
from ..utils import wrap_batch


class CMWithLogitsMetric(torch.nn.Module):
    def forward(self, input, target, _, prop, weight=None):
        if weight is None:
            weight = torch.tensor(1, device=input.device)
        else:
            weight = weight.to(input.device)
            
        preds = input.argmax(dim=-1).clone().detach().to(dtype=weight.dtype)
        labels = target.clone().detach().to(dtype=weight.dtype, device=input.device)
        tp = (preds * labels * weight).sum()
        fp = (preds * (1 - labels) * weight).sum()
        tn = ((1 - preds) * (1 - labels) * weight).sum()
        fn = ((1 - preds) * labels * weight).sum()
        
        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


class BinaryCMWithLogitsMetric(CMWithLogitsMetric):
    def forward(self, input, target, data_item, prop, weight=None):
        target = target.argmax(dim=-1)
        return super().forward(input, target, data_item, prop, weight)


class MultiClassCMWithLogitsMetric(CMWithLogitsMetric):
    def __init__(self, num_classes, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, input, target, data_item, prop, weight=None):
        from torch.nn import functional
        target = functional.one_hot(target, num_classes=self.num_classes)
        input = functional.one_hot(input.argmax(dim=-1), num_classes=self.num_classes)
        input = torch.stack((-input, input), dim=-1)
        if weight is None:
            weight = self.weight
        return super().forward(input, target, data_item, prop, weight)


class DatanodeCMMetric(torch.nn.Module):
    def __init__(self, inferType='ILP'):
        super().__init__()
        self.inferType = inferType

    def forward(self, input, target, data_item, prop, weight=None):
        datanode = data_item
        result = datanode.getInferMetrics(prop.name, inferType=self.inferType)
        if len(result.keys())==2:
            if str(prop.name) in result:
                val =  result[str(prop.name)]
                # if len(confusion_matrix(val["labels"], val["preds"]).ravel())<4:
                    # print()
                    # print("here")
                conf_mat = confusion_matrix(val["labels"], val["preds"])
                if conf_mat.size == 1:
                    if val["labels"][0] == 1:
                        TP = conf_mat[0, 0]
                        FP = FN = TN = 0
                    else:
                        TN = conf_mat[0, 0]
                        TP = FP = FN = 0
                else:
                    TN, FP, FN, TP = conf_mat.ravel()
                return {"TP": TP, 'FP': FP, 'TN': TN, 'FN': FN}
            else:
                return None
        else:
            names=list(result.keys())
            names.remove("Total")
            if names:
                names.remove(str(prop.name))
                return {"class_names":names,"labels":result[str(prop.name)]["labels"],"preds":result[str(prop.name)]["preds"]}


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
            func = super().__call__
            value = {self.kprint(k): func(v) for k, v in self.dict.items()}
        else:
            value = None
        if reset:
            self.reset()
        return value

    def __str__(self):
        value = self.value()
        
        if isinstance(value, dict):
            newValue = {}
            for v in value:
                if isinstance(value[v], dict):
                    newV = {}
                    for w in value[v]:
                        if torch.is_tensor(value[v][w]):
                            newV[w] = value[v][w].item()
                        else:
                            newV[w] = value[v][w]
                        
                    newValue[v] = newV   
                else:
                    if torch.is_tensor(value[v]):
                        newValue[v] = value[v].item()
                    else:
                        newValue[v] = value[v]
                   
            value = newValue
                                    
        return str(value)

class MacroAverageTracker(MetricTracker):
    def __init__(self, metric):
        super().__init__(metric)
        
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


class ValueTracker(MetricTracker):
    def forward(self, values):
        return values

class PRF1Tracker(MetricTracker):
    def __init__(self, metric=CMWithLogitsMetric(),confusion_matrix=False):
        super().__init__(metric)
        self.confusion_matrix=confusion_matrix

    def forward(self, values):
        if values[0] and not "class_names" in values[0]:

            CM = wrap_batch(values)

            if isinstance(CM['TP'], list):
                tp = sum(CM['TP'])
            else:
                tp = CM['TP'].sum().float()

            if isinstance(CM['FP'], list):
                fp = sum(CM['FP'])
            else:
                fp = CM['FP'].sum().float()

            if isinstance(CM['FN'], list):
                fn = sum(CM['FN'])
            else:
                fn = CM['FN'].sum().float()

            if isinstance(CM['TN'], list):
                tn = sum(CM['TN'])
            else:
                tn = CM['TN'].sum().float()

            # check if tp, fp, fn, tn are tensors if not make them tensors
            if not torch.is_tensor(tp):
                tp = torch.tensor(tp)
            if not torch.is_tensor(fp):
                fp = torch.tensor(fp)
            if not torch.is_tensor(fn):
                fn = torch.tensor(fn)
            if not torch.is_tensor(tn):
                tn = torch.tensor(tn)
                
            if tp:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
            else:
                p = torch.zeros_like(tp)
                r = torch.zeros_like(tp)
                f1 = torch.zeros_like(tp)
            if (tp + fp + fn + tn):
                accuracy=(tp + tn) / (tp + fp + fn + tn)
            return {'P': p, 'R': r, 'F1': f1,"accuracy":accuracy}
        elif values[0]:
            names=values[0]["class_names"][:]
            n=len(names)
            labels = np.concatenate([batch["labels"] for batch in values])
            preds = np.concatenate([batch["preds"] for batch in values])

            # remove the negative predictions
            labels = labels[preds >= 0]
            preds = preds[preds >= 0]

            report = classification_report(labels, preds, labels=np.arange(n), output_dict=True,zero_division=0)
            report = {**{names[i]: report[str(i)] for i in range(n)}, **{'weighted avg': report['weighted avg'], 'macro avg': report['macro avg'], 'accuracy': report['accuracy']}}
            return report


class BinaryPRF1Tracker(PRF1Tracker):
    def __init__(self, metric=BinaryCMWithLogitsMetric()):
        super().__init__(metric)
