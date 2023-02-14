from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..base import AutoNamed
from ..utils import wrap_batch


class CMWithLogitsMetric(torch.nn.Module):
    def forward(self, input, target, data_item, prop, weight=None):
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


def calc_TP_FP_TN_FN_for_single_class(val):
    y = val["labels"]
    p = val["preds"]
    TP,FP,TN,FN=0,0,0,0
    for i,j in zip(y,p):
        if i==j and i==1:
            TP+=1
        elif i==j and i==0:
            TN+=1
        elif not i==j and i==1:
            FN+=1
        elif not i == j and i == 0:
            FP += 1
    return {"TP": TP, 'FP': FP, 'TN': TN, 'FN': FN}


class DatanodeCMMetric(torch.nn.Module):
    def __init__(self, inferType='ILP'):
        super().__init__()
        self.inferType = inferType

    def forward(self, input, target, data_item, prop, weight=None):
        if (data_item.needsBatchRootDN()):
            data_item.addBatchRootDN()
        datanode = data_item.getDataNode(context=self.inferType)
        result = datanode.getInferMetrics(prop.name, inferType=self.inferType)
        if len(result.keys())==2:
            if str(prop.name) in result:
                val =  result[str(prop.name)]
                return calc_TP_FP_TN_FN_for_single_class(val)
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
            #value = wrap_batch(self.dict)
            #value = super().__call__(value)
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

def frp_from_matrix(i,matrix):
    matrix=np.array(matrix)
    TP=matrix[i][i]
    TN=matrix.sum()-matrix[i].sum()-matrix[:,i].sum()+matrix[i][i]
    FN=matrix[i].sum()-matrix[i][i]
    FP=matrix[:,i].sum()-matrix[i][i]
    return TP,TN,FP,FN

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

            if tp:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
            else:
                p = torch.zeros_like(torch.tensor(tp))
                r = torch.zeros_like(torch.tensor(tp))
                f1 = torch.zeros_like(torch.tensor(tp))
            if (tp + fp + fn + tn):
                accuracy=(tp + tn) / (tp + fp + fn + tn)
            return {'P': p, 'R': r, 'F1': f1,"accuracy":accuracy}
        elif values[0]:
            output={}
            names=values[0]["class_names"][:]
            n=len(names)

            matrix=[[0 for i in range(n)] for j in range(n)]
            for batch in values:
                for label,pred in zip(batch["labels"],batch["preds"]):
                    matrix[label][pred]+=1
            if self.confusion_matrix:
                output[str(names)]=matrix
            for name in names:
                TP,TN,FP,FN=frp_from_matrix(names.index(name),matrix)
                if (TP+FP):
                    output[name+" Precision"]=TP/(TP+FP)
                else:
                    output[name + " Precision"] = 0
                if (TP+FN):
                    output[name + " Recall"] =TP/(TP+FN)
                else:
                    output[name + " Recall"]=0
                if (output[name+" Precision"]+output[name + " Recall"]):
                    output[name + " F1"] =2*(output[name+" Precision"]*output[name + " Recall"])/(output[name+" Precision"]+output[name + " Recall"])
                else:
                    output[name + " F1"]=0
                if (TP+TN+FP+FN):
                    output[name + " Accuracy"] =(TP+TN)/(TP+TN+FP+FN)
                else:
                    output[name + " Accuracy"]=0
            output["Total Accuracy of All Classes"]=sum([matrix[i][i] for i in range(n)])/sum([sum(matrix[i]) for i in range(n)])
            return output


class BinaryPRF1Tracker(PRF1Tracker):
    def __init__(self, metric=BinaryCMWithLogitsMetric()):
        super().__init__(metric)
