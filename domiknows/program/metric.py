from collections import defaultdict
from typing import Any
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from ..base import AutoNamed
from ..utils import wrap_batch


class CMWithLogitsMetric(torch.nn.Module):
    """
    A utility class for computing confusion matrix metrics from logits.

    Inherits from:
        torch.nn.Module
    """

    def forward(self, input, target, _, prop, weight=None):
        """
        Computes True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) values
        from given logits and target.

        Args:
            input (torch.Tensor): The logits tensor.
            target (torch.Tensor): The ground truth labels tensor.
            _ : Placeholder, not used.
            prop: Placeholder, not used.
            weight (torch.Tensor, optional): Weights to apply to the input. Defaults to tensor of value 1.

        Returns:
            dict: A dictionary containing TP, FP, TN, and FN values.
        """
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


class DatanodeCMMetric(torch.nn.Module):
    """
    A utility class for computing confusion matrix metrics using datanode inference results.

    Inherits from:
        torch.nn.Module

    Attributes:
        inferType (str): The type of inference used to derive metrics.
    """
    def __init__(self, inferType='ILP'):
        """
        Initializes the DatanodeCMMetric with a specified inference type.

        Args:
            inferType (str, optional): The type of inference. Defaults to 'ILP'.
        """
        super().__init__()
        self.inferType = inferType

    def forward(self, input, target, data_item, prop, weight=None):
        """
        Computes the confusion matrix metrics using data from the provided datanode.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The ground truth labels.
            data_item: The datanode containing the inference metrics.
            prop: The property associated with the inference.
            weight (torch.Tensor, optional): An optional weight tensor. Defaults to None.

        Returns:
            dict/None: A dictionary containing the TP, FP, TN, FN values, or 
                       information on class names, labels, and predictions; 
                       returns None if the property name is not found in the results.
        """
        datanode = data_item
        result = datanode.getInferMetrics(prop.name, inferType=self.inferType)
        if len(result.keys())==2:
            if str(prop.name) in result:
                val =  result[str(prop.name)]
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
    """
    A utility class for tracking metrics for all datanodes.

    Attributes:
        metric (callable): The metric function to track.
        list (list): A list of metric values for all the datanodes.
        dict (defaultdict): A dictionary of metric values grouped by keys.
    """
    def __init__(self, metric):
        """
        Initializes the MetricTracker with a specified metric function.

        Args:
            metric (callable): The metric function to be tracked.
        """
        super().__init__()
        self.metric = metric
        self.list = []
        self.dict = defaultdict(list)

    def reset(self):
        """
        Resets the internal storage (both list and dict) to their initial empty state.
        """
        self.list.clear()
        self.dict.clear()

    def __call__(self, *args, **kwargs) -> Any:
        """
        Computes the metric using the provided arguments and stores the result in the internal list.

        Returns:
            Any: The computed metric value.
        """
        value = self.metric(*args, **kwargs)
        self.list.append(value)
        return value

    def __call_dict__(self, keys, *args, **kwargs) -> Any:
        """
        Computes the metric using the provided arguments and stores the result in the internal dictionary with the specified key.

        Args:
            keys: The key under which the metric value should be stored.

        Returns:
            Any: The computed metric value.
        """
        value = self.metric(*args, **kwargs)
        self.dict[keys].append(value)
        return value

    def __getitem__(self, keys):
        """
        Returns a lambda function that computes and stores the metric in the internal dictionary for a specified key.

        Args:
            keys: The key under which the metric value should be stored.

        Returns:
            callable: A lambda function to compute and store the metric value.
        """
        return lambda *args, **kwargs: self.__call_dict__(keys, *args, **kwargs)

    def kprint(self, k):
        """
        Custom key printing function based on the type and properties of the key.

        Args:
            k: The key to be printed.

        Returns:
            str: A string representation of the key.
        """
        if (
            isinstance(k, tuple) and
            len(k) == 2 and
            isinstance(k[0], AutoNamed) and 
            isinstance(k[1], AutoNamed)):
            return k[0].sup.name.name
        else:
            return k

    def value(self, reset=False):
        """
        Retrieves the value(s) of the computed metric(s).

        Args:
            reset (bool, optional): If True, resets the internal storage after retrieving the value. Defaults to False.

        Returns:
            Any: The metric value(s), either as a single value, list, or dictionary.
        """
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
        """
        Provides a string representation of the computed metric value(s).

        Returns:
            str: A string representation of the metric value(s).
        """
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
    """
    A utility class that extends the MetricTracker to compute macro-average of metrics for datanodes.

    Inherits from:
        MetricTracker
    """
    def __init__(self, metric):
        """
        Initializes the MacroAverageTracker with a specified metric function.

        Args:
            metric (callable): The metric function to be tracked.
        """
        super().__init__(metric)
        
    def forward(self, values):
        """
        Computes the macro-average for the given values.

        Args:
            values (Any): The input values (can be single value, list, tensor, or dictionary of values).

        Returns:
            Any: The macro-averaged value. The structure (tensor, list, or dictionary) of the output
                 mirrors the structure of the input.
        """
        def func(value):
            """
            Computes the mean of the provided tensor after detaching it.

            Args:
                value (torch.Tensor): A tensor value.

            Returns:
                torch.Tensor: The mean of the tensor.
            """
            return value.clone().detach().mean()
        def apply(value):
            """
            Recursively applies the mean computation based on the type of the input value.

            Args:
                value (Any): The input value to be averaged.

            Returns:
                Any: The averaged value.
            """
            if isinstance(value, dict):
                return {k: apply(v) for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                return func(value)
            else:
                return apply(torch.tensor(value))
        retval = apply(values)
        return retval


class PRF1Tracker(MetricTracker):
    """
    A tracker to calculate and monitor precision, recall, F1 score, and accuracy metrics.
 
    Inherits from the MetricTracker class.
    
    Methods:
    - forward: Processes input values to compute various metrics like precision, recall, F1 score, and accuracy.
    """
    def __init__(self, metric=CMWithLogitsMetric(),confusion_matrix=False):
        super().__init__(metric)
        self.confusion_matrix=confusion_matrix
        """
        Initialize the PRF1Tracker instance.
        
        Parameters:
        - metric (Metric, optional): An instance of the metric to be tracked. Defaults to CMWithLogitsMetric().
        - confusion_matrix (bool, optional): Whether to create confusion matrix values or not. Defaults to False.
        """
    def forward(self, values):
        """
        Processes the input values and computes precision, recall, F1 score, and accuracy metrics.

        Parameters:
        - values: Input data containing raw class names and predictions.
        
        Returns:
        - dict: A dictionary containing calculated metrics.

        If the input contains class names it means it is for a multiclass concept:
            Returns a classification report with metrics for each class and overall metrics 
            like 'weighted avg', 'macro avg', and 'accuracy' after negative classes are removed.

        Else:

            Returns metrics: 'P' (Precision), 'R' (Recall), 'F1' (F1 Score), and 'accuracy' for the bincaryclass concept.
        """
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
            labels = labels[preds >= 0]
            preds = preds[preds >= 0]

            report = classification_report(labels, preds, labels=np.arange(n), output_dict=True,zero_division=0)
            report = {**{names[i]: report[str(i)] for i in range(n)}, **{'weighted avg': report['weighted avg'], 'macro avg': report['macro avg'], 'accuracy': report['accuracy']}}
            return report
