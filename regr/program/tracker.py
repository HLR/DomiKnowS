import pprint
from collections import defaultdict
import torch

from ..base import AutoNamed
from ..utils import wrap_batch


class TrackerPrinter(pprint.PrettyPrinter):
    def _format(self, obj, *args, **kwargs):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                if (isinstance(k, tuple) and
                    len(k) == 2 and
                    isinstance(k[0], AutoNamed) and 
                    isinstance(k[1], AutoNamed)):
                    k = k[0].sup.name.name
                new_obj[k] = v
            obj = new_obj
        return super()._format(obj, *args, **kwargs)


class Tracker(dict):
    printer = TrackerPrinter()

    def reset(self):
        self.clear()

    def reduce(self):
        raise NotImplementedError

    def value(self, reset=False):
        value = {k: self.reduce(v) for k, v in self.items()}
 
        if reset:
            self.reset()

        return value

    def __setitem__(self, name, value):
        value = value.detach()
        self[name].append(value.view(-1))

    def __getitem__(self, name):
        return super().setdefault(name, list())

    def __str__(self):
        return self.printer.pformat(self.value())

    def append(self, dict_):
        for k, v in dict_.items():
            self[k] = v

class MacroAverageTracker(Tracker):
    def reduce(self, values):
        return torch.cat(values, dim=0).mean()


class ConfusionMatrixTracker(Tracker):
    def __init__(self):
        defaultdict.__init__(lambda : defaultdict(list))

    def __setitem__(self, name, value):
        self[name]['TP'].append(value['TP'])
        self[name]['TN'].append(value['TN'])
        self[name]['FN'].append(value['FN'])
        self[name]['FP'].append(value['FP'])

    def reduce(self, values):
        tp = torch.stack(values['TP']).sum().float()
        fp = torch.stack(values['FP']).sum().float()
        fn = torch.stack(values['FN']).sum().float()
        if tp > 0:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
        else:
            p = torch.zeros_like(tp)
            r = torch.zeros_like(tp)
            f1 = torch.zeros_like(tp)
        return {'P': p, 'R': r, 'F1': f1}
