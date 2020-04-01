def seed(s=1, deterministic=True):
    import os
    import random
    import numpy as np
    import torch

    os.environ['PYTHONHASHSEED'] = str(s)  # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)  # this function will call torch.cuda.manual_seed_all(s) also

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Namespace(dict):
    def __init__(self, __dict={}, **kwargs):
        self.__dict__ = self
        self.update(__dict)
        self.update(kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Namespace(v)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ','.join('\'{}\':{}'.format(k,v) for k,v in self.items()))

    def clone(self):
        from copy import copy
        return Namespace(copy(self))

    def deepclone(self):
        from copy import deepcopy
        return Namespace(deepcopy(self))

    def __getitem__(self, key):
        return self.get(key, None)


def caller_source():
    import inspect

    for frame in inspect.getouterframes(inspect.currentframe(), context=1)[2:]:
        if frame.code_context is not None:
            try:
                with open(frame.filename, 'r') as fin:
                    return fin.read()
                break
            except FileNotFoundError as ex:
                ex = type(ex)('{}\n'
                              'Please run from a file base environment, '
                              'rather than something like notebook.'.format(ex))
                raise ex
    else:
        raise RuntimeError('Who is calling?')


def dict_zip(*dicts, fillvalue=None):  # https://codereview.stackexchange.com/a/160584
    all_keys = {k for d in dicts for k in d.keys()}
    return {k: [d.get(k, fillvalue) for d in dicts] for k in all_keys}


def wrap_batch(values, fillvalue=0):
    import torch

    if isinstance(values, (list, tuple)):
        if isinstance(values[0], dict):
            values = dict_zip(*values, fillvalue=fillvalue)
            values = {k: wrap_batch(v, fillvalue=fillvalue) for k, v in values.items()}
        elif isinstance(values[0], torch.Tensor):
            values = torch.stack(values)
    elif isinstance(values, dict):
        values = {k: wrap_batch(v, fillvalue=fillvalue) for k, v in values.items()}
    return values


# consume(it) https://stackoverflow.com/q/50937966
import sys

if sys.implementation.name == 'cpython':
    import collections
    def consume(it):
        collections.deque(it, maxlen=0)
else:
    def consume(it):
        for _ in it:
            pass


def print_result(model, epoch=None, phase=None):
    header = ''
    if epoch is not None:
        header += 'Epoch {} '.format(epoch)
    if phase is not None:
        header += '{} '.format(phase)
    print('{}Loss:'.format(header))
    loss = model.loss.value()
    for (pred, _), value in loss.items():
        print(' - ', pred.sup.prop_name.name, value.item())
    print('{}Metrics:'.format(header))
    metrics = model.metric.value()
    for (pred, _), value in metrics.items():
        print(' - ', pred.sup.prop_name.name, str({k: v.item() for k, v in value.items()}))
