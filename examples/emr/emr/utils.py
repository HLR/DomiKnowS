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


def print_reformat(value, formatter=None):
    import torch
    #from pprint import pformat

    reformatter = {
        dict: lambda value: ', '.join('%s: %s' % (print_reformat(k, formatter), print_reformat(v, formatter)) for k, v in value.items()),
        torch.Tensor: lambda value: value.cpu().numpy(),
        None: str
    }
    if formatter:
        reformatter.update(formatter)
    for value_type, value_func in reformatter.items():
        if value_type is not None and isinstance(value, value_type):
            return value_func(value)
    return reformatter[None](value)


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
