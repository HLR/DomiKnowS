import inspect
import keyword

from collections import OrderedDict


def extract_args(*args, **kwargs):
    if '_stack_back_level_' in kwargs and kwargs['_stack_back_level_']:
        level = kwargs['_stack_back_level_']
    else:
        level = 0
    frame = inspect.currentframe()
    for _ in range(level + 1):
        frame = frame.f_back
    # print(frame.f_code.co_names)
    names = frame.f_code.co_names[1:]  # skip the first one, the callee name
    names = [name for name in names if not keyword.iskeyword(
        name) and not name in dir(__builtins__)]  # skip python keywords and buildins
    # print(names)
    if not all(name in frame.f_locals for name in names):
        raise TypeError(('Please do not use any expression, but direct variable names,' +
                         ' in caller on Line {} in File {}.')
                        .format(frame.f_lineno, frame.f_code.co_filename))
    return OrderedDict((name, frame.f_locals[name]) for name in names)


def log(*args, **kwargs):
    args = extract_args(_stack_back_level_=1)
    for k, v in args.items():
        v = '{}'.format(v)
        if '\n' in v:
            print('{}:\n{}'.format(k, v))
        else:
            print('{}: {}'.format(k, v))
