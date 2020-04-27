import inspect
import keyword
from functools import reduce
import operator
from collections import OrderedDict, Counter
from typing import Iterable
from contextlib import contextmanager
import warnings


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


def printablesize(ni):
    if hasattr(ni, 'shape'):
        return 'tensor' + str(tuple(ni.shape)) + ''
    elif isinstance(ni, Iterable):
        if len(ni) > 0:
            return 'iterable(' + str(len(ni)) + ')' + '[' + printablesize(ni[0]) + ']'
        else:
            return 'iterable(' + str(len(ni)) + ')[]'
    else:
        return str(type(ni))


def entuple(args):
    if isinstance(args, tuple):
        return args
    return (args,)


def enum(inst, cls=None, offset=0):
    if isinstance(inst, cls):
        enum = {offset: inst}.items()
    elif isinstance(inst, OrderedDict):
        enum = inst.items()
    elif isinstance(inst, dict):
        enum = inst.items()
        if inst:  # if dict not empty
            # NB: stacklevel
            #     1 - utils.enum()
            #     2 - concept.Concept.relation_type.<local>.update.<local>.create()
            #     3 - concept.Concept.__getattr__.<local>.handle()
            #     4 - __main__()
            warnings.warn('Please use OrderedDict rather than dict to prevent unpredictable order of arguments.' +
                          'For this instance, {} is used.'
                          .format(OrderedDict((k, v.name) for k, v in inst.items())),
                          stacklevel=4)
    elif isinstance(inst, Iterable):
        enum = enumerate(inst, offset)
    else:
        raise TypeError('Unsupported type of instance ({}). Use cls specified type, OrderedDict or other Iterable.'
                        .format(type(inst)))

    return enum


@contextmanager
def hide_class(inst, clsinfo, sub=True):  # clsinfo is a type of a tuple of types
    if isinstance(inst, clsinfo):
        if isinstance(clsinfo, type):
            clsinfo = (clsinfo,)

        from six.moves import builtins
        isinstance_orig = builtins.isinstance

        def _isinstance(inst_, clsinfo_):  # clsinfo_ is a type of a tuple of types
            if inst_ is inst:
                if isinstance_orig(clsinfo_, type):
                    clsinfo_ = (clsinfo_,)
                clsinfo_ = [cls_
                            for cls_ in clsinfo_
                            if not (
                                sub and issubclass(cls_, clsinfo)
                            ) and not (
                                not sub and cls_ in clsinfo
                            )]
                clsinfo_ = tuple(clsinfo_)
                # NB: isinstance(inst, ()) == False
            return isinstance_orig(inst_, clsinfo_)

        builtins.isinstance = _isinstance

        try:
            yield inst
        finally:
            builtins.isinstance = isinstance_orig
    else:
        yield inst


@contextmanager
# clsinfo is a type of a tuple of types
def hide_inheritance(cls, clsinfo, sub=True, hidesub=True):
    if issubclass(cls, clsinfo):
        if isinstance(clsinfo, type):
            clsinfo = (clsinfo,)

        from six.moves import builtins
        isinstance_orig = builtins.isinstance
        issubclass_orig = builtins.issubclass

        def _isinstance(inst, clsinfo_):
            if (hidesub and isinstance_orig(inst, cls)
                ) or (
                not hidesub and type(inst) is cls
            ):
                # not sure would this hurt somewhere?
                # the following issubclass is dynamic!
                return any(issubclass(cls_, clsinfo_) for cls_ in {type(inst), inst.__class__})
            return isinstance_orig(inst, clsinfo_)

        def _issubclass(cls_, clsinfo_):  # clsinfo_ is a type of a tuple of types
            if (hidesub and issubclass_orig(cls_, cls)
                ) or (
                not hidesub and cls_ is cls
            ):
                if isinstance_orig(clsinfo_, type):
                    clsinfo_ = (clsinfo_,)
                clsinfo_ = [cls__
                            for cls__ in clsinfo_
                            if not (
                                sub and issubclass_orig(cls__, clsinfo)
                            ) and not (
                                not sub and cls__ in clsinfo
                            )]
                clsinfo_ = tuple(clsinfo_)
            return issubclass_orig(cls_, clsinfo_)

        builtins.isinstance = _isinstance
        builtins.issubclass = _issubclass

        try:
            yield
        finally:
            builtins.isinstance = isinstance_orig
            builtins.issubclass = issubclass_orig
    else:
        yield

        from contextlib import contextmanager


def singleton(cls, getter=None, setter=None):
    if getter is None:
        def getter(*args, **kwargs):
            if hasattr(cls, '__singleton__'):
                return cls.__singleton__
            return None
    if setter is None:
        def setter(obj):
            cls.__singleton__ = obj

    __old_new__ = cls.__new__

    def __new__(cls, *args, **kwargs):
        obj = getter(*args, **kwargs)
        if obj is None:
            obj = __old_new__(cls, *args, **kwargs)
            obj.__i_am_the_new_singoton__ = True
            setter(obj)
            return obj
        else:
            return obj

    __old_init__ = cls.__init__

    def __init__(self, *args, **kwargs):
        if hasattr(self, '__i_am_the_new_singoton__') and self.__i_am_the_new_singoton__:
            del self.__i_am_the_new_singoton__
            __old_init__(self, *args, **kwargs)

    cls.__new__ = staticmethod(__new__)
    cls.__init__ = __init__
    return cls


class WrapperMetaClass(type):
    def __call__(cls, inst, *args, **kwargs):
        if not isinstance(inst, tuple(cls.mro())):
            raise TypeError(
                'Only cast from {}, while {} is given.'.format(super(cls), type(inst)))

        inst.__class__ = cls
        # no need to call cls.__new__ because we do not need new instance
        inst.__init__(*args, **kwargs)
        return inst


def optional_arg_decorator(fn, test=None):
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and (test is None or test(args[0])):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args, **kwargs)
            return real_decorator
    return wrapped_decorator


def optional_arg_decorator_for(test):
    return lambda fn: optional_arg_decorator(fn, test)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def guess_device(context):
    import torch
    poll = Counter()
    for value in context.values():
        if isinstance(value, dict):
            poll += guess_device(value)
        elif isinstance(value, torch.Tensor):
            poll[value.device] += 1
        else:
            poll[None] += 1
    return poll


def find_base(s, n):
    from scipy.optimize import minimize, minimize_scalar
    # NB: `n` here is the number of terms in this "geometric series", including 0 and last k.
    #     So the "n+1" in the original formula is `n` here.
    length = lambda b: (1 - b ** n) / (1 - b)
    res = minimize_scalar(lambda b : (length(b) - s) ** 2, method='bounded', bounds=(1, (s-1)**(1./n)))
    return res.x

def isbad(x):
    return (
        x != x or  # nan
        abs(x) == float('inf')  # inf
    )
