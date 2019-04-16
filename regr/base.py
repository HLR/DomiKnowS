import abc
from collections import Counter, defaultdict, Iterable
from contextlib import contextmanager
from threading import Lock
import pprint


class Scoped(object):
    '''
    A lock that count all trials of acquiring and require same number of releasing to really unlock.
    Who else need this...? Well, let me leave it nested inside then...
    '''
    class LevelLock(object):
        # note: threading.Lock is a `builtin_function_or_method`, not a class!
        def __init__(self):
            self.__lock = Lock()
            self.__level = 0

        @property
        def level(self):
            return self.__level

        def acquire(self, blocking=False):  # in this setting, you don't usually want blocking
            retval = self.__lock.acquire(blocking)
            self.__level += 1
            return retval

        def release(self):
            self.__level -= 1
            if self.__level == 0:
                self.__lock.release()

        def __repr__(self):
            return repr(self.__level)

    __locks = defaultdict(LevelLock)

    def __init__(self, blocking=False):
        self._blocking = blocking

    @contextmanager
    def scope(self, blocking=None):
        if blocking is None:
            blocking = self._blocking
        lock = self.__context.__locks[self.scope_key]
        try:
            yield lock.acquire(self._blocking)
        finally:
            lock.release()

    @property
    def scope_key(self):
        return type(self)


Scoped._Scoped__context = Scoped


class Named(Scoped):
    def __init__(self, name):
        Scoped.__init__(self, blocking=False)
        self.name = name

    def __repr__(self):
        cls = type(self)
        with self.scope() as need_detail:
            if need_detail and callable(getattr(self, 'what', None)):
                repr_str = '{class_name}(name=\'{name}\', what={what})'.format(class_name=cls.__name__,
                                                                               name=self.name,
                                                                               what=pprint.pformat(self.what(), width=8))
            else:
                repr_str = '{class_name}(name=\'{name}\')'.format(class_name=cls.__name__,
                                                                  name=self.name)
        return repr_str


def local_names_and_objs(cls):
    cls._names = Counter()
    cls._objs = dict()
    return cls


def named_singleton(cls):
    class singleton():
        def __new__(cls_, name):
            if name in cls._objs:
                return cls.get()
            cls.__new__(cls, name)
    return cls


@local_names_and_objs
class AutoNamed(Named):
    @classmethod
    def clear(cls):
        cls._names.clear()
        cls._objs.clear()

    @classmethod
    def get(cls, name, value=None):
        return cls._objs.get(name, value)

    @classmethod
    def suggest_name(cls):
        return cls.__name__.lower()

    def assign_suggest_name(self, name=None):
        cls = type(self)
        if name is None:
            name = cls.suggest_name()
        if cls._names[name] > 0:
            while True:
                name_attempt = '{}-{}'.format(name, cls._names[name])
                cls._names[name] += 1
                if cls._names[name_attempt] == 0:
                    name = name_attempt
                    break  # while True
        assert cls._names[name] == 0
        cls._names[name] += 1
        self.name = name
        cls._objs[name] = self

    def __init__(self, name=None):
        Named.__init__(self, name)  # temporary name may apply
        self.assign_suggest_name(name)


class Scorable(AutoNamed):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def score(self, prop): pass

    def __call__(self, prop=None, *args, **kwargs):
        if prop is None:
            return sum([self(prop) for prop, _ in self])
        return self.score(prop, *args, **kwargs)


class Tree(AutoNamed):
    default = None

    def __init__(self, name=None):
        AutoNamed.__init__(self, name)
        self._subs = []
        self._sup = None

    @property
    def scope_key(self):
        # claim recursive check at instance level so it is possible to show children
        return self

    @property
    def sup(self):
        return self._sup

    @sup.setter
    def sup(self, sup):
        # TODO: resolve and prevent recursive definition
        if sup is not None:
            self._sup = sup
            sup._subs.append(self)

    def __enter__(self):
        self.sup = type(self).default
        type(self).default = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        type(self).default = self.sup

    def what(self):
        return {'sup': self.super,
                'sub': self.sub}


class SubScorable(Scorable, Tree):
    def __init__(self, name=None):
        Scorable.__init__(self, name)
        Tree.__init__(self, name)

    def score(self, prop, depth=float('inf')):
        score = 0
        if depth > 0:
            score += sum([sub(prop, depth - 1) for sub in self._subs])
        return score


class Propertied(AutoNamed):
    def __init__(self, name=None):
        AutoNamed.__init__(self, name)
        self._props = defaultdict(list)

    @property
    def props(self):
        return self._props

    def __iter__(self):
        return self.props.items()

    def __getitem__(self, prop):
        return self.props[prop]

    def __setitem__(self, prop, value):
        self.props[prop] = value

    def __delitem__(self, prop):
        del self.props[prop]

    def __len__(self, prop):
        return len(self.props)

    def release(self, prop=None):
        if prop is None:
            for prop, _ in self.props:
                self.release(prop)
        del self[prop]


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


def named_singleton(cls):
    if not issubclass(cls, AutoNamed):
        raise TypeError('named_singleton can be applied to subclasses of AutoNamed,' +
                        ' but MRO of {} is given.'.format(cls.mro()))

    def getter(name=None):
        return cls.get(name, None)

    def setter(obj):
        pass  # AutoNamed save them already

    cls = singleton(cls, getter, setter)
    return cls
