import abc
from collections import Counter, defaultdict, OrderedDict
from contextlib import contextmanager
from threading import Lock
from itertools import chain
import pprint


def entuple(args):
    if isinstance(args, tuple):
        return args
    return (args,)


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
        lock = type(self).__context.__locks[self.scope_key]
        try:
            yield lock.acquire(self._blocking)
        finally:
            lock.release()

    @property
    def scope_key(self):
        return type(self)

    @staticmethod
    def class_scope(cls_):
        # cls_ is not caller (Scoped) class! This is a static function
        cls_.scope_key = Scoped.scope_key
        return cls_

    @staticmethod
    def instance_scope(cls_):
        # cls_ is not caller (Scoped) class! This is a static function
        def scope_key(self_):
            # claim recursive check at instance level so it is possible to show properties of the same type
            return self_
        cls_.scope_key = property(scope_key)
        return cls_


Scoped._Scoped__context = Scoped


class Named(Scoped):
    def __init__(self, name):
        Scoped.__init__(self, blocking=False)
        self.name = name

    def __repr__(self):
        cls = type(self)
        with self.scope() as detailed:
            if detailed and callable(getattr(self, 'what', None)):
                repr_str = '{class_name}(name=\'{name}\', what={what!r})'.format(class_name=cls.__name__,
                                                                               name=self.name,
                                                                               what=self.what())
            else:
                repr_str = '{class_name}(name=\'{name}\')'.format(class_name=cls.__name__,
                                                                  name=self.name)
        return repr_str


class AutoNamed(Named):
    _names = Counter()
    _objs = dict()

    @classmethod
    def clear(cls):
        cls._names.clear()
        cls._objs.clear()

    @staticmethod
    def localize_namespace(cls_):
        # cls_ is not caller (AutoNamed) class! This is a static function
        cls_._names = Counter()
        cls_._objs = dict()
        return cls_

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

    @staticmethod
    def named_singleton(cls_):
        # cls_ is not caller (AutoNamed) class! This is a static function
        if not issubclass(cls_, AutoNamed):
            raise TypeError('named_singleton can be applied to subclasses of AutoNamed,' +
                            ' but MRO of {} is given.'.format(cls_.mro()))

        def getter(name=None):
            return cls_.get(name, None)

        def setter(obj):
            pass  # AutoNamed save them already

        cls_ = singleton(cls_, getter, setter)
        return cls_


class Propertied(object):
    def __init__(self):
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
            for prop in list(self.props.keys()):
                self.release(prop)
        else:
            del self[prop]


class Scorable(AutoNamed):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def score(self, *args, **kwargs): pass

    def __call__(self, *args, **kwargs):
        return self.score(*args, **kwargs)


class NamedTreeNode(Named):
    def __init__(self, name=None):
        Named.__init__(self, name)
        cls = type(self)
        self.sup = cls.context

    context = None

    @staticmethod
    def localize_context(cls_):
        # cls_ is not caller (NamedTree) class! This is a static function
        cls_.context = None
        return cls_

    def __enter__(self):
        cls = type(self)
        cls.context = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cls = type(self)
        cls.context = self.sup

    @property
    def sup(self):
        return self._sup

    @sup.setter
    def sup(self, sup):
        if sup is None:
            self._sup = None
            return
        sup.attach(self)

    @property
    def sups(self):
        yield self.sup
        if self.sup is not None:
            yield self.sup.sup

    @property
    def fullname(self, delim='/'):
        if self.sup is None:
            return self.name
        return self.sup.fullname + delim + self.name

    def what(self):
        return {'sup': self.sup}


@Scoped.instance_scope
class NamedTree(NamedTreeNode, OrderedDict):
    def __hash__(self):
        return hash((type(self), self.name))

    def __init__(self, name=None):
        NamedTreeNode.__init__(self, name)
        OrderedDict.__init__(self)

    def attach(self, sub):
        # TODO: resolve and prevent recursive definition
        if sub in self.sups:
            raise ValueError('Recursive definition detected for attaching {} to {} with sups {}'.format(
                sub.name, self.name, list(self.sups)))
        if isinstance(sub, NamedTreeNode):
            sub.detach(sub)
            sub._sup = self
        if isinstance(sub, Named):
            self[sub.name] = sub
        else:
            raise TypeError(
                'Attach Named instance to NamedTree, {} instance given.'.format(type(obj)))

    def detach(self, sub=None, all=False):
        if sub is None:
            if all:
                return self.clear()
            # detach all leaves
            for sub in self.values():
                if isinstance(sub, NamedTree):
                    sub.detach()
                else:
                    self.detach(sub)
            return
        # detach specific sub
        if sub.name in self:
            del self[sub.name]

    def traverse_apply(self, names, func):
        if len(names) > 1:
            return self[names[0]].traverse_apply(names[1:], func)
        # this is only one layer above the leaf layer
        return func(self, names[0])

    def parse_traverse_apply(self, names, func, delim='/', trim=True):
        names = list(chain(*(name.split(delim) for name in names)))
        if trim:
            names = [name.strip() for name in names]
        return self.traverse_apply(names, func)

    def get_sub(self, *names, delim='/', trim=True):
        return self.parse_traverse_apply(names, lambda d, k: OrderedDict.__getitem__(d, k), delim, trim)

    def set_sub(self, *names, sub, delim='/', trim=True):
        # NB: obj is keyword arg because it is after a list arg
        return self.parse_traverse_apply(names, lambda d, k: OrderedDict.__setitem__(d, k, sub), delim, trim)

    def del_sub(self, *names, delim='/', trim=True):
        return self.parse_traverse_apply(names, lambda d, k: OrderedDict.__delitem__(d, k), delim, trim)

    def __getitem__(self, name):
        return self.get_sub(*entuple(name))

    def __setitem__(self, name, obj):
        return self.set_sub(*entuple(name), sub=obj)

    def __delitem__(self, name):
        return self.del_sub(*entuple(name))

    def what(self):
        wht = NamedTreeNode.what(self)
        #wht['subs'] = self.keys()
        #import pdb
        #pdb.set_trace()
        return wht


class SubScorable(Scorable, NamedTree):
    def __init__(self, name=None):
        Scorable.__init__(self, name)
        NamedTree.__init__(self, name)

    def score(self, depth=float('inf'), *args, **kwargs):
        score = 0
        if depth > 0:
            score += sum([sub(depth - 1, *args, **kwargs)
                          for sub in self._subs])
        return score
