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


from contextlib import contextmanager


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
                repr_str = '{class_name}(name=\'{name}\', what={what})'.format(class_name=cls.__name__,
                                                                               name=self.name,
                                                                               what=pprint.pformat(self.what(), width=8))
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
    def localize_context(cls_, default=None):
        # cls_ is not caller (NamedTree) class! This is a static function
        cls_.context = default
        return cls_

    def __enter__(self):
        cls = type(self)
        self.sup = cls.context # TODO: this could lead to switching sup to context
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
        self._sup = None # NB: sup.attach will check _sup, so keep this line here
        if sup is not None:
            sup.attach(self)

    @property
    def sups(self):
        if self.sup is not None:
            yield self.sup
            yield from self.sup.sups

    @property
    def fullname(self, delim='/'):
        if self.sup is None:
            return self.name
        return self.sup.fullname + delim + self.name

    def what(self):
        return {'sup': self.sup}


@Scoped.instance_scope
class NamedTree(NamedTreeNode, OrderedDict):
    def __repr__(self):
        with hide_inheritance(NamedTree, dict):
            # prevent pprint over optimize OrderedDict(dict) on NamedTree instance
            return NamedTreeNode.__repr__(self)

    def __hash__(self): # NB: OrderedDict is unhashable. We want NamedTree hashable, by name
        return hash((type(self), self.name))

    #def __eq__(self): # TODO: OrderedDict has __eq__, what do we want for Tree?
    #    return ...

    def __init__(self, name=None):
        NamedTreeNode.__init__(self, name)
        OrderedDict.__init__(self)

    def attach(self, sub):
        # resolve and prevent recursive definition
        if sub is self or sub in self.sups:
            raise ValueError('Recursive definition detected for attaching {} to {} with sups {}'.format(
                sub.name, self.name, list(self.sups)))
        if isinstance(sub, NamedTreeNode):
            if sub._sup is not None:
                sub._sup.detach(sub)
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

    def query_apply(self, names, func):
        if len(names) > 1:
            return self[names[0]].query_apply(names[1:], func)
        # this is only one layer above the leaf layer
        return func(names[0])

    def parse_query_apply(self, names, func, delim='/', trim=True):
        names = list(chain(*(name.split(delim) for name in names)))
        if trim:
            names = [name.strip() for name in names]
        return self.query_apply(names, func)

    def get_apply(self, name):
        return OrderedDict.__getitem__(self, name)

    def get_sub(self, *names, delim='/', trim=True):
        return self.parse_query_apply(names, self.get_apply, delim, trim)

    def set_apply(self, name, sub):
        OrderedDict.__setitem__(self, name, sub)

    def set_sub(self, *names, sub, delim='/', trim=True):
        # NB: sub is keyword arg because it is after a list arg
        return self.parse_query_apply(names, lambda k: self.set_apply(k, sub), delim, trim)

    def del_apply(self, name):
        return OrderedDict.__delitem__(self, name)

    def del_sub(self, *names, delim='/', trim=True):
        return self.parse_query_apply(names, self.del_apply, delim, trim)

    def __getitem__(self, name):
        return self.get_sub(*entuple(name))

    def __setitem__(self, name, obj):
        return self.set_sub(*entuple(name), sub=obj)

    def __delitem__(self, name):
        return self.del_sub(*entuple(name))

    def what(self):
        wht = NamedTreeNode.what(self)
        wht['subs'] = dict(self)
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
