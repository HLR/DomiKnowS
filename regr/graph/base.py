import abc
from collections import Counter, defaultdict, OrderedDict
from contextlib import contextmanager
from threading import Lock
from itertools import chain
import pprint
from ..utils import entuple, singleton, hide_inheritance


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


class NamedTreeNode(Named):
    _context = []

    @staticmethod
    def localize_context(cls_, default=None):
        # cls_ is not caller (NamedTree) class! This is a static function
        cls_._context = []
        return cls_

    def __init__(self, name=None):
        Named.__init__(self, name)
        cls = type(self)
        self.attach_to_context()

    def attach_to_context(self):
        cls = type(self)
        if len(cls._context) == 0:
            self.sup = None
        else:
            self.sup = cls._context[-1]

    def __enter__(self):
        cls = type(self)
        #self.sup = cls._context
        self.attach_to_context()  # TODO: this could lead to switching sup to context
        cls._context.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cls = type(self)
        last = cls._context.pop()
        assert last is self

    @property
    def sup(self):
        return self._sup

    @sup.setter
    def sup(self, sup):
        self._sup = None  # NB: sup.attach will check _sup, so keep this line here
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

    def __hash__(self):  # NB: OrderedDict is unhashable. We want NamedTree hashable, by name
        return hash((type(self), self.name))

    # def __eq__(self): # TODO: OrderedDict has __eq__, what do we want for Tree?
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


@NamedTree.localize_context
class BaseGraphTree(AutoNamed, NamedTree):
    def __init__(self, name=None, ontology=None):
        AutoNamed.__init__(self, name)  # name may be update
        NamedTree.__init__(self, self.name)
