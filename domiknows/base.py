import abc
import inspect
from collections import Counter, defaultdict, OrderedDict
from contextlib import contextmanager
from threading import Lock
from itertools import chain
import pprint
from .utils import entuple, singleton, hide_inheritance


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

    @classmethod
    def clear(cls):
        cls.__context.__locks.clear()

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
        super().__init__(blocking=False)
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

    def __str__(self):
        return str(self.name)


class AutoNamed(Named):
    _names = Counter()
    _objs = dict()

    @classmethod
    def clear(cls):
        super().clear()
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
    def localize_context(cls_):
        # cls_ is not caller (NamedTreeNode) class! This is a static function
        cls_._context = []
        return cls_

    @classmethod
    def share_context(cls, cls_):
        # cls is the caller (NamedTreeNode) class
        # cls_ is another class that need to share the same context
        cls_._context = cls._context
        return cls_

    @classmethod
    def clear(cls):
        super().clear()
        cls._context.clear()

    @classmethod
    def default(cls):
        if cls._context:
            return cls._context[-1]
        return None

    def __init__(self, name=None):
        super().__init__(name)
        cls = type(self)
        self._sup = None
        self.attach_to_context()

    def attach_to_context(self, name=None):
        cls = type(self)
        if len(cls._context) == 0:
            context = None
        else:
            context = cls._context[-1]
            context.attach(self, name)

    def attached(self, sup):
        pass

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

    @property
    def sups(self):
        if self.sup is not None:
            yield self.sup
            yield from self.sup.sups

    @property
    def fullname(self):
        return self.get_fullname()

    def get_fullname(self, delim='/'):
        if self.sup is None:
            return self.name
        return self.sup.get_fullname(delim) + delim + self.name

    def what(self):
        return {'sup': self.sup}


@Scoped.instance_scope
class NamedTree(NamedTreeNode, OrderedDict):
    def __repr__(self):
        with hide_inheritance(NamedTree, dict):
            # prevent pprint over optimize OrderedDict(dict) on NamedTree instance
            return super().__repr__()

    def __hash__(self):  # NB: OrderedDict is unhashable.
        # dict override __hash__ by setting it to None. Using super().__hash__() gives problem.
        return NamedTreeNode.__hash__(self)

    def __eq__(self, other): # NB: OrderedDict has __eq__, empty == empty, which is not what we expected
        # dict override __eq__ in an waired way also
        return NamedTreeNode.__eq__(self, other)

    def __init__(self, name=None):
        super().__init__(name)
        super(NamedTreeNode, self).__init__(self.name)

    def attach(self, sub, name=None):
        # resolve and prevent recursive definition
        if sub is self or sub in self.sups:
            raise ValueError('Recursive definition detected for attaching {} to {} with sups {}'.format(
                sub.name, self.name, [sup.name for sup in self.sups]))
        if isinstance(sub, NamedTreeNode):
            if sub.sup is not None:
                sub.sup.detach(sub)
            sub._sup = self # FIXME: using private method
        if isinstance(sub, Named):
            if name is None:
                name = sub.name
            self[name] = sub
        else:
            raise TypeError(
                'Attach Named instance to NamedTree, {} instance given.'.format(type(sub)))
        sub.attached(self)

    def detach(self, sub=None, all=False):
        if sub is None:
            if all:
                for sub in dict(self).values():
                    # dict() makes a shallow copy to avoid runtime error of change
                    self.detach(sub)
                #return self.clear()
            else:
                # detach all leaves
                for sub in dict(self).values():
                    # dict() makes a shallow copy to avoid runtime error of change
                    if isinstance(sub, NamedTree):
                        sub.detach()
                    else:
                        self.detach(sub)
            return
        # detach specific sub
        # NB: sub.name is not reliable!
        for key, value in dict(self).items():
            # dict() makes a shallow copy to avoid runtime error of change
            if value == sub:
                del self[key]
                sub._sup = None # TODO: what else to have?

    def traversal_apply(self, func, filter_fn=lambda x: x is not None, order='pre', first='depth'):
        if order.lower() not in ['pre', 'post']:
            raise ValueError('Options for order are pre or post, {} given.'.format(order))
        if first.lower() not in ['depth', 'breadth']:
            raise ValueError('Options for first are depth or breadth, {} given.'.format(first))

        to_traversal = [self, ]
        to_apply = []

        while to_traversal:
            if first.lower() == 'depth':
                current = to_traversal.pop()
            else:
                current = to_traversal.pop(0)
            to_apply.append(current)

            if order.lower() == 'pre':
                current_apply = to_apply.pop()
                if inspect.isgeneratorfunction(func):
                    # func: yield
                    if filter_fn:
                        yield from filter(filter_fn, func(current_apply))
                    else:
                        yield from func(current_apply)
                else:
                    # func: return
                    retval = func(current_apply)
                    if not filter_fn or filter_fn(retval):
                        yield retval

            if isinstance(current, NamedTree):
                subs = [current[name] for name in current] # compatible to subclasses that override the iterator
                if (first.lower() == 'depth' and order.lower() == 'pre') or (
                    first.lower() == 'breadth' and order.lower() == 'post'):
                    subs = reversed(subs)

                to_traversal.extend(subs)

        if order.lower() == 'post':
            while to_apply:
                current_apply = to_apply.pop()
                if inspect.isgeneratorfunction(func):
                    # func: yield
                    if filter_fn:
                        yield from filter(filter_fn, func(current_apply))
                    else:
                        yield from func(current_apply)
                else:
                    # func: return
                    retval = func(current_apply)
                    if not filter_fn or filter_fn(retval):
                        yield retval

    def extract_name(self, *names, delim='/', trim=True):
        if isinstance(names[0], str):
            name0s = names[0].split(delim)
            name = name0s[0]
            if trim:
                name = name.strip()
            names = list(chain(name0s[1:], names[1:]))
            # match typed name string
            if name[0] == '<' and name[-1] == '>':
                for key in self:
                    if key.name == name[1:-1]:
                        name = key
                        break
        else:
            name = names[0]
            names = names[1:]
        return name, names

    def cutGraphName(self, names):
        gNames = self._names.copy()
        index = 0
        
        for i, name in enumerate(names):
            index = i
            if (name in gNames) and (gNames[name] > 0):
                gNames[name] -= 1
            else:
                break
            
        return index# names[index:]
        
    def parse_query_apply(self, func, *names, delim='/', trim=True):
        name, names = self.extract_name(*names, delim=delim, trim=trim)
        if names:
            return self[name].parse_query_apply(func, *names, delim=delim, trim=trim)
        return func(self, name)

    def get_apply(self, name):
        return OrderedDict.__getitem__(self, name)

    def get_sub(self, *names, delim='/', trim=True):
        return self.parse_query_apply(lambda s, name: s.get_apply(name), *names, delim=delim, trim=trim)

    def set_apply(self, name, sub):
        OrderedDict.__setitem__(self, name, sub)

    def set_sub(self, *names, sub, delim='/', trim=True):
        # NB: sub is keyword arg because it is after a list arg
        return self.parse_query_apply(lambda s, k: s.set_apply(k, sub), *names, delim=delim, trim=trim)

    def del_apply(self, name):
        return OrderedDict.__delitem__(self, name)

    def del_sub(self, *names, delim='/', trim=True):
        return self.parse_query_apply(lambda s, name: s.del_apply(name), *names, delim=delim, trim=trim)

    def __getitem__(self, name):
        try:
            return self.get_sub(*entuple(name))
        except KeyError as e:
            raise type(e)(name)

    def __setitem__(self, name, obj):
        return self.set_sub(*entuple(name), sub=obj)

    def __delitem__(self, name):
        try:
            return self.del_sub(*entuple(name))
        except KeyError as e:
            raise type(e)(name)

    def what(self):
        wht = NamedTreeNode.what(self)
        wht['subs'] = dict(self)
        return wht
