from collections import Iterable, OrderedDict
from .base import AutoNamed, local_names_and_objs
from .entity import Entity


@local_names_and_objs
class Relation(AutoNamed):
    def __init__(self, src, dst, T=None, name=None):
        AutoNamed.__init__(self, name)
        self._src = src
        self._dst = dst
        self.T = T

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        if isinstance(src, Entity):
            self._src = src
        else:
            raise TypeError('Unsupported type of relation src {} in relation {}. Supported types is Entity.'
                            .format(type(src), self, ))

    @property
    def dst(self):
        if isinstance(self._dst, Entity):
            enum = {0: self._dst}.items()
        elif isinstance(self._dst, Iterable):
            if isinstance(self._dst, dict):
                enum = self._dst.items()
            else:
                enum = enumerate(self._dst)
        else:
            raise TypeError(
                'Unsupported type of relation dst {} in relation {}. Supported types are Entity or Iterable (including dict).'
                .format(type(self._dst), self, ))

        for k, v in enum:
            yield (k, v)

    @dst.setter
    def dst(self, dst):
        if isinstance(dst, Entity):
            self._dst = OrderedDict({0: dst})
        elif isinstance(dst, OrderedDict):
            self._dst = dst
        elif isinstance(dst, dict):
            self._dst = OrderedDict(dst)
            raise UserWarning(
                'Please use OrderedDict rather than dict to prevent unpredictable order of arguments in the relationship. For this instance, {} is used.'
                .format(self._dst))
        elif isinstance(self._dst, Iterable):
            self._dst = OrderedDict(enumerate(dst))
        else:
            raise TypeError(
                'Unsupported type of relation dst {} in relation {}. Supported types are Entity or Iterable (including dict but please use OrderedDict for consistence argument order).'
                .format(type(dst), self, ))

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, T):
        self._T = T

    def what(self):
        return {
            'src': self._src,
            'dst': self._dst,
        }


@Entity.register_rel_types('be')
class Be(Relation):
    def __init__(self, src, dst, name=None):
        # force T=None as identical
        Relation.__init__(self, src, dst, T=None, name=name)


@Entity.register_rel_types('have')
class Have(Relation):
    pass
