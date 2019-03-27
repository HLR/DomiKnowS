from collections import OrderedDict
from .base import AutoNamed, local_names_and_objs
from .concept import Concept, enum


@local_names_and_objs
class Relation(AutoNamed):
    def __init__(self, src, dst, T=None, name=None):
        AutoNamed.__init__(self, name)
        self.src = src
        self.dst = dst
        self.T = T

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        if isinstance(src, Concept):
            self._src = src
        else:
            raise TypeError('Unsupported type of relation src {} in relation {}. Supported types is Concept.'
                            .format(type(src), self, ))

    @property
    def dst(self):
        return enum(self._dst)

    @dst.setter
    def dst(self, dst):
        self._dst = OrderedDict(enum(dst))

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


@Concept.register_rel_types('be')
class Be(Relation):
    def __init__(self, src, dst, name=None):
        # force T=None as identical
        Relation.__init__(self, src, dst, T=None, name=name)


@Concept.register_rel_types('have')
class Have(Relation):
    pass
