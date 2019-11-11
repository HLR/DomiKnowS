import abc
from collections import OrderedDict
from itertools import chain
if __package__ is None or __package__ == '':
    from base import BaseGraphShallowTree
    from concept import Concept
else:
    from .base import BaseGraphShallowTree
    from .concept import Concept


@BaseGraphShallowTree.localize_namespace
class Relation(BaseGraphShallowTree):
    @classmethod
    def name(cls):  # complicated to use class property, just function
        return cls.__name__

    def __init__(self, src, dst, argument_name):
        cls = type(self)
        name = '{}-{}-{}-{}'.format(src.name,
                                    cls.name(), argument_name, dst.name)
        BaseGraphShallowTree.__init__(self, name)
        self.src = src
        self.dst = dst
        src._out.setdefault(cls.name(), []).append(self)
        dst._in.setdefault(cls.name(), []).append(self)

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        if isinstance(src, Concept):
            self._src = src
        else:
            raise TypeError('Unsupported type of relation src {} in relation. Supported type is Concept.'
                            .format(type(src)))

    @property
    def dst(self):
        return self._dst

    @dst.setter
    def dst(self, dst):
        self._dst = dst

    def what(self):
        return {'src': self.src, 'dst': self.dst}

    def T(self, dst, prop, hops=0):  # default to no more hop because this is a hop already
        src = self.src
        src_val = src[prop, hops]  # ( batch, slen(...srank), v(...) )
        # dst_val = dst[prop, 0] # ( batch, dlen(...drank), v(...) ) # TODO: hops need? I need the shape of this prop
        dshape = dst.vshape(prop)
        ret_vals = []
        ret_confs = []
        offset = 1  # 0 for batch
        srank = src.rank
        drank = dst.rank
        v = len(dshape) - 1 - drank
        for i, cur in self.dst:
            if cur is dst:
                # use offset, get a new view of src_val
                #
                #  ( batch, sl(...sr), v(...) )
                #  |
                #  V
                #  ( sl(1...o-1), sl(o+dr...sr), batch, sl(o...o+dr-1), v(...) )
                #
                new_axis = (range(1, offset),
                            range(offset + drank, srank + 1),
                            (0,),
                            range(offset, offset + drank),
                            range(srank + 1, srank + 1 + v))
                new_axis = list(chain(*new_axis))
                src_view = src.b.transpose(src.b.copy(src_val), new_axis)

                new_shape = ((-1,), dshape)
                new_shape = list(chain(*new_shape))
                src_view = src.b.reshape(src_view, new_shape)

                ret_vals.append(self._T(src_view))
                ret_confs.append(None)  # TODO: just... how?
            offset += cur.rank
        ret_vals = src.b.concatenate(*ret_vals)
        return ret_vals, ret_confs

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _T(self, src_view): pass


class OTORelation(Relation):
    pass


class OTMRelation(Relation):
    pass


@Concept.relation_type('is_a')
class IsA(OTORelation):
    def _T(self, src_view):
        # identical transform
        return src_view


@Concept.relation_type('not_a')
class NotA(OTORelation):
    def _T(self, src_view):
        # identical transform
        return not src_view


@Concept.relation_type('has_a')
class HasA(OTORelation):
    pass


@Concept.relation_type('has_many')
class HasMany(OTMRelation):
    pass


@Concept.relation_type('contains')
class Contains(OTMRelation):
    pass
