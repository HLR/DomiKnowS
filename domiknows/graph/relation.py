import abc
from collections import OrderedDict
from itertools import chain, permutations
from domiknows.graph.logicalConstrain import nandL

import torch

if __package__ is None or __package__ == '':
    from base import BaseGraphTree
    from concept import Concept
else:
    from .base import BaseGraphTree
    from .concept import Concept


class Transformed():
    def __init__(self, relation, property, fn=None):
        self.relation = relation
        if isinstance(property, (str, Relation)):
            property = self.relation.src[property]
        self.property = property
        self.fn = fn

    def __call__(self, data_item, device=None):
        value = self.property(data_item)
        try:
            mapping = self.relation.dst[self.relation](data_item)
        except KeyError:
            mapping = self.relation.src[self.relation.reversed](data_item).T
        mapping = mapping.to(dtype=torch.float, device=device)
        value = value.to(dtype=torch.float, device=device)
        if self.fn is None:
            return mapping.matmul(value)
        # mapping (N,M)
        # value (M,...)
        mapping = mapping.view(*(mapping.shape + (1,)*(len(value.shape)-1)))  # (N,M,...)
        value = value.unsqueeze(dim=0)  # (1,M,...)
        return self.fn(mapping * value)


@BaseGraphTree.localize_namespace
class Relation(BaseGraphTree):
    @classmethod
    def name(cls):  # complicated to use class property, just function
        return cls.__name__

    def __init__(self, src, dst, argument_name, reverse_of=None, auto_constraint=None):
        cls = type(self)
        if isinstance(argument_name, str):
            name = argument_name
        else:
            name = '{}-{}-{}-{}'.format(src.name, cls.name(), argument_name, dst.name)
        BaseGraphTree.__init__(self, name)
        self.src = src
        self.dst = dst
        if reverse_of is None:
            self.is_reversed = False
            src._out.setdefault(cls.name(), []).append(self)
            dst._in.setdefault(cls.name(), []).append(self)
            reverse_of = Relation(dst, src, f'{name}.reversed', self)
        else:
            self.is_reversed = True
        self.reversed = reverse_of
        self.auto_constraint = auto_constraint

    @property
    def mode(self):
        return 'backward' if self.is_reversed else 'forward'

    @property
    def auto_constraint(self):
        if self._auto_constraint is None and self.sup is not None:
            return self.sup.auto_constraint
        return self._auto_constraint or False  # if None, return False instead

    @auto_constraint.setter
    def auto_constraint(self, value):
        self._auto_constraint = value

    def __call__(self, *props, fn=None):
        return Transformed(self, props[0], fn=fn)
        # TODO: support mapping multiple props together?
        # for prop in props:
        #     assert prop.sup == self.src
        #     yield Transformed(self.relation, prop, fn=fn)

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

    __metaclass__ = abc.ABCMeta

    def set_apply(self, name, sub):
        from ..sensor import Sensor
        from .property import Property
        if isinstance(sub, Property):
            # call usually come from attach, further from constructor of property
            BaseGraphTree.set_apply(self, name, sub)
        elif isinstance(sub, Sensor):
            if name not in self:
                with self:
                    prop = Property(prop_name=name)
            self.get_apply(name).attach(sub)


class OTORelation(Relation):
    pass


class OTMRelation(Relation):
    pass


class MTORelation(Relation):
    pass


class MTMRelation(Relation):
    pass


@Concept.relation_type('is_a')
class IsA(OTORelation):
    pass


@Concept.relation_type('not_a')
class NotA(OTORelation):
    def __init__(self, src, dst, *args, **kwargs):
        super().__init__(src, dst, *args, **kwargs)
        nandL(src, dst)


def disjoint(*concepts):
    rels = []
    for c1, c2 in permutations(concepts, r=2):
        rels.extend(c1.not_a(c2))
    return rels


@Concept.relation_type('has_a')
class HasA(MTORelation):
    pass


@Concept.relation_type('has_many')
class HasMany(OTMRelation):
    pass

@Concept.relation_type('contains')
class Contains(OTMRelation):
    pass

@Concept.relation_type('equal')
class Equal(OTORelation):
    pass
