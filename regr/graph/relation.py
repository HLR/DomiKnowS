import abc
from collections import OrderedDict
from itertools import chain, permutations

import torch

if __package__ is None or __package__ == '':
    from base import BaseGraphTree
    from concept import Concept
else:
    from .base import BaseGraphTree
    from .concept import Concept


class Transformed():
    def __init__(self, relationfunction, property, fn=None):
        self.relationfunction = relationfunction
        if isinstance(property, (str, RelationFunction)):
            property = self.relationfunction.src[property]
        self.property = property
        self.fn = fn

    def __call__(self, data_item, device=-1):
        value = self.property(data_item)
        try:
            mapping = self.relationfunction.dst[self.relationfunction](data_item)
        except KeyError:
            mapping = self.relationfunction.src[self.relationfunction.T](data_item).T
        mapping = mapping.to(dtype=torch.float, device=device)
        value = value.to(dtype=torch.float, device=device)
        if self.fn is None:
            return mapping.matmul(value)
        # mapping (N,M)
        # value (M,...)
        mapping = mapping.view(*(mapping.shape + (1,)*(len(value.shape)-1)))  # (N,M,...)
        value = value.unsqueeze(dim=0)  # (1,M,...)
        return self.fn(mapping * value)


class RelationFunction():
    def __init__(self, relation):
        self.relation = relation
        self.mode = "forward"

    @property
    def src(self):
        return self.relation.src

    @property
    def dst(self):
        return self.relation.dst

    @property
    def T(self):
        return self.relation.bacward
    
    @property
    def name(self):
        return f'{self.relation.name}.{self.mode}'

    def __call__(self, *props, fn=None):
        return Transformed(self, props[0], fn=fn)
        # TODO: support mapping multiple props together?
        # for prop in props:
        #     assert prop.sup == self.src
        #     yield Transformed(self.relation, prop, fn=fn)

    def __str__(self):
        return f'{str(self.relation)}.{self.mode}'

    def __repr__(self):
        return f'{repr(self.relation)}.{self.mode}'


class RelationBackwardFunction(RelationFunction):
    def __init__(self, relation):
        super().__init__(relation)
        self.mode = "backward"

    @property
    def src(self):
        return self.relation.dst

    @property
    def dst(self):
        return self.relation.src

    @property
    def T(self):
        return self.relation.forward

@BaseGraphTree.localize_namespace
class Relation(BaseGraphTree):
    @classmethod
    def name(cls):  # complicated to use class property, just function
        return cls.__name__

    def __init__(self, src, dst, argument_name):
        cls = type(self)
        if isinstance(argument_name, str):
            name = argument_name
        else:
            name = '{}-{}-{}-{}'.format(src.name, cls.name(), argument_name, dst.name)
        BaseGraphTree.__init__(self, name)
        self.src = src
        self.dst = dst
        src._out.setdefault(cls.name(), []).append(self)
        dst._in.setdefault(cls.name(), []).append(self)
        self.forward = RelationFunction(self)
        self.backward = RelationBackwardFunction(self)

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
    pass


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
